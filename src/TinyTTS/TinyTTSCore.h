#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "TinyTTS/Alignment.h"
#include "TinyTTS/CmuDict.h"
#include "TinyTTS/Flow.h"
#include "TinyTTS/DictionaryModel.h"
#include "TinyTTS/Mat.h"
#include "TinyTTS/PhonemeEncoder.h"
#include "TinyTTS/TextG2P.h"
#include "TinyTTS/WeightStore.h"

namespace tinytts {

/// Metadata about a completed synthesize() call -- NOT the audio itself.
/// Audio is streamed out via the AudioChunkFn callback as each decoder chunk
/// is ready, never buffered as a whole utterance (an utterance's audio can be
/// large; there's no reason to hold all of it in RAM when the caller is
/// just going to stream it to I2S/a file/etc. as it arrives).
struct SynthesisInfo {
  int y_len = 0;  // total vocoder frames synthesized
  std::vector<int> durations;
};

/**
 * @brief Public API for TinyTTS: text -> speech, on an ESP32-S3.
 *
 * Two of the four model stages (text_encoder, flow) run as hand-written C++
 * here (Attention/TransformerBlock/PhonemeEncoder/Flow) -- the ONNX->TFLite
 * conversion path (onnx2tf) cannot handle their windowed relative-position
 * attention module, so they're implemented directly instead. The other two
 * stages (duration_predictor, decoder/vocoder) convert to TFLite cleanly and
 * run through TFLite Micro; this class does not link TFLM itself, it takes
 * those two stages as callbacks (`setDurationPredictor`/`setDecoder`) so
 * TinyTTS.h has no hard dependency on any particular TFLM binding -- the
 * sketch wires up the actual `tflm_esp32` interpreters and passes them in.
 *
 * The decoder runs in fixed-size frame windows (`decoderChunkFrames`, see
 * setDecoderChunkFrames()), not one call over the whole utterance: TFLite
 * Micro's MicroInterpreter has no input-resize API, so a TFLM-backed decoder
 * is necessarily a fixed shape. This also naturally gives streaming
 * synthesis -- audio for the first chunk is available (and can start
 * playing) well before the whole utterance has been generated.
 *
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class TinyTTSCore {
 public:
  /// x: [T_x, hidden] phoneme encoder hidden state, g: [1, gin_channels]
  /// speaker embedding -> logw: [T_x] (one log-duration value per phoneme).
  using DurationPredictorFn = std::function<std::vector<float>(const Mat& x, const Mat& g)>;

  /// z: [decoderChunkFrames, inter_channels] (always exactly this many rows
  /// -- zero-padded for a short final chunk, see synthesize()), g: [1,
  /// gin_channels] speaker embedding -> interleaved PCM samples for that
  /// chunk.
  using DecoderFn = std::function<std::vector<float>(const Mat& z, const Mat& g)>;

  /// Called once per decoder chunk, with that chunk's PCM samples, as soon
  /// as they're ready -- e.g. write them straight to an I2SStream. Never
  /// called with a whole-utterance buffer.
  using AudioChunkFn = std::function<void(const float* samples, size_t count)>;

  /// Loads model weights and the CMU dictionary from in-memory buffers
  /// (flash-embedded const arrays, or buffers read from LittleFS/SD at
  /// startup -- this class doesn't care which). Both buffers must outlive
  /// this object. Returns false if either buffer is malformed/truncated.
  ///
  /// `dictionary_model_buf`/`dictionary_model_len` are optional
  /// (research/export_dictionary_model.py's dictionary_model.bin) -- if
  /// given, DictionaryModel handles words not in the dictionary instead of
  /// the crude character-level fallback. Must also outlive this object.
  bool begin(const uint8_t* weights_buf, size_t weights_len, const uint8_t* cmudict_buf, size_t cmudict_len,
             int n_heads = 2, int window_size = 4, int n_flows = 4,
             const uint8_t* dictionary_model_buf = nullptr, size_t dictionary_model_len = 0) {
    if (!weights_.begin(weights_buf, weights_len)) return false;
    if (!cmudict_.begin(cmudict_buf, cmudict_len)) return false;
    encoder_.begin(weights_, n_heads, window_size);
    flow_.begin(weights_, n_flows, n_heads, window_size);

    const DictionaryModel* dictionary_model_ptr = nullptr;
    if (dictionary_model_buf != nullptr && dictionary_model_len > 0) {
      if (!dictionary_model_.begin(dictionary_model_buf, dictionary_model_len)) return false;
      dictionary_model_ptr = &dictionary_model_;
    }
    g2p_.begin(cmudict_, dictionary_model_ptr);

    const WeightStore::Entry* emb_g = weights_.get("emb_g.weight");
    gin_channels_ = emb_g ? emb_g->shape[1] : 0;
    emb_g_ = weights_.embedding("emb_g.weight");
    started_ = true;
    return true;
  }

  void setDurationPredictor(DurationPredictorFn fn) { duration_predictor_ = std::move(fn); }
  void setDecoder(DecoderFn fn) { decoder_ = std::move(fn); }

  /// Must match the fixed frame count the decoder TFLite model was built
  /// for (e.g. 96 -- see research/'s -ois export). Default 96 matches this
  /// project's own reference export; set explicitly if yours differs.
  void setDecoderChunkFrames(int frames) { decoder_chunk_frames_ = frames; }

  /// Samples per vocoder frame (HOP_LENGTH in tiny_tts.utils.config) --
  /// needed to trim the zero-padded tail of a short final chunk down to its
  /// real sample count.
  void setHopLength(int hop_length) { hop_length_ = hop_length; }

  bool isReady() const { return started_ && duration_predictor_ && decoder_; }

  /// Synthesizes `text` for speaker `speaker_id` (index into the model's
  /// speaker embedding table; 0 for single-speaker models), streaming audio
  /// out through `on_audio` one decoder chunk at a time. noise_scale
  /// controls the flow's stochasticity (0 = deterministic), length_scale
  /// stretches/compresses durations (1 = normal speed), rng_seed seeds the
  /// noise sampling (same seed -> same output, for reproducible testing).
  SynthesisInfo synthesize(const std::string& text, const AudioChunkFn& on_audio, int speaker_id = 0,
                            float noise_scale = 0.667f, float length_scale = 1.0f, uint32_t rng_seed = 0) const {
    SynthesisInfo info;
    if (!isReady()) return info;

    // Two separate blank-insertion steps, matching the reference exactly:
    // g2p_.process() pads the phrase with a single '_' at each end (its own
    // pad_start_end step), then insertBlanks() interleaves a '_' between
    // EVERY symbol on top of that (commons.insert_blanks) -- both stages
    // are real and both are required, not redundant.
    G2POutput g2p_out = g2p_.process(text);
    std::vector<int> phone_ids = TextG2P::insertBlanks(g2p_out.phone_ids);
    std::vector<int> tone_ids = TextG2P::insertBlanks(g2p_out.tone_ids);
    std::vector<int> language_ids = TextG2P::insertBlanks(g2p_out.language_ids);

    Mat g(1, gin_channels_);
    for (int c = 0; c < gin_channels_; c++) g.at(0, c) = emb_g_.at(speaker_id, c);

    PhonemeEncoderOutput enc_out = encoder_.forward(phone_ids, tone_ids, language_ids, g);

    std::vector<float> logw = duration_predictor_(enc_out.x, g);
    std::vector<int> durations = alignment::durationsFromLogw(logw, length_scale);
    int t_y = alignment::totalDuration(durations);

    Mat m_p_exp = alignment::expandByDuration(enc_out.m_p, durations, t_y);
    Mat logs_p_exp = alignment::expandByDuration(enc_out.logs_p, durations, t_y);

    std::mt19937 rng(rng_seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Mat z_p(t_y, m_p_exp.cols());
    for (int t = 0; t < t_y; t++)
      for (int c = 0; c < m_p_exp.cols(); c++)
        z_p.at(t, c) = m_p_exp.at(t, c) + normal(rng) * std::exp(logs_p_exp.at(t, c)) * noise_scale;

    Mat z = flow_.reverse(z_p, g);

    // Decode in fixed-size windows, streaming each chunk's audio out as
    // soon as it's ready (see class doc for why: TFLM has no resize API).
    int chunk_frames = decoder_chunk_frames_;
    for (int start = 0; start < t_y; start += chunk_frames) {
      int valid = std::min(chunk_frames, t_y - start);
      Mat z_chunk(chunk_frames, z.cols(), 0.0f);  // zero-padded if this is a short final chunk
      for (int t = 0; t < valid; t++) std::copy(z.row(start + t), z.row(start + t) + z.cols(), z_chunk.row(t));

      std::vector<float> audio = decoder_(z_chunk, g);
      size_t valid_samples = (size_t)valid * hop_length_;
      if (valid_samples > audio.size()) valid_samples = audio.size();  // defensive, shouldn't happen
      if (on_audio) on_audio(audio.data(), valid_samples);
    }

    info.durations = std::move(durations);
    info.y_len = t_y;
    return info;
  }

  // ---- component access, for testing / advanced use ----
  const PhonemeEncoder& encoder() const { return encoder_; }
  const Flow& flow() const { return flow_; }
  const TextG2P& g2p() const { return g2p_; }
  const WeightStore& weights() const { return weights_; }

 private:
  WeightStore weights_;
  CmuDict cmudict_;
  DictionaryModel dictionary_model_;
  PhonemeEncoder encoder_;
  Flow flow_;
  TextG2P g2p_;
  Mat emb_g_;
  int gin_channels_ = 0;
  int decoder_chunk_frames_ = 96;
  int hop_length_ = 512;
  bool started_ = false;
  DurationPredictorFn duration_predictor_;
  DecoderFn decoder_;
};

}  // namespace tinytts
