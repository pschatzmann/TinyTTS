#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>

#ifdef ARDUINO
#include <Arduino.h>
#endif

#include "TinyTTS/Alignment.h"
#include "TinyTTS/CmuDict.h"
#include "TinyTTS/Decoder.h"
#include "TinyTTS/DurationPredictor.h"
#include "TinyTTS/Flow.h"
#include "TinyTTS/DictionaryModel.h"
#include "TinyTTS/Mat.h"
#include "TinyTTS/PhonemeEncoder.h"
#include "TinyTTS/TextG2P.h"
#include "TinyTTS/WeightStore.h"

namespace tinytts {

/// Metadata about a completed synthesize() call -- NOT the audio itself.
struct SynthesisInfo {
  int y_len = 0;  // total vocoder frames synthesized
  std::vector<int> durations;
};

/**
 * @brief Public API for TinyTTS: text -> speech, on an ESP32-S3.
 *
 * All four model stages run as hand-written C++ here:
 *   - text_encoder/flow (PhonemeEncoder/Flow): attention-bearing, and the
 *     ONNX->TFLite conversion path (onnx2tf) can't handle their windowed
 *     relative-position attention module at all.
 *   - duration_predictor (DurationPredictor): plain Conv1d/ChannelNorm/ReLU,
 *     converts to TFLite cleanly on its own, but is hand-written anyway --
 *     reusing the same primitives (Ops.h) once the attention path already
 *     needed them -- because doing so removes TFLite Micro's
 *     fixed-input-shape limitation for this stage entirely.
 *   - decoder (Decoder): the HiFi-GAN-style vocoder, likewise plain
 *     Conv1d/ConvTranspose1d/LeakyReLU (no attention) and hand-written for
 *     the same reason.
 *
 * There is no TFLite Micro (or any other inference-runtime) dependency
 * anywhere in this class -- see docs/architecture.md for the full mechanical
 * breakdown of each stage and why it ended up this way.
 *
 * Because none of the four stages has a TFLite Micro fixed-input-shape
 * constraint, synthesize() runs the whole utterance through in a single
 * pass, then hands the complete PCM buffer to `on_audio` once -- there is no
 * chunk-by-chunk streaming-while-decoding here (that was a side effect of
 * TFLM's fixed-shape decoder needing a window in the first place, not an
 * independent design goal).
 *
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class TinyTTSCore {
 public:
  /// Called once, with the whole utterance's PCM samples, when synthesis
  /// finishes -- e.g. write them straight to an I2SStream.
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
    duration_predictor_.begin(weights_);
    decoder_.begin(weights_);

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

  bool isReady() const { return started_; }

  /// Synthesizes `text` for speaker `speaker_id` (index into the model's
  /// speaker embedding table; 0 for single-speaker models), handing the
  /// whole utterance's PCM to `on_audio` once synthesis completes.
  /// noise_scale controls the flow's stochasticity (0 = deterministic),
  /// length_scale stretches/compresses durations (1 = normal speed),
  /// rng_seed seeds the noise sampling (same seed -> same output, for
  /// reproducible testing).
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

#ifdef ARDUINO
    uint32_t t_enc0 = millis();
#endif
    PhonemeEncoderOutput enc_out = encoder_.forward(phone_ids, tone_ids, language_ids, g);
#ifdef ARDUINO
    Serial.printf("[TinyTTS] encoder: %lu ms\n", millis() - t_enc0);
    uint32_t t_dp0 = millis();
#endif

    std::vector<float> logw = duration_predictor_.forward(enc_out.x, g);
    std::vector<int> durations = alignment::durationsFromLogw(logw, length_scale);
    int t_y = alignment::totalDuration(durations);
#ifdef ARDUINO
    Serial.printf("[TinyTTS] duration_predictor: %lu ms, t_y=%d\n", millis() - t_dp0, t_y);
#endif

    Mat m_p_exp = alignment::expandByDuration(enc_out.m_p, durations, t_y);
    Mat logs_p_exp = alignment::expandByDuration(enc_out.logs_p, durations, t_y);

    std::mt19937 rng(rng_seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Mat z_p(t_y, m_p_exp.cols());
    for (int t = 0; t < t_y; t++)
      for (int c = 0; c < m_p_exp.cols(); c++)
        z_p.at(t, c) = m_p_exp.at(t, c) + normal(rng) * std::exp(logs_p_exp.at(t, c)) * noise_scale;

#ifdef ARDUINO
    uint32_t t_flow0 = millis();
#endif
    Mat z = flow_.reverse(z_p, g);
#ifdef ARDUINO
    Serial.printf("[TinyTTS] flow: %lu ms\n", millis() - t_flow0);
    uint32_t t_dec0 = millis();
#endif
    auto audio = decoder_.forward(z, g);
#ifdef ARDUINO
    Serial.printf("[TinyTTS] decoder: %lu ms, samples=%u\n", millis() - t_dec0, (unsigned)audio.size());
#endif
    if (on_audio) on_audio(audio.data(), audio.size());

    info.durations = std::move(durations);
    info.y_len = t_y;
    return info;
  }

  // ---- component access, for testing / advanced use ----
  const PhonemeEncoder& encoder() const { return encoder_; }
  const Flow& flow() const { return flow_; }
  const DurationPredictor& durationPredictor() const { return duration_predictor_; }
  const Decoder& decoder() const { return decoder_; }
  const TextG2P& g2p() const { return g2p_; }
  const WeightStore& weights() const { return weights_; }

 private:
  WeightStore weights_;
  CmuDict cmudict_;
  DictionaryModel dictionary_model_;
  PhonemeEncoder encoder_;
  Flow flow_;
  DurationPredictor duration_predictor_;
  Decoder decoder_;
  TextG2P g2p_;
  Mat emb_g_;
  int gin_channels_ = 0;
  bool started_ = false;
};

}  // namespace tinytts
