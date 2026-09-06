#pragma once
#include <cmath>
#include <string>
#include <vector>

#include "TinyTTS/DataBuffer.h"
#include "TinyTTS/TinyTTSCore.h"

#ifdef ARDUINO
#include <Print.h>
#endif

namespace tinytts {

/**
 * @brief TinyTTS: text -> speech for Microcontrollers, the simple way.
 *
 *   #include <TinyTTS.h>
 *   #include "weights_data.h"       // your data -- see below
 *   #include "cmudict_data.h"
 *
 *   tinytts::TinyTTS tts;
 *   tts.setWeights(default_weights, default_weights_len);
 *   tts.setDictionary(default_cmudict, default_cmudict_len);
 *   tts.begin(i2s_out);   // i2s_out: any Print (e.g. audio_tools::I2SStream)
 *   tts.speak("Hello world!");
 *   tts.end();
 *
 * TinyTTS does NOT embed a default model itself -- providing the two data
 * buffers (weights, dictionary) is the sketch's responsibility, via the
 * setters below, called before begin(). `research/`'s export scripts (see
 * docs/research.md) produce ready-to-use headers; `examples/tts_i2s_output/`
 * includes and wires up a full set. Keeping this out of the library itself
 * means a sketch that doesn't need the default dataset doesn't pay for it,
 * and a sketch that wants a different checkpoint/dictionary doesn't need to
 * fight a built-in default to replace it.
 *
 * Each setter has two forms:
 *   - `(const uint8_t* data, size_t len)`: BORROWED -- e.g. a flash const
 *     array. `data` must outlive this TinyTTS object; nothing is copied or
 *     freed.
 *   - `(File& file)`: OWNED -- reads the whole file into a new buffer this
 *     object allocates and frees automatically (via DataBuffer's
 *     unique_ptr -- no manual cleanup, no leak risk even across repeated
 *     calls; see DataBuffer.h for where that buffer comes from). Use this
 *     to load from LittleFS/FFat/SD instead of compiling data into flash.
 *
 * All four model stages (text_encoder, flow, duration_predictor, decoder)
 * are hand-written C++ (TinyTTSCore.h's doc has the full breakdown) -- no
 * TFLite Micro, no external inference-runtime dependency at all, so this
 * class is just a thin Print/File convenience wrapper around TinyTTSCore.
 * Only the File-based setters, the Print-taking constructor, and the
 * Print-based begin() overloads stay ARDUINO-only, since File/Print have no
 * host equivalent; use begin(const AudioChunkFn&) directly in a host build.
 *
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class TinyTTS {
 public:
  TinyTTS() = default;

#ifdef ARDUINO
  /// Remembers `output` for the no-arg begin() below, so you don't have to
  /// pass it again at the call site.
  TinyTTS(Print& output) : output_(&output) {}
#endif

  ~TinyTTS() { end(); }

  // ---- model data: call before begin() ----

  void setWeights(const uint8_t* data, size_t len) {
    weights_.setBorrowed(data, len);
  }

  void setDictionary(const uint8_t* data, size_t len) {
    dict_.setBorrowed(data, len);
  }

  /**
   * Optional: the neural G2P fallback model
   * (research/export_dictionary_model.py's dictionary_model.bin) for words
   * not in the dictionary. Unlike the two setters above, this one may be
   * left unset -- begin() then falls back to TextG2P's crude
   * character-level fallback for out-of-dictionary words instead.
   */
  void setDictionaryModel(const uint8_t* data, size_t len) {
    dictionary_model_.setBorrowed(data, len);
  }

#ifdef ARDUINO
  bool setWeights(File& file) { return weights_.loadFromFile(file); }
  bool setDictionary(File& file) { return dict_.loadFromFile(file); }
  bool setDictionaryModel(File& file) {
    return dictionary_model_.loadFromFile(file);
  }
#endif

  void setSpeakerId(int speaker_id) { speaker_id_ = speaker_id; }
  void setNoiseScale(float noise_scale) { noise_scale_ = noise_scale; }
  void setLengthScale(float length_scale) { length_scale_ = length_scale; }

  /// How many participants (calling thread + workers) the decoder's hot
  /// conv1d()/convTranspose1d() loops split across -- see
  /// ops::setNumWorkers()'s doc (Ops.h) for the full explanation, most
  /// importantly: this must be called before the first speak()/synthesize()
  /// call, since the underlying worker pool is created lazily on first use
  /// and persists for the process/device lifetime -- calling this after
  /// synthesis has already started has no effect. No-op on a build without
  /// arduino-audio-tools' Task support available (single-core fallback).
  void setNumWorkers(int n) { ops::setNumWorkers(n); }

  /// Scales the PCM data written to the Print output (begin(Print&)/the
  /// no-arg begin()) -- 1.0 is the model's native (full-scale) loudness,
  /// 0.0 is silence. Perceived loudness is logarithmic, not linear (see
  /// volumeToGain() below), so this value is mapped through a dB taper
  /// rather than multiplied onto the samples directly: a straight linear
  /// gain of 0.2 is still only about -14dB, which barely sounds quieter
  /// than full volume to the ear -- confirmed uncomfortably loud on real
  /// hardware even at that setting. Default 0.4, log-scaled (confirmed a
  /// comfortable level on real hardware at this setting). Doesn't
  /// affect begin(const AudioChunkFn&)'s raw float samples -- those stay
  /// the model's true, unscaled output.
  void setVolume(float volume) { volume_ = volume; }

  // ---- output PCM format produced by speak() -- always mono/16-bit at the
  // model's native rate, so callers can configure their audio output (e.g.
  // I2SStream) directly from these instead of duplicating the values. ----
  int getAudioChannels() const { return 1; }
  int getAudioBitsPerSample() const { return 16; }
  int getAudioSampleRate() const { return sample_rate_; }
  void setAudioSampleRate(int sample_rate) { sample_rate_ = sample_rate; }

  /**
   * Public callback type: called once with the whole utterance's raw float
   * PCM samples (range [-1,1]) once synthesis finishes. Same shape as
   * TinyTTSCore::AudioChunkFn, just re-exposed here so a caller doesn't need
   * to reach into core().
   */
  using AudioChunkFn = TinyTTSCore::AudioChunkFn;

  /**
   * Loads the model data set via the setters above (BOTH weights and
   * dictionary must have been set, or this fails) and stores `on_audio` for
   * speak() to hand the synthesized audio to. Returns false on failure
   * (check Serial for the reason).
   *
   * This is the portable entry point -- no Print/Arduino type in its
   * signature, so it (and everything above it) can run in a plain host
   * build for testing. begin(Print&) below is a thin, ARDUINO-only
   * convenience wrapper around this.
   */
  bool begin(const AudioChunkFn& on_audio) {
    if (!weights_.valid() || !dict_.valid()) {
      return false;  // see class doc: call the set*() methods before begin()
    }
    audio_callback_ = on_audio;

    if (!core_.begin(weights_.data(), weights_.size(), dict_.data(),
                     dict_.size(), /*n_heads=*/2,
                     /*window_size=*/4, /*n_flows=*/4, dictionary_model_.data(),
                     dictionary_model_.size()))
      return false;

    started_ = true;
    return true;
  }

#ifdef ARDUINO

  /// Uses the `Print` given to the TinyTTS(Print&) constructor instead of
  /// taking one here -- for when that constructor was used. Returns false
  /// if it wasn't.
  bool begin() {
    if (output_ == nullptr && audio_callback_ == nullptr) return false;
    return begin(*output_);
  }

  /**
   * Arduino convenience overload: streams PCM (int16, clipped) to `output`
   * via Print::write() instead of handing you raw float samples. Just
   * begin(AudioChunkFn) above wired to writeAudio() -- see that overload
   * for what "portable" means here.
   */
  bool begin(Print& output) {
    output_ = &output;
    return begin([this](const float* samples, size_t count) {
      writeAudio(samples, count);
    });
  }
#endif

  /// @brief  Defines the callback to be used for audio output. Must be called before begin().
  /// @param on_audio 
  void setCallback(const AudioChunkFn& on_audio) { audio_callback_ = on_audio; }

  /**
   * Synthesizes `text` and hands the resulting PCM to the callback given to
   * begin(). Returns false if begin() wasn't called successfully.
   */
  bool speak(const std::string& text) {
    if (!started_) return false;
    core_.synthesize(text, audio_callback_, speaker_id_, noise_scale_,
                     length_scale_);
    return true;
  }

  /**
   * Marks this object as no longer ready to speak(). The model data set via
   * the set*() methods is NOT cleared -- a subsequent begin() reuses it
   * without needing to be set again.
   */
  void end() {
#ifdef ARDUINO
    output_ = nullptr;
#endif
    started_ = false;
    audio_callback_ = nullptr;
  }

  /**
   * Direct access to the underlying orchestration/model layer, for
   * advanced use (introspecting durations, using a different audio sink
   * than a Print, etc).
   */
  TinyTTSCore& core() { return core_; }

 private:
#ifdef ARDUINO
  // Maps a 0..1 volume setting to a linear gain via a dB taper -- see
  // setVolume()'s doc for why (perceived loudness is logarithmic).
  // kDynamicRangeDb=40 means volume=0 -> -40dB (near-silent, not a hard
  // mute) and volume=1 -> 0dB (unity gain, the model's native loudness).
  static float volumeToGain(float volume) {
    if (volume <= 0.0f) return 0.0f;
    if (volume >= 1.0f) return 1.0f;
    constexpr float kDynamicRangeDb = 40.0f;
    return std::pow(10.0f, (volume - 1.0f) * kDynamicRangeDb / 20.0f);
  }

  void writeAudio(const float* samples, size_t count) {
    if (!output_) return;
    pcm_buf_.resize(count);
    float gain = volumeToGain(volume_);
    for (size_t i = 0; i < count; i++) {
      float s = samples[i] * gain;
      if (s > 1.0f) s = 1.0f;
      if (s < -1.0f) s = -1.0f;
      pcm_buf_[i] = (int16_t)(s * 32767.0f);
    }
    output_->write((const uint8_t*)pcm_buf_.data(),
                   pcm_buf_.size() * sizeof(int16_t));
  }
#endif  // ARDUINO

  DataBuffer<> weights_;
  DataBuffer<> dict_;
  DataBuffer<> dictionary_model_;  // optional -- see setDictionaryModel()

  int speaker_id_ = 0;
  float noise_scale_ = 0.667f;
  float length_scale_ = 1.0f;
  float volume_ = 0.4f;
  int sample_rate_ = 44100;  // tiny_tts.utils.config.SAMPLING_RATE

#ifdef ARDUINO
  Print* output_ = nullptr;
  std::vector<int16_t> pcm_buf_;
#endif

  TinyTTSCore core_;
  AudioChunkFn audio_callback_ = nullptr;
  bool started_ = false;
};

}  // namespace tinytts

/// Arduino sketch can `#include <TinyTTS.h>` and use the class directly without
#ifdef ARDUINO
using namespace tinytts;
#endif
