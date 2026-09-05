#pragma once
#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "TinyTTS/TFLite.h"
#include "TinyTTS/DataBuffer.h"
#include "TinyTTS/TinyTTSCore.h"

namespace tinytts {

/**
 * @brief Default allocator TinyTTS uses for its arenas and TFLite Micro
 * interpreter objects: on ARDUINO, explicit PSRAM (heap_caps_malloc +
 * MALLOC_CAP_SPIRAM); off ARDUINO (host/test builds), plain malloc/free, so
 * the exact same allocation/ownership logic (arenas, placement-new'd
 * interpreters, the matching deleters) is compiled and exercised there too
 * -- under a sanitizer if you like -- instead of only being provable by
 * flashing real hardware. Pass a different type as TinyTTS's template
 * argument to use a different allocator (e.g. plain internal RAM instead of
 * PSRAM, or a pooled/tracked allocator for testing); it just needs static
 * `allocate(size_t)`/`deallocate(void*)` methods with these signatures.
 */
struct PsramAllocator {
  static void* allocate(size_t n) {
#ifdef ESP32
    return heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
#else
    return malloc(n);
#endif
  }
  static void deallocate(void* p) {
#ifdef ESP32
    heap_caps_free(p);
#else
    free(p);
#endif
  }
};

/**
 * @brief TinyTTS: text -> speech on an ESP32-S3, the simple way.
 *
 *   #include <TinyTTS.h>
 *   #include "weights_data.h"       // your data -- see below
 *   #include "cmudict_data.h"
 *   #include "dp_model_data.h"
 *   #include "decoder_model_data.h"
 *
 *   tinytts::TinyTTS<> tts;
 *   tts.setWeights(default_weights, default_weights_len);
 *   tts.setDictionary(default_cmudict, default_cmudict_len);
 *   tts.setDurationPredictorModel(default_duration_predictor_model, default_duration_predictor_model_len, 32);
 *   tts.setDecoderModel(default_decoder_model, default_decoder_model_len, 96);
 *   tts.begin(i2s_out);              // i2s_out: any Print (e.g. audio_tools::I2SStream)
 *   tts.speak("Hello world!");
 *   tts.end();
 *
 * TinyTTS does NOT embed a default model itself -- providing the four data
 * buffers (weights, dictionary, duration_predictor, decoder) is the
 * sketch's responsibility, via the setters below, called before begin().
 * `research/`'s export scripts (see docs/research.md) produce ready-to-use
 * headers; `examples/tts_i2s_output/` includes and wires up a full set.
 * Keeping this out of the library itself means a sketch that doesn't need
 * the default ~7MB dataset doesn't pay for it, and a sketch that wants a
 * different checkpoint/quantization/dictionary doesn't need to fight a
 * built-in default to replace it.
 *
 * Each setter has two forms:
 *   - `(const uint8_t* data, size_t len, ...)`: BORROWED -- e.g. a flash
 *     const array. `data` must outlive this TinyTTS object; nothing is
 *     copied or freed.
 *   - `(File& file, ...)`: OWNED -- reads the whole file into a new PSRAM
 *     buffer this object allocates and frees automatically (via
 *     DataBuffer's unique_ptr -- no manual cleanup, no leak risk even
 *     across repeated calls). Use this to load from LittleFS/FFat/SD
 *     instead of compiling data into flash.
 *
 * Internally owns two TFLite Micro interpreters (duration_predictor,
 * decoder) and the hand-written C++ text_encoder/flow (TinyTTSCore) -- see
 * TinyTTSCore.h's doc for why those two are hand-written rather than TFLite
 * graphs. On ARDUINO builds the TFLM types come from `tflm_esp32`
 * (precompiled for ESP32-S3); on a plain host build they come from a
 * fetched copy of the real TFLite Micro source (see
 * test/cmake/FetchTFLiteMicro.cmake) instead -- same `tflite::` API either
 * way, so this whole class compiles and runs on both. That's what makes it
 * possible to test the exact interpreter-build/arena/deleter logic here --
 * under ASan if you like -- rather than only being provable by flashing
 * real hardware. Only the File-based setters and the Print-based begin()
 * overload stay ARDUINO-only, since File/Print have no host equivalent;
 * use begin(const AudioChunkFn&) directly in a host build.
 *
 * Templated on `Allocator` (default PsramAllocator, see its own doc above)
 * so arenas/interpreter objects can go somewhere other than PSRAM if you
 * need that -- most sketches just want `TinyTTS<>`.
 *
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
template <typename Allocator = PsramAllocator>
class TinyTTS {
 public:
  // ---- model data: call before begin() ----

  void setWeights(const uint8_t* data, size_t len) { weights_.setBorrowed(data, len); }
  void setDictionary(const uint8_t* data, size_t len) { dict_.setBorrowed(data, len); }

  /// `max_phonemes` must match the fixed phoneme-window shape the model was
  /// exported for (see research/'s -ois export).
  void setDurationPredictorModel(const uint8_t* data, size_t len, int max_phonemes) {
    dp_model_.setBorrowed(data, len);
    max_phonemes_ = max_phonemes;
  }

  /// `chunk_frames` must match the fixed frame-window shape the model was
  /// exported for.
  void setDecoderModel(const uint8_t* data, size_t len, int chunk_frames) {
    decoder_model_.setBorrowed(data, len);
    decoder_chunk_frames_ = chunk_frames;
  }

  /// Optional: the neural G2P fallback model
  /// (research/export_dictionary_model.py's dictionary_model.bin) for words
  /// not in the dictionary. Unlike the four setters above, this one may be
  /// left unset -- begin() then falls back to TextG2P's crude
  /// character-level fallback for out-of-dictionary words instead.
  void setDictionaryModel(const uint8_t* data, size_t len) { dictionary_model_.setBorrowed(data, len); }

#ifdef ARDUINO
  bool setWeights(File& file) { return weights_.loadFromFile(file); }
  bool setDictionary(File& file) { return dict_.loadFromFile(file); }
  bool setDurationPredictorModel(File& file, int max_phonemes) {
    max_phonemes_ = max_phonemes;
    return dp_model_.loadFromFile(file);
  }
  bool setDecoderModel(File& file, int chunk_frames) {
    decoder_chunk_frames_ = chunk_frames;
    return decoder_model_.loadFromFile(file);
  }
  bool setDictionaryModel(File& file) { return dictionary_model_.loadFromFile(file); }

#endif

  void setArenaSizes(size_t duration_predictor_bytes, size_t decoder_bytes) {
    dp_arena_size_ = duration_predictor_bytes;
    decoder_arena_size_ = decoder_bytes;
  }

  /// Samples per vocoder frame (HOP_LENGTH in tiny_tts.utils.config).
  void setHopLength(int hop_length) { hop_length_ = hop_length; }

  /// duration_predictor's receptive field, each side (see
  /// runDurationPredictor()'s doc) -- only needed if you swap in a
  /// duration_predictor model with a different conv architecture than this
  /// project's default (two stacked kernel_size=3 layers, margin 2).
  void setDurationPredictorMargin(int margin) { dp_margin_ = margin; }
  void setSpeakerId(int speaker_id) { speaker_id_ = speaker_id; }
  void setNoiseScale(float noise_scale) { noise_scale_ = noise_scale; }
  void setLengthScale(float length_scale) { length_scale_ = length_scale; }

  // ---- output PCM format produced by speak() -- always mono/16-bit at the
  // model's native rate, so callers can configure their audio output (e.g.
  // I2SStream) directly from these instead of duplicating the values. ----
  int getAudioChannels() const { return 1; }
  int getAudioBitsPerSample() const { return 16; }
  int getAudioSampleRate() const { return sample_rate_; }
  /// Only meaningful if you're using a decoder model with a different
  /// native sample rate than the project default (44100).
  void setAudioSampleRate(int sample_rate) { sample_rate_ = sample_rate; }

  /// Public callback type: called once per decoder chunk, with that chunk's
  /// raw float PCM samples (range [-1,1]), as soon as they're ready -- never
  /// with a whole-utterance buffer. Same shape as TinyTTSCore::AudioChunkFn,
  /// just re-exposed here so a caller doesn't need to reach into core().
  using AudioChunkFn = TinyTTSCore::AudioChunkFn;

  /// Wires up the TFLite Micro interpreters from the model data set via the
  /// setters above (ALL FOUR must have been set -- weights, dictionary,
  /// duration_predictor model, decoder model -- or this fails) and stores
  /// `on_audio` for speak() to stream decoder chunks to. Returns false on
  /// failure (check Serial for the reason).
  ///
  /// This is the portable entry point -- no Print/Arduino type in its
  /// signature, so it (and everything above it) can run in a plain host
  /// build for testing. begin(Print&) below is a thin, ARDUINO-only
  /// convenience wrapper around this.
  bool begin(const AudioChunkFn& on_audio) {
    if (!weights_.valid() || !dict_.valid() || !dp_model_.valid() || !decoder_model_.valid()) {
      return false;  // see class doc: call the set*() methods before begin()
    }
    audio_callback_ = on_audio;

    dp_arena_.reset((uint8_t*)Allocator::allocate(dp_arena_size_));
    decoder_arena_.reset((uint8_t*)Allocator::allocate(decoder_arena_size_));
    if (!dp_arena_ || !decoder_arena_) return false;

    registerDurationPredictorOps();
    registerDecoderOps();

    dp_interpreter_ = buildInterpreter(dp_model_.data(), dp_arena_.get(), dp_arena_size_, dp_resolver_);
    decoder_interpreter_ =
        buildInterpreter(decoder_model_.data(), decoder_arena_.get(), decoder_arena_size_, dec_resolver_);
    if (!dp_interpreter_ || !decoder_interpreter_) return false;

    if (!core_.begin(weights_.data(), weights_.size(), dict_.data(), dict_.size(), /*n_heads=*/2,
                     /*window_size=*/4, /*n_flows=*/4, dictionary_model_.data(), dictionary_model_.size()))
      return false;
    core_.setDurationPredictor([this](const Mat& x, const Mat& g) { return runDurationPredictor(x, g); });
    core_.setDecoder([this](const Mat& z, const Mat& g) { return runDecoder(z, g); });
    core_.setDecoderChunkFrames(decoder_chunk_frames_);
    core_.setHopLength(hop_length_);

    started_ = true;
    return true;
  }

#ifdef ARDUINO
  /// Arduino convenience overload: streams PCM (int16, clipped) to `output`
  /// via Print::write() instead of handing you raw float samples. Just
  /// begin(AudioChunkFn) above wired to writeAudio() -- see that overload
  /// for what "portable" means here.
  bool begin(Print& output) {
    output_ = &output;
    return begin([this](const float* samples, size_t count) { writeAudio(samples, count); });
  }
#endif

  /// Synthesizes `text` and streams the resulting PCM to the callback given
  /// to begin(), one decoder chunk at a time (never buffers a whole
  /// utterance). Returns false if begin() wasn't called successfully.
  bool speak(const std::string& text) {
    if (!started_) return false;
    core_.synthesize(text, audio_callback_, speaker_id_, noise_scale_, length_scale_);
    return true;
  }

  /// Releases the TFLite Micro interpreters and their PSRAM tensor arenas.
  /// The model data set via the set*() methods is NOT cleared -- a
  /// subsequent begin() reuses it without needing to be set again.
  void end() {
    dp_interpreter_.reset();
    decoder_interpreter_.reset();
    dp_arena_.reset();
    decoder_arena_.reset();
#ifdef ARDUINO
    output_ = nullptr;
#endif
    started_ = false;
    audio_callback_ = nullptr;
  }

  /// Direct access to the underlying orchestration/model layer, for
  /// advanced use (introspecting durations, using a different audio sink
  /// than a Print, etc).
  TinyTTSCore& core() { return core_; }

 private:
  struct PsramDeleter {
    void operator()(uint8_t* p) const {
      if (p) Allocator::deallocate(p);
    }
  };
  using PsramBuffer = std::unique_ptr<uint8_t[], PsramDeleter>;

  // Plain `new tflite::MicroInterpreter(...)` goes wherever the default
  // allocator sends it, not explicitly through Allocator like the tensor
  // arena above -- placement-new it into an explicitly-allocated block
  // instead, with a matching deleter (explicit destructor call +
  // Allocator::deallocate(), since `delete` on a placement-new'd object
  // must not call operator delete).
  struct InterpreterDeleter {
    void operator()(tflite::MicroInterpreter* p) const {
      if (p) {
        p->~MicroInterpreter();
        Allocator::deallocate(p);
      }
    }
  };
  using InterpreterPtr = std::unique_ptr<tflite::MicroInterpreter, InterpreterDeleter>;

  static InterpreterPtr buildInterpreter(const uint8_t* model_data, uint8_t* arena, size_t arena_size,
                                          tflite::MicroOpResolver& resolver) {
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) return nullptr;

    void* mem = Allocator::allocate(sizeof(tflite::MicroInterpreter));
    if (!mem) return nullptr;
    auto* interpreter = new (mem) tflite::MicroInterpreter(model, resolver, arena, arena_size);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
      interpreter->~MicroInterpreter();
      Allocator::deallocate(mem);
      return nullptr;
    }
    return InterpreterPtr(interpreter);
  }

  // Union of ops seen across every duration_predictor/decoder variant this
  // project has validated (float32, dynamic-range int8, full int8 with
  // int8 or int16 activations -- see the project plan's quantization
  // study), so swapping the model via the setters above doesn't also
  // require reworking op registration.
  void registerDurationPredictorOps() {
    dp_resolver_.AddAdd();
    dp_resolver_.AddConv2D();
    dp_resolver_.AddDiv();
    dp_resolver_.AddExpandDims();
    dp_resolver_.AddGather();
    dp_resolver_.AddMean();
    dp_resolver_.AddMul();
    dp_resolver_.AddPad();
    dp_resolver_.AddRelu();
    dp_resolver_.AddReshape();
    dp_resolver_.AddRsqrt();
    dp_resolver_.AddShape();
    dp_resolver_.AddSqrt();
    dp_resolver_.AddSqueeze();
    dp_resolver_.AddSub();
    dp_resolver_.AddTranspose();
  }

  void registerDecoderOps() {
    dec_resolver_.AddAdd();
    dec_resolver_.AddConcatenation();
    dec_resolver_.AddConv2D();
    dec_resolver_.AddDequantize();
    dec_resolver_.AddExpandDims();
    dec_resolver_.AddGather();
    dec_resolver_.AddLeakyRelu();
    dec_resolver_.AddMul();
    dec_resolver_.AddPad();
    dec_resolver_.AddReshape();
    dec_resolver_.AddShape();
    dec_resolver_.AddSlice();
    dec_resolver_.AddSqueeze();
    dec_resolver_.AddStridedSlice();
    dec_resolver_.AddSub();
    dec_resolver_.AddTanh();
    dec_resolver_.AddTranspose();
    dec_resolver_.AddTransposeConv();
  }

  // Quantizes a float buffer to the tensor's native int8/int16 type using
  // its (scale, zero_point). Leaves float32 tensors untouched.
  static void setQuantized(TfLiteTensor* t, const float* data, int n) {
    if (t->type == kTfLiteFloat32) {
      std::memcpy(t->data.f, data, n * sizeof(float));
      return;
    }
    float scale = t->params.scale;
    int32_t zero_point = t->params.zero_point;
    for (int i = 0; i < n; i++) {
      int32_t q = (int32_t)lroundf(data[i] / scale) + zero_point;
      if (t->type == kTfLiteInt8) {
        t->data.int8[i] = (int8_t)std::max<int32_t>(-128, std::min<int32_t>(127, q));
      } else {
        t->data.i16[i] = (int16_t)std::max<int32_t>(-32768, std::min<int32_t>(32767, q));
      }
    }
  }

  static void getDequantized(const TfLiteTensor* t, float* out, int n) {
    if (t->type == kTfLiteFloat32) {
      std::memcpy(out, t->data.f, n * sizeof(float));
      return;
    }
    float scale = t->params.scale;
    int32_t zero_point = t->params.zero_point;
    for (int i = 0; i < n; i++) {
      int32_t q = (t->type == kTfLiteInt8) ? t->data.int8[i] : t->data.i16[i];
      out[i] = (q - zero_point) * scale;
    }
  }

  // Runs one fixed-size duration_predictor window over x rows
  // [start, start+window_len) (window_len <= max_phonemes_; the rest of the
  // window is zero-padded and masked out via x_mask). Returns exactly
  // max_phonemes_ dequantized logw values -- callers decide which of them
  // are trustworthy (see runDurationPredictor()'s sliding-window doc).
  std::vector<float> runDurationPredictorWindow(const Mat& x, int start, int window_len, const Mat& g) {
    // Input order matches the exported signature: g, x, x_mask (index 0,1,2).
    TfLiteTensor* in_g = dp_interpreter_->input(0);
    TfLiteTensor* in_x = dp_interpreter_->input(1);
    TfLiteTensor* in_mask = dp_interpreter_->input(2);

    std::vector<float> x_padded((size_t)max_phonemes_ * x.cols(), 0.0f);
    std::copy(x.data().begin() + (size_t)start * x.cols(), x.data().begin() + (size_t)(start + window_len) * x.cols(),
              x_padded.begin());
    std::vector<float> mask_padded(max_phonemes_, 0.0f);
    std::fill(mask_padded.begin(), mask_padded.begin() + window_len, 1.0f);

    setQuantized(in_x, x_padded.data(), (int)x_padded.size());
    setQuantized(in_mask, mask_padded.data(), (int)mask_padded.size());
    setQuantized(in_g, g.data().data(), (int)g.data().size());

    dp_interpreter_->Invoke();

    std::vector<float> logw_padded(max_phonemes_);
    getDequantized(dp_interpreter_->output(0), logw_padded.data(), max_phonemes_);
    return logw_padded;
  }

  // x: [T,hidden] + g: [1,gin] -> logw: [T], for ANY T -- not just T <=
  // max_phonemes_. duration_predictor's own receptive field is small
  // (dp_margin_ positions each side: this project's shipped model is two
  // stacked kernel_size=3 Conv1d layers, giving a receptive field of
  // 1+2+2=5, i.e. +-2), so rather than truncating text longer than one
  // window, this slides overlapping windows across x and keeps only each
  // window's "trusted" middle -- the part whose dp_margin_ neighbors on
  // both sides come from real phoneme data, not zero-padding -- except at
  // the true start/end of the whole utterance, where the model's own
  // edge-of-sequence behavior IS what's wanted (there's no more real
  // context beyond the utterance boundary regardless of windowing).
  // Consecutive windows' kept ranges tile [0,T) exactly, no gap or overlap
  // in the output, so every phoneme gets a real answer -- not just the
  // first max_phonemes_ of them.
  std::vector<float> runDurationPredictor(const Mat& x, const Mat& g) {
    int T = x.rows();
    int margin = std::min(dp_margin_, (max_phonemes_ - 1) / 2);  // degenerate-config safety clamp
    std::vector<float> logw(T);

    int written = 0;
    int start = 0;
    while (written < T) {
      int window_len = std::min(max_phonemes_, T - start);
      std::vector<float> logw_window = runDurationPredictorWindow(x, start, window_len, g);

      bool is_first = (start == 0);
      bool is_last = (start + window_len >= T);
      int copy_from = is_first ? 0 : margin;
      int copy_len = (is_last ? window_len : window_len - margin) - copy_from;

      std::copy(logw_window.begin() + copy_from, logw_window.begin() + copy_from + copy_len,
                logw.begin() + written);
      written += copy_len;
      if (is_last) break;
      start = written - margin;  // next window starts `margin` early, for real left-context
    }
    return logw;
  }

  // z: [decoder_chunk_frames_,inter_channels] + g: [1,gin] -> PCM for this chunk.
  std::vector<float> runDecoder(const Mat& z, const Mat& g) {
    // Input order matches the exported signature: g, z (index 0,1).
    TfLiteTensor* in_g = decoder_interpreter_->input(0);
    TfLiteTensor* in_z = decoder_interpreter_->input(1);

    setQuantized(in_z, z.data().data(), (int)z.data().size());
    setQuantized(in_g, g.data().data(), (int)g.data().size());

    decoder_interpreter_->Invoke();

    const TfLiteTensor* out = decoder_interpreter_->output(0);
    int n_samples =
        out->type == kTfLiteFloat32 ? (int)(out->bytes / sizeof(float)) : (int)(out->bytes / sizeof(int8_t));
    std::vector<float> audio(n_samples);
    getDequantized(out, audio.data(), n_samples);
    return audio;
  }

#ifdef ARDUINO
  void writeAudio(const float* samples, size_t count) {
    if (!output_) return;
    pcm_buf_.resize(count);
    for (size_t i = 0; i < count; i++) {
      float s = samples[i];
      if (s > 1.0f) s = 1.0f;
      if (s < -1.0f) s = -1.0f;
      pcm_buf_[i] = (int16_t)(s * 32767.0f);
    }
    output_->write((const uint8_t*)pcm_buf_.data(), pcm_buf_.size() * sizeof(int16_t));
  }
#endif  // ARDUINO

  DataBuffer weights_;
  DataBuffer dict_;
  DataBuffer dp_model_;
  DataBuffer decoder_model_;
  DataBuffer dictionary_model_;  // optional -- see setDictionaryModel()

  int max_phonemes_ = 32;
  // duration_predictor's receptive field, each side -- see
  // runDurationPredictor()'s doc. This project's shipped model is two
  // stacked kernel_size=3 Conv1d layers (receptive field 1+2+2=5, i.e. +-2);
  // override via setDurationPredictorMargin() if you swap in a model with a
  // different architecture.
  int dp_margin_ = 2;
  int decoder_chunk_frames_ = 96;
  int hop_length_ = 512;
  int speaker_id_ = 0;
  float noise_scale_ = 0.667f;
  float length_scale_ = 1.0f;
  int sample_rate_ = 44100;  // tiny_tts.utils.config.SAMPLING_RATE

  size_t dp_arena_size_ = 128 * 1024;
  // 384KB was undersized for the (necessarily fixed-point, see below)
  // decoder model -- confirmed via a host-side TFLite Micro build that it
  // actually needs ~1.5MB; 2MB leaves headroom. PSRAM has plenty of room.
  size_t decoder_arena_size_ = 2048 * 1024;
  PsramBuffer dp_arena_;
  PsramBuffer decoder_arena_;

  static constexpr int kNumDpOps = 16;
  static constexpr int kNumDecOps = 18;
  tflite::MicroMutableOpResolver<kNumDpOps> dp_resolver_;
  tflite::MicroMutableOpResolver<kNumDecOps> dec_resolver_;
  InterpreterPtr dp_interpreter_;
  InterpreterPtr decoder_interpreter_;

#ifdef ARDUINO
  Print* output_ = nullptr;
  std::vector<int16_t> pcm_buf_;
#endif

  TinyTTSCore core_;
  AudioChunkFn audio_callback_;
  bool started_ = false;
};

}  // namespace tinytts

/// Arduino sketch can `#include <TinyTTS.h>` and use the class directly without
#ifdef ARDUINO
using namespace tinytts;
#endif
