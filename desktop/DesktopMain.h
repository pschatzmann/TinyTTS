#pragma once
// Desktop-only CLI wrapper -- NOT part of the Arduino/ESP-IDF library
// surface (nothing under src/ includes this), only built when
// -DTINYTTS_BUILD_DESKTOP_MAIN=ON (see desktop/CMakeLists.txt). Needs
// IS_MIN_DESKTOP defined (by the CMake target, before this header) so
// AudioTools' desktop platform config (AudioTools/PlatformConfig/desktop.h)
// pulls in its own lightweight Emulation/{Arduino,Time,Main}.h -- no
// separate Arduino-Emulator/WiFi dependency needed for this.
//
// Uses TinyTTS's portable begin(const AudioChunkFn&) rather than the
// Print&-based constructor/begin(): the latter needs ARDUINO defined,
// which pulls in more of TinyTTSCore.h's ARDUINO-gated code than AudioTools'
// IS_MIN_DESKTOP emulation actually provides (e.g. TinyTTSCore.h's
// timingLog() calls Serial.printf(), an ESP32-specific HardwareSerial
// extension the emulation's base Print/Stream API doesn't have) -- chasing
// that down further isn't worth it for what the portable callback path
// already does just as well.
//
// ODR note: MiniAudioStream.h does `#define MINIAUDIO_IMPLEMENTATION`
// immediately before `#include "miniaudio.h"` -- the single-header
// library's actual implementation gets compiled wherever that header is
// included. This file must therefore be included from EXACTLY ONE
// translation unit in the whole desktop target (desktop/main.cpp) -- do
// not #include this from a second .cpp, or the build fails with duplicate
// miniaudio symbol definitions at link time.
#include "AudioTools.h"
#include "AudioTools/AudioLibs/MiniAudioStream.h"
#include "TinyTTS.h"
#include "TinyTTS/data/default_cmudict_slim_data.h"
#include "TinyTTS/data/default_dictionary_model_data.h"
#include "TinyTTS/data/default_weights_data.h"

#include <unistd.h>

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace tinytts {

/**
 * @brief CLI entry point for the desktop build: owns the TinyTTS instance
 * and the MiniAudioStream output sink for the process's lifetime, so the
 * actual free `main()` (desktop/main.cpp) stays a one-line argv-parsing
 * shim -- `main()` itself can't be a class method.
 *
 * Supports Unix-style piping: `echo "hi" | tinytts --stdout > out.wav`
 * reads text from stdin (when no positional TEXT/--file is given) and writes
 * WAV bytes to stdout instead of playing -- see run()'s --help text for the
 * full option list.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class DesktopMain {
 public:
  int run(int argc, char** argv) {
    Options opt;
    if (!parseArgs(argc, argv, opt)) return opt.help_requested ? 0 : 1;

    std::string text;
    if (!resolveText(opt, text)) return 1;

    tts_.setVolume(opt.volume);
    tts_.setNoiseScale(opt.noise_scale);
    tts_.setLengthScale(opt.length_scale);
    tts_.setSpeakerId(opt.speaker);
    // Must happen before the first speak() -- see setNumWorkers()'s own
    // doc for why (the underlying worker pool is created lazily on first
    // use and persists for the process's lifetime).
    tts_.setNumWorkers(opt.threads);

    bool should_play = !opt.no_play && opt.output_file.empty() && !opt.to_stdout;
    if (should_play) {
      auto cfg = out_.defaultConfig(audio_tools::TX_MODE);
      cfg.sample_rate = tts_.getAudioSampleRate();
      // TinyTTS is always mono; MiniAudioConfig::defaultConfig() defaults
      // to channels=2 -- must override explicitly or every other sample
      // is silence (channel-swapped/interleaved garbage), an easy miss
      // since it isn't the default.
      cfg.channels = tts_.getAudioChannels();
      cfg.bits_per_sample = tts_.getAudioBitsPerSample();
      if (!out_.begin(cfg)) {
        std::fprintf(stderr, "MiniAudioStream.begin() failed\n");
        return 1;
      }
    }

    // Same three embedded data buffers examples/tts_i2s_output uses -- see
    // that sketch's doc for why this combo (slimmed dictionary + neural
    // G2P fallback) is the recommended one.
    tts_.setWeights(default_weights, default_weights_len);
    tts_.setDictionary(default_cmudict_slim, default_cmudict_slim_len);
    tts_.setDictionaryModel(default_dictionary_model, default_dictionary_model_len);
    if (!tts_.begin([this, should_play](const float* samples, size_t count) {
          onAudio(samples, count, should_play);
        })) {
      std::fprintf(stderr, "TinyTTS.begin() failed -- did you call all setters above?\n");
      return 1;
    }

    // Chunked, not one speak() call for the whole input: TinyTTS::speak()
    // synthesizes its entire argument as one batch (all four stages run
    // once over the whole phoneme sequence) and only calls the
    // AudioChunkFn once, at the very end, with the complete waveform --
    // for a whole file (`-f README.md`), nothing would reach the speaker
    // until everything finished. Splitting on sentence boundaries and
    // calling speak() once per sentence means the first sentence starts
    // playing as soon as *it* is ready, not after the last one -- and per
    // docs/text-input.md, splitting text across multiple speak() calls on
    // sentence/word boundaries never mispronounces anything (only
    // mid-word splits do).
    total_samples_ = 0;
    all_samples_.clear();
    for (const std::string& chunk : splitIntoChunks(text)) {
      if (!tts_.speak(chunk)) {
        std::fprintf(stderr, "TinyTTS.speak() failed on: %s\n", chunk.c_str());
        return 1;
      }
    }

    if (!opt.output_file.empty()) {
      std::ofstream f(opt.output_file, std::ios::binary);
      if (!f) {
        std::fprintf(stderr, "cannot open %s for writing\n", opt.output_file.c_str());
        return 1;
      }
      writeWav(f, all_samples_, tts_.getAudioSampleRate(), opt.volume);
    }
    if (opt.to_stdout) {
      // Diagnostics (TinyTTSCore.h's timingLog()) already go to stderr, not
      // stdout, specifically so they never interleave with this.
      writeWav(std::cout, all_samples_, tts_.getAudioSampleRate(), opt.volume);
    }

    if (should_play) {
      // speak() returning means synthesis finished, NOT that playback
      // finished -- MiniAudioStream::write() (called from onAudio() above)
      // only fills a ring buffer; a real-time callback thread on its own
      // drains it asynchronously. Sleep for the known audio duration plus
      // a fixed margin for MiniAudio's own internal buffering latency,
      // rather than polling MiniAudioStream's internal state
      // (is_playing/buffer_out aren't a stable public API).
      double seconds = (double)total_samples_ / tts_.getAudioSampleRate();
      std::this_thread::sleep_for(std::chrono::duration<double>(seconds + 0.3));
    }
    return 0;
  }

 private:
  struct Options {
    std::string text;
    bool have_text = false;
    std::string input_file;
    std::string output_file;
    bool to_stdout = false;
    bool no_play = false;
    float volume = 1.0f;
    float noise_scale = 0.667f;
    float length_scale = 1.0f;
    int speaker = 0;
    int threads = 2;  // matches TinyTTS::setNumWorkers()'s own default
    bool help_requested = false;
  };

  static void printUsage(const char* prog) {
    std::fprintf(stderr,
                  "Usage: %s [options] [TEXT]\n"
                  "\n"
                  "Input:\n"
                  "  TEXT                  Text to speak. If omitted (and --file isn't given),\n"
                  "                        read from stdin -- e.g. echo \"hi\" | %s\n"
                  "  -f, --file FILE       Read text from FILE instead.\n"
                  "\n"
                  "Output:\n"
                  "  -o, --output FILE     Write synthesized audio to FILE as a WAV file,\n"
                  "                        instead of playing it.\n"
                  "  --stdout              Write WAV bytes to stdout instead of playing --\n"
                  "                        for piping, e.g. %s --stdout | aplay, or > out.wav\n"
                  "  --no-play             Skip playback (implied by -o/--stdout).\n"
                  "\n"
                  "Voice tuning:\n"
                  "  --volume VALUE        0.0-1.0, default 1.0\n"
                  "  --noise-scale VALUE   default 0.667\n"
                  "  --length-scale VALUE  default 1.0 (speaking rate; >1 slower, <1 faster)\n"
                  "  --speaker ID          default 0\n"
                  "  --threads N           worker count for the decoder's parallel conv loops,\n"
                  "                        default 2 (this machine has %u hardware threads)\n"
                  "\n"
                  "  -h, --help            Show this help text.\n",
                  prog, prog, prog, std::thread::hardware_concurrency());
  }

  static bool parseArgs(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i];
      auto next = [&](const char* flag) -> const char* {
        if (i + 1 >= argc) {
          std::fprintf(stderr, "%s requires a value\n", flag);
          return nullptr;
        }
        return argv[++i];
      };
      if (a == "-h" || a == "--help") {
        printUsage(argv[0]);
        opt.help_requested = true;
        return false;
      } else if (a == "-f" || a == "--file") {
        const char* v = next(a.c_str());
        if (!v) return false;
        opt.input_file = v;
      } else if (a == "-o" || a == "--output") {
        const char* v = next(a.c_str());
        if (!v) return false;
        opt.output_file = v;
      } else if (a == "--stdout") {
        opt.to_stdout = true;
      } else if (a == "--no-play") {
        opt.no_play = true;
      } else if (a == "--volume") {
        const char* v = next(a.c_str());
        if (!v) return false;
        opt.volume = std::stof(v);
      } else if (a == "--noise-scale") {
        const char* v = next(a.c_str());
        if (!v) return false;
        opt.noise_scale = std::stof(v);
      } else if (a == "--length-scale") {
        const char* v = next(a.c_str());
        if (!v) return false;
        opt.length_scale = std::stof(v);
      } else if (a == "--speaker") {
        const char* v = next(a.c_str());
        if (!v) return false;
        opt.speaker = std::atoi(v);
      } else if (a == "--threads") {
        const char* v = next(a.c_str());
        if (!v) return false;
        opt.threads = std::atoi(v);
      } else if (!a.empty() && a[0] == '-' && a != "-") {
        std::fprintf(stderr, "unknown option: %s\n", a.c_str());
        printUsage(argv[0]);
        return false;
      } else {
        opt.text = a;
        opt.have_text = true;
      }
    }
    return true;
  }

  /// Splits `text` into sentence-sized pieces so run() can call speak()
  /// once per sentence instead of once for the whole input -- see run()'s
  /// doc for why that matters for time-to-first-sound. Splits on
  /// sentence-ending punctuation (./!/?) followed by whitespace or
  /// end-of-input, and on blank lines (paragraph breaks); a chunk that's
  /// still very long after that (e.g. a markdown code block or a single
  /// long line with no sentence punctuation at all) gets further split on
  /// the nearest word boundary at or before kMaxChunkChars, so one
  /// unpunctuated wall of text can't become a single, slow-to-first-sound
  /// chunk. Never splits mid-word (see docs/text-input.md for why that's
  /// the one real constraint on multi-call splitting).
  static std::vector<std::string> splitIntoChunks(const std::string& text) {
    constexpr size_t kMaxChunkChars = 300;
    std::vector<std::string> raw;
    size_t chunk_start = 0, last_space = std::string::npos;
    for (size_t i = 0; i < text.size(); i++) {
      char c = text[i];
      if (std::isspace((unsigned char)c)) last_space = i;
      bool sentence_end = (c == '.' || c == '!' || c == '?') &&
                           (i + 1 == text.size() || std::isspace((unsigned char)text[i + 1]));
      bool paragraph_break = c == '\n' && i + 1 < text.size() && text[i + 1] == '\n';
      if (sentence_end || paragraph_break) {
        raw.push_back(text.substr(chunk_start, i + 1 - chunk_start));
        chunk_start = i + 1;
        last_space = std::string::npos;
      } else if (i - chunk_start >= kMaxChunkChars && last_space != std::string::npos && last_space > chunk_start) {
        raw.push_back(text.substr(chunk_start, last_space - chunk_start));
        chunk_start = last_space + 1;
        last_space = std::string::npos;
      }
    }
    if (chunk_start < text.size()) raw.push_back(text.substr(chunk_start));

    std::vector<std::string> chunks;
    for (std::string& s : raw) {
      size_t b = s.find_first_not_of(" \t\r\n");
      size_t e = s.find_last_not_of(" \t\r\n");
      if (b == std::string::npos) continue;  // whitespace-only (blank lines, markdown spacing, ...)
      chunks.push_back(s.substr(b, e - b + 1));
    }
    if (chunks.empty()) chunks.push_back("");  // preserve speak()'s own behavior on genuinely empty input
    return chunks;
  }

  /// Resolves the text to speak from (in priority order) --file, the
  /// positional argument, or stdin -- the standard Unix piping convention
  /// for a tool with no other required input.
  static bool resolveText(const Options& opt, std::string& out) {
    if (!opt.input_file.empty()) {
      std::ifstream f(opt.input_file);
      if (!f) {
        std::fprintf(stderr, "cannot open %s\n", opt.input_file.c_str());
        return false;
      }
      std::ostringstream ss;
      ss << f.rdbuf();
      out = ss.str();
      return true;
    }
    if (opt.have_text) {
      out = opt.text;
      return true;
    }
    if (isatty(fileno(stdin))) {
      std::fprintf(stderr, "No text given (and stdin is a terminal, not a pipe) -- pass TEXT, --file, or pipe input.\n");
      return false;
    }
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    out = ss.str();
    return true;
  }

  // Same RIFF/WAVE header layout as test/main.cpp's write_wav(), just
  // targeting an arbitrary ostream (a file or std::cout) instead of always
  // a path, so -o/--output and --stdout can share one implementation. Volume
  // mirrors TinyTTS.h's own volumeToGain() dB taper (see this file's
  // onAudio() for why that's duplicated here rather than reused).
  static void writeWav(std::ostream& f, const std::vector<float>& samples, int sample_rate, float volume) {
    float gain = volumeToGain(volume);
    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
      float s = samples[i] * gain;
      if (s > 1.0f) s = 1.0f;
      if (s < -1.0f) s = -1.0f;
      pcm[i] = (int16_t)(s * 32767.0f);
    }
    uint32_t data_bytes = (uint32_t)(pcm.size() * sizeof(int16_t));
    uint32_t byte_rate = (uint32_t)sample_rate * 1 /*channels*/ * 2 /*bytes/sample*/;
    uint16_t block_align = 2;
    uint16_t bits_per_sample = 16;
    uint32_t riff_size = 36 + data_bytes;

    f.write("RIFF", 4);
    f.write((const char*)&riff_size, 4);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    uint32_t fmt_size = 16;
    f.write((const char*)&fmt_size, 4);
    uint16_t audio_format = 1;  // PCM
    uint16_t num_channels = 1;
    f.write((const char*)&audio_format, 2);
    f.write((const char*)&num_channels, 2);
    f.write((const char*)&sample_rate, 4);
    f.write((const char*)&byte_rate, 4);
    f.write((const char*)&block_align, 2);
    f.write((const char*)&bits_per_sample, 2);
    f.write("data", 4);
    f.write((const char*)&data_bytes, 4);
    f.write((const char*)pcm.data(), data_bytes);
    f.flush();
  }

  // Mirrors TinyTTS.h's own volumeToGain() (a dB taper -- perceived
  // loudness is logarithmic, so a linear gain barely sounds quieter than
  // full volume -- see that file's doc for the full reasoning) --
  // duplicated here rather than reused because that's a private member of
  // the ARDUINO-only Print-output path, which isn't available under
  // IS_MIN_DESKTOP; the portable begin(const AudioChunkFn&) path this
  // class uses deliberately hands back the model's raw, unscaled samples.
  static float volumeToGain(float volume) {
    if (volume <= 0.0f) return 0.0f;
    if (volume >= 1.0f) return 1.0f;
    constexpr float kDynamicRangeDb = 40.0f;
    return std::pow(10.0f, (volume - 1.0f) * kDynamicRangeDb / 20.0f);
  }

  void onAudio(const float* samples, size_t count, bool should_play) {
    // Appends, not assign(): run() now calls speak() once per chunk (see
    // its doc), and each call's callback must add to the running total
    // for -o/--stdout (one WAV covering every chunk) and the play-drain
    // sleep below, not overwrite what earlier chunks already produced.
    all_samples_.insert(all_samples_.end(), samples, samples + count);
    total_samples_ += count;
    if (!should_play) return;

    pcm_buf_.resize(count);
    float gain = volumeToGain(1.0f);
    for (size_t i = 0; i < count; i++) {
      float s = samples[i] * gain;
      if (s > 1.0f) s = 1.0f;
      if (s < -1.0f) s = -1.0f;
      pcm_buf_[i] = (int16_t)(s * 32767.0f);
    }
    // A single write() call handing MiniAudioStream this much data in one
    // shot (TinyTTS's AudioChunkFn callback fires once per utterance, not
    // incrementally) overwhelms its internal ring-buffer backpressure --
    // confirmed via a minimal, TinyTTS-free repro: one huge write() times
    // out after filling exactly one ring-buffer's worth and never
    // draining further, while the same bytes written in ~2KB pieces (no
    // extra pacing needed) succeed completely. BufferedStream gives us
    // that chunking for free instead of hand-rolling a loop here.
    buffered_out_.write((const uint8_t*)pcm_buf_.data(), pcm_buf_.size() * sizeof(int16_t));
    buffered_out_.flush();  // pcm_buf_.size() isn't guaranteed a multiple of kWriteChunkBytes
  }

  // Small enough to reliably avoid the ring-buffer-overwhelm issue above
  // (the repro's proven-good size was ~2KB); large enough to keep
  // BufferedStream's per-byte write() loop (see its own implementation)
  // from mattering performance-wise for a one-shot CLI tool.
  static constexpr size_t kWriteChunkBytes = 4096;

  audio_tools::MiniAudioStream out_;
  audio_tools::BufferedStream buffered_out_{out_, kWriteChunkBytes};
  TinyTTS tts_;
  std::vector<int16_t> pcm_buf_;
  std::vector<float> all_samples_;
  size_t total_samples_ = 0;
};

}  // namespace tinytts
