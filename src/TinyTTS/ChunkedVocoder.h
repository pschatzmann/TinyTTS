#pragma once
#include <algorithm>
#include <functional>

#include "TinyTTS/Mat.h"
#include "TinyTTS/PsramStlAllocator.h"

namespace tinytts {

/**
 * @brief Runs the vocoder as a series of fixed-size, overlapping frame
 * windows instead of one pass over the whole utterance -- for inference
 * engines that only run static shapes, such as the Milk-V Duo's TPU
 * (a TPU-MLIR cvimodel, see docs/milkv-duo.md).
 *
 * Each call to the window function decodes `window` frames: the
 * `window - 2*context` frames this chunk is responsible for, plus `context`
 * frames on each side so the convolutions see their real neighbours. Only the
 * middle part of each window's output is kept. Frames before the start or
 * past the end of the utterance are zero, and the window function is told
 * which frames are real (`valid_begin`/`valid_end`): it must treat the rest
 * as padding at every layer, as the one-pass decode does at the utterance's
 * ends (the TPU model takes a mask input for this). The vocoder is fully
 * convolutional, so this matches the one-pass result: with 12 frames of
 * context to float rounding (>110dB SNR), with 10 frames >57dB -- see
 * research/milkv_duo/export_decoder_tpu.py --sweep.
 *
 * Audio is handed to `on_audio` window by window, so playback can start
 * after the first window instead of after the whole utterance.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class ChunkedVocoder {
 public:
  /// Decodes one window: `z` is [window, inter_channels], of which rows
  /// [valid_begin, valid_end) are part of the utterance, `g` the [1,
  /// gin_channels] speaker embedding. Writes window * hop samples to
  /// `audio`. Returns false on failure.
  using WindowFn =
      std::function<bool(const Mat& z, int valid_begin, int valid_end, const Mat& g, float* audio)>;
  using AudioChunkFn = std::function<void(const float* samples, size_t count)>;

  /// `hop`: output samples per frame (the product of the upsample rates).
  bool begin(WindowFn fn, int window, int context, int hop = 512) {
    if (!fn || context < 0 || window - 2 * context <= 0 || hop <= 0) return false;
    fn_ = std::move(fn);
    window_ = window;
    context_ = context;
    hop_ = hop;
    return true;
  }

  int window() const { return window_; }
  int context() const { return context_; }
  int hop() const { return hop_; }

  /// z: [T, inter_channels] (the flow's output), g: [1, gin_channels].
  /// Calls `on_audio` once per window with T * hop samples in total.
  bool forward(const Mat& z, const Mat& g, const AudioChunkFn& on_audio) const {
    if (!fn_) return false;
    const int T = z.rows(), C = z.cols();
    const int core = window_ - 2 * context_;
    Mat win(window_, C);
    PsramVector<float> audio((size_t)window_ * hop_);
    for (int s = 0; s < T; s += core) {
      std::fill(win.data().begin(), win.data().end(), 0.0f);
      const int lo = s - context_;
      const int src_lo = std::max(lo, 0), src_hi = std::min(lo + window_, T);
      for (int t = src_lo; t < src_hi; t++) std::copy(z.row(t), z.row(t) + C, win.row(t - lo));
      if (!fn_(win, src_lo - lo, src_hi - lo, g, audio.data())) return false;
      const int n = std::min(core, T - s);
      if (on_audio) on_audio(audio.data() + (size_t)context_ * hop_, (size_t)n * hop_);
    }
    return true;
  }

 private:
  WindowFn fn_;
  int window_ = 0;
  int context_ = 0;
  int hop_ = 512;
};

}  // namespace tinytts
