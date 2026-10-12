#pragma once
// Linux only: needs the Sophgo/CVITEK TPU runtime (cviruntime.h,
// libcviruntime.so) from the Milk-V Duo TPU SDK
// (https://github.com/milkv-duo/tpu-sdk-sg200x) -- deliberately not included
// by TinyTTS.h. See docs/milkv-duo.md.
#include <cviruntime.h>

#include <cstdio>
#include <cstring>

#include "TinyTTS/ChunkedVocoder.h"
#include "TinyTTS/Mat.h"
#include "TinyTTS/TinyTTSCore.h"

namespace tinytts {

/**
 * @brief Runs the vocoder on the TPU of the Milk-V Duo 256M / Duo S
 * (SG2002/SG2000) through the CVITEK runtime, as a ChunkedVocoder.
 *
 * Loads the cvimodel built by research/milkv_duo/build_cvimodel.sh: inputs
 * z [1, inter_channels, W, 1], g [1, gin_channels, 1, 1] and mask [1, 1, W,
 * 1], output audio [1, 1, W * 512, 1], all float32 (the conversion to the TPU's INT8/BF16
 * happens inside the model). W is read from the model.
 *
 *   CviTpuVocoder tpu;
 *   if (tpu.begin("decoder_w128_bf16.cvimodel")) tts.setVocoder(tpu.vocoderFn());
 *
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class CviTpuVocoder {
 public:
  CviTpuVocoder() = default;
  CviTpuVocoder(const CviTpuVocoder&) = delete;
  CviTpuVocoder& operator=(const CviTpuVocoder&) = delete;
  ~CviTpuVocoder() { end(); }

  /// `context`: overlap frames on each side of a window, see ChunkedVocoder
  /// (12 matches the one-pass decode to float rounding, 10 to >57dB).
  bool begin(const char* cvimodel_path, int context = 12) {
    end();
    if (CVI_NN_RegisterModel(cvimodel_path, &model_) != CVI_RC_SUCCESS) {
      model_ = nullptr;
      return fail("cannot load %s", cvimodel_path);
    }
    if (CVI_NN_GetInputOutputTensors(model_, &inputs_, &n_inputs_, &outputs_, &n_outputs_) != CVI_RC_SUCCESS)
      return fail("CVI_NN_GetInputOutputTensors failed");
    z_ = CVI_NN_GetTensorByName("z", inputs_, n_inputs_);
    g_ = CVI_NN_GetTensorByName("g", inputs_, n_inputs_);
    mask_ = CVI_NN_GetTensorByName("mask", inputs_, n_inputs_);
    audio_ = n_outputs_ == 1 ? &outputs_[0] : nullptr;
    if (!z_ || !g_ || !mask_ || !audio_) return fail("expected inputs z, g, mask and one output");
    for (CVI_TENSOR* t : {z_, g_, mask_, audio_})
      if (t->fmt != CVI_FMT_FP32)
        return fail("tensor %s is not float32 -- build the cvimodel without --quant_input/--quant_output",
                    CVI_NN_TensorName(t));

    // TPU-MLIR runs the 1-D convolutions as 2-D ones, so the model's tensors
    // are [1, C, T, 1] rather than the ONNX model's [1, C, T]. Same layout.
    CVI_SHAPE zs = CVI_NN_TensorShape(z_);
    bool z_ok = zs.dim_size >= 3 && zs.dim[0] == 1;
    for (size_t i = 3; i < zs.dim_size; i++) z_ok = z_ok && zs.dim[i] == 1;
    if (!z_ok) return fail("unexpected z shape");
    channels_ = zs.dim[1];
    window_ = zs.dim[2];
    if (CVI_NN_TensorCount(mask_) != (size_t)window_) return fail("mask doesn't match z");
    if (CVI_NN_TensorCount(audio_) % window_ != 0) return fail("output isn't a whole number of frames");
    const int hop = (int)(CVI_NN_TensorCount(audio_) / window_);
    if (!chunked_.begin([this](const Mat& z, int vb, int ve, const Mat& g, float* audio) {
          return runWindow(z, vb, ve, g, audio);
        },
                        window_, context, hop))
      return fail("context %d too large for a %d-frame window", context, window_);
    return true;
  }

  void end() {
    if (model_) CVI_NN_CleanupModel(model_);
    model_ = nullptr;
    inputs_ = outputs_ = nullptr;
    z_ = g_ = mask_ = audio_ = nullptr;
    n_inputs_ = n_outputs_ = 0;
  }

  /// Frames per TPU call, from the model.
  int window() const { return window_; }

  /// For TinyTTS::setVocoder() / TinyTTSCore::setVocoder(). This object must
  /// outlive its use there.
  TinyTTSCore::VocoderFn vocoderFn() {
    return [this](const Mat& z, const Mat& g, const TinyTTSCore::AudioChunkFn& on_audio) {
      return chunked_.forward(z, g, on_audio);
    };
  }

 private:
  bool runWindow(const Mat& z, int valid_begin, int valid_end, const Mat& g, float* audio) {
    if (z.cols() != channels_ || (size_t)g.cols() != CVI_NN_TensorCount(g_))
      return fail("z/g don't match the model (%d/%d channels)", z.cols(), g.cols());
    // Mat is [T, C]; the model takes channel-first [1, C, T].
    float* zp = (float*)CVI_NN_TensorPtr(z_);
    for (int t = 0; t < window_; t++)
      for (int c = 0; c < channels_; c++) zp[(size_t)c * window_ + t] = z.at(t, c);
    std::memcpy(CVI_NN_TensorPtr(g_), g.row(0), sizeof(float) * g.cols());
    float* mp = (float*)CVI_NN_TensorPtr(mask_);
    for (int t = 0; t < window_; t++) mp[t] = (t >= valid_begin && t < valid_end) ? 1.0f : 0.0f;

    if (CVI_NN_Forward(model_, inputs_, n_inputs_, outputs_, n_outputs_) != CVI_RC_SUCCESS)
      return fail("CVI_NN_Forward failed");
    std::memcpy(audio, CVI_NN_TensorPtr(audio_), CVI_NN_TensorSize(audio_));
    return true;
  }

  template <typename... Args>
  static bool fail(const char* fmt, Args... args) {
    std::fprintf(stderr, "[CviTpuVocoder] ");
    std::fprintf(stderr, fmt, args...);
    std::fprintf(stderr, "\n");
    return false;
  }
  static bool fail(const char* msg) { return fail("%s", msg); }

  CVI_MODEL_HANDLE model_ = nullptr;
  CVI_TENSOR* inputs_ = nullptr;
  CVI_TENSOR* outputs_ = nullptr;
  int32_t n_inputs_ = 0, n_outputs_ = 0;
  CVI_TENSOR *z_ = nullptr, *g_ = nullptr, *mask_ = nullptr, *audio_ = nullptr;
  int channels_ = 0;
  int window_ = 0;
  ChunkedVocoder chunked_;
};

}  // namespace tinytts
