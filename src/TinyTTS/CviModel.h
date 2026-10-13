#pragma once
// Linux only: needs the Sophgo/CVITEK TPU runtime (cviruntime.h,
// libcviruntime.so) from the Milk-V Duo TPU SDK
// (https://github.com/milkv-duo/tpu-sdk-sg200x) -- deliberately not included
// by TinyTTS.h. See docs/milkv-duo.md.
#include <cviruntime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "TinyTTS/Mat.h"

namespace tinytts {

/**
 * @brief One cvimodel loaded into the CVITEK TPU runtime, for the models
 * built by research/milkv_duo/: float32 inputs and outputs (the conversion to
 * the TPU's BF16 happens inside the model), in channel-first [1, C, L]
 * layout. TPU-MLIR runs 1-D convolutions as 2-D ones, so the tensors show up
 * as [1, C, L, 1]; the memory layout is the same.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class CviModel {
 public:
  CviModel() = default;
  CviModel(const CviModel&) = delete;
  CviModel& operator=(const CviModel&) = delete;
  ~CviModel() { end(); }

  /// Loads `path` and checks that every input and output is float32.
  bool begin(const char* path) {
    end();
    name_ = path;
    if (CVI_NN_RegisterModel(path, &model_) != CVI_RC_SUCCESS) {
      model_ = nullptr;
      return fail("cannot load the model");
    }
    if (CVI_NN_GetInputOutputTensors(model_, &inputs_, &n_inputs_, &outputs_, &n_outputs_) != CVI_RC_SUCCESS)
      return fail("CVI_NN_GetInputOutputTensors failed");
    for (int i = 0; i < n_inputs_ + n_outputs_; i++) {
      CVI_TENSOR* t = i < n_inputs_ ? &inputs_[i] : &outputs_[i - n_inputs_];
      if (t->fmt != CVI_FMT_FP32)
        return fail("tensor %s is not float32 -- build the cvimodel without --quant_input/--quant_output",
                    CVI_NN_TensorName(t));
    }
    return true;
  }

  void end() {
    if (model_) CVI_NN_CleanupModel(model_);
    model_ = nullptr;
    inputs_ = outputs_ = nullptr;
    n_inputs_ = n_outputs_ = 0;
  }

  /// nullptr (and an error message) if there is no such input.
  CVI_TENSOR* input(const char* name) {
    CVI_TENSOR* t = model_ ? CVI_NN_GetTensorByName(name, inputs_, n_inputs_) : nullptr;
    if (!t) fail("no input named %s", name);
    return t;
  }

  /// The only output; nullptr (and an error message) if there isn't exactly one.
  CVI_TENSOR* output() {
    if (n_outputs_ != 1) fail("expected one output, the model has %d", (int)n_outputs_);
    return n_outputs_ == 1 ? &outputs_[0] : nullptr;
  }

  bool forward() {
    if (CVI_NN_Forward(model_, inputs_, n_inputs_, outputs_, n_outputs_) != CVI_RC_SUCCESS)
      return fail("CVI_NN_Forward failed");
    return true;
  }

  /// Channels and length of a [1, C, L] (or [1, C, L, 1]) tensor.
  bool shape(CVI_TENSOR* t, int& channels, int& length) {
    CVI_SHAPE s = CVI_NN_TensorShape(t);
    bool ok = s.dim_size >= 3 && s.dim[0] == 1;
    for (size_t i = 3; i < s.dim_size; i++) ok = ok && s.dim[i] == 1;
    if (!ok) return fail("tensor %s isn't [1, C, L]", CVI_NN_TensorName(t));
    channels = s.dim[1];
    length = s.dim[2];
    return true;
  }

  /// Writes rows [0, rows) of the [T, C] `m` into the [1, C, length] tensor
  /// `t`, channel-first; the rest of the tensor is zero.
  static void setChannelFirst(CVI_TENSOR* t, const Mat& m, int rows, int length) {
    float* p = (float*)CVI_NN_TensorPtr(t);
    std::memset(p, 0, sizeof(float) * (size_t)m.cols() * length);
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < m.cols(); c++) p[(size_t)c * length + r] = m.at(r, c);
  }

  /// The sample rate in the model's file name (research/milkv_duo/ writes
  /// decoder_<rate>hz_w<W>_*.cvimodel and flow_<rate>hz_l<L>_*.cvimodel), or
  /// 0 if the name has none.
  int sampleRateFromName() const {
    size_t hz = name_.rfind("hz_");
    if (hz == std::string::npos) return 0;
    size_t start = hz;
    while (start > 0 && name_[start - 1] >= '0' && name_[start - 1] <= '9') start--;
    return start < hz ? std::atoi(name_.substr(start, hz - start).c_str()) : 0;
  }

  /// False (and an error message) if the file name says the model is for
  /// another sample rate than `sample_rate`, i.e. another checkpoint than the
  /// loaded weights.bin.
  bool checkSampleRate(int sample_rate) const {
    int named = sampleRateFromName();
    if (named != 0 && named != sample_rate)
      return fail("model is for %d Hz, the weights are %d Hz -- export both from the same checkpoint", named,
                  sample_rate);
    return true;
  }

  /// Fills `t` with 1 for [begin, end) and 0 elsewhere.
  static void setMask(CVI_TENSOR* t, int begin, int end) {
    float* p = (float*)CVI_NN_TensorPtr(t);
    const int n = (int)CVI_NN_TensorCount(t);
    for (int i = 0; i < n; i++) p[i] = (i >= begin && i < end) ? 1.0f : 0.0f;
  }

  /// Prints "[TinyTTS TPU] <model>: <message>" to stderr; returns false.
  template <typename... Args>
  bool fail(const char* fmt, Args... args) const {
    std::fprintf(stderr, "[TinyTTS TPU] %s: ", name_.c_str());
    std::fprintf(stderr, fmt, args...);
    std::fprintf(stderr, "\n");
    return false;
  }
  bool fail(const char* msg) const { return fail("%s", msg); }

 private:
  std::string name_;
  CVI_MODEL_HANDLE model_ = nullptr;
  CVI_TENSOR* inputs_ = nullptr;
  CVI_TENSOR* outputs_ = nullptr;
  int32_t n_inputs_ = 0, n_outputs_ = 0;
};

}  // namespace tinytts
