#pragma once
// Linux only, needs the Milk-V Duo TPU SDK -- see CviModel.h.
#include <cstring>

#include "TinyTTS/ChunkedVocoder.h"
#include "TinyTTS/CviModel.h"
#include "TinyTTS/TinyTTSCore.h"

namespace tinytts {

/**
 * @brief Runs the vocoder on the TPU of the Milk-V Duo 256M / Duo S
 * (SG2002/SG2000) through the CVITEK runtime, as a ChunkedVocoder.
 *
 * Loads the cvimodel built by research/milkv_duo/build_cvimodel.sh: inputs
 * z [1, inter_channels, W], g [1, gin_channels, 1] and mask [1, 1, W], output
 * audio [1, 1, W * 512], all float32. W is read from the model.
 *
 *   CviTpuVocoder tpu;
 *   if (tpu.begin("decoder_44100hz_w128_bf16.cvimodel")) tts.setVocoder(tpu.vocoderFn());
 *   ...  // tts.begin()
 *   if (!tpu.checkModel(tts.core())) ...  // decoder from another checkpoint
 *
 * The decoder only fits the weights.bin of the checkpoint it was exported
 * from: a model trained at another sample rate has another number of output
 * samples per frame (hop), and with a mismatch the audio silently plays at
 * the wrong speed and pitch. checkModel() catches that.
 *
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class CviTpuVocoder {
 public:
  /// `context`: overlap frames on each side of a window, see ChunkedVocoder
  /// (12 matches the one-pass decode to float rounding, 10 to >57dB).
  bool begin(const char* cvimodel_path, int context = 12) {
    if (!model_.begin(cvimodel_path)) return false;
    z_ = model_.input("z");
    g_ = model_.input("g");
    mask_ = model_.input("mask");
    audio_ = model_.output();
    if (!z_ || !g_ || !mask_ || !audio_ || !model_.shape(z_, channels_, window_)) return false;
    if (CVI_NN_TensorCount(mask_) != (size_t)window_) return model_.fail("mask doesn't match z");
    if (CVI_NN_TensorCount(audio_) % window_ != 0) return model_.fail("output isn't a whole number of frames");
    const int hop = (int)(CVI_NN_TensorCount(audio_) / window_);
    if (!chunked_.begin([this](const Mat& z, int vb, int ve, const Mat& g, float* audio) {
          return runWindow(z, vb, ve, g, audio);
        },
                        window_, context, hop))
      return model_.fail("context %d too large for a %d-frame window", context, window_);
    return true;
  }

  void end() { model_.end(); }

  /// Frames per TPU call, from the model.
  int window() const { return window_; }

  /// Output samples per frame, from the model.
  int hop() const { return chunked_.hop(); }

  /// After core.begin(): false (and an error message) if this decoder doesn't
  /// belong to the loaded weights -- a different number of output samples per
  /// frame, or a different sample rate in the model's file name. Models for
  /// 16, 22.05 and 24 kHz all have 256 samples per frame, so only the file
  /// name tells them apart.
  bool checkModel(const TinyTTSCore& core) const {
    if (hop() != core.vocoder().hopLength())
      return model_.fail("%d samples per frame, the weights' vocoder has %d -- export both from the same checkpoint",
                         hop(), core.vocoder().hopLength());
    return model_.checkSampleRate(core.sampleRate());
  }

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
      return model_.fail("z/g don't match the model (%d/%d channels)", z.cols(), g.cols());
    CviModel::setChannelFirst(z_, z, window_, window_);
    std::memcpy(CVI_NN_TensorPtr(g_), g.row(0), sizeof(float) * g.cols());
    CviModel::setMask(mask_, valid_begin, valid_end);
    if (!model_.forward()) return false;
    std::memcpy(audio, CVI_NN_TensorPtr(audio_), CVI_NN_TensorSize(audio_));
    return true;
  }

  CviModel model_;
  CVI_TENSOR *z_ = nullptr, *g_ = nullptr, *mask_ = nullptr, *audio_ = nullptr;
  int channels_ = 0;
  int window_ = 0;
  ChunkedVocoder chunked_;
};

}  // namespace tinytts
