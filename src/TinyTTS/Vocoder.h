#pragma once
#include <cmath>
#include <string>
#include <vector>

#include "TinyTTS/Mat.h"
#include "TinyTTS/Ops.h"
#include "TinyTTS/PsramStlAllocator.h"
#include "TinyTTS/WeightStore.h"

namespace tinytts {

/**
 * @brief Hand-written C++ port of tiny_tts.models.synthesizer.WaveformDecoder
 * (the HiFi-GAN-style vocoder, `net_g.dec`): Conv1d pre + N upsample stages
 * (LeakyReLU + ConvTranspose1d + 3 summed dilated ConvResBlocks) + LeakyReLU +
 * Conv1d post + tanh, with speaker conditioning added once before the first
 * upsample stage. Plain Conv1d/ConvTranspose1d/LeakyReLU -- no attention, no
 * TFLite Micro fixed-input-shape constraint, so an utterance of any length
 * runs in one pass (see docs/architecture.md for why this replaced the
 * TFLite Micro decoder).
 *
 * The released 44.1 kHz model has 5 stages with strides {8, 8, 2, 2, 2} (x512
 * samples per frame). Models trained at a lower sample rate
 * (research/training/audio_config.py) drop stages from the end; their
 * weights.bin carries `meta.upsample_rates`, and the stage count/kernel
 * sizes follow from that and the weight shapes. Weights without it are the
 * released layout.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class Vocoder {
 public:
  /// `ws` must outlive this object (conv weights are referenced, not copied).
  void begin(const WeightStore& ws, const std::string& prefix = "dec") {
    conv_pre_w_ = ws.convWeight(prefix + ".conv_pre.weight");
    conv_pre_b_ = ws.vec(prefix + ".conv_pre.bias");
    cond_w_ = ws.linearWeight(prefix + ".cond.weight");
    cond_b_ = ws.vec(prefix + ".cond.bias");

    std::vector<float> rates = ws.vec("meta.upsample_rates");
    if (rates.empty()) rates = {8, 8, 2, 2, 2};  // released 44.1 kHz model
    num_upsamples_ = (int)rates.size() < kMaxUpsamples ? (int)rates.size() : kMaxUpsamples;
    hop_length_ = 1;
    for (int i = 0; i < num_upsamples_; i++) {
      std::string p = prefix + ".ups." + std::to_string(i);
      ups_[i].weight = ws.convWeight(p + ".weight");
      ups_[i].bias = ws.vec(p + ".bias");
      ups_[i].stride = (int)rates[i];
      // ConvTranspose1d weight is [Cin, Cout, K]; upstream pads (K - stride) / 2.
      int kernel = ups_[i].weight ? ups_[i].weight->shape[2] : 0;
      ups_[i].padding = (kernel - ups_[i].stride) / 2;
      hop_length_ *= ups_[i].stride;
    }

    for (int idx = 0; idx < num_upsamples_ * kNumKernels; idx++) {
      std::string p = prefix + ".resblocks." + std::to_string(idx);
      for (int c = 0; c < 3; c++) {
        resblocks_[idx].convs1[c] = ws.convWeight(p + ".convs1." + std::to_string(c) + ".weight");
        resblocks_[idx].convs1_b[c] = ws.vec(p + ".convs1." + std::to_string(c) + ".bias");
        resblocks_[idx].convs2[c] = ws.convWeight(p + ".convs2." + std::to_string(c) + ".weight");
        resblocks_[idx].convs2_b[c] = ws.vec(p + ".convs2." + std::to_string(c) + ".bias");
      }
    }

    conv_post_w_ = ws.convWeight(prefix + ".conv_post.weight");
  }

  /// z: [T, inter_channels] (the flow's output). g: [1, gin_channels] speaker
  /// embedding. Returns interleaved mono PCM samples (range [-1,1]), length
  /// T * hopLength() (product of the upsample rates, 512 for the released
  /// 44.1 kHz model) --
  /// PSRAM-backed (PsramVector, see PsramStlAllocator.h): for anything but
  /// very short text this is hundreds of KB to a few MB, too big for
  /// internal DRAM.
  PsramVector<float> forward(const Mat& z, const Mat& g) const {
    Mat x = ops::conv1d(z, *conv_pre_w_, conv_pre_b_);
    Mat g_proj = ops::linear(g, cond_w_, cond_b_);  // [1, upsample_initial_channel]
    for (int t = 0; t < x.rows(); t++)
      for (int c = 0; c < x.cols(); c++) x.at(t, c) += g_proj.at(0, c);

    for (int i = 0; i < num_upsamples_; i++) {
      ops::leakyReluInplace(x, 0.1f);
      x = ops::convTranspose1d(x, *ups_[i].weight, ups_[i].bias, ups_[i].stride, ups_[i].padding);

      Mat xs;
      for (int j = 0; j < kNumKernels; j++) {
        Mat r = resblockForward(resblocks_[i * kNumKernels + j], x);
        if (j == 0) {
          xs = std::move(r);
        } else {
          ops::addInplace(xs, r);
        }
      }
      for (auto& v : xs.data()) v /= (float)kNumKernels;
      x = std::move(xs);
    }

    ops::leakyReluInplace(x, 0.01f);  // default F.leaky_relu() slope
    Mat audio_mat = ops::conv1d(x, *conv_post_w_, {});  // conv_post has no bias
    PsramVector<float> audio((size_t)audio_mat.rows());
    for (int t = 0; t < audio_mat.rows(); t++) audio[t] = std::tanh(audio_mat.at(t, 0));
    return audio;
  }

  /// Output samples per spectrogram frame (product of the upsample rates).
  int hopLength() const { return hop_length_; }
  int numUpsamples() const { return num_upsamples_; }

 private:
  static constexpr int kMaxUpsamples = 6;
  static constexpr int kNumKernels = 3;  // resblock_kernel_sizes: 3, 7, 11
  static constexpr int kDilations[3] = {1, 3, 5};

  struct UpsampleLayer {
    const WeightStore::Entry* weight = nullptr;
    std::vector<float> bias;
    int stride = 1;
    int padding = 0;
  };

  struct ResBlockWeights {
    const WeightStore::Entry* convs1[3] = {nullptr, nullptr, nullptr};
    std::vector<float> convs1_b[3];
    const WeightStore::Entry* convs2[3] = {nullptr, nullptr, nullptr};
    std::vector<float> convs2_b[3];
  };

  // Matches ConvResBlock.forward(): 3 (conv1(dilated) -> LeakyReLU ->
  // conv2(dilation=1)) residual pairs, no mask (single unpadded utterance).
  TINYTTS_HOT static Mat resblockForward(const ResBlockWeights& rb, const Mat& x_in) {
    Mat x = x_in;
    for (int c = 0; c < 3; c++) {
      Mat xt = x;
      ops::leakyReluInplace(xt, 0.1f);
      xt = ops::conv1d(xt, *rb.convs1[c], rb.convs1_b[c], kDilations[c]);
      ops::leakyReluInplace(xt, 0.1f);
      xt = ops::conv1d(xt, *rb.convs2[c], rb.convs2_b[c], /*dilation=*/1);
      ops::addInplace(xt, x);
      x = std::move(xt);
    }
    return x;
  }

  const WeightStore::Entry* conv_pre_w_ = nullptr;
  std::vector<float> conv_pre_b_;
  Mat cond_w_;
  std::vector<float> cond_b_;
  int num_upsamples_ = 0;
  int hop_length_ = 1;
  UpsampleLayer ups_[kMaxUpsamples];
  ResBlockWeights resblocks_[kMaxUpsamples * kNumKernels];
  const WeightStore::Entry* conv_post_w_ = nullptr;
};

}  // namespace tinytts
