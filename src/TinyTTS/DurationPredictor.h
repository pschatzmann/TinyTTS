#pragma once
#include <vector>

#include "TinyTTS/Mat.h"
#include "TinyTTS/Ops.h"
#include "TinyTTS/WeightStore.h"

namespace tinytts {

/**
 * @brief Hand-written C++ port of tiny_tts.models.synthesizer.DurationEstimator:
 * Conv1d(k=3)+ChannelNorm+ReLU, twice, then a Conv1d(k=1) projection to a
 * single duration-logit channel, with speaker conditioning added before the
 * first conv. Plain Conv1d/LayerNorm/ReLU -- no attention, no TFLite Micro
 * fixed-window constraint, so an utterance of any length runs in one pass.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class DurationPredictor {
 public:
  /// `ws` must outlive this object (kernel_size=3 conv weights are referenced,
  /// not copied).
  void begin(const WeightStore& ws, const std::string& prefix = "dp") {
    conv1_w_ = ws.convWeight(prefix + ".conv_1.weight");
    conv1_b_ = ws.vec(prefix + ".conv_1.bias");
    norm1_gamma_ = ws.vec(prefix + ".norm_1.gamma");
    norm1_beta_ = ws.vec(prefix + ".norm_1.beta");
    conv2_w_ = ws.convWeight(prefix + ".conv_2.weight");
    conv2_b_ = ws.vec(prefix + ".conv_2.bias");
    norm2_gamma_ = ws.vec(prefix + ".norm_2.gamma");
    norm2_beta_ = ws.vec(prefix + ".norm_2.beta");
    proj_w_ = ws.linearWeight(prefix + ".proj.weight");
    proj_b_ = ws.vec(prefix + ".proj.bias");
    cond_w_ = ws.linearWeight(prefix + ".cond.weight");
    cond_b_ = ws.vec(prefix + ".cond.bias");
  }

  /// x: [T, in_channels] (the encoder's hidden state). g: [1, gin_channels]
  /// speaker embedding. Returns logw: length-T duration logits (one value
  /// per phoneme, matching the reference's `proj` output channel count of 1).
  std::vector<float> forward(const Mat& x_in, const Mat& g) const {
    Mat x = x_in;
    Mat g_proj = ops::linear(g, cond_w_, cond_b_);  // [1, in_channels]
    for (int t = 0; t < x.rows(); t++)
      for (int c = 0; c < x.cols(); c++) x.at(t, c) += g_proj.at(0, c);

    Mat h = ops::conv1d(x, *conv1_w_, conv1_b_);
    ops::reluInplace(h);
    ops::channelLayerNormInplace(h, norm1_gamma_, norm1_beta_);

    h = ops::conv1d(h, *conv2_w_, conv2_b_);
    ops::reluInplace(h);
    ops::channelLayerNormInplace(h, norm2_gamma_, norm2_beta_);

    Mat proj = ops::linear(h, proj_w_, proj_b_);  // [T, 1]
    std::vector<float> logw((size_t)proj.rows());
    for (int t = 0; t < proj.rows(); t++) logw[t] = proj.at(t, 0);
    return logw;
  }

 private:
  const WeightStore::Entry* conv1_w_ = nullptr;
  std::vector<float> conv1_b_;
  std::vector<float> norm1_gamma_, norm1_beta_;
  const WeightStore::Entry* conv2_w_ = nullptr;
  std::vector<float> conv2_b_;
  std::vector<float> norm2_gamma_, norm2_beta_;
  Mat proj_w_;
  std::vector<float> proj_b_;
  Mat cond_w_;
  std::vector<float> cond_b_;
};

}  // namespace tinytts
