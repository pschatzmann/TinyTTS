#pragma once
#include <vector>

#include "TinyTTS/Attention.h"
#include "TinyTTS/Mat.h"
#include "TinyTTS/Ops.h"
#include "TinyTTS/WeightStore.h"

namespace tinytts {

/**
 * @brief A stack of (windowed relative-position self-attention + FeedForward
 * + LayerNorm) layers, with optional speaker-embedding conditioning injected
 * before one layer (cond_layer_idx). Used by both PhonemeEncoder's own
 * transformer and each Flow coupling layer's internal transformer.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class TransformerBlock {
 public:
  void begin(const WeightStore& ws, const std::string& prefix, int n_layers, int n_heads, int window_size,
             int cond_layer_idx = 2) {
    cond_layer_idx_ = cond_layer_idx;
    spk_emb_linear_w_ = ws.linear2dWeight(prefix + ".spk_emb_linear.weight");
    spk_emb_linear_b_ = ws.vec(prefix + ".spk_emb_linear.bias");

    layers_.resize(n_layers);
    for (int i = 0; i < n_layers; i++) {
      Layer& lw = layers_[i];
      std::string lp = prefix + ".attn_layers." + std::to_string(i);
      lw.attn.begin(ws, lp, n_heads, window_size);

      lw.norm1_gamma = ws.vec(prefix + ".norm_layers_1." + std::to_string(i) + ".gamma");
      lw.norm1_beta = ws.vec(prefix + ".norm_layers_1." + std::to_string(i) + ".beta");

      std::string fp = prefix + ".ffn_layers." + std::to_string(i);
      lw.ffn_conv1 = ws.convWeight(fp + ".conv_1.weight");
      lw.ffn_bias1 = ws.vec(fp + ".conv_1.bias");
      lw.ffn_conv2 = ws.convWeight(fp + ".conv_2.weight");
      lw.ffn_bias2 = ws.vec(fp + ".conv_2.bias");

      lw.norm2_gamma = ws.vec(prefix + ".norm_layers_2." + std::to_string(i) + ".gamma");
      lw.norm2_beta = ws.vec(prefix + ".norm_layers_2." + std::to_string(i) + ".beta");
    }
  }

  /// x: [T, hidden]. g: [1, gin_channels] (the speaker embedding), or
  /// nullptr if unconditioned.
  Mat forward(Mat x, const Mat* g) const {
    for (int i = 0; i < (int)layers_.size(); i++) {
      if (i == cond_layer_idx_ && g != nullptr) {
        Mat g_proj = ops::linear(*g, spk_emb_linear_w_, spk_emb_linear_b_);  // [1, hidden]
        for (int t = 0; t < x.rows(); t++)
          for (int c = 0; c < x.cols(); c++) x.at(t, c) += g_proj.at(0, c);
      }
      const Layer& lw = layers_[i];

      Mat y = lw.attn.forward(x);
      ops::addInplace(y, x);
      ops::channelLayerNormInplace(y, lw.norm1_gamma, lw.norm1_beta);
      x = y;

      Mat ff = ops::conv1d(x, *lw.ffn_conv1, lw.ffn_bias1);
      ops::reluInplace(ff);
      ff = ops::conv1d(ff, *lw.ffn_conv2, lw.ffn_bias2);
      ops::addInplace(ff, x);
      ops::channelLayerNormInplace(ff, lw.norm2_gamma, lw.norm2_beta);
      x = ff;
    }
    return x;
  }

 private:
  struct Layer {
    Attention attn;
    std::vector<float> norm1_gamma, norm1_beta;
    const WeightStore::Entry* ffn_conv1 = nullptr;
    std::vector<float> ffn_bias1;
    const WeightStore::Entry* ffn_conv2 = nullptr;
    std::vector<float> ffn_bias2;
    std::vector<float> norm2_gamma, norm2_beta;
  };

  std::vector<Layer> layers_;
  Mat spk_emb_linear_w_;
  std::vector<float> spk_emb_linear_b_;
  int cond_layer_idx_ = 2;
};

}  // namespace tinytts
