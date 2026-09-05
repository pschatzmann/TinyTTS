#pragma once
#include <cmath>
#include <vector>

#include "TinyTTS/Mat.h"
#include "TinyTTS/Ops.h"
#include "TinyTTS/WeightStore.h"

namespace tinytts {

/**
 * @brief Windowed relative-position multi-head self-attention (VITS/Bert-VITS2
 * style). A from-scratch C++ implementation -- not derived from onnx2tf
 * output, which cannot convert this module (see project plan). Formulas
 * verified bit-exact against the PyTorch reference during development.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class Attention {
 public:
  /// Loads weights for one attention layer at `prefix` (e.g.
  /// "enc_p.encoder.attn_layers.0" or "flow.flows.0.enc.attn_layers.1").
  void begin(const WeightStore& ws, const std::string& prefix, int n_heads, int window_size) {
    conv_q_ = ws.linearWeight(prefix + ".conv_q.weight");
    bias_q_ = ws.vec(prefix + ".conv_q.bias");
    conv_k_ = ws.linearWeight(prefix + ".conv_k.weight");
    bias_k_ = ws.vec(prefix + ".conv_k.bias");
    conv_v_ = ws.linearWeight(prefix + ".conv_v.weight");
    bias_v_ = ws.vec(prefix + ".conv_v.bias");
    conv_o_ = ws.linearWeight(prefix + ".conv_o.weight");
    bias_o_ = ws.vec(prefix + ".conv_o.bias");

    const WeightStore::Entry* rel_k = ws.get(prefix + ".emb_rel_k");
    const WeightStore::Entry* rel_v = ws.get(prefix + ".emb_rel_v");
    emb_rel_k_ = Mat(rel_k->shape[1], rel_k->shape[2]);
    emb_rel_k_.data() = rel_k->data;
    emb_rel_v_ = Mat(rel_v->shape[1], rel_v->shape[2]);
    emb_rel_v_.data() = rel_v->data;

    n_heads_ = n_heads;
    k_channels_ = conv_q_.rows() / n_heads;
    window_size_ = window_size;
  }

  Mat forward(const Mat& x) const {
    int t_len = x.rows();
    int c = x.cols();
    int h = n_heads_, k = k_channels_;
    float scale = 1.0f / std::sqrt((float)k);

    Mat q = ops::linear(x, conv_q_, bias_q_);
    Mat key = ops::linear(x, conv_k_, bias_k_);
    Mat v = ops::linear(x, conv_v_, bias_v_);

    Mat rel_k = relativeEmbeddings(emb_rel_k_, t_len);
    Mat rel_v = relativeEmbeddings(emb_rel_v_, t_len);
    int width = 2 * t_len - 1;

    Mat merged(t_len, c);
    Mat scores(t_len, t_len);

    for (int head = 0; head < h; head++) {
      int off = head * k;
      for (int i = 0; i < t_len; i++) {
        for (int j = 0; j < t_len; j++) {
          float acc = 0.0f;
          for (int ch = 0; ch < k; ch++) acc += q.at(i, off + ch) * key.at(j, off + ch);
          int r = j - i + (t_len - 1);
          float rel_acc = 0.0f;
          for (int ch = 0; ch < k; ch++) rel_acc += q.at(i, off + ch) * rel_k.at(r, ch);
          scores.at(i, j) = (acc + rel_acc) * scale;
        }
      }
      for (int i = 0; i < t_len; i++) ops::softmaxInplace(scores.row(i), t_len);

      for (int i = 0; i < t_len; i++) {
        for (int ch = 0; ch < k; ch++) {
          float acc = 0.0f;
          for (int j = 0; j < t_len; j++) acc += scores.at(i, j) * v.at(j, off + ch);
          for (int r = 0; r < width; r++) {
            int j = i + r - (t_len - 1);
            if (j < 0 || j >= t_len) continue;
            acc += scores.at(i, j) * rel_v.at(r, ch);
          }
          merged.at(i, off + ch) = acc;
        }
      }
    }

    return ops::linear(merged, conv_o_, bias_o_);
  }

  int outChannels() const { return conv_o_.rows(); }

 private:
  // table: [2*window_size+1, k_channels] entries for relative offsets
  // [-window_size .. +window_size]. Returns [2T-1, k_channels] for offsets
  // [-(T-1) .. +(T-1)], zero for any offset beyond the trained window.
  Mat relativeEmbeddings(const Mat& table, int t_len) const {
    int width = 2 * t_len - 1;
    Mat result(width, table.cols(), 0.0f);
    for (int r = 0; r < width; r++) {
      int offset = r - (t_len - 1);
      if (offset >= -window_size_ && offset <= window_size_) {
        int tidx = offset + window_size_;
        for (int c = 0; c < table.cols(); c++) result.at(r, c) = table.at(tidx, c);
      }
    }
    return result;
  }

  Mat conv_q_, conv_k_, conv_v_, conv_o_;
  std::vector<float> bias_q_, bias_k_, bias_v_, bias_o_;
  Mat emb_rel_k_, emb_rel_v_;
  int n_heads_ = 1, k_channels_ = 1, window_size_ = 4;
};

}  // namespace tinytts
