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
    for (size_t i = 0; i < rel_k->count; i++) emb_rel_k_.data()[i] = rel_k->at(i);
    emb_rel_v_ = Mat(rel_v->shape[1], rel_v->shape[2]);
    for (size_t i = 0; i < rel_v->count; i++) emb_rel_v_.data()[i] = rel_v->at(i);

    n_heads_ = n_heads;
    k_channels_ = conv_q_.rows() / n_heads;
    window_size_ = window_size;
  }

  TINYTTS_HOT Mat forward(const Mat& x) const {
    int t_len = x.rows();
    int c = x.cols();

    Mat q = ops::linear(x, conv_q_, bias_q_);
    Mat key = ops::linear(x, conv_k_, bias_k_);
    Mat v = ops::linear(x, conv_v_, bias_v_);

    Mat merged(t_len, c);
    {
      TINYTTS_PROFILE_SCOPE(kAttentionScores);
      Mat scores(t_len, t_len);
      for (int head = 0; head < n_heads_; head++) {
#ifdef TINYTTS_FIXED_POINT
        headFixed(q, key, v, head, scores, merged);
#else
        headFloat(q, key, v, head, scores, merged);
#endif
      }
    }

    return ops::linear(merged, conv_o_, bias_o_);
  }

  int outChannels() const { return conv_o_.rows(); }

 private:
  // The relative-position embeddings emb_rel_k_/emb_rel_v_ have one row per
  // offset j - i in [-window_size_, window_size_] (row offset +
  // window_size_) and are zero beyond that, so both heads below only visit
  // |j - i| <= window_size_ for them - the same result as looping over all
  // 2T-1 offsets, at a fraction of the work for long inputs.

  /// One head in float: scores = softmax((q.k + q.rel_k) / sqrt(k)),
  /// merged = scores.v + scores.rel_v.
  TINYTTS_HOT void headFloat(const Mat& q, const Mat& key, const Mat& v, int head, Mat& scores,
                             Mat& merged) const {
    int t_len = q.rows(), k = k_channels_, off = head * k, w = window_size_;
    float scale = 1.0f / std::sqrt((float)k);
    for (int i = 0; i < t_len; i++) {
      const float* qi = q.row(i) + off;
      float* si = scores.row(i);
      for (int j = 0; j < t_len; j++) {
        const float* kj = key.row(j) + off;
        float acc = 0.0f;
        for (int ch = 0; ch < k; ch++) acc += qi[ch] * kj[ch];
        int d = j - i;
        if (d >= -w && d <= w) {
          const float* rk = emb_rel_k_.row(d + w);
          for (int ch = 0; ch < k; ch++) acc += qi[ch] * rk[ch];
        }
        si[j] = acc * scale;
      }
      ops::softmaxInplace(si, t_len);
    }
    for (int i = 0; i < t_len; i++) {
      const float* si = scores.row(i);
      float* mi = merged.row(i) + off;
      for (int ch = 0; ch < k; ch++) {
        float acc = 0.0f;
        for (int j = 0; j < t_len; j++) acc += si[j] * v.at(j, off + ch);
        for (int d = -w; d <= w; d++) {
          int j = i + d;
          if (j < 0 || j >= t_len) continue;
          acc += si[j] * emb_rel_v_.at(d + w, ch);
        }
        mi[ch] = acc;
      }
    }
  }

#ifdef TINYTTS_FIXED_POINT
  /// Quantizes rows x cols values (row stride `stride`) to 16 bits with one
  /// shared scale; returns the scale (value ~= q * scale).
  static float quantize16(const float* src, int rows, int cols, int stride, std::vector<int16_t>& out) {
    float m = 0.0f;
    for (int r = 0; r < rows; r++) m = std::max(m, ops::detail::absMaxBits(src + (size_t)r * stride, cols));
    float s = m > 0.0f ? m / 32767.0f : 1.0f;
    float inv = 1.0f / s;
    out.resize((size_t)rows * cols);
    for (int r = 0; r < rows; r++)
      for (int ch = 0; ch < cols; ch++) {
        float x = src[(size_t)r * stride + ch] * inv;
        out[(size_t)r * cols + ch] = (int16_t)(x + (x >= 0.0f ? 0.5f : -0.5f));
      }
    return s;
  }

  /// One head as headFloat(), with the dot products in integers (see
  /// TINYTTS_FIXED_POINT in Ops.h): q, k, v and the relative embeddings as
  /// 16-bit values with one scale each, the softmax weights as Q15, sums in
  /// 64 bits - one float conversion per score and per output instead of a
  /// float multiply and add per term.
  TINYTTS_HOT void headFixed(const Mat& q, const Mat& key, const Mat& v, int head, Mat& scores,
                             Mat& merged) const {
    int t_len = q.rows(), k = k_channels_, off = head * k, w = window_size_;
    int nrel = 2 * w + 1;
    std::vector<int16_t> qq, kq, vq, rkq, rvq, sw((size_t)t_len);
    float sqs = quantize16(q.data().data() + off, t_len, k, q.cols(), qq);
    float sks = quantize16(key.data().data() + off, t_len, k, key.cols(), kq);
    float svs = quantize16(v.data().data() + off, t_len, k, v.cols(), vq);
    float srk = quantize16(emb_rel_k_.data().data(), nrel, k, emb_rel_k_.cols(), rkq);
    float srv = quantize16(emb_rel_v_.data().data(), nrel, k, emb_rel_v_.cols(), rvq);
    float scale = 1.0f / std::sqrt((float)k);
    float f_qk = sqs * sks * scale, f_qr = sqs * srk * scale;
    for (int i = 0; i < t_len; i++) {
      const int16_t* qi = qq.data() + (size_t)i * k;
      float* si = scores.row(i);
      for (int j = 0; j < t_len; j++) {
        const int16_t* kj = kq.data() + (size_t)j * k;
        int64_t acc = 0;
        for (int ch = 0; ch < k; ch++) acc += (int32_t)qi[ch] * kj[ch];
        float sc = (float)acc * f_qk;
        int d = j - i;
        if (d >= -w && d <= w) {
          const int16_t* rk = rkq.data() + (size_t)(d + w) * k;
          int64_t racc = 0;
          for (int ch = 0; ch < k; ch++) racc += (int32_t)qi[ch] * rk[ch];
          sc += (float)racc * f_qr;
        }
        si[j] = sc;
      }
      ops::softmaxInplace(si, t_len);
    }
    const float f_v = svs / 32767.0f, f_rv = srv / 32767.0f;
    for (int i = 0; i < t_len; i++) {
      const float* si = scores.row(i);
      for (int j = 0; j < t_len; j++) sw[j] = (int16_t)(si[j] * 32767.0f + 0.5f);  // softmax output is in [0, 1]
      float* mi = merged.row(i) + off;
      for (int ch = 0; ch < k; ch++) {
        int64_t acc = 0;
        for (int j = 0; j < t_len; j++) acc += (int32_t)sw[j] * vq[(size_t)j * k + ch];
        int64_t racc = 0;
        for (int d = -w; d <= w; d++) {
          int j = i + d;
          if (j < 0 || j >= t_len) continue;
          racc += (int32_t)sw[j] * rvq[(size_t)(d + w) * k + ch];
        }
        mi[ch] = (float)acc * f_v + (float)racc * f_rv;
      }
    }
  }
#endif

  Mat conv_q_, conv_k_, conv_v_, conv_o_;
  std::vector<float> bias_q_, bias_k_, bias_v_, bias_o_;
  Mat emb_rel_k_, emb_rel_v_;
  int n_heads_ = 1, k_channels_ = 1, window_size_ = 4;
};

}  // namespace tinytts
