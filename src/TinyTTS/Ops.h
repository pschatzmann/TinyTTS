#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "TinyTTS/Mat.h"
#include "TinyTTS/WeightStore.h"

namespace tinytts {

/**
 * @brief Primitive tensor ops shared by Attention/TransformerBlock/PhonemeEncoder/Flow.
 * All activations are [T, C] (time-major, channel-last). Batch=1 always
 * (single-utterance embedded inference).
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
namespace ops {

/// Conv1d(kernel_size=1) == per-timestep Linear. y[t,co] = sum_ci x[t,ci]*W[co,ci] + b[co]
inline Mat linear(const Mat& x, const Mat& w, const std::vector<float>& bias) {
  Mat y(x.rows(), w.rows());
  for (int t = 0; t < x.rows(); t++) {
    const float* xr = x.row(t);
    float* yr = y.row(t);
    for (int co = 0; co < w.rows(); co++) {
      float acc = bias.empty() ? 0.0f : bias[co];
      const float* wr = w.row(co);
      for (int ci = 0; ci < w.cols(); ci++) acc += xr[ci] * wr[ci];
      yr[co] = acc;
    }
  }
  return y;
}

/// General Conv1d, weight shape [Cout, Cin, K], "same" padding (pad=(K-1)/2), stride 1.
inline Mat conv1d(const Mat& x, const WeightStore::Entry& w, const std::vector<float>& bias) {
  int cout = w.shape[0], cin = w.shape[1], k = w.shape[2];
  int pad = (k - 1) / 2;
  Mat y(x.rows(), cout);
  for (int t = 0; t < x.rows(); t++) {
    float* yr = y.row(t);
    for (int co = 0; co < cout; co++) {
      float acc = bias.empty() ? 0.0f : bias[co];
      for (int kk = 0; kk < k; kk++) {
        int ti = t + kk - pad;
        if (ti < 0 || ti >= x.rows()) continue;
        const float* xr = x.row(ti);
        const float* wbase = w.data.data() + ((size_t)co * cin) * k + kk;
        for (int ci = 0; ci < cin; ci++) acc += xr[ci] * wbase[(size_t)ci * k];
      }
      yr[co] = acc;
    }
  }
  return y;
}

/// LayerNorm over the channel dim, applied independently per timestep (==
/// the model's "ChannelNorm"/"ChannelLayerNorm" -- both are plain
/// per-timestep layer-norm over C despite the different class names in the
/// Python source).
inline void channelLayerNormInplace(Mat& x, const std::vector<float>& gamma, const std::vector<float>& beta,
                                     float eps = 1e-5f) {
  for (int t = 0; t < x.rows(); t++) {
    float* r = x.row(t);
    float mean = 0.0f;
    for (int c = 0; c < x.cols(); c++) mean += r[c];
    mean /= x.cols();
    float var = 0.0f;
    for (int c = 0; c < x.cols(); c++) {
      float d = r[c] - mean;
      var += d * d;
    }
    var /= x.cols();
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int c = 0; c < x.cols(); c++) r[c] = (r[c] - mean) * inv_std * gamma[c] + beta[c];
  }
}

inline void reluInplace(Mat& x) {
  for (auto& v : x.data()) v = std::max(0.0f, v);
}

inline void addInplace(Mat& a, const Mat& b) {
  for (size_t i = 0; i < a.data().size(); i++) a.data()[i] += b.data()[i];
}

inline void softmaxInplace(float* row, int n) {
  float mx = row[0];
  for (int i = 1; i < n; i++) mx = std::max(mx, row[i]);
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    row[i] = std::exp(row[i] - mx);
    sum += row[i];
  }
  for (int i = 0; i < n; i++) row[i] /= sum;
}

}  // namespace ops
}  // namespace tinytts
