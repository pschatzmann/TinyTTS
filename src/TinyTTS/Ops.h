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

/// General Conv1d, weight shape [Cout, Cin, K], "same" padding
/// (pad=dilation*(K-1)/2, K odd), stride 1.
inline Mat conv1d(const Mat& x, const WeightStore::Entry& w, const std::vector<float>& bias, int dilation = 1) {
  int cout = w.shape[0], cin = w.shape[1], k = w.shape[2];
  int pad = dilation * (k - 1) / 2;
  Mat y(x.rows(), cout);
  for (int t = 0; t < x.rows(); t++) {
    float* yr = y.row(t);
    for (int co = 0; co < cout; co++) {
      float acc = bias.empty() ? 0.0f : bias[co];
      for (int kk = 0; kk < k; kk++) {
        int ti = t + kk * dilation - pad;
        if (ti < 0 || ti >= x.rows()) continue;
        const float* xr = x.row(ti);
        size_t wbase = ((size_t)co * cin) * k + kk;
        for (int ci = 0; ci < cin; ci++) acc += xr[ci] * w.at(wbase + (size_t)ci * k);
      }
      yr[co] = acc;
    }
  }
  return y;
}

/// ConvTranspose1d, weight shape [Cin, Cout, K] (PyTorch's ConvTranspose1d
/// layout -- note the axis order differs from conv1d()'s [Cout, Cin, K]),
/// stride/padding as in PyTorch (dilation=1, output_padding=0). Output
/// length = (T_in-1)*stride - 2*padding + K.
inline Mat convTranspose1d(const Mat& x, const WeightStore::Entry& w, const std::vector<float>& bias, int stride,
                            int padding) {
  int cin = w.shape[0], cout = w.shape[1], k = w.shape[2];
  int t_out_len = (x.rows() - 1) * stride - 2 * padding + k;
  Mat y(t_out_len, cout);
  for (int co = 0; co < cout; co++) {
    float b = bias.empty() ? 0.0f : bias[co];
    for (int t = 0; t < t_out_len; t++) y.at(t, co) = b;
  }
  for (int ti = 0; ti < x.rows(); ti++) {
    const float* xr = x.row(ti);
    for (int kk = 0; kk < k; kk++) {
      int t_out = ti * stride - padding + kk;
      if (t_out < 0 || t_out >= t_out_len) continue;
      float* yr = y.row(t_out);
      for (int ci = 0; ci < cin; ci++) {
        float xv = xr[ci];
        size_t wbase = ((size_t)ci * cout) * k + kk;
        for (int co = 0; co < cout; co++) yr[co] += xv * w.at(wbase + (size_t)co * k);
      }
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

inline void leakyReluInplace(Mat& x, float slope = 0.1f) {
  for (auto& v : x.data())
    if (v < 0.0f) v *= slope;
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
