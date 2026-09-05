#pragma once
#include <string>
#include <vector>

#include "TinyTTS/Mat.h"
#include "TinyTTS/Ops.h"
#include "TinyTTS/TransformerBlock.h"
#include "TinyTTS/WeightStore.h"

namespace tinytts {

/**
 * @brief The normalizing flow that maps the prior latent z_p to the
 * posterior latent z consumed by the decoder/vocoder: 4 transformer-based
 * affine coupling layers interleaved with channel-flips, run in reverse
 * (inference direction only -- training-direction forward pass is not
 * implemented, it's not needed on-device). Ported from
 * tiny_tts.models.synthesizer.AttentionFlowBlock + nn.modules.TransformerCouplingLayer.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class Flow {
 public:
  /// `ws` must outlive this object.
  void begin(const WeightStore& ws, int n_flows, int n_heads, int window_size) {
    layers_.resize(n_flows);
    for (int i = 0; i < n_flows; i++) {
      std::string prefix = "flow.flows." + std::to_string(i * 2);
      CouplingLayer& cw = layers_[i];
      cw.pre_w = ws.linearWeight(prefix + ".pre.weight");
      cw.pre_b = ws.vec(prefix + ".pre.bias");
      cw.enc.begin(ws, prefix + ".enc", /*n_layers=*/3, n_heads, window_size);
      cw.post_w = ws.linearWeight(prefix + ".post.weight");
      cw.post_b = ws.vec(prefix + ".post.bias");
      cw.half_channels = cw.pre_w.cols();
    }
  }

  /// AttentionFlowBlock.forward(x, x_mask, g, reverse=True):
  ///   for flow in reversed(self.flows): x = flow(x, x_mask, g=g, reverse=True)
  /// flows = [CL0,Flip,CL1,Flip,CL2,Flip,CL3,Flip] (8 entries); reversed =
  /// [Flip,CL3,Flip,CL2,Flip,CL1,Flip,CL0] -- starts AND ends with a
  /// Flip/CouplingLayer pair, no trailing flip after the last one.
  Mat reverse(Mat x, const Mat& g) const {
    for (int i = (int)layers_.size() - 1; i >= 0; i--) {
      x = flipChannels(x);
      x = couplingLayerReverse(x, layers_[i], g);
    }
    return x;
  }

 private:
  struct CouplingLayer {
    Mat pre_w;  // [hidden, half]
    std::vector<float> pre_b;
    TransformerBlock enc;
    Mat post_w;  // [half, hidden] (mean_only -> out = half, not 2*half)
    std::vector<float> post_b;
    int half_channels = 16;
  };

  // Reverse-mode affine coupling (mean_only, so logs==0 and the inverse is
  // simply x1 -= m).
  static Mat couplingLayerReverse(const Mat& x, const CouplingLayer& w, const Mat& g) {
    int t_len = x.rows();
    int half = w.half_channels;
    Mat x0(t_len, half), x1(t_len, half);
    for (int t = 0; t < t_len; t++) {
      for (int c = 0; c < half; c++) {
        x0.at(t, c) = x.at(t, c);
        x1.at(t, c) = x.at(t, c + half);
      }
    }
    Mat h = ops::linear(x0, w.pre_w, w.pre_b);
    h = w.enc.forward(h, &g);
    Mat m = ops::linear(h, w.post_w, w.post_b);  // [T, half]

    Mat out(t_len, 2 * half);
    for (int t = 0; t < t_len; t++) {
      for (int c = 0; c < half; c++) {
        out.at(t, c) = x0.at(t, c);
        out.at(t, c + half) = x1.at(t, c) - m.at(t, c);
      }
    }
    return out;
  }

  static Mat flipChannels(const Mat& x) {
    Mat out(x.rows(), x.cols());
    for (int t = 0; t < x.rows(); t++)
      for (int c = 0; c < x.cols(); c++) out.at(t, c) = x.at(t, x.cols() - 1 - c);
    return out;
  }

  std::vector<CouplingLayer> layers_;
};

}  // namespace tinytts
