#pragma once
// Linux on a Raspberry Pi 1 / Zero only -- see QpuDevice.h and
// docs/rpi-zero-gpu.md.
#include <cmath>
#include <string>
#include <vector>

#include "TinyTTS/QpuConv.h"
#include "TinyTTS/TinyTTSCore.h"

namespace tinytts {

/**
 * @brief The flow (Flow.h, reverse direction) on the GPU of a Raspberry Pi
 * 1 / Zero: 4 coupling layers, each a linear layer, a 3-layer transformer
 * (windowed relative-position attention, FFN, LayerNorm) and a linear layer,
 * all on the 12 QPUs (QpuConv). Only z_p goes in and z comes out.
 *
 *   QpuFlow flow;
 *   ...  // tts.begin()
 *   if (flow.begin(tts.core())) tts.setFlow(flow.flowFn());
 *
 * The channel flips between the coupling layers cost nothing: they are
 * folded into the order of the pre/post weights. The speaker conditioning
 * is added to a LayerNorm's beta. Utterances longer than `max_frames` fall
 * back to the CPU flow (TinyTTSCore does that when flowFn() returns false).
 * Needs the model's flow shape: 32 channels, 2 heads of 16, window 4.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class QpuFlow {
 public:
  static constexpr int kChannels = 32;
  static constexpr int kHeads = 2;
  static constexpr int kLayers = 3;    // transformer layers per coupling layer
  static constexpr int kCondLayer = 2; // speaker conditioning before this layer

  /// GPU memory begin() needs for `max_frames`.
  static uint32_t arenaBytes(int max_frames) {
    const size_t row = QpuConv::rowLength(max_frames), T16 = QpuConv::t16(max_frames);
    const size_t buffers = (6 * kChannels + 128) * row + kHeads * (16 + 32) * T16;
    return (uint32_t)(buffers * 4 + (4u << 20));  // + weights (~2.5 MB), kernels, jobs
  }

  /// After core.begin(), with its own GPU memory.
  bool begin(const TinyTTSCore& core, int max_frames = 512, int n_flows = 4) {
    if (!own_.begin(arenaBytes(max_frames))) return false;
    return begin(core.weights(), own_, max_frames, n_flows);
  }

  /// In `conv`'s GPU memory (shared with e.g. a QpuVocoder; see
  /// arenaBytes()). `ws` and `conv` must outlive this object.
  bool begin(const WeightStore& ws, QpuConv& conv, int max_frames = 512, int n_flows = 4) {
    conv_ = &conv;
    max_frames_ = max_frames;
    const int row = QpuConv::rowLength(max_frames), T16 = QpuConv::t16(max_frames);
    for (auto* b : {&x_, &h_, &q_, &m_, &y_, &x1_})
      if (!conv.allocBuffer(*b, kChannels, max_frames)) return fail("out of GPU memory");
    if (!conv.allocBuffer(f_, 128, max_frames)) return fail("out of GPU memory");
    k_.assign(kHeads, nullptr);
    kv_.assign(kHeads, nullptr);
    for (int h = 0; h < kHeads; h++) {
      k_[h] = (float*)conv.alloc((size_t)T16 * 16 * 4);
      kv_[h] = (float*)conv.alloc((size_t)T16 * 32 * 4);
      if (!k_[h] || !kv_[h]) return fail("out of GPU memory");
    }

    couplings_.assign(n_flows, Coupling());
    for (int i = 0; i < n_flows; i++) {
      Coupling& cp = couplings_[i];
      const std::string p = "flow.flows." + std::to_string(2 * i);
      // reverse() flips the channels before each coupling layer, from the last
      // to the first: before layer i, n_flows - i flips have happened.
      cp.reversed = (n_flows - i) % 2 == 1;
      const bool rev = cp.reversed;
      Mat pre_w = ws.linearWeight(p + ".pre.weight"), post_w = ws.linearWeight(p + ".post.weight");
      std::vector<float> pre_b = ws.vec(p + ".pre.bias"), post_b = ws.vec(p + ".post.bias");
      const int half = pre_w.cols();
      if (half != kChannels / 2 || pre_w.rows() != kChannels) return fail("unexpected flow shape");
      // x0 = logical channels 0..15: physical rows 16..31 reversed when flipped.
      if (!conv.prepareConv(cp.pre, half, kChannels, 1, 1, row, pre_b,
                            [&](int co, int ci, int) { return pre_w.at(co, rev ? half - 1 - ci : ci); }))
        return fail("pre");
      // out x1 = x1 - post(h): weights negated, x1 index a of the physical rows.
      std::vector<float> nb(half);
      for (int a = 0; a < half; a++) nb[a] = -post_b[rev ? half - 1 - a : a];
      if (!conv.prepareConv(cp.post, kChannels, half, 1, 1, row, nb,
                            [&](int a, int ci, int) { return -post_w.at(rev ? half - 1 - a : a, ci); }))
        return fail("post");

      const std::string e = p + ".enc";
      cp.spk_w = ws.linear2dWeight(e + ".spk_emb_linear.weight");
      cp.spk_b = ws.vec(e + ".spk_emb_linear.bias");
      for (int l = 0; l < kLayers; l++) {
        Layer& ly = cp.layers[l];
        const std::string a = e + ".attn_layers." + std::to_string(l), L = std::to_string(l);
        Mat wq = ws.linearWeight(a + ".conv_q.weight"), wk = ws.linearWeight(a + ".conv_k.weight"),
            wv = ws.linearWeight(a + ".conv_v.weight"), wo = ws.linearWeight(a + ".conv_o.weight");
        std::vector<float> bq = ws.vec(a + ".conv_q.bias"), bk = ws.vec(a + ".conv_k.bias"),
                           bv = ws.vec(a + ".conv_v.bias"), bo = ws.vec(a + ".conv_o.bias");
        // Scores in log2 units for the kernel's exp2: q * log2(e) / sqrt(16).
        const float qs = 1.4426950408889634f / std::sqrt(16.0f);
        std::vector<float> bqs(bq);
        for (float& v : bqs) v *= qs;
        // Keys and values per head, interleaved: K h0, V h0, K h1, V h1.
        auto kv_row = [](int co, int& part) {
          part = co % 32 / 16;
          return co / 32 * 16 + co % 16;
        };
        std::vector<float> bkv(64);
        for (int co = 0; co < 64; co++) {
          int part, r = kv_row(co, part);
          bkv[co] = part ? bv[r] : bk[r];
        }
        const WeightStore::Entry* rel_k = ws.get(a + ".emb_rel_k");
        const WeightStore::Entry* rel_v = ws.get(a + ".emb_rel_v");
        const WeightStore::Entry* ffn1 = ws.get(e + ".ffn_layers." + L + ".conv_1.weight");
        const WeightStore::Entry* ffn2 = ws.get(e + ".ffn_layers." + L + ".conv_2.weight");
        if (!rel_k || !rel_v || !ffn1 || !ffn2 || wq.rows() != kChannels) return fail("missing transformer weights");
        if (!conv.prepareConv(ly.q, kChannels, kChannels, 1, 1, row, bqs,
                              [&](int co, int ci, int) { return wq.at(co, ci) * qs; }) ||
            !conv.prepareConv(ly.k, kChannels, kChannels, 1, 1, row, bk,
                              [&](int co, int ci, int) { return wk.at(co, ci); }) ||
            !conv.prepareConv(ly.kv, kChannels, 64, 1, 1, row, bkv,
                              [&](int co, int ci, int) {
                                int part, r = kv_row(co, part);
                                return part ? wv.at(r, ci) : wk.at(r, ci);
                              }) ||
            !conv.prepareConv(ly.o, kChannels, kChannels, 1, 1, row, bo,
                              [&](int co, int ci, int) { return wo.at(co, ci); }) ||
            !conv.prepareAttention(ly.rel, *rel_k, *rel_v) ||
            !conv.prepareConv(ly.ffn1, *ffn1, ws.vec(e + ".ffn_layers." + L + ".conv_1.bias"), 1, row) ||
            !conv.prepareConv(ly.ffn2, *ffn2, ws.vec(e + ".ffn_layers." + L + ".conv_2.bias"), 1, row))
          return fail("transformer layer");
        ly.beta2 = ws.vec(e + ".norm_layers_2." + L + ".beta");
        if (!conv.prepareLayerNorm(ly.norm1, ws.vec(e + ".norm_layers_1." + L + ".gamma"),
                                   ws.vec(e + ".norm_layers_1." + L + ".beta")) ||
            !conv.prepareLayerNorm(ly.norm2, ws.vec(e + ".norm_layers_2." + L + ".gamma"), ly.beta2))
          return fail("layer norm");
      }
    }
    return true;
  }

  /// z_p [T, 32] -> z, as Flow::reverse(). False if T > max_frames.
  bool reverse(const Mat& z_p, const Mat& g, Mat& z) {
    const int T = z_p.rows();
    if (T > max_frames_ || z_p.cols() != kChannels) return false;
    QpuConv& conv = *conv_;
    // Speaker conditioning: x += spk_emb_linear(g) before layer kCondLayer,
    // i.e. on the previous layer's LayerNorm output.
    for (Coupling& cp : couplings_) {
      Mat g_proj = ops::linear(g, cp.spk_w, cp.spk_b);
      std::vector<float> beta = cp.layers[kCondLayer - 1].beta2;
      for (int c = 0; c < kChannels; c++) beta[c] += g_proj.at(0, c);
      conv.setLayerNormBeta(cp.layers[kCondLayer - 1].norm2, beta);
    }

    put(z_p, x_, T);
    const int half = kChannels / 2;
    for (int i = (int)couplings_.size() - 1; i >= 0; i--) {
      Coupling& cp = couplings_[i];
      QpuConv::Buffer x0 = rows(x_, cp.reversed ? half : 0, half), x1 = rows(x_, cp.reversed ? 0 : half, half);
      if (!conv.run(cp.pre, QpuConv::kOut0, x0, T, &h_, nullptr)) return false;
      for (Layer& ly : cp.layers) {
        if (!conv.run(ly.q, QpuConv::kOut0, h_, T, &q_, nullptr) ||
            !conv.runTimeMajor(ly.k, h_, T, [&](int g) { return k_[g]; }, 16 * 4) ||
            !conv.runTimeMajor(ly.kv, h_, T, [&](int g) { return kv_[g / 2] + g % 2 * 16; }, 32 * 4) ||
            !conv.attention(ly.rel, q_, k_, kv_, m_, T) ||
            !conv.run(ly.o, QpuConv::kRes | QpuConv::kOut0, m_, T, &y_, nullptr, &h_) ||
            !conv.layerNorm(ly.norm1, y_, x1_, T) ||
            !conv.run(ly.ffn1, QpuConv::kOut1, x1_, T, nullptr, &f_, nullptr, nullptr, 0.0f) ||  // ReLU
            !conv.run(ly.ffn2, QpuConv::kRes | QpuConv::kOut0, f_, T, &y_, nullptr, &x1_) ||
            !conv.layerNorm(ly.norm2, y_, h_, T))
          return false;
      }
      if (!conv.run(cp.post, QpuConv::kRes | QpuConv::kOut0, h_, T, &x1, nullptr, &x1)) return false;
    }

    // Logical channel c is physical row c, or 31 - c after an odd number of flips.
    const bool rev = couplings_.size() % 2 == 1;
    z = Mat(T, kChannels);
    std::vector<float> r((size_t)T);
    for (int c = 0; c < kChannels; c++) {
      std::memcpy(r.data(), x_.at(rev ? kChannels - 1 - c : c, 0), (size_t)T * 4);
      for (int t = 0; t < T; t++) z.at(t, c) = r[t];
    }
    return true;
  }

  /// For TinyTTS::setFlow() / TinyTTSCore::setFlow(). This object must
  /// outlive its use there.
  TinyTTSCore::FlowFn flowFn() {
    return [this](const Mat& z_p, const Mat& g, Mat& z) { return reverse(z_p, g, z); };
  }

 private:
  struct Layer {
    QpuConv::Layer q, k, kv, o, ffn1, ffn2;
    QpuConv::Relative rel;
    QpuConv::Norm norm1, norm2;
    std::vector<float> beta2;
  };
  struct Coupling {
    bool reversed = false;  // channels flipped (physical row = 31 - logical)
    QpuConv::Layer pre, post;
    Mat spk_w;
    std::vector<float> spk_b;
    Layer layers[kLayers];
  };

  QpuConv own_;
  QpuConv* conv_ = nullptr;
  int max_frames_ = 0;
  std::vector<Coupling> couplings_;
  QpuConv::Buffer x_, h_, q_, m_, y_, x1_, f_;
  std::vector<float*> k_, kv_;

  static QpuConv::Buffer rows(const QpuConv::Buffer& b, int first, int count) {
    QpuConv::Buffer v = b;
    v.data = b.data + (size_t)first * b.row;
    v.channels = count;
    return v;
  }

  static void put(const Mat& m, const QpuConv::Buffer& b, int T) {
    QpuConv::zeroMargins(b, T);
    std::vector<float> r((size_t)T);
    for (int c = 0; c < m.cols(); c++) {
      for (int t = 0; t < T; t++) r[t] = m.at(t, c);
      std::memcpy(b.at(c, 0), r.data(), (size_t)T * 4);
    }
  }

  static bool fail(const char* msg) {
    std::fprintf(stderr, "QpuFlow: %s\n", msg);
    return false;
  }
};

}  // namespace tinytts
