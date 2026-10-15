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
 * @brief The vocoder (Vocoder.h's HiFi-GAN decoder) on the GPU of a
 * Raspberry Pi 1 / Zero: every layer runs on the 12 QPUs (QpuConv) and the
 * activations stay in GPU memory; only z goes in and the audio comes out.
 * The speaker conditioning and the final tanh run on the CPU.
 *
 *   QpuVocoder qpu;
 *   ...  // tts.begin()
 *   if (qpu.begin(tts.core())) tts.setVocoder(qpu.vocoderFn());
 *
 * Weights and buffers are prepared for up to `max_frames` spectrogram frames
 * (512 frames: ~6 s of audio at 44.1 kHz); longer utterances run on the CPU
 * Vocoder instead. Needs root and the vc4-kms-v3d overlay disabled, see
 * docs/rpi-zero-gpu.md.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class QpuVocoder {
 public:
  /// GPU memory begin() needs for the weights in `ws` and `max_frames`
  /// (0 if `ws` has no decoder).
  static uint32_t arenaBytes(const WeightStore& ws, int max_frames) {
    size_t floats = 0;
    int out_row = 0;
    if (!bufferSizes(ws, max_frames, floats, out_row)) return 0;
    // Kernels + weight streams + 9 buffers + audio.
    return (uint32_t)((6u << 20) + 9 * (floats * 4 + 64) + 2 * out_row * 4 + (1 << 16));
  }

  /// After core.begin(), with its own GPU memory. `core` must outlive this
  /// object.
  bool begin(const TinyTTSCore& core, int max_frames = 512) { return begin(core.weights(), core.vocoder(), max_frames); }

  /// `ws` holds the decoder weights; `cpu` (the same weights) decodes
  /// utterances longer than `max_frames`. Both must outlive this object.
  bool begin(const WeightStore& ws, const Vocoder& cpu, int max_frames = 512) {
    if (!own_.begin(arenaBytes(ws, max_frames)))
      return fail("can't get the GPU memory (gpu_mem too small? try a smaller max_frames)");
    return begin(ws, cpu, own_, max_frames);
  }

  /// In `conv`'s GPU memory (shared with e.g. a QpuFlow; see arenaBytes()),
  /// which must outlive this object.
  bool begin(const WeightStore& ws, const Vocoder& cpu, QpuConv& conv, int max_frames = 512) {
    conv_ = &conv;
    cpu_ = &cpu;
    max_frames_ = max_frames;
    const std::string p = "dec";
    const WeightStore::Entry* conv_pre = ws.get(p + ".conv_pre.weight");
    const WeightStore::Entry* conv_post = ws.get(p + ".conv_post.weight");
    if (!conv_pre || !conv_post) return fail("no decoder weights");
    cond_w_ = ws.linearWeight(p + ".cond.weight");
    cond_b_ = ws.vec(p + ".cond.bias");
    conv_pre_b_ = ws.vec(p + ".conv_pre.bias");

    std::vector<float> rates = ws.vec("meta.upsample_rates");
    if (rates.empty()) rates = {8, 8, 2, 2, 2};  // released 44.1 kHz model, as in Vocoder
    stages_.assign(rates.size(), Stage());

    size_t floats = 0;
    if (!bufferSizes(ws, max_frames, floats, out_row_)) return fail("missing ups weights");
    int T16 = QpuConv::t16(max_frames);
    for (size_t i = 0; i < stages_.size(); i++) {
      Stage& s = stages_[i];
      s.ups_w = ws.get(p + ".ups." + std::to_string(i) + ".weight");
      if (!s.ups_w) return fail("missing ups weights");
      s.stride = (int)rates[i];
      s.cin = s.ups_w->shape[0];
      s.cout = s.ups_w->shape[1];
      s.in_row = T16 + 2 * QpuConv::kMargin;
      T16 *= s.stride;
      s.row = T16 + 2 * QpuConv::kMargin;
    }

    for (auto* b : {&z_, &x_, &lx_, &a_, &l_, &r1_, &r2_, &s_, &n_}) {
      b->data = (float*)conv_->alloc(floats * 4);
      if (!b->data) return fail("out of GPU memory for buffers");
    }
    if (!conv_->allocBuffer(audio_, 2, out_row_ - 2 * QpuConv::kMargin)) return fail("out of GPU memory");

    // conv_pre reads z (rows for max_frames).
    if (!conv_->prepareConv(pre_, *conv_pre, conv_pre_b_, 1, QpuConv::rowLength(max_frames)))
      return fail("conv_pre");
    for (size_t i = 0; i < stages_.size(); i++) {
      Stage& s = stages_[i];
      std::string ups = p + ".ups." + std::to_string(i);
      int k = s.ups_w->shape[2];
      if (!conv_->prepareConvTranspose(s.ups, *s.ups_w, ws.vec(ups + ".bias"), s.stride, (k - s.stride) / 2,
                                      s.in_row))
        return fail("ups");
      for (int j = 0; j < 3; j++)
        for (int c = 0; c < 3; c++) {
          std::string rb = p + ".resblocks." + std::to_string(i * 3 + j);
          const WeightStore::Entry* w1 = ws.get(rb + ".convs1." + std::to_string(c) + ".weight");
          const WeightStore::Entry* w2 = ws.get(rb + ".convs2." + std::to_string(c) + ".weight");
          if (!w1 || !w2) return fail("missing resblock weights");
          if (!conv_->prepareConv(s.convs1[j][c], *w1, ws.vec(rb + ".convs1." + std::to_string(c) + ".bias"),
                                 kDilations[c], s.row) ||
              !conv_->prepareConv(s.convs2[j][c], *w2, ws.vec(rb + ".convs2." + std::to_string(c) + ".bias"), 1,
                                 s.row))
            return fail("resblock");
        }
    }
    if (!conv_->prepareConv(post_, *conv_post, {}, 1, out_row_)) return fail("conv_post");
    return true;
  }

  void end() { own_.end(); }

  /// Decodes z [T, inter_channels] with speaker embedding g [1, gin_channels].
  bool forward(const Mat& z, const Mat& g, const TinyTTSCore::AudioChunkFn& on_audio) {
    const int T = z.rows();
    if (T > max_frames_) {
      std::fprintf(stderr, "QpuVocoder: %d frames, prepared for %d -- using the CPU vocoder\n", T, max_frames_);
      PsramVector<float> audio = cpu_->forward(z, g);
      if (on_audio) on_audio(audio.data(), audio.size());
      return true;
    }
    // conv_pre's bias plus the speaker conditioning, added once (Vocoder::forward).
    Mat g_proj = ops::linear(g, cond_w_, cond_b_);
    std::vector<float> bias = conv_pre_b_;
    for (size_t c = 0; c < bias.size(); c++) bias[c] += g_proj.at(0, (int)c);
    conv_->setBias(pre_, bias);

    QpuConv::Buffer zb = view(z_, z.cols(), pre_.in_row);
    QpuConv::zeroMargins(zb, T);
    std::vector<float> row(T);
    for (int c = 0; c < z.cols(); c++) {
      for (int t = 0; t < T; t++) row[t] = z.at(t, c);
      std::memcpy(zb.at(c, 0), row.data(), T * 4);
    }

    // x = leaky_relu(conv_pre(z) + g_proj, 0.1): the first stage's input.
    QpuConv::Buffer in = view(n_, pre_.outChannels(), stages_[0].in_row);
    if (!conv_->run(pre_, QpuConv::kOut1, zb, T, nullptr, &in, nullptr, nullptr, 0.1f)) return false;

    int t_in = T;
    for (size_t i = 0; i < stages_.size(); i++) {
      Stage& s = stages_[i];
      const bool last = i + 1 == stages_.size();
      const int t_out = t_in * s.stride;
      const int ch = s.ups.outChannels();
      QpuConv::Buffer x = view(x_, ch, s.row), lx = view(lx_, ch, s.row), a = view(a_, ch, s.row),
                      l = view(l_, ch, s.row), r1 = view(r1_, ch, s.row), r2 = view(r2_, ch, s.row),
                      sum = view(s_, ch, s.row);
      // The next stage's input (or conv_post's): leaky_relu(mean of the 3 resblocks).
      QpuConv::Buffer next = view(n_, ch, last ? out_row_ : stages_[i + 1].in_row);
      if (!conv_->run(s.ups, QpuConv::kOut0 | QpuConv::kOut1, in, t_in, &x, &lx, nullptr, nullptr, 0.1f))
        return false;
      for (int j = 0; j < 3; j++) {
        const QpuConv::Buffer* res = &x;  // the resblock's input
        const QpuConv::Buffer* cin = &lx; // leaky_relu of it
        QpuConv::Buffer* outs[2] = {&r1, &r2};
        for (int c = 0; c < 3; c++) {
          if (!conv_->run(s.convs1[j][c], QpuConv::kOut1, *cin, t_out, nullptr, &a, nullptr, nullptr, 0.1f))
            return false;
          const QpuConv::Layer& l2 = s.convs2[j][c];
          bool ok;
          if (c < 2) {
            ok = conv_->run(l2, QpuConv::kRes | QpuConv::kOut0 | QpuConv::kOut1, a, t_out, outs[c], &l, res,
                           nullptr, 0.1f);
            res = outs[c];
            cin = &l;
          } else if (j == 0) {
            ok = conv_->run(l2, QpuConv::kRes | QpuConv::kSum | QpuConv::kOut0, a, t_out, &sum, nullptr, res,
                           nullptr, 0.1f, 1.0f / 3);
          } else if (j == 1) {
            ok = conv_->run(l2, QpuConv::kRes | QpuConv::kSum | QpuConv::kSumAcc | QpuConv::kOut0, a, t_out, &sum,
                           nullptr, res, &sum, 0.1f, 1.0f / 3);
          } else {
            // Before conv_post: F.leaky_relu's default slope, as in Vocoder::forward.
            ok = conv_->run(l2, QpuConv::kRes | QpuConv::kSum | QpuConv::kSumAcc | QpuConv::kOut1, a, t_out,
                           nullptr, &next, res, &sum, last ? 0.01f : 0.1f, 1.0f / 3);
          }
          if (!ok) return false;
        }
      }
      in = next;
      t_in = t_out;
    }

    if (!conv_->run(post_, QpuConv::kOut0, in, t_in, &audio_, nullptr)) return false;
    audio_out_.resize(t_in);
    std::memcpy(audio_out_.data(), audio_.at(0, 0), (size_t)t_in * 4);
    for (float& v : audio_out_) v = std::tanh(v);
    if (on_audio) on_audio(audio_out_.data(), audio_out_.size());
    return true;
  }

  /// For TinyTTS::setVocoder() / TinyTTSCore::setVocoder(). This object must
  /// outlive its use there.
  TinyTTSCore::VocoderFn vocoderFn() {
    return [this](const Mat& z, const Mat& g, const TinyTTSCore::AudioChunkFn& on_audio) {
      return forward(z, g, on_audio);
    };
  }

 private:
  static constexpr int kDilations[3] = {1, 3, 5};

  struct Stage {
    const WeightStore::Entry* ups_w = nullptr;
    int stride = 1, cin = 0, cout = 0;
    int in_row = 0, row = 0;  // row lengths of the stage's input and of its activations
    QpuConv::Layer ups;
    QpuConv::Layer convs1[3][3], convs2[3][3];  // [resblock][pair]
  };

  const Vocoder* cpu_ = nullptr;
  int max_frames_ = 0;
  QpuConv own_;
  QpuConv* conv_ = nullptr;
  Mat cond_w_;
  std::vector<float> cond_b_, conv_pre_b_;
  std::vector<Stage> stages_;
  QpuConv::Layer pre_, post_;
  int out_row_ = 0;
  QpuConv::Buffer z_, x_, lx_, a_, l_, r1_, r2_, s_, n_, audio_;
  std::vector<float> audio_out_;

  /// Buffer sizes: every buffer is reused by every stage, so it gets the
  /// largest channels * row length of any stage.
  static bool bufferSizes(const WeightStore& ws, int max_frames, size_t& floats, int& out_row) {
    const WeightStore::Entry* conv_pre = ws.get("dec.conv_pre.weight");
    if (!conv_pre) return false;
    std::vector<float> rates = ws.vec("meta.upsample_rates");
    if (rates.empty()) rates = {8, 8, 2, 2, 2};
    floats = (size_t)conv_pre->shape[1] * QpuConv::rowLength(max_frames);
    int T16 = QpuConv::t16(max_frames);
    for (size_t i = 0; i < rates.size(); i++) {
      const WeightStore::Entry* w = ws.get("dec.ups." + std::to_string(i) + ".weight");
      if (!w) return false;
      const int in_row = T16 + 2 * QpuConv::kMargin;
      T16 *= (int)rates[i];
      const int row = T16 + 2 * QpuConv::kMargin;
      floats = std::max(floats, (size_t)std::max(w->shape[0] * in_row, w->shape[1] * row));
    }
    out_row = T16 + 2 * QpuConv::kMargin;
    return true;
  }

  QpuConv::Buffer view(const QpuConv::Buffer& b, int channels, int row) const {
    QpuConv::Buffer v;
    v.data = b.data;
    v.channels = channels;
    v.row = row;
    return v;
  }

  static bool fail(const char* msg) {
    std::fprintf(stderr, "QpuVocoder: %s\n", msg);
    return false;
  }
};

}  // namespace tinytts
