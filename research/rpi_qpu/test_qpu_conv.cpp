// Checks every QpuConv layer type and epilogue against the CPU ops
// (tinytts::ops). Run as root on the Pi Zero:  sudo ./test_qpu_conv
#include <cmath>
#include <cstdio>
#include <random>

#include "TinyTTS/Ops.h"
#include "TinyTTS/QpuConv.h"

using namespace tinytts;

static std::mt19937 rng(1);

struct Weights {
  std::vector<float> w, b;
  WeightStore::Entry e;
  Weights(int d0, int d1, int k) : w((size_t)d0 * d1 * k), b(0) {
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    float s = 1.0f / std::sqrt((float)d1 * k);
    for (auto& v : w) v = uni(rng) * s;
    e.shape = {d0, d1, k};
    e.raw = (const uint8_t*)w.data();
    e.count = w.size();
    e.dtype = 0;
  }
  void bias(int n) {
    std::uniform_real_distribution<float> uni(-0.1f, 0.1f);
    b.resize(n);
    for (auto& v : b) v = uni(rng);
  }
};

static Mat random(int T, int C) {
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  Mat m(T, C);
  for (auto& v : m.data()) v = uni(rng);
  return m;
}

static void put(const Mat& m, const QpuConv::Buffer& b) {
  QpuConv::zeroMargins(b, m.rows());
  for (int c = 0; c < m.cols(); c++)
    for (int t = 0; t < m.rows(); t++) *b.at(c, t) = m.at(t, c);
}

static Mat get(const QpuConv::Buffer& b, int T, int C) {
  Mat m(T, C);
  for (int c = 0; c < C; c++)
    for (int t = 0; t < T; t++) m.at(t, c) = *b.at(c, t);
  return m;
}

static Mat leaky(Mat m, float a) {
  ops::leakyReluInplace(m, a);
  return m;
}

static int failures = 0;

static void ran(bool ok) {
  if (!ok) {
    std::printf("run() failed\n");
    failures++;
  }
}

static void check(const char* what, const Mat& got, const Mat& want, const QpuConv::Buffer* b = nullptr) {
  float err = 0, mx = 1e-6f;
  for (size_t i = 0; i < want.data().size(); i++) {
    err = std::max(err, std::fabs(got.data()[i] - want.data()[i]));
    mx = std::max(mx, std::fabs(want.data()[i]));
  }
  // The tail up to the next block of 16 must be zero (the next layer's padding).
  float tail = 0;
  if (b)
    for (int c = 0; c < want.cols(); c++)
      for (int t = want.rows(); t < QpuConv::t16(want.rows()); t++) tail = std::max(tail, std::fabs(*b->at(c, t)));
  bool ok = err / mx < 1e-4f && tail == 0;
  failures += !ok;
  std::printf("%-44s rel err %.2e, tail %g  %s\n", what, err / mx, tail, ok ? "OK" : "FAIL");
}

int main() {
  QpuConv gpu;
  if (!gpu.begin(16 << 20)) return 1;

  for (int T : {117, 936, 1000}) {
    for (int C : {32, 16, 8, 4, 2}) {
      for (int k : {3, 11}) {
        const int d = k == 3 ? 5 : 3;
        size_t mark = gpu.mark();
        Weights w(C, C, k);
        w.bias(C);
        Mat x = random(T, C), r = random(T, C), s0 = random(T, C);
        QpuConv::Buffer bx, bo0, bo1, br, bs;
        QpuConv::Layer l;
        if (!gpu.allocBuffer(bx, C, T) || !gpu.allocBuffer(bo0, C, T) || !gpu.allocBuffer(bo1, C, T) ||
            !gpu.allocBuffer(br, C, T) || !gpu.allocBuffer(bs, C, T) || !gpu.prepareConv(l, w.e, w.b, d, bx.row))
          return 1;
        put(x, bx);
        put(r, br);
        put(s0, bs);
        Mat y = ops::conv1d(x, w.e, w.b, d);
        char what[96];

        ran(gpu.run(l, QpuConv::kOut0, bx, T, &bo0, nullptr));
        std::snprintf(what, sizeof what, "conv T=%d C=%d k=%d d=%d out0", T, C, k, d);
        check(what, get(bo0, T, C), y, &bo0);
        ran(gpu.run(l, QpuConv::kOut1, bx, T, nullptr, &bo1, nullptr, nullptr, 0.1f));
        std::snprintf(what, sizeof what, "conv T=%d C=%d k=%d d=%d leaky", T, C, k, d);
        check(what, get(bo1, T, C), leaky(y, 0.1f), &bo1);

        Mat yr = y;
        ops::addInplace(yr, r);
        ran(gpu.run(l, QpuConv::kRes | QpuConv::kOut0 | QpuConv::kOut1, bx, T, &bo0, &bo1, &br, nullptr, 0.01f));
        std::snprintf(what, sizeof what, "conv T=%d C=%d +residual", T, C);
        check(what, get(bo0, T, C), yr, &bo0);
        std::snprintf(what, sizeof what, "conv T=%d C=%d +residual, leaky .01", T, C);
        check(what, get(bo1, T, C), leaky(yr, 0.01f), &bo1);

        Mat ys = yr;
        for (size_t i = 0; i < ys.data().size(); i++) ys.data()[i] = yr.data()[i] / 3 + s0.data()[i];
        ran(gpu.run(l, QpuConv::kRes | QpuConv::kSum | QpuConv::kSumAcc | QpuConv::kOut0, bx, T, &bs, nullptr, &br,
                    &bs, 0.1f, 1.0f / 3));
        std::snprintf(what, sizeof what, "conv T=%d C=%d +residual, /3 + S in place", T, C);
        check(what, get(bs, T, C), ys, &bs);
        gpu.reset(mark);
      }
    }
  }

  // ConvTranspose1d as in the 44.1 kHz decoder: (cin, cout, K, stride, padding).
  const int ups[][5] = {{64, 32, 16, 8, 4}, {32, 16, 16, 8, 4}, {16, 8, 8, 2, 3}, {8, 4, 2, 2, 0}, {4, 2, 2, 2, 0}};
  for (int T : {117, 50}) {
    for (const auto& u : ups) {
      size_t mark = gpu.mark();
      Weights w(u[0], u[1], u[2]);
      w.bias(u[1]);
      Mat x = random(T, u[0]);
      QpuConv::Buffer bx, bo0, bo1;
      QpuConv::Layer l;
      const int To = QpuConv::t16(T) * u[3];
      if (!gpu.allocBuffer(bx, u[0], T) || !gpu.prepareConvTranspose(l, w.e, w.b, u[3], u[4], bx.row) ||
          !gpu.allocBuffer(bo0, l.outChannels(), To) || !gpu.allocBuffer(bo1, l.outChannels(), To))
        return 1;
      put(x, bx);
      Mat y = ops::convTranspose1d(x, w.e, w.b, u[3], u[4]);
      ran(gpu.run(l, QpuConv::kOut0 | QpuConv::kOut1, bx, T, &bo0, &bo1, nullptr, nullptr, 0.1f));
      char what[96];
      std::snprintf(what, sizeof what, "convT T=%d %d->%d K=%d s=%d out0", T, u[0], u[1], u[2], u[3]);
      check(what, get(bo0, y.rows(), u[1]), y, &bo0);
      std::snprintf(what, sizeof what, "convT T=%d %d->%d K=%d s=%d leaky", T, u[0], u[1], u[2], u[3]);
      check(what, get(bo1, y.rows(), u[1]), leaky(y, 0.1f), &bo1);
      gpu.reset(mark);
    }
  }

  // conv_post: one output channel (padded to a tile of 2).
  {
    const int T = 5000;
    Weights w(1, 2, 7);
    Mat x = random(T, 2);
    QpuConv::Buffer bx, bo;
    QpuConv::Layer l;
    if (!gpu.allocBuffer(bx, 2, T) || !gpu.prepareConv(l, w.e, {}, 1, bx.row) ||
        !gpu.allocBuffer(bo, l.outChannels(), T))
      return 1;
    put(x, bx);
    ran(gpu.run(l, QpuConv::kOut0, bx, T, &bo, nullptr));
    check("conv_post 2->1 k=7", get(bo, T, 1), ops::conv1d(x, w.e, {}, 1), &bo);
  }

  // Flow pieces (QpuFlow): time-major outputs, LayerNorm, attention.
  for (int T : {128, 77}) {
    size_t mark = gpu.mark();
    const int C = 32;
    Weights w(C, C, 1);
    w.bias(C);
    Mat x = random(T, C);
    QpuConv::Buffer bx, bo;
    QpuConv::Layer l;
    const int T16 = QpuConv::t16(T);
    float* tm[2] = {(float*)gpu.alloc((size_t)T16 * 32 * 4), (float*)gpu.alloc((size_t)T16 * 32 * 4)};
    if (!gpu.allocBuffer(bx, C, T) || !gpu.allocBuffer(bo, C, T) || !gpu.prepareConv(l, w.e, w.b, 1, bx.row))
      return 1;
    put(x, bx);
    Mat y = ops::conv1d(x, w.e, w.b, 1);
    // Tile g -> tm[g] with time rows of 32 floats, channels at offset 8.
    ran(gpu.runTimeMajor(l, bx, T, [&](int g) { return tm[g] + 8; }, 32 * 4));
    Mat got(T, C);
    for (int t = 0; t < T; t++)
      for (int c = 0; c < C; c++) got.at(t, c) = tm[c / 16][t * 32 + 8 + c % 16];
    char what[96];
    std::snprintf(what, sizeof what, "time-major conv T=%d", T);
    check(what, got, y);

    std::vector<float> gamma(C), beta(C);
    for (int c = 0; c < C; c++) {
      gamma[c] = 0.5f + 0.03f * c;
      beta[c] = 0.1f * (c % 5) - 0.2f;
    }
    QpuConv::Norm norm;
    if (!gpu.prepareLayerNorm(norm, gamma, beta)) return 1;
    for (auto& v : x.data()) v = v * 3 + 0.5f;
    put(x, bx);
    ran(gpu.layerNorm(norm, bx, bo, T));
    Mat ln = x;
    ops::channelLayerNormInplace(ln, gamma, beta);
    if (getenv("DEBUG"))
      for (int t : {0, 1, 17}) {
        Mat gm = get(bo, T, C);
        for (int c : {0, 1, 16, 31}) std::printf("  ln t=%d c=%d got %g want %g x %g\n", t, c, gm.at(t, c), ln.at(t, c), x.at(t, c));
      }
    std::snprintf(what, sizeof what, "layernorm T=%d", T);
    check(what, get(bo, T, C), ln, &bo);

    // Attention, 2 heads of 16 channels, window 4 (Attention::headFloat).
    Mat q = random(T, C), k = random(T, C), v = random(T, C);
    std::vector<float> relk(9 * 16), relv(9 * 16);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    for (auto& r : relk) r = uni(rng);
    for (auto& r : relv) r = uni(rng);
    WeightStore::Entry ek, ev;
    ek.shape = {1, 9, 16};
    ek.raw = (const uint8_t*)relk.data();
    ek.count = relk.size();
    ev = ek;
    ev.raw = (const uint8_t*)relv.data();
    QpuConv::Relative rel;
    if (!gpu.prepareAttention(rel, ek, ev)) return 1;
    const float qs = 1.4426950408889634f / 4.0f;
    Mat qscaled = q;
    for (auto& e : qscaled.data()) e *= qs;
    put(qscaled, bx);
    std::vector<float*> kh = {(float*)gpu.alloc((size_t)T16 * 16 * 4), (float*)gpu.alloc((size_t)T16 * 16 * 4)};
    std::vector<float*> kvh = {tm[0], tm[1]};
    for (int h = 0; h < 2; h++)
      for (int t = 0; t < T; t++)
        for (int c = 0; c < 16; c++) {
          kh[h][t * 16 + c] = k.at(t, h * 16 + c);
          kvh[h][t * 32 + c] = k.at(t, h * 16 + c);
          kvh[h][t * 32 + 16 + c] = v.at(t, h * 16 + c);
        }
    ran(gpu.attention(rel, bx, kh, kvh, bo, T));
    Mat want(T, C);
    std::vector<float> sc(T);
    for (int h = 0; h < 2; h++)
      for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
          float acc = 0;
          for (int c = 0; c < 16; c++) acc += q.at(i, h * 16 + c) * k.at(j, h * 16 + c);
          int d = j - i;
          if (d >= -4 && d <= 4)
            for (int c = 0; c < 16; c++) acc += q.at(i, h * 16 + c) * relk[(d + 4) * 16 + c];
          sc[j] = acc * 0.25f;
        }
        ops::softmaxInplace(sc.data(), T);
        for (int c = 0; c < 16; c++) {
          float acc = 0;
          for (int j = 0; j < T; j++) acc += sc[j] * v.at(j, h * 16 + c);
          for (int d = -4; d <= 4; d++)
            if (i + d >= 0 && i + d < T) acc += sc[i + d] * relv[(d + 4) * 16 + c];
          want.at(i, h * 16 + c) = acc;
        }
      }
    std::snprintf(what, sizeof what, "attention T=%d", T);
    check(what, get(bo, T, C), want, &bo);
    gpu.reset(mark);
  }

  std::printf("%s (%d failures)\n", failures ? "FAIL" : "ALL OK", failures);
  return failures ? 3 : 0;
}
