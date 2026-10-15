// Benchmarks the QPU conv1d against tinytts::ops::conv1d for every resblock
// conv shape of the 44.1 kHz decoder ("Hello world!", t_y = 117), checks
// the results match, and projects the decoder's resblock time. Run as root.
//
//   sudo ./bench_conv1d [t_y]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "TinyTTS/Ops.h"
#include "TinyTTS/QpuConv.h"

using namespace tinytts;
using Clock = std::chrono::steady_clock;

static double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Result {
  double cpu = 0, qpu = 0;
  float err = 0;
};

static Result benchShape(QpuConv& gpu, int T, int C, int k, int dil, std::mt19937& rng) {
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  Mat x(T, C);
  for (auto& v : x.data()) v = uni(rng);
  std::vector<float> wv((size_t)C * C * k), bias(C);
  const float ws = 1.0f / std::sqrt((float)C * k);
  for (auto& v : wv) v = uni(rng) * ws;
  for (auto& v : bias) v = uni(rng) * 0.1f;
  WeightStore::Entry w;
  w.shape = {C, C, k};
  w.raw = (const uint8_t*)wv.data();
  w.count = wv.size();
  w.dtype = 0;

  Result r;
  auto t0 = Clock::now();
  Mat ref = ops::conv1d(x, w, bias, dil);
  r.cpu = ms(t0, Clock::now());

  size_t mark = gpu.mark();
  QpuConv::Buffer bx, by;
  QpuConv::Layer layer;
  if (!gpu.allocBuffer(bx, C, T) || !gpu.allocBuffer(by, C, T) || !gpu.prepareConv(layer, w, bias, dil, bx.row)) {
    std::fprintf(stderr, "out of GPU memory\n");
    std::exit(1);
  }
  QpuConv::zeroMargins(bx, T);
  for (int c = 0; c < C; c++)
    for (int t = 0; t < T; t++) *bx.at(c, t) = x.at(t, c);
  r.qpu = 1e30;
  for (int rep = 0; rep < 3; rep++) {
    auto a = Clock::now();
    if (!gpu.run(layer, QpuConv::kOut0, bx, T, &by, nullptr)) std::exit(2);
    r.qpu = std::min(r.qpu, ms(a, Clock::now()));
  }
  float mx = 1e-6f;
  for (int c = 0; c < C; c++)
    for (int t = 0; t < T; t++) {
      r.err = std::max(r.err, std::fabs(*by.at(c, t) - ref.at(t, c)));
      mx = std::max(mx, std::fabs(ref.at(t, c)));
    }
  r.err /= mx;  // relative to the output's range
  gpu.reset(mark);
  return r;
}

int main(int argc, char** argv) {
  const int t_y = argc > 1 ? atoi(argv[1]) : 117;
  QpuConv gpu;
  if (!gpu.begin(8 << 20)) return 1;
  std::mt19937 rng(42);

  const int kernels[3] = {3, 7, 11};
  const int dilations[3] = {1, 3, 5};
  double cpu_total = 0, qpu_total = 0;
  float worst = 0;
  std::printf("%6s %3s %3s %4s %10s %10s %8s %9s\n", "T", "C", "k", "dil", "cpu ms", "qpu ms", "speedup", "rel err");
  int T = t_y, C = 64;
  for (int stage = 0; stage < 5; stage++) {
    T *= stage < 2 ? 8 : 2;
    C /= 2;
    for (int k : kernels) {
      // Each resblock: convs1 with dilation 1, 3, 5, convs2 3x with dilation 1.
      for (int d : dilations) {
        Result r = benchShape(gpu, T, C, k, d, rng);
        const int times = d == 1 ? 4 : 1;
        cpu_total += r.cpu * times;
        qpu_total += r.qpu * times;
        worst = std::max(worst, r.err);
        std::printf("%6d %3d %3d %4d %10.1f %10.2f %7.1fx %9.2e\n", T, C, k, d, r.cpu, r.qpu, r.cpu / r.qpu, r.err);
      }
    }
  }
  std::printf("\nresblock convs (90 per utterance): cpu %.0f ms, qpu %.0f ms (%.1fx)\n", cpu_total, qpu_total,
              cpu_total / qpu_total);
  std::printf("worst relative error: %.2e -- %s\n", worst, worst < 1e-4f ? "OK" : "TOO LARGE");
  return worst < 1e-4f ? 0 : 3;
}
