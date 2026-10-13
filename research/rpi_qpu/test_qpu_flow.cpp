// Compares QpuFlow against the CPU Flow on the same z_p and times both.
// Run as root on the Pi Zero:
//
//   sudo ./test_qpu_flow weights.bin [frames]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>

#include "TinyTTS/Flow.h"
#include "TinyTTS/QpuFlow.h"

using namespace tinytts;
using Clock = std::chrono::steady_clock;

static double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s weights.bin [frames]\n", argv[0]);
    return 1;
  }
  std::ifstream f(argv[1], std::ios::binary);
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const int T = argc > 2 ? std::atoi(argv[2]) : 128;

  WeightStore ws;
  if (!ws.begin(buf.data(), buf.size())) return 1;
  Flow cpu;
  cpu.begin(ws, /*n_flows=*/4, /*n_heads=*/2, /*window_size=*/4);
  QpuConv conv;
  if (!conv.begin(QpuFlow::arenaBytes(512))) return 2;
  QpuFlow qpu;
  auto t0 = Clock::now();
  if (!qpu.begin(ws, conv)) return 2;
  std::printf("QpuFlow.begin: %.0f ms\n", ms(t0, Clock::now()));

  // z_p as the duration-expanded prior gives it: roughly unit scale.
  std::mt19937 rng(3);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  Mat z_p(T, QpuFlow::kChannels);
  for (auto& v : z_p.data()) v = normal(rng);
  Mat emb = ws.embedding("emb_g.weight");
  Mat g(1, emb.cols());
  for (int c = 0; c < emb.cols(); c++) g.at(0, c) = emb.at(0, c);

  Mat got;
  for (int rep = 0; rep < 3; rep++) {
    auto a = Clock::now();
    if (!qpu.reverse(z_p, g, got)) return 3;
    std::printf("QPU flow: %.1f ms\n", ms(a, Clock::now()));
  }
  auto b = Clock::now();
  Mat want = cpu.reverse(z_p, g);
  std::printf("CPU flow: %.0f ms\n", ms(b, Clock::now()));

  double sig = 0, noise = 0;
  float mx = 0;
  int worst = 0;
  for (size_t i = 0; i < want.data().size(); i++) {
    float d = want.data()[i] - got.data()[i];
    sig += (double)want.data()[i] * want.data()[i];
    noise += (double)d * d;
    if (std::fabs(d) > mx) {
      mx = std::fabs(d);
      worst = (int)i;
    }
  }
  double snr = 10 * std::log10(sig / std::max(noise, 1e-30));
  std::printf("SNR vs CPU: %.1f dB, max abs diff %.2e at t=%d c=%d -- %s\n", snr, mx, worst / want.cols(),
              worst % want.cols(), snr > 60 ? "OK" : "FAIL");
  return snr > 60 ? 0 : 5;
}
