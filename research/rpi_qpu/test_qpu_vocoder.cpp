// Compares QpuVocoder against the CPU Vocoder on the same z and times both.
// Run as root on the Pi Zero:
//
//   sudo ./test_qpu_vocoder weights.bin [frames] [--no-cpu]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>

#include "TinyTTS/QpuVocoder.h"

using namespace tinytts;
using Clock = std::chrono::steady_clock;

static double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s weights.bin [frames] [--no-cpu]\n", argv[0]);
    return 1;
  }
  std::ifstream f(argv[1], std::ios::binary);
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const int T = argc > 2 ? std::atoi(argv[2]) : 117;
  const bool cpu_run = !(argc > 3 && std::strcmp(argv[3], "--no-cpu") == 0);

  WeightStore ws;
  if (!ws.begin(buf.data(), buf.size())) return 1;
  Vocoder cpu;
  cpu.begin(ws);
  QpuVocoder qpu;
  auto t0 = Clock::now();
  if (!qpu.begin(ws, cpu)) return 2;
  std::printf("QpuVocoder.begin: %.0f ms\n", ms(t0, Clock::now()));

  // A z of realistic scale (the flow's output is roughly unit variance) and speaker 0.
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  Mat z(T, ws.get("dec.conv_pre.weight")->shape[1]);
  for (auto& v : z.data()) v = normal(rng);
  Mat emb = ws.embedding("emb_g.weight");
  Mat g(1, emb.cols());
  for (int c = 0; c < emb.cols(); c++) g.at(0, c) = emb.at(0, c);

  std::vector<float> got;
  for (int rep = 0; rep < 3; rep++) {
    got.clear();
    auto a = Clock::now();
    if (!qpu.forward(z, g, [&](const float* s, size_t n) { got.insert(got.end(), s, s + n); })) return 3;
    std::printf("QPU vocoder: %.0f ms, %zu samples\n", ms(a, Clock::now()), got.size());
  }
  if (!cpu_run) return 0;

  auto b = Clock::now();
  PsramVector<float> want = cpu.forward(z, g);
  std::printf("CPU vocoder: %.0f ms, %zu samples\n", ms(b, Clock::now()), want.size());
  if (want.size() != got.size()) {
    std::printf("FAIL: sample count differs\n");
    return 4;
  }
  double sig = 0, noise = 0;
  float mx = 0;
  for (size_t i = 0; i < want.size(); i++) {
    sig += (double)want[i] * want[i];
    noise += (double)(want[i] - got[i]) * (want[i] - got[i]);
    mx = std::max(mx, std::fabs(want[i] - got[i]));
  }
  double snr = 10 * std::log10(sig / std::max(noise, 1e-30));
  std::printf("SNR vs CPU: %.1f dB, max abs diff %.2e -- %s\n", snr, mx, snr > 60 ? "OK" : "FAIL");
  return snr > 60 ? 0 : 5;
}
