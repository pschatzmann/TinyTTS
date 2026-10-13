// Go/no-go check for QPU access on the Pi Zero: runs hello.qasm
// (out = in * 2 + 1) on all 12 QPUs and checks every lane. Run as root.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "TinyTTS/QpuDevice.h"

static const uint32_t kCode[] = {
#include "hello.hex"
};

int main(int argc, char** argv) {
  const int num_qpus = argc > 1 ? atoi(argv[1]) : 12;
  const int vectors = argc > 2 ? atoi(argv[2]) : 64;  // per QPU
  const int reps = argc > 3 ? atoi(argv[3]) : 1;
  const int n = num_qpus * vectors * 16;

  struct Layout {
    uint32_t code[sizeof(kCode) / 4];
    uint32_t unif[12][4];
    uint32_t mail[12][2];
  };
  tinytts::QpuDevice gpu;
  if (!gpu.begin(sizeof(Layout) + 2 * n * sizeof(float) + 64)) return 1;
  Layout* l = (Layout*)gpu.data();
  float* in = (float*)(gpu.data() + ((sizeof(Layout) + 63) & ~63));
  float* out = in + n;

  std::memcpy(l->code, kCode, sizeof(kCode));
  for (int i = 0; i < n; i++) {
    in[i] = (float)i * 0.25f - 100.0f;
    out[i] = -12345.0f;
  }
  for (int q = 0; q < num_qpus; q++) {
    l->unif[q][0] = q;
    l->unif[q][1] = vectors;
    l->unif[q][2] = gpu.busAddress(in + q * vectors * 16);
    l->unif[q][3] = gpu.busAddress(out + q * vectors * 16);
    l->mail[q][0] = gpu.busAddress(l->unif[q]);
    l->mail[q][1] = gpu.busAddress(l->code);
  }
  int bad = 0;
  for (int rep = 0; rep < reps; rep++) {
    for (int i = 0; i < n; i++) out[i] = -12345.0f;
    auto t0 = std::chrono::steady_clock::now();
    if (!gpu.execute(num_qpus, &l->mail[0][0])) return 2;
    double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    int rep_bad = 0;
    for (int i = 0; i < n; i++) {
      float want = in[i] * 2 + 1;
      if (std::fabs(out[i] - want) > 1e-3f) {
        if (rep_bad < 3) std::printf("run %d: mismatch at %d: got %g, want %g\n", rep, i, out[i], want);
        rep_bad++;
      }
    }
    if (reps > 1) std::printf("run %d: %.2f ms, %d mismatches\n", rep, t, rep_bad);
    bad += rep_bad;
  }
  std::printf("%d QPUs, %d values: %s (%d mismatches)\n", num_qpus, n, bad ? "FAIL" : "OK", bad);
  return bad ? 3 : 0;
}
