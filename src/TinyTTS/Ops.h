#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "TinyTTS/AlignedStlAllocator.h"
#include "TinyTTS/Concurrency/TileSplitter.h"
#include "TinyTTS/InternalStlAllocator.h"
#include "TinyTTS/Mat.h"
#include "TinyTTS/Profile.h"
#include "TinyTTS/PsramStlAllocator.h"
#include "TinyTTS/WeightStore.h"

// Optional: SIMD-accelerated dot products for ESP32-S3/P4. Float32
// (dsps_dotprod_f32) comes from the Arduino-ESP32 core's OWN bundled
// esp-dsp component (~3.3.x+ cores ship one) -- found automatically via
// <dsps_dotprod.h>, nothing to vendor for that part. Both conv1d() and
// convTranspose1d() transpose their weight tiles into a per-tap-contiguous
// layout (see their own docs) specifically so every hot-path call is the
// plain unstrided dsps_dotprod_f32 -- esp-dsp's strided dsps_dotprode_f32
// and its mulc()/add() pair (formerly used for convTranspose1d's scatter,
// see docs/performance.md) are no longer needed at all. INT8
// (tinytts::dspsDotProdS8, used by conv1d()'s INT8-activation path) is
// vendored in src/int8-dotprod/ instead, via a quoted, project-relative
// include, NOT <dsps_dotprod.h> -- the core's bundled component doesn't
// declare dsps_dp_s8 at all, and even if it did, an angle-bracket include
// from this file would resolve to the core's copy (its include dirs
// precede this library's own src/ on the compiler command line), not
// whatever this project vendors -- see src/int8-dotprod/NOTICE.md and
// dsps_dp_s8.h's own doc for the full story. Still guarded by
// __has_include rather than a hard library.properties dependency, so a
// host/desktop build still compiles and just uses the plain scalar loop
// below.
#if defined(ESP32) && __has_include(<dsps_dotprod.h>)
#include <dsps_dotprod.h>
#include "int8-dotprod/dsps_dp_s8.h"
#define TINYTTS_HAVE_ESP_DSP 1
#endif

// Optional: SIMD-accelerated float32 dot products via plain ARM NEON --
// e.g. a Raspberry Pi 4's Cortex-A72 (desktop CLI build only; ESP32 chips
// are Xtensa/RISC-V, never ARM, hence mutually exclusive with the
// TINYTTS_HAVE_ESP_DSP block above). Same per-tap weight transpose as the
// esp-dsp path above means every hot-path NEON call here is a plain
// contiguous vld1q_f32 load, no manual strided gather needed.
#if !defined(TINYTTS_HAVE_ESP_DSP) && \
    (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)) && __has_include(<arm_neon.h>)
#include <arm_neon.h>
#define TINYTTS_HAVE_NEON 1
#endif

// Optional: the Tang Nano 20K core's on-chip INT8 dot-product engine
// (AIAccelerator library) for conv1d()'s INT8-activation path. Only when
// Tools > AI Accelerator is enabled -- the core then defines
// TANGNANO20K_AI_ACCEL=1, and only then is the engine in the bitstream.
// Define TINYTTS_NO_TANG_AI to keep the scalar loop regardless.
#if defined(ARDUINO_ARCH_TANGNANO20K) && defined(TANGNANO20K_AI_ACCEL) && TANGNANO20K_AI_ACCEL && \
    !defined(TINYTTS_NO_TANG_AI)
#include <AIAccelerator.h>
#define TINYTTS_HAVE_TANG_AI 1
#endif

// TINYTTS_HOT marks the inner loops synthesis spends its time in. On the
// Tang Nano 20K it's the core's SRAM_CODE: with Tools > Boot Mode: ... +
// SDRAM everything else in a sketch runs from SDRAM, about 2.4x slower
// than the internal SRAM these functions are then kept in (see the core's
// docs/PERIPHERALS.md "Code in SDRAM"). Nothing anywhere else.
#if defined(ARDUINO_ARCH_TANGNANO20K) && defined(SRAM_CODE)
#define TINYTTS_HOT SRAM_CODE
#else
#define TINYTTS_HOT
#endif

// TINYTTS_FIXED_POINT: integer versions of the hot float loops, for CPUs
// without an FPU, where every float operation is a software routine of
// hundreds of cycles (defaults to on for RISC-V without F, e.g. the Tang
// Nano 20K's picorv32; CPUs with an FPU keep the float code):
// - the INT8 conv paths sum their per-tap results in 64-bit integers, with
//   the per-timestep activation scales as Q24 fixed-point fractions of the
//   layer's largest scale, and convert to float once per output (error at
//   most 2^-25 of the largest scale);
// - INT8 quantization finds the abs-max on the float bit patterns and
//   computes round(x * 127 / max) from the mantissas with integer
//   multiplies (same results, except rarely at an exact .5);
// - attention (Attention.h) quantizes q/k/v and the relative embeddings
//   to 16 bits per head and the softmax weights to Q15, and sums its dot
//   products in 64-bit integers.
#if !defined(TINYTTS_FIXED_POINT) && defined(__riscv) && !defined(__riscv_flen)
#define TINYTTS_FIXED_POINT 1
#endif

// DMA-prefetching weight tiles (GDMA async memcpy, overlapping the fetch
// of the NEXT tile with CPU compute on the current one) was tried and
// reverted: real-hardware testing showed it silently corrupts data when
// the source is the embedded-data default's flash-mapped `const` weight
// array (default_weights_data.h) -- duration_predictor produced garbage
// (t_y=6637 instead of the correct 128) and the pipeline crashed
// downstream. ESP-IDF's own async_memcpy docs say DMA requires
// "DMA-accessible" buffers and call out flash-mapped/MMU-cached read-only
// data as not necessarily qualifying, with no error returned on
// violation -- consistent with what was observed. Since the flash-embedded
// weights are TinyTTS's real, default usage pattern (not an edge case),
// this isn't safe to ship even opt-in. See git history if a future
// PSRAM-only weight source makes this worth revisiting.

namespace tinytts {

/**
 * @brief Primitive tensor ops shared by Attention/TransformerBlock/PhonemeEncoder/Flow.
 * All activations are [T, C] (time-major, channel-last). Batch=1 always
 * (single-utterance embedded inference).
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
namespace ops {

#ifdef TINYTTS_HAVE_NEON
namespace detail {

// Portable 4-lane horizontal sum -- vaddvq_f32 is AArch64-only, this form
// (vadd_f32 of the two 2-lane halves, then vpadd_f32 to fold those) works
// identically on both 32-bit ARMv7 NEON and AArch64.
inline float neonHsum(float32x4_t v) {
  float32x2_t lo = vget_low_f32(v);
  float32x2_t hi = vget_high_f32(v);
  float32x2_t s = vadd_f32(lo, hi);
  s = vpadd_f32(s, s);
  return vget_lane_f32(s, 0);
}

/// Contiguous dot product: *dot = sum(a[i]*b[i]) for i in [0,n) -- both
/// sides unit-stride, e.g. linear()'s Mat rows. Overwrites *dot (matches
/// dsps_dotprod_f32's semantics, not an accumulate) so call sites written
/// against the esp-dsp API need no change beyond which function they call.
inline void neonDotProd(const float* a, const float* b, float* dot, int n) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i + 4 <= n; i += 4) acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
  float sum = neonHsum(acc);
  for (; i < n; i++) sum += a[i] * b[i];
  *dot = sum;
}

/// Fused scale-and-accumulate: y[i] += xv*w[i] for i in [0,n), both sides
/// contiguous -- convTranspose1d()'s scatter update, after its weight tile
/// has been transposed into a per-tap-contiguous layout (mirroring
/// conv1d()'s own weight transpose, see conv1d()'s doc). A real vector FMA
/// per 4 elements, unlike esp-dsp which has no float32 multiply-accumulate
/// primitive at all (checked directly against its vendored source).
inline void neonMla(float* y, const float* w, int n, float xv) {
  float32x4_t vxv = vdupq_n_f32(xv);
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t vy = vmlaq_f32(vld1q_f32(y + i), vxv, vld1q_f32(w + i));
    vst1q_f32(y + i, vy);
  }
  for (; i < n; i++) y[i] += xv * w[i];
}

}  // namespace detail
#endif  // TINYTTS_HAVE_NEON

/// Whether linear(), float conv1d() and convTranspose1d() run on the Tang
/// Nano 20K's AI accelerator (TINYTTS_HAVE_TANG_AI) -- default on there.
/// Those ops have float weights, so the accelerated path quantizes both
/// sides to INT8 on the fly (weights per output channel, activations per
/// timestep, symmetric abs-max/127, like the decoder's INT8 path): a
/// speed/quality tradeoff. Turn it off to keep them in float. Always false,
/// and setting it a no-op, on builds without the accelerator.
inline bool& accelerateFloatOpsRef() {
#ifdef TINYTTS_HAVE_TANG_AI
  static bool on = true;
#else
  static bool on = false;
#endif
  return on;
}
inline void setAccelerateFloatOps(bool on) {
#ifdef TINYTTS_HAVE_TANG_AI
  accelerateFloatOpsRef() = on;
#else
  (void)on;
#endif
}
inline bool accelerateFloatOps() { return accelerateFloatOpsRef(); }

namespace detail {

/// Per-timestep activation scales as Q24 fixed-point fractions of their
/// maximum (see TINYTTS_FIXED_POINT): scale[t] ~= q[t] * unit.
struct FixedScales {
  InternalVector<int32_t> q;
  float unit = 0.0f;
};

TINYTTS_HOT inline void toFixedScales(const float* scale, int n, FixedScales& out) {
  float mx = 0.0f;
  for (int i = 0; i < n; i++) mx = std::max(mx, scale[i]);
  if (mx <= 0.0f) mx = 1.0f;
  const float kOne = 16777216.0f;  // 2^24
  out.unit = mx / kOne;
  float inv = kOne / mx;
  out.q.resize((size_t)n);
  for (int i = 0; i < n; i++) out.q[i] = (int32_t)(scale[i] * inv + 0.5f);
}

inline uint32_t floatBits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

inline float bitsFloat(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

/// max |x[i]| without float operations: for non-negative IEEE floats the
/// bit patterns compare like the values.
TINYTTS_HOT inline float absMaxBits(const float* x, int n) {
  uint32_t m = 0;
  for (int i = 0; i < n; i++) {
    uint32_t u = floatBits(x[i]) & 0x7fffffffu;
    if (u > m) m = u;
  }
  return bitsFloat(m);
}

/// A positive float factor split for quantizeInt(): 24-bit mantissa (with
/// the implicit 1) and biased exponent.
struct IntFactor {
  uint32_t mant;
  int exp;
};

inline IntFactor intFactor(float f) {
  uint32_t u = floatBits(f);
  return {(u & 0x7fffffu) | 0x800000u, (int)((u >> 23) & 0xff)};
}

/// round(x * f) clamped to [-127, 127], with integer instructions only:
/// x * f = Mx * Mf * 2^(Ex + Ef - 300); the 48-bit mantissa product comes
/// from one 32x32->64 multiply (mulhu on RV32IM), of which the top 32 bits
/// are kept, then shifted with round-half-away-from-zero. Denormal x (and
/// 0) give 0.
TINYTTS_HOT inline int8_t quantizeInt(float x, IntFactor f) {
  uint32_t u = floatBits(x);
  int ex = (int)((u >> 23) & 0xff);
  if (ex == 0) return 0;
  uint32_t mx = (u & 0x7fffffu) | 0x800000u;
  uint32_t hi = (uint32_t)(((uint64_t)(mx << 8) * (uint64_t)(f.mant << 8)) >> 32);  // (mx * mant) >> 16
  int sh = 300 - ex - f.exp - 16;
  int q;
  if (sh <= 0)
    q = 127;
  else if (sh > 32)
    q = 0;
  else
    q = (int)(((hi >> (sh - 1)) + 1) >> 1);
  if (q > 127) q = 127;
  return (int8_t)((u >> 31) ? -q : q);
}

/// Symmetric per-row INT8 quantization: scale = abs-max/127.
TINYTTS_HOT inline void quantizeRow(const float* src, int n, int8_t* dst, float* scale) {
#ifdef TINYTTS_FIXED_POINT
  float s = std::max(absMaxBits(src, n), 1e-8f) / 127.0f;
  *scale = s;
  IntFactor f = intFactor(1.0f / s);
  for (int i = 0; i < n; i++) dst[i] = quantizeInt(src[i], f);
#else
  float m = 0.0f;
  for (int i = 0; i < n; i++) m = std::max(m, std::fabs(src[i]));
  float s = std::max(m, 1e-8f) / 127.0f;
  *scale = s;
  float inv = 1.0f / s;
  for (int i = 0; i < n; i++) {
    float v = src[i] * inv;
    int q = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));  // no libm lround() call
    dst[i] = (int8_t)std::max(-127, std::min(127, q));
  }
#endif
}

}  // namespace detail

#ifdef TINYTTS_HAVE_TANG_AI
namespace detail {

// AIAccelerator's fixed gateware limits (see AIAccelerator.h): at most 8
// weight rows and 16 taps per tile, and k * cinPadded <= 1024 bytes.
constexpr int kAccelRows = 8;
constexpr int kAccelMaxTaps = 16;
constexpr int kAccelMaxWindow = 1024;

// Below these channel counts the engine costs more than it saves: input
// channels are padded to kSimdAlign (16) bytes per tap and every call
// computes and returns all 8 rows, so e.g. a 2-channel vocoder layer
// makes 2-8 bus accesses per useful multiply-accumulate -- work the CPU
// does with one integer multiply. Such layers stay on the CPU loop.
constexpr int kAccelMinInChannels = 16;
constexpr int kAccelMinOutChannels = 8;
inline bool accelWorthIt(int cin, int cout) { return cin >= kAccelMinInChannels && cout >= kAccelMinOutChannels; }


inline int padTo(int n, int align) { return (n + align - 1) / align * align; }

/// Activations quantized per row (timestep), rows `stride` bytes apart
/// (cols rounded up to kSimdAlign, zero-padded).
struct QuantRows {
  InternalVector<int8_t> q;
  InternalVector<float> scale;
  int stride = 0;
};

inline void quantizeMat(const Mat& x, QuantRows& out) {
  out.stride = padTo(x.cols(), (int)kSimdAlign);
  out.q.assign((size_t)x.rows() * out.stride, 0);
  out.scale.resize((size_t)x.rows());
  for (int t = 0; t < x.rows(); t++)
    quantizeRow(x.row(t), x.cols(), out.q.data() + (size_t)t * out.stride, &out.scale[t]);
}

/// Which input row each tap of an accelerated op reads: for output row t
/// and tap kk it's (t + offset[kk]) / stride, used only when t + offset[kk]
/// is >= 0 and divisible by stride and the row is < rows_in. conv1d():
/// stride 1, offset[kk] = kk * dilation - pad; linear(): stride 1, offset
/// 0; convTranspose1d(): its stride, offset[j] = padding - (the j-th tap).
struct AccelTaps {
  int k = 0;
  int stride = 1;
  int rows_in = 0;
  int offset[kAccelMaxTaps] = {};
};

/// An accelerated op's INT8 weights: fill() writes output channels
/// co0..co0+rows, input channels c0..c0+cn into the zeroed tile, laid out
/// [row][tap][chunk]; scale[co] is output channel co's weight scale. A
/// function pointer rather than a template parameter, so accelGather() is
/// one ordinary function: GCC ignores TINYTTS_HOT's section attribute on
/// template instantiations.
struct AccelWeights {
  void (*fill)(const void* ctx, int co0, int rows, int c0, int cn, int chunk, int8_t* tile);
  const void* ctx;
  const float* scale;
};

/// The one engine loop every accelerated op goes through:
///   y[t][co] += scale[co] * sum_kk dot(xq[src], W[co][kk]) * xs[src]
/// for t in [0,T_out), co in [0,cout), src the input row taps.offset/stride
/// give (taps without one are skipped). xq/xs: activations quantized per
/// row, rows `xstride` bytes apart (a multiple of kSimdAlign, zero-padded
/// past cin). 8 output channels x k taps per compute() call; windows over
/// 1KB are split into input-channel chunks, summing the partial results.
/// Returns false (y untouched) if the layer doesn't suit the engine or its
/// buffers can't be allocated -- the caller then uses its own loop.
TINYTTS_HOT inline bool accelGather(const int8_t* xq, const float* xs, int xstride, int cin, int cout, int T_out,
                                    const AccelTaps& taps, const AccelWeights& w, Mat& y) {
  int k = taps.k;
  if (k <= 0 || k > kAccelMaxTaps || !accelWorthIt(cin, cout)) return false;
  int chunk = (kAccelMaxWindow / k) / (int)kSimdAlign * (int)kSimdAlign;
  if (chunk <= 0) return false;
  chunk = std::min(chunk, xstride);

  AIAccelerator accel;
  if (!accel.begin((uint16_t)chunk, (uint8_t)k, (uint8_t)kAccelRows)) return false;
  InternalVector<int8_t> wtile((size_t)kAccelRows * k * chunk);
  InternalVector<int8_t> window((size_t)k * chunk);
  float scales[kAccelRows];
  int src[kAccelMaxTaps];
  // Integer tap sums, see TINYTTS_FIXED_POINT (always used here: this
  // only runs on the Tang Nano 20K, which has no FPU).
  FixedScales xs_q;
  toFixedScales(xs, taps.rows_in, xs_q);

  for (int co0 = 0; co0 < cout; co0 += kAccelRows) {
    int tile_rows = std::min(kAccelRows, cout - co0);
    for (int r = 0; r < tile_rows; r++) scales[r] = w.scale[co0 + r] * xs_q.unit;
    for (int c0 = 0; c0 < cin; c0 += chunk) {
      int cn = std::min(chunk, cin - c0);
      std::fill(wtile.begin(), wtile.end(), (int8_t)0);
      w.fill(w.ctx, co0, tile_rows, c0, cn, chunk, wtile.data());
      accel.loadWeights(wtile.data(), wtile.size());

      // xq rows are xstride wide; the last chunk may reach past that.
      int copy = std::min(chunk, xstride - c0);
      for (int t = 0; t < T_out; t++) {
        bool any = false;
        for (int kk = 0; kk < k; kk++) {
          int num = t + taps.offset[kk];
          int s = -1;
          if (num >= 0) {
            if (taps.stride == 1)
              s = num;
            else if (num % taps.stride == 0)
              s = num / taps.stride;
          }
          if (s >= taps.rows_in) s = -1;
          src[kk] = s;
          int8_t* dst = window.data() + (size_t)kk * chunk;
          if (s >= 0) {
            std::memcpy(dst, xq + (size_t)s * xstride + c0, (size_t)copy);
            std::memset(dst + copy, 0, (size_t)(chunk - copy));
            any = true;
          } else {
            std::memset(dst, 0, (size_t)chunk);
          }
        }
        if (!any) continue;
        const int32_t* res;
        {
          TINYTTS_PROFILE_SCOPE(kEngine);
          res = accel.compute(window.data(), window.size());
        }
        float* yr = y.row(t);
        for (int r = 0; r < tile_rows; r++) {
          int64_t acc = 0;
          for (int kk = 0; kk < k; kk++)
            if (src[kk] >= 0) acc += (int64_t)res[(size_t)r * k + kk] * xs_q.q[src[kk]];
          yr[co0 + r] += (float)acc * scales[r];
        }
      }
    }
  }
  return true;
}

/// Weights requantized per output channel, laid out [co][tap][cin].
struct QuantWeights {
  InternalVector<int8_t> q;
  InternalVector<float> scale;
  int cin = 0, k = 0;
  const int8_t* at(int co, int kk) const { return q.data() + ((size_t)co * k + kk) * cin; }
};

/// AccelWeights::fill for QuantWeights: engine tap j is weight tap taps[j].
struct QuantWeightsTaps {
  const QuantWeights* w;
  const int* taps;
  int k;
};

inline void fillQuantWeights(const void* ctx, int co0, int rows, int c0, int cn, int chunk, int8_t* tile) {
  const QuantWeightsTaps& q = *static_cast<const QuantWeightsTaps*>(ctx);
  for (int r = 0; r < rows; r++)
    for (int j = 0; j < q.k; j++)
      std::memcpy(tile + ((size_t)r * q.k + j) * chunk, q.w->at(co0 + r, q.taps[j]) + c0, (size_t)cn);
}

/// AccelWeights::fill for a dtype 2 (INT8) conv1d weight, used as stored.
inline void fillInt8Entry(const void* ctx, int co0, int rows, int c0, int cn, int chunk, int8_t* tile) {
  const WeightStore::Entry& w = *static_cast<const WeightStore::Entry*>(ctx);
  int cin = w.shape[1], k = w.shape[2];
  size_t row_size = (size_t)cin * k;
  for (int r = 0; r < rows; r++) {
    size_t base = (size_t)(co0 + r) * row_size;
    for (int kk = 0; kk < k; kk++) {
      int8_t* dst = tile + ((size_t)r * k + kk) * chunk;
      for (int ci = 0; ci < cn; ci++) dst[ci] = (int8_t)w.raw[base + (size_t)(c0 + ci) * k + kk];
    }
  }
}

inline AccelTaps conv1dTaps(int k, int dilation, int pad, int rows_in) {
  AccelTaps taps;
  taps.k = k;
  taps.rows_in = rows_in;
  for (int kk = 0; kk < k; kk++) taps.offset[kk] = kk * dilation - pad;
  return taps;
}

inline void initBias(Mat& y, const std::vector<float>& bias) {
  for (int t = 0; t < y.rows(); t++) {
    float* yr = y.row(t);
    for (int co = 0; co < y.cols(); co++) yr[co] = bias.empty() ? 0.0f : bias[co];
  }
}

/// linear() on the engine: k=1, weights w [cout][cin].
inline bool linearAccel(const Mat& x, const Mat& w, const std::vector<float>& bias, Mat& y) {
  if (!accelWorthIt(w.cols(), w.rows())) return false;
  TINYTTS_PROFILE_SCOPE(kLinearAccel);
  QuantWeights qw;
  qw.cin = w.cols();
  qw.k = 1;
  qw.q.resize((size_t)w.rows() * w.cols());
  qw.scale.resize((size_t)w.rows());
  for (int co = 0; co < w.rows(); co++)
    quantizeRow(w.row(co), w.cols(), qw.q.data() + (size_t)co * w.cols(), &qw.scale[co]);
  QuantRows xq;
  quantizeMat(x, xq);
  static const int kTap0[] = {0};
  QuantWeightsTaps ctx{&qw, kTap0, 1};
  AccelWeights weights{fillQuantWeights, &ctx, qw.scale.data()};
  initBias(y, bias);
  return accelGather(xq.q.data(), xq.scale.data(), xq.stride, w.cols(), w.rows(), x.rows(),
                     conv1dTaps(1, 1, 0, x.rows()), weights, y);
}

/// conv1d() with float/float16 weights on the engine (weights [Cout, Cin, K]).
inline bool conv1dFloatAccel(const Mat& x, const WeightStore::Entry& w, const std::vector<float>& bias,
                             int dilation, int pad, Mat& y) {
  int cout = w.shape[0], cin = w.shape[1], k = w.shape[2];
  if (k > kAccelMaxTaps || !accelWorthIt(cin, cout)) return false;
  TINYTTS_PROFILE_SCOPE(kConv1dFloatAccel);
  size_t row_size = (size_t)cin * k;
  QuantWeights qw;
  qw.cin = cin;
  qw.k = k;
  qw.q.resize((size_t)cout * row_size);
  qw.scale.resize((size_t)cout);
  InternalVector<float> row(row_size), tr(row_size);
  for (int co = 0; co < cout; co++) {
    w.decodeRun((size_t)co * row_size, row_size, row.data());  // [ci][kk]
    for (int kk = 0; kk < k; kk++)
      for (int ci = 0; ci < cin; ci++) tr[(size_t)kk * cin + ci] = row[(size_t)ci * k + kk];
    quantizeRow(tr.data(), (int)row_size, qw.q.data() + (size_t)co * row_size, &qw.scale[co]);
  }
  QuantRows xq;
  quantizeMat(x, xq);
  int tap_index[kAccelMaxTaps];
  for (int kk = 0; kk < k; kk++) tap_index[kk] = kk;
  QuantWeightsTaps ctx{&qw, tap_index, k};
  AccelWeights weights{fillQuantWeights, &ctx, qw.scale.data()};
  initBias(y, bias);
  return accelGather(xq.q.data(), xq.scale.data(), xq.stride, cin, cout, x.rows(),
                     conv1dTaps(k, dilation, pad, x.rows()), weights, y);
}

/// conv1d()'s INT8-activation path (dtype 2 weights) on the engine: the
/// stored INT8 weights and per-row scales are used as they are, with the
/// caller's per-row quantized activations -- the same numbers the scalar
/// INT8 loop multiplies. `y` must already hold the bias.
inline bool conv1dInt8Accel(const int8_t* xq, const float* xs, int xstride, const WeightStore::Entry& w,
                            int dilation, int pad, Mat& y) {
  int cout = w.shape[0], cin = w.shape[1], k = w.shape[2];
  if (k > kAccelMaxTaps || !accelWorthIt(cin, cout)) return false;
  TINYTTS_PROFILE_SCOPE(kConv1dInt8Accel);
  InternalVector<float> scale((size_t)cout);  // row_scale may sit in slow flash: read it once
  for (int co = 0; co < cout; co++) scale[co] = w.rowScale(co);
  AccelWeights weights{fillInt8Entry, &w, scale.data()};
  return accelGather(xq, xs, xstride, cin, cout, y.rows(), conv1dTaps(k, dilation, pad, y.rows()), weights, y);
}

/// convTranspose1d() on the engine, as a gather: output t_out receives
/// x[t] * W[ci][co][kk] for every kk with t_out = t*stride - padding + kk,
/// i.e. only taps kk congruent to (t_out + padding) mod stride. One engine
/// pass per residue class, with just that class's taps, so no zero taps
/// are sent. `y` must already hold the bias; on false it may hold partial
/// sums.
inline bool convTranspose1dAccel(const Mat& x, const WeightStore::Entry& w, int stride, int padding, Mat& y) {
  int cin = w.shape[0], cout = w.shape[1], k = w.shape[2];
  if ((k + stride - 1) / stride > kAccelMaxTaps || !accelWorthIt(cin, cout)) return false;
  TINYTTS_PROFILE_SCOPE(kConvTranspose1dAccel);
  size_t row_size = (size_t)cout * k;
  // Transpose [ci][co][kk] to [co][kk][ci], then quantize per co.
  InternalVector<float> wt((size_t)cout * k * cin), row(row_size);
  for (int ci = 0; ci < cin; ci++) {
    w.decodeRun((size_t)ci * row_size, row_size, row.data());
    for (int co = 0; co < cout; co++)
      for (int kk = 0; kk < k; kk++) wt[((size_t)co * k + kk) * cin + ci] = row[(size_t)co * k + kk];
  }
  QuantWeights qw;
  qw.cin = cin;
  qw.k = k;
  qw.q.resize(wt.size());
  qw.scale.resize((size_t)cout);
  for (int co = 0; co < cout; co++)
    quantizeRow(wt.data() + (size_t)co * k * cin, k * cin, qw.q.data() + (size_t)co * k * cin, &qw.scale[co]);
  wt.clear();
  wt.shrink_to_fit();

  QuantRows xq;
  quantizeMat(x, xq);
  for (int rho = 0; rho < stride && rho < k; rho++) {
    int tap_index[kAccelMaxTaps];
    AccelTaps taps;
    taps.stride = stride;
    taps.rows_in = x.rows();
    for (int kk = rho; kk < k; kk += stride) {
      tap_index[taps.k] = kk;
      taps.offset[taps.k] = padding - kk;
      taps.k++;
    }
    QuantWeightsTaps ctx{&qw, tap_index, taps.k};
    AccelWeights weights{fillQuantWeights, &ctx, qw.scale.data()};
    bool ok = accelGather(xq.q.data(), xq.scale.data(), xq.stride, cin, cout, y.rows(), taps, weights, y);
    // A later class failing leaves y partly accumulated -- the caller
    // resets it to the bias before falling back.
    if (!ok) return false;
  }
  return true;
}

}  // namespace detail
#endif  // TINYTTS_HAVE_TANG_AI

/// Conv1d(kernel_size=1) == per-timestep Linear. y[t,co] = sum_ci x[t,ci]*W[co,ci] + b[co]
TINYTTS_HOT inline Mat linear(const Mat& x, const Mat& w, const std::vector<float>& bias) {
  TINYTTS_PROFILE_SCOPE(kLinear);
  Mat y(x.rows(), w.rows());
#ifdef TINYTTS_HAVE_TANG_AI
  // Falls through to the float loop below (which overwrites y) on failure.
  if (accelerateFloatOps() && detail::linearAccel(x, w, bias, y)) return y;
#endif
  for (int t = 0; t < x.rows(); t++) {
    const float* xr = x.row(t);
    float* yr = y.row(t);
    for (int co = 0; co < w.rows(); co++) {
      float acc = bias.empty() ? 0.0f : bias[co];
      const float* wr = w.row(co);
#ifdef TINYTTS_HAVE_ESP_DSP
      // Both xr and wr are plain contiguous Mat rows here (unlike
      // conv1d()'s tiled/strided weight layout), so the unstrided,
      // slightly-faster dsps_dotprod_f32 applies directly. NOTE: despite
      // upstream's doc comment claiming "*dest += ...", every real
      // implementation (ansi/ae32/aes3, checked directly in the vendored
      // source) does `*dest = acc` -- an overwrite, not an accumulate.
      // Passing `&acc` directly here would silently discard the bias
      // already stored in it -- dot into a scratch var and add instead.
      float dot = 0.0f;
      dsps_dotprod_f32(xr, wr, &dot, w.cols());
      acc += dot;
#elif defined(TINYTTS_HAVE_NEON)
      float dot = 0.0f;
      detail::neonDotProd(xr, wr, &dot, w.cols());
      acc += dot;
#else
      for (int ci = 0; ci < w.cols(); ci++) acc += xr[ci] * wr[ci];
#endif
      yr[co] = acc;
    }
  }
  return y;
}

// Weight tensors here never exceed this many kernel positions (decoder's
// biggest upsample kernel is 16) -- a fixed stack buffer avoids a heap
// allocation in these hot loops. Generous headroom over the real max.
constexpr int kMaxKernelSize = 64;

// Bound on cin*k for conv1d() / cout*k for convTranspose1d() (one weight
// row's element count, a.k.a. row_size below) -- used to size small, fixed
// internal-heap scratch buffers (InternalVector/AlignedInternalVector, see
// their call sites) for both ops' weight-tile transpose, conv1d's
// per-timestep input-gather, and its INT8-activation per-window quantize
// buffer (see their own docs for why each exists). This is the actual
// verified maximum across every layer in the shipped decoder, not a padded
// guess -- deliberately NOT given extra headroom the way kMaxKernelSize
// above is, to keep these buffers' internal-RAM footprint tight (see
// kWeightTileRows's doc for the internal-RAM-exhaustion story this
// project's own history already warns about).
//
// These were plain STACK arrays (`float buf[kMaxRowSize]`) at first --
// switched to InternalVector after a real ESP32-S3 crash (Guru Meditation,
// "Stack canary watchpoint triggered (loopTask)") once the INT8-activation
// path added one more such buffer: `conv1d()`'s tile lambda can run on the
// calling thread directly (the Arduino main loopTask) rather than always on
// TileSplitter's dedicated 16KB worker-task stack, and loopTask's own,
// much smaller default stack was already mostly spent by the rest of the
// synthesis call chain before conv1d() even runs. Internal-heap allocation
// (same allocator wtile/wtile_i8 already use) sidesteps the question of
// which thread's stack budget applies entirely.
// Verified by direct runtime instrumentation across a full speak() call
// (every conv1d()/convTranspose1d() call site, not just the ones a static
// read of the model source suggested -- an earlier attempt at this bound,
// based on reading research/tiny_tts's Python reference config, missed
// DurationPredictor's and TransformerBlock's FFN conv1d calls entirely and
// separately got the shipped model's actual filter_channels wrong (256, not
// the config file's 128) -- both mistakes caused a real stack-buffer
// overflow/crash before this was caught and re-measured directly).
// Real max: conv1d's largest is DurationPredictor's conv_2 (Cin=256, K=3 ->
// 768); convTranspose1d's largest is the decoder's stage-0 upsample
// (Cout=32, K=16 -> 512). Channel counts and kernel sizes are fixed by the
// model architecture, not by input text, so this holds for every possible
// input, not just the utterance used to measure it -- but if the model
// architecture ever changes (more channels, larger kernels, a new module
// that also calls these ops), this constant must be re-verified against the
// new real max, ideally the same way (instrument and run, not just read the
// source) -- undersizing it means these stack buffers silently overflow.
constexpr int kMaxRowSize = 768;

// Rows (dim0 of the weight tensor -- Cout for conv1d, Cin for
// convTranspose1d) decoded into internal RAM at a time. A whole-tensor cache
// was tried first (see git history) and crashed on real ESP32-S3 hardware
// with std::bad_alloc: internal RAM left over after FreeRTOS/I2S/WiFi-BT
// reservations turned out to be far less than the ~130KB a full tensor can
// need, and even a single conv1d's much smaller ~49KB tensor failed to
// allocate. A small fixed tile keeps the per-allocation footprint tiny
// (worst case in this model: 8 rows * 512 floats/row * 4 bytes = 16KB) while
// still turning most of the T-fold redundant PSRAM weight refetch (the
// original problem the user's tiled-GEMM suggestion was pointing at) into a
// per-tile fetch instead -- each row is decoded once per tile pass rather
// than once per output timestep.
constexpr int kWeightTileRows = 8;

#ifdef TINYTTS_HAVE_TASK
namespace detail {
inline int& numWorkersRef() {
  static int n = 2;  // matches ESP32's two real cores
  return n;
}
}  // namespace detail
#endif

/// How many participants (calling thread + workers) conv1d()/
/// convTranspose1d() split their tile loop across via the shared
/// TileSplitter (sharedSplitter(), below) -- 2 (default) matches ESP32's
/// two real physical cores; on desktop this can be raised to use more.
/// Must be called before the first synthesize() call: sharedSplitter()'s
/// worker pool is created lazily on first use and persists for the
/// process/device lifetime (see TileSplitter's own doc for why), so
/// changing this afterward has no effect. No-op when TINYTTS_HAVE_TASK
/// isn't defined (a single-core build has nothing to configure).
inline void setNumWorkers(int n) {
#ifdef TINYTTS_HAVE_TASK
  detail::numWorkersRef() = n;
#else
  (void)n;
#endif
}

inline int numWorkers() {
#ifdef TINYTTS_HAVE_TASK
  return detail::numWorkersRef();
#else
  return 1;
#endif
}

/// `conv1d()`'s activation precision -- kFloat32 (default) matches every
/// other op in this file; kInt8Activations quantizes activations to INT8
/// per-timestep (symmetric, abs-max/127 over each input row independently)
/// and does a real INT8xINT8 dot product against the already-INT8 decoder
/// weights (dtype 2, see WeightStore.h), rescaling the int32 result back to
/// float once at the end -- a genuine speed/quality tradeoff, not just
/// weights-only quantization's float-activation scheme. Only `conv1d()`
/// honors this (see its own doc for why `convTranspose1d()` doesn't: its
/// scatter pattern has no single dot-product primitive to swap in). On
/// ESP32 (TINYTTS_HAVE_ESP_DSP), the dot product itself runs as real SIMD
/// (dsps_dp_s8); on the Tang Nano 20K with Tools > AI Accelerator enabled
/// (TINYTTS_HAVE_TANG_AI) it runs on the on-chip INT8 engine, which also
/// makes kInt8Activations the default there; everywhere else it's the
/// identical quantization math via a plain scalar loop -- same audible quality tradeoff, just not the
/// speed one (see docs/desktop.md).
///
/// Deliberately per-ROW, not per-window (one shared scale across the whole
/// k-tap receptive field, which is what conv1d()'s float32 path uses to
/// fuse all k taps into a single dot-product call): per-window quantization
/// was implemented and measured on real ESP32-S3 hardware, and reverted --
/// see conv1d()'s `xq`/`x_scale` doc for the numbers. It was slower AND
/// lower quality, not a tradeoff worth keeping.
enum class DecoderPrecision { kFloat32, kInt8Activations };

namespace detail {
inline DecoderPrecision& decoderPrecisionRef() {
#ifdef TINYTTS_HAVE_TANG_AI
  // Enabling the accelerator only pays off on the INT8 path.
  static DecoderPrecision p = DecoderPrecision::kInt8Activations;
#else
  static DecoderPrecision p = DecoderPrecision::kFloat32;
#endif
  return p;
}
}  // namespace detail

inline void setDecoderPrecision(DecoderPrecision p) { detail::decoderPrecisionRef() = p; }
inline DecoderPrecision decoderPrecision() { return detail::decoderPrecisionRef(); }

/// The one TileSplitter shared by conv1d() and convTranspose1d() -- they're
/// never called concurrently with each other (one sequential synthesis
/// pipeline, stage by stage), so sharing a single pool avoids paying for
/// 2x the worker tasks two separate pools would otherwise need.
inline TileSplitter& sharedSplitter() {
  static TileSplitter splitter(numWorkers());
  return splitter;
}

// Forward-declared: defined further below, but convTranspose1d() (just
// below conv1d()) needs it to merge its per-participant accumulators.
inline void addInplace(Mat& a, const Mat& b);

/// General Conv1d, weight shape [Cout, Cin, K], "same" padding
/// (pad=dilation*(K-1)/2, K odd), stride 1. Tiles over Cout in
/// kWeightTileRows-row chunks (see that constant's doc): each output
/// channel's result only depends on its own weight row, so tiling the outer
/// loop over co is exact, not an approximation -- just trades one full pass
/// over x per tile (still PSRAM traffic, but far less than the weight-side
/// traffic this avoids) for a bounded, internal-RAM-safe weight cache.
TINYTTS_HOT inline Mat conv1d(const Mat& x, const WeightStore::Entry& w, const std::vector<float>& bias, int dilation = 1) {
  TINYTTS_PROFILE_SCOPE(kConv1d);
  int cout = w.shape[0], cin = w.shape[1], k = w.shape[2];
  int pad = dilation * (k - 1) / 2;
  int T = x.rows();
  Mat y(T, cout);
  size_t row_size = (size_t)cin * k;
  int num_tiles = (cout + kWeightTileRows - 1) / kWeightTileRows;

  // Only decoder conv weights are ever dtype 2 (see WeightStore.h), but
  // check it explicitly rather than assume -- this op is generic.
  bool use_int8 = decoderPrecision() == DecoderPrecision::kInt8Activations && w.dtype == 2;

  // cin padded up to a multiple of kSimdAlign (16) with zero bytes --
  // lets the INT8 SIMD kernel (esp_nn_dot_s8_aligned_esp32s3, see
  // src/int8-dotprod/) always run over a full 16-multiple length, no
  // separate remainder/tail path needed: zero-valued padding elements
  // don't change a dot product's result. Both xq (below) and wtile_i8
  // (in the tile lambda) are laid out with this padded stride.
  int cin_padded = use_int8 ? (int)(((size_t)cin + kSimdAlign - 1) / kSimdAlign * kSimdAlign) : cin;

  // Quantize x once, per-timestep (row) -- not per-tensor, and not redone
  // per weight tile -- before entering the tile loop below. Per-row (not
  // the historical experiment's per-tensor scale, see docs/performance.md)
  // costs nothing extra here since this loop is already row-major; doing
  // it once up front (rather than implicitly once per tile, the way the
  // float32 path re-reads x from PSRAM per tile) avoids num_tiles-fold
  // redundant quantization work.
  //
  // Deliberately per-ROW, not per-window: a per-window scale (one scale
  // per receptive-field, matching float32's fused single-dot-product
  // trick above) was tried and reverted -- it requires re-scanning/
  // re-quantizing k*cin elements from scratch for every output timestep,
  // redundantly per tile, instead of once per input row shared globally
  // across every conv1d() call that reads it. Measured on real ESP32-S3
  // hardware: ~78.3s decoder time, dramatically SLOWER than both this
  // per-row scheme's own prior number (~51.5s dual-core) and float32's
  // fused 29.8s -- the fused SIMD-call savings were completely swamped by
  // the added quantization overhead. It was ALSO lower audio quality
  // (18.36dB SNR / 0.9927 cosine similarity vs. this scheme's 23.37dB /
  // 0.9977) -- worse on both axes, not a tradeoff, so not kept. See
  // docs/performance.md for the full writeup.
  //
  // AlignedPsramVector, NOT PsramVector/InternalVector -- two distinct
  // requirements: (1) PSRAM, not internal RAM (InternalVector means
  // on-chip internal SRAM specifically, only ~300KB total -- using it
  // here once crashed hard on real ESP32-S3 hardware, std::bad_alloc
  // resizing to a mere 128KB, once T got large enough; xq/x_scale scale
  // with T, the full frame count, unlike wtile/wtile_i8's fixed
  // per-layer-channel-count size); (2) 16-byte pointer alignment --
  // esp_nn_dot_s8_aligned_esp32s3 requires it (undefined behavior
  // otherwise, it's a 128-bit vector load), which plain PsramVector's
  // heap_caps_malloc doesn't guarantee. See AlignedStlAllocator.h.
  //
  // function-local static, not a fresh alloc/free every call: conv1d() is
  // called many times per synthesis (once per decoder layer), so this
  // avoids that many PSRAM alloc/free cycles on top of the existing
  // per-tile wtile churn. Safe as a static despite conv1d() participating
  // in TileSplitter's parallel tile loop below: these are computed once,
  // sequentially, BEFORE that parallel section starts, and only ever read
  // (never written) from within it -- unlike wtile/wtile_i8 below, which
  // genuinely must stay per-participant-local (see their own doc).
  static AlignedPsramVector<int8_t> xq;
  static PsramVector<float> x_scale;  // plain PsramVector: never touched by the SIMD kernel, no alignment need
  if (use_int8) {
    TINYTTS_PROFILE_SCOPE(kConv1dQuantize);
    xq.resize((size_t)T * cin_padded);
    x_scale.resize((size_t)T);
    int pad_cols = cin_padded - cin;
    for (int t = 0; t < T; t++) {
      const float* xr = x.row(t);
      int8_t* qr = xq.data() + (size_t)t * cin_padded;
#ifdef TINYTTS_FIXED_POINT
      float s;
      detail::quantizeRow(xr, cin, qr, &s);
      x_scale[t] = s;
#else
      float m = 0.0f;
      for (int ci = 0; ci < cin; ci++) m = std::max(m, std::fabs(xr[ci]));
      float s = std::max(m, 1e-8f) / 127.0f;
      x_scale[t] = s;
      // One division per row, then a multiply and an add per element --
      // no per-element division or lround() call, which matters on cores
      // without an FPU (both are software routines there).
      float inv = 1.0f / s;
      for (int ci = 0; ci < cin; ci++) {
        float v = xr[ci] * inv;
        int q = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
        qr[ci] = (int8_t)std::max(-127, std::min(127, q));
      }
#endif
      // xq is a persistent static buffer reused (not re-zeroed) across
      // calls with DIFFERENT cin_padded strides -- explicitly zero the
      // pad columns every time rather than relying on leftover state
      // from a previous, differently-shaped layer's use.
      if (pad_cols > 0) std::memset(qr + cin, 0, (size_t)pad_cols);
    }
  }
#ifdef TINYTTS_FIXED_POINT
  // Read-only inside the tile loop below, like xq/x_scale.
  static detail::FixedScales x_scale_q;
  if (use_int8) detail::toFixedScales(x_scale.data(), T, x_scale_q);
#endif

#ifdef TINYTTS_HAVE_TANG_AI
  // On failure the tile loop below overwrites every y cell, so a partial
  // accelerated result can't leak through.
  if (use_int8) {
    detail::initBias(y, bias);
    if (detail::conv1dInt8Accel(xq.data(), x_scale.data(), cin_padded, w, dilation, pad, y)) return y;
  } else if (accelerateFloatOps() && detail::conv1dFloatAccel(x, w, bias, dilation, pad, y)) {
    return y;
  }
#endif

  // Each output channel co is written by exactly one tile, so splitting the
  // tile range across two workers (see TileSplitter) writes disjoint
  // columns of y -- no race, as long as each half decodes its weight rows
  // into its OWN wtile rather than a buffer shared between them (hence
  // wtile being local to this lambda, not a shared static, when running
  // split -- see TileSplitter's doc for why that matters).
  size_t row_size_i8 = (size_t)cin_padded * k;  // wtile_i8's per-output-row size (padded stride), vs. row_size (cin*k, unpadded, used to index w.raw)
  auto tileRange = [&](int tile0, int tile1) TINYTTS_HOT {
    InternalVector<float> wtile;
    // INT8 path only: the tile's weights, reshaped from the raw [cin,k]
    // interleaved layout into k separate CONTIGUOUS, zero-padded (to
    // cin_padded) [cin_padded] arrays -- esp_nn_dot_s8_aligned_esp32s3
    // (unlike dsps_dotprode_f32) has no strided/step variant and requires
    // a length that's a multiple of 16, so this reshape+pad is what makes
    // a single SIMD dot-product call per tap possible at all. Read
    // straight from w.raw (no float dequant at all, unlike decodeRun() --
    // cheaper than the float32 path, not just different). AlignedInternal-
    // Vector, not InternalVector: same 16-byte-alignment requirement as
    // xq above, see AlignedStlAllocator.h. A fresh vector each tileRange()
    // call (per-participant, not static -- see this lambda's own doc), so
    // resize()'s zero-initialization already covers the padding columns,
    // no explicit memset needed here (unlike xq, which is static/reused).
    AlignedInternalVector<int8_t> wtile_i8;
    if (use_int8) {
      wtile_i8.resize((size_t)kWeightTileRows * row_size_i8);
    } else {
      wtile.resize((size_t)kWeightTileRows * row_size);
    }
    const int8_t* xqrows[kMaxKernelSize];
    float wscales[kWeightTileRows];  // INT8 path: this tile's per-row weight scales
    for (int tile = tile0; tile < tile1; tile++) {
      int co0 = tile * kWeightTileRows;
      int tile_rows = std::min(kWeightTileRows, cout - co0);
      if (use_int8) {
        for (int r = 0; r < tile_rows; r++) {
          int co = co0 + r;
          // Read once per tile, not once per timestep: row_scale may live
          // in slow memory (e.g. memory-mapped flash on the Tang Nano 20K).
          wscales[r] = w.rowScale(co);
          for (int kk = 0; kk < k; kk++) {
            int8_t* dst = wtile_i8.data() + (size_t)r * row_size_i8 + (size_t)kk * cin_padded;
            for (int ci = 0; ci < cin; ci++) dst[ci] = (int8_t)w.raw[(size_t)co * row_size + (size_t)ci * k + kk];
            // dst[cin..cin_padded) stays zero (fresh vector, see above).
          }
        }
      } else {
        // Decode this row in its native [cin,k] order into a small scratch
        // buffer, then transpose into wtile as [k,cin] -- lets the
        // per-timestep dot product below run as ONE fully-contiguous call
        // over row_size (cin*k) elements instead of k separate
        // strided/gathered calls, one per kernel tap. `rowbuf` is
        // internal-heap-backed (InternalVector, same allocator as
        // wtile/wtile_i8), NOT a raw stack array -- a real ESP32-S3 stack
        // overflow crash (Guru Meditation, "Stack canary watchpoint
        // triggered") was hit with this and xtap below as a plain
        // `float buf[kMaxRowSize]` stack array during INT8-fusion
        // development: the calling thread here can be the Arduino main
        // loopTask, whose default stack is much smaller than
        // TileSplitter's dedicated 16KB worker-task stack (see
        // kMaxRowSize's doc) -- unlike that worker stack, the main task's
        // budget is already mostly spent by the rest of the synthesis call
        // chain before conv1d() even runs. Resized once per tileRange()
        // call (per-participant, matching wtile_i8's own doc for why that
        // matters), not per-row/per-timestep.
        InternalVector<float> rowbuf;
        rowbuf.resize((size_t)kMaxRowSize);
        for (int r = 0; r < tile_rows; r++) {
          w.decodeRun((size_t)(co0 + r) * row_size, row_size, rowbuf.data());
          float* dst = wtile.data() + (size_t)r * row_size;
          for (int kk = 0; kk < k; kk++)
            for (int ci = 0; ci < cin; ci++) dst[(size_t)kk * cin + ci] = rowbuf[(size_t)ci * k + kk];
        }
      }

      // Float32 path only: this tile's full receptive-field window for the
      // CURRENT timestep, gathered into one contiguous [k,cin] buffer
      // (zero-padded for out-of-range taps, matching wtile's transposed
      // layout above) -- rebuilt per tile (redundant across tiles, like
      // wtile's own per-tile weight decode), since caching it for every t
      // up front would need a T*row_size buffer -- infeasible given T runs
      // into the tens of thousands of frames deep in the decoder's upsample
      // stages (see xq/x_scale's own doc above for the identical reasoning
      // applied to activation quantization). Internal-heap-backed, not a
      // stack array -- see rowbuf's doc just above for why (the real crash
      // this caused).
      InternalVector<float> xtap;
      xtap.resize((size_t)kMaxRowSize);
      for (int t = 0; t < T; t++) {
        for (int kk = 0; kk < k; kk++) {
          int ti = t + kk * dilation - pad;
          bool valid = ti >= 0 && ti < T;
          xqrows[kk] = (use_int8 && valid) ? xq.data() + (size_t)ti * cin_padded : nullptr;
          if (!use_int8) {
            float* dst = xtap.data() + (size_t)kk * cin;
            if (valid)
              std::memcpy(dst, x.row(ti), (size_t)cin * sizeof(float));
            else
              std::memset(dst, 0, (size_t)cin * sizeof(float));
          }
        }
        float* yr = y.row(t);
        for (int r = 0; r < tile_rows; r++) {
          int co = co0 + r;
          float acc = bias.empty() ? 0.0f : bias[co];
          if (use_int8) {
            const int8_t* wrow = wtile_i8.data() + (size_t)r * row_size_i8;  // [k][cin_padded], contiguous per tap
            // sum_kk dot_kk * x_scale[ti] * wscale == wscale * sum_kk dot_kk * x_scale[ti]:
            // one weight-scale multiply per output instead of one per tap.
#ifdef TINYTTS_FIXED_POINT
            int64_t taps = 0;  // sum_kk dot_kk * x_scale_q[ti], see TINYTTS_FIXED_POINT
#else
            float taps = 0.0f;
#endif
            for (int kk = 0; kk < k; kk++) {
              if (!xqrows[kk]) continue;
              int32_t dot = 0;
#ifdef TINYTTS_HAVE_ESP_DSP
              // Real INT8 SIMD on ESP32-S3 (esp_nn_dot_s8_aligned_esp32s3,
              // ported from espressif/esp-nn -- see dsps_dp_s8.h/NOTICE.md
              // for why this project rolled its own instead of trusting
              // esp-dsp's own dsps_dp_s8_aes3, which was found broken on
              // real hardware). Scalar ansi fallback elsewhere (e.g.
              // ESP32-P4, or if the alignment/length preconditions this
              // op guarantees ever stop holding).
              dspsDotProdS8(xqrows[kk], wrow + (size_t)kk * cin_padded, &dot, cin_padded);
#else
              // Desktop/other: identical quantization math, scalar loop --
              // same audible quality tradeoff as the SIMD path, just not
              // the speed one (real SIMD only compiles for ESP32-S3).
              const int8_t* wk = wrow + (size_t)kk * cin_padded;
              for (int ci = 0; ci < cin; ci++) dot += (int32_t)xqrows[kk][ci] * (int32_t)wk[ci];
#endif
              int ti = t + kk * dilation - pad;
#ifdef TINYTTS_FIXED_POINT
              taps += (int64_t)dot * x_scale_q.q[ti];
#else
              taps += (float)dot * x_scale[ti];
#endif
            }
#ifdef TINYTTS_FIXED_POINT
            acc += (float)taps * (x_scale_q.unit * wscales[r]);
#else
            acc += taps * wscales[r];
#endif
          } else {
            // wrow is [k,cin], transposed+contiguous (see the weight-decode
            // step above); xtap is this timestep's gathered [k,cin] input
            // window in the identical layout -- so the whole tap loop
            // collapses into ONE dot product over row_size (cin*k)
            // elements, fully contiguous on both sides. This replaces what
            // used to be k separate strided/gathered dot-product calls
            // (one per kernel tap), each paying its own per-call overhead
            // -- see kMaxRowSize's doc for the memory-safety reasoning
            // behind how xtap/wtile are sized and reused.
            const float* wrow = wtile.data() + (size_t)r * row_size;
            float dot = 0.0f;
#ifdef TINYTTS_HAVE_ESP_DSP
            dsps_dotprod_f32(xtap.data(), wrow, &dot, (int)row_size);
#elif defined(TINYTTS_HAVE_NEON)
            detail::neonDotProd(xtap.data(), wrow, &dot, (int)row_size);
#else
            for (size_t i = 0; i < row_size; i++) dot += xtap[i] * wrow[i];
#endif
            acc += dot;
          }
          yr[co] = acc;
        }
      }
    }
  };

  sharedSplitter().run(num_tiles, [&](int tile0, int tile1, int /*participant*/) { tileRange(tile0, tile1); });
  return y;
}

/// ConvTranspose1d, weight shape [Cin, Cout, K] (PyTorch's ConvTranspose1d
/// layout -- note the axis order differs from conv1d()'s [Cout, Cin, K]),
/// stride/padding as in PyTorch (dilation=1, output_padding=0). Output
/// length = (T_in-1)*stride - 2*padding + K. Tiles over Cin (the weight
/// tensor's row dimension here) in kWeightTileRows-row chunks -- see that
/// constant's doc. Unlike conv1d's independent-per-output-channel gather,
/// this op scatters each input channel's contribution across multiple
/// output positions via +=, so tiling here means summing contributions from
/// each Cin-tile in turn into `y` (already bias-initialized once, up front)
/// -- correct because addition is commutative/associative regardless of
/// which ci values are processed in which pass.
inline Mat convTranspose1d(const Mat& x, const WeightStore::Entry& w, const std::vector<float>& bias, int stride,
                            int padding) {
  TINYTTS_PROFILE_SCOPE(kConvTranspose1d);
  int cin = w.shape[0], cout = w.shape[1], k = w.shape[2];
  int t_out_len = (x.rows() - 1) * stride - 2 * padding + k;
  Mat y(t_out_len, cout);
  for (int co = 0; co < cout; co++) {
    float b = bias.empty() ? 0.0f : bias[co];
    for (int t = 0; t < t_out_len; t++) y.at(t, co) = b;
  }
#ifdef TINYTTS_HAVE_TANG_AI
  if (accelerateFloatOps()) {
    if (detail::convTranspose1dAccel(x, w, stride, padding, y)) return y;
    detail::initBias(y, bias);  // the scatter below accumulates onto y
  }
#endif
  size_t row_size = (size_t)cout * k;
  int num_tiles = (cin + kWeightTileRows - 1) / kWeightTileRows;

  // Unlike conv1d's per-output-channel gather (independent, disjoint
  // writes), this op's inner loop scatters read-modify-write additions
  // into `acc` (yrows[kk][co] += ...) -- every ci tile can touch the same
  // (t_out, co) cell, since that's the whole point of a transposed-conv
  // scatter. Splitting the ci tile range across two workers writing into
  // the SAME Mat would be a genuine, silent, nondeterministic lost-update
  // race (worse than this project's earlier silent-corruption bugs,
  // because scheduling-dependent nondeterminism could pass a sanity check
  // on some runs and not others). So each half accumulates into its own
  // private, zero-initialized (no bias) Mat instead, merged into the real,
  // already-bias-initialized `y` via addInplace() once both finish.
  auto tileRange = [&](Mat& acc, int tile0, int tile1) TINYTTS_HOT {
    InternalVector<float> wtile;
    wtile.resize((size_t)kWeightTileRows * row_size);
    float* yrows[kMaxKernelSize];
    for (int tile = tile0; tile < tile1; tile++) {
      int ci0 = tile * kWeightTileRows;
      int tile_rows = std::min(kWeightTileRows, cin - ci0);
      // Decode each row in its native [cout,k] order into a small scratch
      // buffer, then transpose into wtile as [k,cout] -- so the scatter
      // loop below reads/writes contiguous cout-length runs for a fixed
      // tap kk instead of a stride-k gather, mirroring conv1d()'s own
      // weight transpose (see its doc for the same reasoning). Internal-
      // heap-backed (InternalVector), NOT a raw stack array -- see
      // conv1d()'s rowbuf doc for the real ESP32-S3 stack-overflow crash
      // this class of buffer caused when it was a plain stack array on the
      // Arduino main loopTask's much smaller stack.
      InternalVector<float> rowbuf;
      rowbuf.resize((size_t)kMaxRowSize);
      for (int r = 0; r < tile_rows; r++) {
        w.decodeRun((size_t)(ci0 + r) * row_size, row_size, rowbuf.data());
        float* dst = wtile.data() + (size_t)r * row_size;
        for (int kk = 0; kk < k; kk++)
          for (int co = 0; co < cout; co++) dst[(size_t)kk * cout + co] = rowbuf[(size_t)co * k + kk];
      }

      for (int ti = 0; ti < x.rows(); ti++) {
        const float* xr = x.row(ti);
        for (int kk = 0; kk < k; kk++) {
          int t_out = ti * stride - padding + kk;
          yrows[kk] = (t_out < 0 || t_out >= t_out_len) ? nullptr : acc.row(t_out);
        }
        for (int r = 0; r < tile_rows; r++) {
          int ci = ci0 + r;
          float xv = xr[ci];
          const float* wrow = wtile.data() + (size_t)r * row_size;  // [k, cout], transposed+contiguous
          // This op scatters (yrows[kk][co] += xv*w[co,kk]), unlike
          // conv1d's gather, so there's no single dot-product primitive for
          // it -- but with wrow transposed above, a fixed tap kk's cout
          // values ARE now contiguous, same as conv1d()'s fused dot. esp-dsp
          // has no float32 multiply-accumulate op though (checked the
          // vendored source directly, same way the dotprod
          // overwrite-vs-accumulate bug was found -- not trusting doc
          // comments alone this time), so esp32 gets no dedicated SIMD path
          // here: an earlier two-call mulc()+add() version (real esp-dsp
          // SIMD, but writing/re-reading a full `scaled[cout]` scratch
          // buffer through memory between the two calls) measured as a wash
          // against a plain scalar loop (~41.06s vs. ~40.75s, see
          // docs/performance.md) -- removed rather than kept as dead-weight
          // complexity for no real gain.
#if defined(TINYTTS_HAVE_NEON)
          for (int kk = 0; kk < k; kk++) {
            if (!yrows[kk]) continue;
            detail::neonMla(yrows[kk], wrow + (size_t)kk * cout, cout, xv);
          }
#else
          for (int kk = 0; kk < k; kk++) {
            if (!yrows[kk]) continue;
            const float* wk = wrow + (size_t)kk * cout;
            for (int co = 0; co < cout; co++) yrows[kk][co] += xv * wk[co];
          }
#endif
        }
      }
    }
  };

#ifdef TINYTTS_HAVE_TASK
  TileSplitter& splitter = sharedSplitter();
  int n = splitter.participantsFor(num_tiles);
  if (n > 1) {
    // One private, zero-initialized (no bias -- Mat's ctor fills with
    // 0.0f by default) accumulator per participant, indexed by the
    // `participant` TileSplitter hands to the callback -- see this
    // function's own doc above for why a shared accumulator would race.
    std::vector<Mat> accs;
    accs.reserve(n);
    for (int i = 0; i < n; i++) accs.emplace_back(t_out_len, cout);
    splitter.run(num_tiles,
                 [&](int tile0, int tile1, int participant) { tileRange(accs[participant], tile0, tile1); });
    for (int i = 0; i < n; i++) addInplace(y, accs[i]);
    return y;
  }
#endif
  tileRange(y, 0, num_tiles);
  return y;
}

/// LayerNorm over the channel dim, applied independently per timestep (==
/// the model's "ChannelNorm"/"ChannelLayerNorm" -- both are plain
/// per-timestep layer-norm over C despite the different class names in the
/// Python source).
TINYTTS_HOT inline void channelLayerNormInplace(Mat& x, const std::vector<float>& gamma, const std::vector<float>& beta,
                                     float eps = 1e-5f) {
  TINYTTS_PROFILE_SCOPE(kElementwise);
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

TINYTTS_HOT inline void reluInplace(Mat& x) {
  TINYTTS_PROFILE_SCOPE(kElementwise);
  for (auto& v : x.data()) v = std::max(0.0f, v);
}

TINYTTS_HOT inline void leakyReluInplace(Mat& x, float slope = 0.1f) {
  TINYTTS_PROFILE_SCOPE(kElementwise);
  for (auto& v : x.data())
    if (v < 0.0f) v *= slope;
}

inline void addInplace(Mat& a, const Mat& b) {
  TINYTTS_PROFILE_SCOPE(kElementwise);
  for (size_t i = 0; i < a.data().size(); i++) a.data()[i] += b.data()[i];
}

TINYTTS_HOT inline void softmaxInplace(float* row, int n) {
  TINYTTS_PROFILE_SCOPE(kElementwise);
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
