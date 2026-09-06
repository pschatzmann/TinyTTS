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
#include "TinyTTS/PsramStlAllocator.h"
#include "TinyTTS/WeightStore.h"

// Optional: SIMD-accelerated dot products for ESP32-S3/P4. Float32
// (dsps_dotprod_f32/dsps_dotprode_f32/dsps_mulc_f32/dsps_add_f32) comes
// from the Arduino-ESP32 core's OWN bundled esp-dsp component (~3.3.x+
// cores ship one) -- found automatically via <dsps_dotprod.h>/<dsps_mulc.h>/
// <dsps_add.h>, nothing to vendor for that part. INT8
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
#include <dsps_mulc.h>
#include <dsps_add.h>
#include "int8-dotprod/dsps_dp_s8.h"
#define TINYTTS_HAVE_ESP_DSP 1
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

/// Conv1d(kernel_size=1) == per-timestep Linear. y[t,co] = sum_ci x[t,ci]*W[co,ci] + b[co]
inline Mat linear(const Mat& x, const Mat& w, const std::vector<float>& bias) {
  Mat y(x.rows(), w.rows());
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
/// per-timestep (symmetric, abs-max/127) and does a real INT8xINT8 dot
/// product against the already-INT8 decoder weights (dtype 2, see
/// WeightStore.h), rescaling the int32 result back to float once at the
/// end -- a genuine speed/quality tradeoff, not just weights-only
/// quantization's float-activation scheme. Only `conv1d()` honors this
/// (see its own doc for why `convTranspose1d()` doesn't: its scatter
/// pattern has no single dot-product primitive to swap in). On ESP32
/// (TINYTTS_HAVE_ESP_DSP), the dot product itself runs as real SIMD
/// (dsps_dp_s8); everywhere else it's the identical quantization math via
/// a plain scalar loop -- same audible quality tradeoff, just not the
/// speed one (see docs/desktop.md).
enum class DecoderPrecision { kFloat32, kInt8Activations };

namespace detail {
inline DecoderPrecision& decoderPrecisionRef() {
  static DecoderPrecision p = DecoderPrecision::kFloat32;
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
inline Mat conv1d(const Mat& x, const WeightStore::Entry& w, const std::vector<float>& bias, int dilation = 1) {
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
    xq.resize((size_t)T * cin_padded);
    x_scale.resize((size_t)T);
    int pad_cols = cin_padded - cin;
    for (int t = 0; t < T; t++) {
      const float* xr = x.row(t);
      float m = 0.0f;
      for (int ci = 0; ci < cin; ci++) m = std::max(m, std::fabs(xr[ci]));
      float s = std::max(m, 1e-8f) / 127.0f;
      x_scale[t] = s;
      int8_t* qr = xq.data() + (size_t)t * cin_padded;
      for (int ci = 0; ci < cin; ci++) {
        int q = (int)std::lround(xr[ci] / s);
        qr[ci] = (int8_t)std::max(-127, std::min(127, q));
      }
      // xq is a persistent static buffer reused (not re-zeroed) across
      // calls with DIFFERENT cin_padded strides -- explicitly zero the
      // pad columns every time rather than relying on leftover state
      // from a previous, differently-shaped layer's use.
      if (pad_cols > 0) std::memset(qr + cin, 0, (size_t)pad_cols);
    }
  }

  // Each output channel co is written by exactly one tile, so splitting the
  // tile range across two workers (see TileSplitter) writes disjoint
  // columns of y -- no race, as long as each half decodes its weight rows
  // into its OWN wtile rather than a buffer shared between them (hence
  // wtile being local to this lambda, not a shared static, when running
  // split -- see TileSplitter's doc for why that matters).
  size_t row_size_i8 = (size_t)cin_padded * k;  // wtile_i8's per-output-row size (padded stride), vs. row_size (cin*k, unpadded, used to index w.raw)
  auto tileRange = [&](int tile0, int tile1) {
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
    const float* xrows[kMaxKernelSize];
    const int8_t* xqrows[kMaxKernelSize];
    for (int tile = tile0; tile < tile1; tile++) {
      int co0 = tile * kWeightTileRows;
      int tile_rows = std::min(kWeightTileRows, cout - co0);
      if (use_int8) {
        for (int r = 0; r < tile_rows; r++) {
          int co = co0 + r;
          for (int kk = 0; kk < k; kk++) {
            int8_t* dst = wtile_i8.data() + (size_t)r * row_size_i8 + (size_t)kk * cin_padded;
            for (int ci = 0; ci < cin; ci++) dst[ci] = (int8_t)w.raw[(size_t)co * row_size + (size_t)ci * k + kk];
            // dst[cin..cin_padded) stays zero (fresh vector, see above).
          }
        }
      } else {
        for (int r = 0; r < tile_rows; r++)
          w.decodeRun((size_t)(co0 + r) * row_size, row_size, wtile.data() + (size_t)r * row_size);
      }

      for (int t = 0; t < T; t++) {
        for (int kk = 0; kk < k; kk++) {
          int ti = t + kk * dilation - pad;
          bool valid = ti >= 0 && ti < T;
          xrows[kk] = valid ? x.row(ti) : nullptr;
          xqrows[kk] = (use_int8 && valid) ? xq.data() + (size_t)ti * cin_padded : nullptr;
        }
        float* yr = y.row(t);
        for (int r = 0; r < tile_rows; r++) {
          int co = co0 + r;
          float acc = bias.empty() ? 0.0f : bias[co];
          if (use_int8) {
            const int8_t* wrow = wtile_i8.data() + (size_t)r * row_size_i8;  // [k][cin_padded], contiguous per tap
            float wscale = w.rowScale(co);
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
              acc += (float)dot * x_scale[ti] * wscale;
            }
          } else {
            const float* wrow = wtile.data() + (size_t)r * row_size;  // [cin, k]
#ifdef TINYTTS_HAVE_ESP_DSP
            // Per kernel tap kk: dot(xrows[kk][0..cin), wrow[ci*k+kk] for
            // ci in [0..cin)) -- the weight side isn't contiguous (stride k
            // within a [cin,k] row), so this uses dsps_dotprode_f32's step2
            // rather than needing to transpose the tile layout. Every real
            // implementation of dsps_dotprod(e)_f32 (ansi/ae32/aes3, checked
            // directly in the vendored source) does `*dest = acc` -- an
            // overwrite, not the "*dest += ..." upstream's doc comment
            // claims -- so each kk's partial dot goes into a scratch var and
            // gets added to acc explicitly; passing &acc straight through
            // would silently drop the bias and every kk but the last.
            for (int kk = 0; kk < k; kk++) {
              if (!xrows[kk]) continue;
              float dot = 0.0f;
              dsps_dotprode_f32(xrows[kk], wrow + kk, &dot, cin, 1, k);
              acc += dot;
            }
#else
            for (int ci = 0; ci < cin; ci++) {
              const float* wk = wrow + (size_t)ci * k;
              for (int kk = 0; kk < k; kk++)
                if (xrows[kk]) acc += xrows[kk][ci] * wk[kk];
            }
#endif
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
  int cin = w.shape[0], cout = w.shape[1], k = w.shape[2];
  int t_out_len = (x.rows() - 1) * stride - 2 * padding + k;
  Mat y(t_out_len, cout);
  for (int co = 0; co < cout; co++) {
    float b = bias.empty() ? 0.0f : bias[co];
    for (int t = 0; t < t_out_len; t++) y.at(t, co) = b;
  }
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
  auto tileRange = [&](Mat& acc, int tile0, int tile1) {
    InternalVector<float> wtile;
    wtile.resize((size_t)kWeightTileRows * row_size);
#ifdef TINYTTS_HAVE_ESP_DSP
    // Scratch for the mulc+add two-step below (see its doc). cout-sized,
    // tiny (largest in this model is a few hundred floats).
    InternalVector<float> scaled;
    scaled.resize((size_t)cout);
#endif
    float* yrows[kMaxKernelSize];
    for (int tile = tile0; tile < tile1; tile++) {
      int ci0 = tile * kWeightTileRows;
      int tile_rows = std::min(kWeightTileRows, cin - ci0);
      for (int r = 0; r < tile_rows; r++)
        w.decodeRun((size_t)(ci0 + r) * row_size, row_size, wtile.data() + (size_t)r * row_size);

      for (int ti = 0; ti < x.rows(); ti++) {
        const float* xr = x.row(ti);
        for (int kk = 0; kk < k; kk++) {
          int t_out = ti * stride - padding + kk;
          yrows[kk] = (t_out < 0 || t_out >= t_out_len) ? nullptr : acc.row(t_out);
        }
        for (int r = 0; r < tile_rows; r++) {
          int ci = ci0 + r;
          float xv = xr[ci];
          const float* wrow = wtile.data() + (size_t)r * row_size;  // [cout, k]
#ifdef TINYTTS_HAVE_ESP_DSP
          // This op scatters (yrows[kk][co] += xv*w[co,kk]), unlike
          // conv1d's gather, so there's no single dot-product primitive
          // for it. esp-dsp has no float32 multiply-accumulate op either
          // (checked the vendored source directly, same way the dotprod
          // overwrite-vs-accumulate bug was found -- not trusting doc
          // comments alone this time), so this is two SIMD calls instead
          // of one: scale w[:,kk] by xv into `scaled` (dsps_mulc_f32,
          // strided read since w's row is [cout,k]), then add that into
          // yrows[kk] in place (dsps_add_f32) -- both are plain elementwise
          // overwrites of their output, confirmed against the assembly, so
          // no accumulate-semantics trap here.
          for (int kk = 0; kk < k; kk++) {
            if (!yrows[kk]) continue;
            dsps_mulc_f32(wrow + kk, scaled.data(), cout, xv, k, 1);
            dsps_add_f32(yrows[kk], scaled.data(), yrows[kk], cout, 1, 1, 1);
          }
#else
          for (int co = 0; co < cout; co++) {
            const float* wk = wrow + (size_t)co * k;
            for (int kk = 0; kk < k; kk++)
              if (yrows[kk]) yrows[kk][co] += xv * wk[kk];
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
