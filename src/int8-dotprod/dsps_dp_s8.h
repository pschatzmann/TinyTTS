#pragma once
// INT8 dot product for Ops.h's conv1d() INT8-activation path.
//
// Originally vendored from espressif/esp-dsp (dsps_dp_s8_aes3 + ansi
// fallback). The aes3 (ESP32-S3 SIMD) kernel was found to be genuinely
// broken on real hardware -- a self-test (see docs/performance.md) caught
// it returning flatly wrong answers (e.g. expected 2360, got 263, then
// different wrong values on other inputs) -- consistent with a leftover
// "DEBUG: always ANSI; remove before release" comment found in that
// vendored file. Rather than hand-debug unfamiliar Xtensa PIE assembly (the
// exact class of mistake that produced the bug in the first place),
// esp_nn_dot_s8_aligned_esp32s3 (vendored into this directory from
// espressif/esp-nn, Apache-2.0, same as everything else here) is used
// instead: Espressif's own production quantized-NN kernel library, used
// for real int8 conv/FC inference, far more scrutinized than one orphaned
// esp-dsp function. dsps_dp_s8_ansi (portable scalar, still vendored here)
// remains as both the correctness fallback and the self-test's reference.
//
// Deliberately NOT declared via the shared dsps_dotprod.h / esp-nn's own
// common_functions.h: Arduino-ESP32 cores from ~3.3.x onward bundle their
// own precompiled esp-dsp component, and that core's own include
// directories are placed before any library's own src/ on the compiler
// command line -- so an angle-bracket include of either upstream header
// would resolve to whatever (possibly dp_s8-lacking) copy the core
// bundles, not this directory's. This header is included via a
// project-relative quoted path ("int8-dotprod/dsps_dp_s8.h", not
// <dsps_dotprod.h>) specifically because a quoted include can't be
// shadowed that way.
#include <cstdint>
#include "esp_err.h"    // real ESP-IDF/Arduino-core framework header, for esp_err_t
#include "sdkconfig.h"  // ditto, for CONFIG_IDF_TARGET_ESP32S3

extern "C" {
esp_err_t dsps_dp_s8_ansi(const int8_t* src1, const int8_t* src2, int32_t* dest, int len);
#if defined(CONFIG_IDF_TARGET_ESP32S3)
// esp_nn's own declared signature (common_functions.h): returns the dot
// product directly, no separate status code, requires len a multiple of
// 16 (>=16) and both pointers 16-byte aligned -- see
// esp_nn_dot_s8_esp32s3.S's own header comment. Undefined behavior if
// those preconditions aren't met, so dspsDotProdS8() below checks them
// before ever calling this.
int32_t esp_nn_dot_s8_aligned_esp32s3(const int8_t* a, const int8_t* b, int32_t len);
#endif
}

namespace tinytts {

/// Dispatches to the ESP32-S3 SIMD kernel when its real preconditions are
/// met (16-byte-aligned pointers, len a multiple of 16 and >=16), the
/// portable scalar kernel otherwise. Ops.h's conv1d() is written to always
/// satisfy the SIMD preconditions for its own buffers (kSimdAlign-aligned
/// allocation, cin padded up to a multiple of 16 with zero bytes -- see
/// AlignedStlAllocator.h and conv1d()'s own doc), so the scalar fallback
/// below is a correctness safety net for any other caller, not the
/// expected hot path.
inline void dspsDotProdS8(const int8_t* a, const int8_t* b, int32_t* dest, int len) {
#if defined(CONFIG_IDF_TARGET_ESP32S3)
  if (len >= 16 && (len % 16) == 0 && ((reinterpret_cast<uintptr_t>(a) & 15) == 0) &&
      ((reinterpret_cast<uintptr_t>(b) & 15) == 0)) {
    *dest = esp_nn_dot_s8_aligned_esp32s3(a, b, len);
    return;
  }
#endif
  dsps_dp_s8_ansi(a, b, dest, len);
}

}  // namespace tinytts
