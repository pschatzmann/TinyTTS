#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>
#include <vector>

#include "TinyTTS/AllocFailure.h"

#ifdef ESP32
#include <esp_heap_caps.h>
#endif

// std::aligned_alloc needs C++17 and a C library that provides it
// (libstdc++ reports the latter as _GLIBCXX_HAVE_ALIGNED_ALLOC).
#if __cplusplus >= 201703L && (!defined(__GLIBCXX__) || defined(_GLIBCXX_HAVE_ALIGNED_ALLOC))
#define TINYTTS_HAVE_ALIGNED_ALLOC 1
#else
#define TINYTTS_HAVE_ALIGNED_ALLOC 0
#endif

namespace tinytts {

/**
 * @brief 16-byte-aligned allocators, PSRAM- and internal-RAM-backed --
 * needed specifically for Ops.h's `conv1d()` INT8 SIMD path
 * (`esp_nn_dot_s8_aligned_esp32s3`, see `src/int8-dotprod/`), which
 * requires both operand pointers 16-byte aligned (undefined behavior
 * otherwise -- Xtensa PIE's `ee.vld.128.ip` is a 128-bit vector load).
 * Plain `PsramStlAllocator`/`InternalStlAllocator` (see those files) don't
 * guarantee any particular alignment beyond what the underlying allocator
 * happens to return, which ESP-IDF's `heap_caps_malloc` does not document
 * as 16-byte for either PSRAM or internal RAM -- hence a dedicated
 * allocator here rather than assuming.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
constexpr size_t kSimdAlign = 16;

struct AlignedPsramAllocator {
  static void* allocate(size_t n) {
#ifdef ESP32
    return heap_caps_aligned_alloc(kSimdAlign, n, MALLOC_CAP_SPIRAM);
#elif TINYTTS_HAVE_ALIGNED_ALLOC
    return std::aligned_alloc(kSimdAlign, n);
#else
    // No std::aligned_alloc (pre-C++17, or a libc such as a bare-metal
    // newlib that doesn't declare it): the 16-byte alignment only matters
    // for the ESP32-S3 SIMD path, so plain malloc is enough here.
    return std::malloc(n);
#endif
  }
  static void deallocate(void* p) {
#ifdef ESP32
    heap_caps_free(p);
#else
    std::free(p);
#endif
  }
};

struct AlignedInternalAllocator {
  static void* allocate(size_t n) {
#ifdef ESP32
    return heap_caps_aligned_alloc(kSimdAlign, n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#elif TINYTTS_HAVE_ALIGNED_ALLOC
    return std::aligned_alloc(kSimdAlign, n);
#else
    // No std::aligned_alloc (pre-C++17, or a libc such as a bare-metal
    // newlib that doesn't declare it): the 16-byte alignment only matters
    // for the ESP32-S3 SIMD path, so plain malloc is enough here.
    return std::malloc(n);
#endif
  }
  static void deallocate(void* p) {
#ifdef ESP32
    heap_caps_free(p);
#else
    std::free(p);
#endif
  }
};

/// `std::allocator`-compatible wrapper, same shape as `PsramStlAllocator`/
/// `InternalStlAllocator` -- see those files' docs for the pattern.
/// `std::aligned_alloc`/`heap_caps_aligned_alloc` both require the
/// requested size to be a multiple of the alignment -- true for every call
/// site here by construction (buffers are always sized as a multiple of
/// kSimdAlign elements for a 1-byte T), but not enforced by this class
/// itself; callers (Ops.h's conv1d()) are responsible for padding.
template <typename T, typename Base>
struct AlignedStlAllocator {
  using value_type = T;

  AlignedStlAllocator() noexcept = default;
  template <typename U>
  AlignedStlAllocator(const AlignedStlAllocator<U, Base>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n == 0) return nullptr;
    void* p = Base::allocate(n * sizeof(T));
    if (!p) allocFailed();
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) noexcept { Base::deallocate(p); }

  template <typename U>
  bool operator==(const AlignedStlAllocator<U, Base>&) const noexcept {
    return true;
  }
  template <typename U>
  bool operator!=(const AlignedStlAllocator<U, Base>&) const noexcept {
    return false;
  }
};

template <typename T>
using AlignedPsramVector = std::vector<T, AlignedStlAllocator<T, AlignedPsramAllocator>>;

template <typename T>
using AlignedInternalVector = std::vector<T, AlignedStlAllocator<T, AlignedInternalAllocator>>;

}  // namespace tinytts
