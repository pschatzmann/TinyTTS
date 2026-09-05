#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>
#include <vector>

#ifdef ESP32
#include <esp_heap_caps.h>
#endif

namespace tinytts {

/**
 * @brief Explicitly requests internal (on-chip) RAM rather than PSRAM, on
 * ESP32 -- plain malloc/free everywhere else (a host build has no
 * PSRAM-vs-internal distinction to make). The counterpart to
 * `PsramAllocator` (see DataBuffer.h/PsramStlAllocator.h): that one exists
 * because some buffers are too big for internal RAM and must go to PSRAM;
 * this one exists because a few small, latency-critical buffers must do the
 * opposite -- go in fast internal RAM even though PSRAM is available,
 * because the platform's default allocator can silently route a large
 * enough allocation to PSRAM (see `esp_heap_caps.h`'s malloc-extmem-enable
 * behavior), which would defeat the entire point of caching something to
 * avoid PSRAM traffic in the first place.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
struct InternalRamAllocator {
  static void* allocate(size_t n) {
#ifdef ESP32
    return heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#else
    return malloc(n);
#endif
  }
  static void deallocate(void* p) {
#ifdef ESP32
    heap_caps_free(p);
#else
    free(p);
#endif
  }
};

/// `std::allocator`-compatible wrapper around `InternalRamAllocator`, same
/// shape as `PsramStlAllocator` -- see that file's doc for the pattern.
template <typename T>
struct InternalStlAllocator {
  using value_type = T;

  InternalStlAllocator() noexcept = default;
  template <typename U>
  InternalStlAllocator(const InternalStlAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n == 0) return nullptr;
    void* p = InternalRamAllocator::allocate(n * sizeof(T));
    if (!p) throw std::bad_alloc();
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) noexcept { InternalRamAllocator::deallocate(p); }

  template <typename U>
  bool operator==(const InternalStlAllocator<U>&) const noexcept {
    return true;
  }
  template <typename U>
  bool operator!=(const InternalStlAllocator<U>&) const noexcept {
    return false;
  }
};

/// A `std::vector<T>` guaranteed to live in internal RAM (not PSRAM) on
/// ESP32 -- for small, hot, reused-every-call scratch buffers where
/// round-trips to external PSRAM would defeat the purpose of caching them
/// at all (see Ops.h's conv1d()/convTranspose1d() weight cache).
template <typename T>
using InternalVector = std::vector<T, InternalStlAllocator<T>>;

}  // namespace tinytts
