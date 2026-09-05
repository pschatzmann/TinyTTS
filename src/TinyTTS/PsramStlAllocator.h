#pragma once
#include <cstddef>
#include <new>
#include <vector>

#include "TinyTTS/DataBuffer.h"

namespace tinytts {

/**
 * @brief A `std::allocator`-compatible wrapper around `PsramAllocator` (see
 * DataBuffer.h), so any standard container can be backed by it, e.g.
 * `std::vector<float, PsramStlAllocator<float>>` (or the `PsramVector<T>`
 * alias below). This is the only place the wrapping happens -- the actual
 * ESP32-specific allocation logic stays entirely inside `PsramAllocator`
 * itself; this template has none of its own.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
template <typename T>
struct PsramStlAllocator {
  using value_type = T;

  PsramStlAllocator() noexcept = default;
  template <typename U>
  PsramStlAllocator(const PsramStlAllocator<U>&) noexcept {}

  // Standard Allocator requirement: allocate() must either succeed or
  // throw -- never silently return null, or a caller like std::vector will
  // go on to write through that null pointer instead of getting a clean,
  // catchable failure.
  T* allocate(std::size_t n) {
    if (n == 0) return nullptr;
    void* p = PsramAllocator::allocate(n * sizeof(T));
    if (!p) throw std::bad_alloc();
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::size_t) noexcept { PsramAllocator::deallocate(p); }

  template <typename U>
  bool operator==(const PsramStlAllocator<U>&) const noexcept {
    return true;
  }
  template <typename U>
  bool operator!=(const PsramStlAllocator<U>&) const noexcept {
    return false;
  }
};

/// A `std::vector<T>` whose storage comes from `PsramAllocator` (PSRAM on
/// ESP32, plain malloc/free elsewhere) instead of the default heap -- used
/// for the buffers big enough to matter: `Mat`'s activations and
/// `WeightStore::Entry`'s parsed tensor data. Otherwise behaves exactly
/// like `std::vector<T>` (same iterators/operator[]/size()/etc.), so most
/// call sites don't need to change; only places that assign a plain
/// `std::vector<T>` into one (or vice versa) need `.assign(begin, end)`
/// instead of `operator=`, since containers with different allocator types
/// aren't implicitly convertible.
template <typename T>
using PsramVector = std::vector<T, PsramStlAllocator<T>>;

}  // namespace tinytts
