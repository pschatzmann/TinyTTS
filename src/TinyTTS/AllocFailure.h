#pragma once
#include <cstdlib>
#include <new>

namespace tinytts {

/**
 * @brief Called by the STL allocators (PsramStlAllocator,
 * InternalStlAllocator, AlignedStlAllocator) when the underlying allocation
 * returns null: throws std::bad_alloc where C++ exceptions are enabled,
 * and aborts on -fno-exceptions builds (e.g. many Arduino cores) instead
 * of letting a container write through a null pointer.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
[[noreturn]] inline void allocFailed() {
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
  throw std::bad_alloc();
#else
  std::abort();
#endif
}

}  // namespace tinytts
