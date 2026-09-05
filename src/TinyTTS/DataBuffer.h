#pragma once
#include <cstdint>
#include <cstdlib>
#include <memory>

#ifdef ARDUINO
#include <FS.h>
#endif
#ifdef ESP32
#include <esp_heap_caps.h>
#endif

namespace tinytts {

/**
 * @brief Default allocator DataBuffer uses for its owned (File-loaded)
 * buffers: on ESP32, explicit PSRAM (heap_caps_malloc + MALLOC_CAP_SPIRAM);
 * everywhere else, plain malloc/free. This is the only ESP32-specific code
 * anywhere in DataBuffer -- pass a different type as DataBuffer's template
 * argument for a different allocator (e.g. plain internal RAM, or a
 * pooled/tracked allocator for testing); it just needs static
 * `allocate(size_t)`/`deallocate(void*)` methods with these signatures.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
struct PsramAllocator {
  static void* allocate(size_t n) {
#ifdef ESP32
    return heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
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

/**
 * @brief A data buffer TinyTTS may or may not own, so the same setter shape
 * works both for a caller-supplied flash const array (borrowed -- the
 * caller keeps ownership and must keep it alive) and for data TinyTTS reads
 * from a File itself (owned -- freed automatically, no manual cleanup, no
 * leak risk even across repeated setWeights()/begin() calls). Templated on
 * `Allocator` (default PsramAllocator, see its own doc above) so the owned
 * buffer can go somewhere other than PSRAM if you need that.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
template <typename Allocator = PsramAllocator>
class DataBuffer {
 public:
  /// Borrowed: `data` must outlive this object (and anything that reads
  /// from it, e.g. a TinyTTS that was begin()'d with it).
  void setBorrowed(const uint8_t* data, size_t len) {
#ifdef ARDUINO
    owned_.reset();
#endif
    data_ = data;
    len_ = len;
  }

#ifdef ARDUINO
  /// Owned: reads `file` fully into a new buffer (via `Allocator`) this
  /// object now owns (replacing anything previously set). Returns false on
  /// read failure or if allocation fails; the buffer is left unset (as if
  /// neither setter had been called) in that case.
  bool loadFromFile(File& file) {
    size_t n = file.size();
    uint8_t* buf = (uint8_t*)Allocator::allocate(n);
    if (!buf) {
      owned_.reset();
      data_ = nullptr;
      len_ = 0;
      return false;
    }
    size_t got = file.read(buf, n);
    if (got != n) {
      Allocator::deallocate(buf);
      owned_.reset();
      data_ = nullptr;
      len_ = 0;
      return false;
    }
    owned_.reset(buf);  // frees any previously-owned buffer automatically
    data_ = buf;
    len_ = n;
    return true;
  }
#endif

  const uint8_t* data() const { return data_; }
  size_t size() const { return len_; }
  bool valid() const { return data_ != nullptr; }

 private:
#ifdef ARDUINO
  struct Deleter {
    void operator()(uint8_t* p) const {
      if (p) Allocator::deallocate(p);
    }
  };

  std::unique_ptr<uint8_t[], Deleter> owned_;  // null for borrowed data
#endif
  const uint8_t* data_ = nullptr;
  size_t len_ = 0;
};

}  // namespace tinytts
