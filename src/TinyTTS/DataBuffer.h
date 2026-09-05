#pragma once
#include <cstdint>
#include <memory>

#ifdef ARDUINO
#include <FS.h>
#include <esp_heap_caps.h>
#endif

namespace tinytts {

/**
 * @brief A data buffer TinyTTS may or may not own, so the same setter shape
 * works both for a caller-supplied flash const array (borrowed -- the
 * caller keeps ownership and must keep it alive) and for data TinyTTS reads
 * from a File itself (owned -- freed automatically, no manual cleanup, no
 * leak risk even across repeated setWeights()/begin() calls).
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
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
  /// Owned: reads `file` fully into a new PSRAM buffer this object now
  /// owns (replacing anything previously set). Returns false on read
  /// failure or if PSRAM allocation fails; the buffer is left unset (as if
  /// neither setter had been called) in that case.
  bool loadFromFile(File& file) {
    size_t n = file.size();
    uint8_t* buf = (uint8_t*)heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
    if (!buf) {
      owned_.reset();
      data_ = nullptr;
      len_ = 0;
      return false;
    }
    size_t got = file.read(buf, n);
    if (got != n) {
      heap_caps_free(buf);
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
  struct PsramDeleter {
    void operator()(uint8_t* p) const {
      if (p) heap_caps_free(p);
    }
  };

  std::unique_ptr<uint8_t[], PsramDeleter> owned_;  // null for borrowed data
#endif
  const uint8_t* data_ = nullptr;
  size_t len_ = 0;
};

}  // namespace tinytts
