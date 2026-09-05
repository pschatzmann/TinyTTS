#pragma once
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "TinyTTS/Mat.h"

namespace tinytts {

// IEEE 754 binary16 -> binary32, standard bit manipulation (no compiler
// intrinsic assumed, so this works identically on host and Xtensa).
inline float halfToFloat(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +/-0
    } else {
      // subnormal half -> normalize into a normal float
      int shift = 0;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        shift++;
      }
      mant &= 0x3FFu;
      uint32_t new_exp = 127 - 15 - shift;
      bits = sign | (new_exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);  // inf/nan
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

/**
 * @brief Parses the named-tensor binary format written by
 * research/export_weights_and_vectors.py:
 *   repeated: [int32 name_len][name bytes][int32 ndims][int32 dims...]
 *             [int32 dtype][data...]
 *   terminated by name_len == 0
 *   dtype 0: data is float32 (4 bytes/element). dtype 1: data is float16
 *   (IEEE 754 binary16, 2 bytes/element). dtype 2: data is
 *   [float32 row_scale[shape[0]]][int8 data[count]] -- symmetric per-row
 *   (dim0) INT8, weights-only (used for `decoder`'s Conv1d/ConvTranspose1d
 *   weights; see docs/architecture.md for why and the accuracy numbers).
 *
 * Zero-copy: `begin()` only records, per tensor, a pointer into the buffer
 * it was given plus its shape/dtype -- it never copies or expands the
 * tensor data itself (`Entry::at()` decodes float16 on read instead). The
 * buffer passed to `begin()` must outlive this object (a flash-embedded
 * const array or a caller-owned PSRAM buffer both work, this class doesn't
 * care which -- same as CmuDict/DictionaryModel's own storage). This
 * matters in practice: weights.bin's tensors are mostly float16 to halve
 * flash usage, but expanding all of them to float32 and copying that into a
 * second, persistent buffer at parse time -- the previous design -- would
 * throw that saving away and then some (float32 is 2x float16, so the
 * "savings" become a straight loss versus not compressing at all): on
 * device that copy was measured at several MB of PSRAM, gone before
 * synthesis ever runs. Reading through a pointer with on-demand float16
 * decode avoids that entirely.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class WeightStore {
 public:
  struct Entry {
    std::vector<int> shape;
    const uint8_t* raw = nullptr;  // dtype-encoded bytes, count elements, into the caller's buffer
    const uint8_t* row_scale = nullptr;  // dtype 2 only: shape[0] per-row float32 scales
    size_t count = 0;
    int32_t dtype = 0;  // 0=float32, 1=float16, 2=int8 (per-row scale)

    float at(size_t i) const {
      if (dtype == 0) {
        float v;
        std::memcpy(&v, raw + i * 4, 4);
        return v;
      }
      if (dtype == 1) {
        uint16_t h;
        std::memcpy(&h, raw + i * 2, 2);
        return halfToFloat(h);
      }
      // dtype == 2
      return (float)(int8_t)raw[i] * rowScale(i / rowSize());
    }

    /// Decodes `n` contiguous elements starting at `start` into `out`.
    /// Branches on dtype once per call instead of once per element -- use
    /// this over repeated at() calls in a hot loop over a contiguous run
    /// (e.g. one kernel position's worth of weights). The run must not
    /// cross a row (dim0) boundary for dtype 2 -- true for every call site
    /// in this project (conv1d()/convTranspose1d() only ever decode one
    /// kernel position's worth of weights for a fixed output/input channel
    /// pair, which is always within one row).
    void decodeRun(size_t start, size_t n, float* out) const {
      if (dtype == 0) {
        std::memcpy(out, raw + start * 4, n * 4);
        return;
      }
      if (dtype == 1) {
        const uint8_t* p = raw + start * 2;
        for (size_t i = 0; i < n; i++) {
          uint16_t h;
          std::memcpy(&h, p + i * 2, 2);
          out[i] = halfToFloat(h);
        }
        return;
      }
      // dtype == 2
      float scale = rowScale(start / rowSize());
      for (size_t i = 0; i < n; i++) out[i] = (float)(int8_t)raw[start + i] * scale;
    }

   private:
    size_t rowSize() const { return count / (size_t)shape[0]; }

    // row_scale may not be 4-byte aligned (it sits right after a
    // variable-length name + shape header in the buffer) -- memcpy instead
    // of dereferencing, same reason at()/decodeRun() memcpy raw for
    // dtype 0/1 rather than casting to a float* directly.
    float rowScale(size_t row) const {
      float v;
      std::memcpy(&v, row_scale + row * 4, 4);
      return v;
    }
  };

  /// Parses all tensors out of the given buffer (recording pointers into
  /// it, not copying -- see class doc). Returns false on a
  /// malformed/truncated buffer.
  bool begin(const uint8_t* buf, size_t len) {
    entries_.clear();
    size_t pos = 0;
    while (pos + 4 <= len) {
      int32_t name_len = read_i32(buf, pos);
      if (name_len == 0) return true;
      if (name_len < 0 || pos + (size_t)name_len > len) return false;
      std::string name((const char*)(buf + pos), name_len);
      pos += name_len;

      if (pos + 4 > len) return false;
      int32_t ndims = read_i32(buf, pos);
      if (ndims < 0) return false;

      std::vector<int> shape(ndims);
      long total = 1;
      for (int i = 0; i < ndims; i++) {
        if (pos + 4 > len) return false;
        int32_t d = read_i32(buf, pos);
        shape[i] = d;
        total *= d;
      }

      if (pos + 4 > len) return false;
      int32_t dtype = read_i32(buf, pos);
      if (dtype != 0 && dtype != 1 && dtype != 2) return false;  // unknown dtype

      Entry e;
      e.shape = shape;
      e.count = (size_t)total;
      e.dtype = dtype;

      if (dtype == 2) {
        if (shape.empty()) return false;
        size_t scale_bytes = (size_t)shape[0] * 4;
        if (pos + scale_bytes + (size_t)total > len) return false;
        e.row_scale = buf + pos;
        pos += scale_bytes;
        e.raw = buf + pos;
        pos += (size_t)total;
      } else {
        size_t elem_size = dtype == 0 ? 4 : 2;
        if (pos + (size_t)total * elem_size > len) return false;
        e.raw = buf + pos;
        pos += (size_t)total * elem_size;
      }
      entries_[name] = e;
    }
    return true;
  }

  bool has(const std::string& name) const { return entries_.count(name) > 0; }

  const Entry* get(const std::string& name) const {
    auto it = entries_.find(name);
    return it == entries_.end() ? nullptr : &it->second;
  }

  /// Conv1d(kernel_size=1) weight [Cout, Cin, 1] -> Mat[Cout, Cin] (== Linear weight)
  Mat linearWeight(const std::string& name) const {
    const Entry* e = get(name);
    if (!e || e->shape.size() != 3 || e->shape[2] != 1) return Mat();
    return toMat(*e, e->shape[0], e->shape[1]);
  }

  /// Linear weight [Cout, Cin] (e.g. spk_emb_linear) -> Mat[Cout, Cin]
  Mat linear2dWeight(const std::string& name) const {
    const Entry* e = get(name);
    if (!e || e->shape.size() != 2) return Mat();
    return toMat(*e, e->shape[0], e->shape[1]);
  }

  /// Small (bias-sized) vectors only -- decodes into a plain (internal-heap)
  /// std::vector<float>, fine for the channel-sized biases this is used for.
  std::vector<float> vec(const std::string& name) const {
    const Entry* e = get(name);
    if (!e) return std::vector<float>();
    std::vector<float> v(e->count);
    for (size_t i = 0; i < e->count; i++) v[i] = e->at(i);
    return v;
  }

  /// Embedding table [N, C] -> Mat[N, C]
  Mat embedding(const std::string& name) const {
    const Entry* e = get(name);
    if (!e) return Mat();
    return toMat(*e, e->shape[0], e->shape[1]);
  }

  /// Conv1d(kernel_size=K>1) weight, shape [Cout, Cin, K] -- returned as the
  /// raw Entry since conv1d() consumes shape + Entry::at() directly.
  const Entry* convWeight(const std::string& name) const { return get(name); }

 private:
  static int32_t read_i32(const uint8_t* buf, size_t& pos) {
    int32_t v;
    std::memcpy(&v, buf + pos, 4);
    pos += 4;
    return v;
  }

  static Mat toMat(const Entry& e, int rows, int cols) {
    Mat m(rows, cols);
    for (size_t i = 0; i < e.count; i++) m.data()[i] = e.at(i);
    return m;
  }

  std::map<std::string, Entry> entries_;
};

}  // namespace tinytts
