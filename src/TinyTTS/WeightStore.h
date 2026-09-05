#pragma once
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "TinyTTS/Mat.h"

namespace tinytts {

/**
 * @brief Parses the named-tensor binary format written by
 * research/export_weights_and_vectors.py:
 *   repeated: [int32 name_len][name bytes][int32 ndims][int32 dims...]
 *             [int32 dtype][data...]
 *   terminated by name_len == 0
 *   dtype 0: data is float32 (4 bytes/element). dtype 1: data is float16
 *   (IEEE 754 binary16, 2 bytes/element), expanded to float32 here at parse
 *   time -- Entry::data is always float32 regardless, so every consumer
 *   (PhonemeEncoder/Flow/Attention/...) is unaffected by which one a given
 *   tensor was stored as; only the source buffer's size differs.
 *
 * Reads from an in-memory buffer (not a file) so the same class works both
 * from a flash-embedded const array on the target device and from a buffer
 * read from LittleFS/SD at startup -- storage is the caller's decision, this
 * class only parses.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class WeightStore {
 public:
  struct Entry {
    std::vector<int> shape;
    std::vector<float> data;
  };

  /// Parses all tensors out of the given buffer. Returns false on a
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

      std::vector<float> data(total);
      if (dtype == 0) {
        if (pos + (size_t)total * 4 > len) return false;
        std::memcpy(data.data(), buf + pos, (size_t)total * 4);
        pos += (size_t)total * 4;
      } else if (dtype == 1) {
        if (pos + (size_t)total * 2 > len) return false;
        uint16_t half;
        for (long i = 0; i < total; i++) {
          std::memcpy(&half, buf + pos + (size_t)i * 2, 2);
          data[i] = halfToFloat(half);
        }
        pos += (size_t)total * 2;
      } else {
        return false;  // unknown dtype
      }

      entries_[name] = Entry{std::move(shape), std::move(data)};
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
    Mat m(e->shape[0], e->shape[1]);
    m.data() = e->data;
    return m;
  }

  /// Linear weight [Cout, Cin] (e.g. spk_emb_linear) -> Mat[Cout, Cin]
  Mat linear2dWeight(const std::string& name) const {
    const Entry* e = get(name);
    if (!e || e->shape.size() != 2) return Mat();
    Mat m(e->shape[0], e->shape[1]);
    m.data() = e->data;
    return m;
  }

  std::vector<float> vec(const std::string& name) const {
    const Entry* e = get(name);
    return e ? e->data : std::vector<float>();
  }

  /// Embedding table [N, C] -> Mat[N, C]
  Mat embedding(const std::string& name) const {
    const Entry* e = get(name);
    if (!e) return Mat();
    Mat m(e->shape[0], e->shape[1]);
    m.data() = e->data;
    return m;
  }

  /// Conv1d(kernel_size=K>1) weight, shape [Cout, Cin, K] -- returned as the
  /// raw Entry since conv1d() consumes shape + flat data directly.
  const Entry* convWeight(const std::string& name) const { return get(name); }

 private:
  static int32_t read_i32(const uint8_t* buf, size_t& pos) {
    int32_t v;
    std::memcpy(&v, buf + pos, 4);
    pos += 4;
    return v;
  }

  // IEEE 754 binary16 -> binary32, standard bit manipulation (no compiler
  // intrinsic assumed, so this works identically on host and Xtensa).
  static float halfToFloat(uint16_t h) {
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

  std::map<std::string, Entry> entries_;
};

}  // namespace tinytts
