#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace tinytts {

/**
 * @brief Binary-search lookup for the compact CMU pronunciation dictionary
 * produced by research/export_cmudict.py (123,463 entries, symbol ids and
 * tones pre-resolved against tiny_tts.text.symbols at export time -- no
 * string/symbol-table logic needed at runtime). Reads from an in-memory
 * buffer, same storage-agnostic pattern as WeightStore.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class CmuDict {
 public:
  struct Entry {
    const uint8_t* symbol_tone_pairs;  // interleaved (symbol_id, tone), `count` pairs
    int count;
  };

  bool begin(const uint8_t* buf, size_t len) {
    if (len < 4) return false;
    size_t pos = 0;
    int32_t n = read_i32(buf, pos);
    if (n < 0) return false;
    word_count_ = n;

    if (pos + (size_t)(n + 1) * 4 > len) return false;
    word_offsets_.assign((const int32_t*)(buf + pos), (const int32_t*)(buf + pos) + n + 1);
    pos += (size_t)(n + 1) * 4;

    if (pos + (size_t)(n + 1) * 4 > len) return false;
    phoneme_offsets_.assign((const int32_t*)(buf + pos), (const int32_t*)(buf + pos) + n + 1);
    pos += (size_t)(n + 1) * 4;

    size_t words_blob_size = word_offsets_[n];
    if (pos + words_blob_size > len) return false;
    words_blob_ = buf + pos;
    pos += words_blob_size;

    size_t phoneme_data_size = phoneme_offsets_[n];
    if (pos + phoneme_data_size > len) return false;
    phoneme_data_ = buf + pos;

    return true;
  }

  int wordCount() const { return word_count_; }

  /// word must already be uppercase ASCII. Returns false if not found.
  bool lookup(const std::string& word, Entry* out) const {
    int lo = 0, hi = word_count_ - 1;
    while (lo <= hi) {
      int mid = lo + (hi - lo) / 2;
      int cmp = compareWord(mid, word);
      if (cmp == 0) {
        int start = phoneme_offsets_[mid];
        int end = phoneme_offsets_[mid + 1];
        out->symbol_tone_pairs = phoneme_data_ + start;
        out->count = (end - start) / 2;
        return true;
      } else if (cmp < 0) {
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return false;
  }

 private:
  static int32_t read_i32(const uint8_t* buf, size_t& pos) {
    int32_t v;
    std::memcpy(&v, buf + pos, 4);
    pos += 4;
    return v;
  }

  int compareWord(int idx, const std::string& word) const {
    int start = word_offsets_[idx];
    int len = word_offsets_[idx + 1] - start;
    const char* w = reinterpret_cast<const char*>(words_blob_ + start);
    size_t min_len = std::min((size_t)len, word.size());
    int cmp = std::memcmp(w, word.data(), min_len);
    if (cmp != 0) return cmp;
    return (int)len - (int)word.size();
  }

  int word_count_ = 0;
  std::vector<int32_t> word_offsets_;
  std::vector<int32_t> phoneme_offsets_;
  const uint8_t* words_blob_ = nullptr;
  const uint8_t* phoneme_data_ = nullptr;
};

}  // namespace tinytts
