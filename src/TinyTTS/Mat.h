#pragma once
#include <algorithm>
#include <vector>

#include "TinyTTS/PsramStlAllocator.h"

namespace tinytts {

/**
 * @brief Minimal row-major 2D float matrix: rows x cols. Used throughout as
 * the [T, C] (time-major, channel-last) activation layout. Backed by
 * `PsramVector<float>` (see PsramStlAllocator.h) rather than a plain
 * `std::vector<float>` -- a decoder-stage `Mat` can be hundreds of KB to a
 * few MB, far more than an ESP32's internal DRAM has room for, so this
 * needs to go to PSRAM explicitly rather than hoping the platform's default
 * allocator routes it there.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class Mat {
 public:
  Mat() = default;
  Mat(int rows, int cols, float fill = 0.0f)
      : rows_(rows), cols_(cols), data_((size_t)rows * cols, fill) {}

  int rows() const { return rows_; }
  int cols() const { return cols_; }

  float& at(int r, int c) { return data_[(size_t)r * cols_ + c]; }
  float at(int r, int c) const { return data_[(size_t)r * cols_ + c]; }

  float* row(int r) { return data_.data() + (size_t)r * cols_; }
  const float* row(int r) const { return data_.data() + (size_t)r * cols_; }

  PsramVector<float>& data() { return data_; }
  const PsramVector<float>& data() const { return data_; }

 private:
  int rows_ = 0, cols_ = 0;
  PsramVector<float> data_;
};

}  // namespace tinytts
