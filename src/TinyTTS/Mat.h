#pragma once
#include <algorithm>
#include <vector>

namespace tinytts {

/**
 * @brief Minimal row-major 2D float matrix: rows x cols. Used throughout as
 * the [T, C] (time-major, channel-last) activation layout.
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

  std::vector<float>& data() { return data_; }
  const std::vector<float>& data() const { return data_; }

 private:
  int rows_ = 0, cols_ = 0;
  std::vector<float> data_;
};

}  // namespace tinytts
