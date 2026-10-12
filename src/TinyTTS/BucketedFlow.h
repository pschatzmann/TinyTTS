#pragma once
#include <algorithm>
#include <functional>
#include <vector>

#include "TinyTTS/Mat.h"

namespace tinytts {

/**
 * @brief Runs the flow on fixed-length models ("buckets"), for inference
 * engines that only run static shapes, such as the Milk-V Duo's TPU (see
 * docs/milkv-duo.md).
 *
 * An utterance of T frames goes to the smallest bucket with length >= T, as
 * T frames followed by zero padding. The bucket function must treat the
 * padding as masked (the TPU models take a y_mask input for this). The flow
 * masks padded frames everywhere, so this matches the unpadded result
 * (>133dB SNR, research/milkv_duo/export_flow_tpu.py). Unlike the decoder,
 * the flow can't be split into windows: its attention spans the whole
 * utterance. reverse() returns false for an utterance longer than the
 * largest bucket, so the caller can fall back to the CPU flow.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class BucketedFlow {
 public:
  /// Runs one bucket: `z_p` is [length, inter_channels], of which rows
  /// [valid, length) are zero padding. Writes `z` with at least `valid` rows
  /// (only those are used). Returns false on failure.
  using BucketFn = std::function<bool(const Mat& z_p, int valid, const Mat& g, Mat& z)>;

  void addBucket(int length, BucketFn fn) {
    buckets_.push_back({length, std::move(fn)});
    std::sort(buckets_.begin(), buckets_.end(), [](const Bucket& a, const Bucket& b) { return a.length < b.length; });
  }

  void clear() { buckets_.clear(); }

  /// 0 without buckets.
  int maxLength() const { return buckets_.empty() ? 0 : buckets_.back().length; }

  /// z_p: [T, inter_channels], g: [1, gin_channels]. Writes z [T,
  /// inter_channels]. Returns false if no bucket fits or the bucket fails.
  bool reverse(const Mat& z_p, const Mat& g, Mat& z) const {
    const int T = z_p.rows(), C = z_p.cols();
    for (const Bucket& b : buckets_) {
      if (b.length < T) continue;
      Mat padded(b.length, C);
      std::copy(z_p.data().begin(), z_p.data().end(), padded.data().begin());
      Mat out;
      if (!b.fn(padded, T, g, out) || out.rows() < T || out.cols() != C) return false;
      z = Mat(T, C);
      std::copy(out.data().begin(), out.data().begin() + (size_t)T * C, z.data().begin());
      return true;
    }
    return false;
  }

 private:
  struct Bucket {
    int length;
    BucketFn fn;
  };
  std::vector<Bucket> buckets_;
};

}  // namespace tinytts
