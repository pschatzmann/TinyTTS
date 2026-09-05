#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "TinyTTS/Mat.h"

namespace tinytts {

/**
 * @brief Duration -> frame-level expansion, the glue between
 * DurationPredictor and Flow. Matches tiny_tts's
 * `_compute_alignment_path_np`, but implemented directly as a per-phoneme
 * repeat: for a single, unpadded utterance, that attention-matrix
 * construction is equivalent to "repeat phoneme i's row duration[i] times."
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
namespace alignment {

/// logw: [T_x] log-duration predictor output (one value per phoneme).
/// Returns durations[i] = ceil(exp(logw[i]) * length_scale), clamped to >= 0.
inline std::vector<int> durationsFromLogw(const std::vector<float>& logw, float length_scale = 1.0f) {
  std::vector<int> durations(logw.size());
  for (size_t i = 0; i < logw.size(); i++) {
    float w = std::exp(logw[i]) * length_scale;
    int d = (int)std::ceil(w);
    durations[i] = d > 0 ? d : 0;
  }
  return durations;
}

inline int totalDuration(const std::vector<int>& durations) {
  int total = 0;
  for (int d : durations) total += d;
  return total > 0 ? total : 1;  // matches the Python path's clamp_min(sum, 1)
}

/// Expands a [T_x, C] phoneme-level tensor to [T_y, C] frame-level by
/// repeating row i exactly durations[i] times. T_y must equal
/// totalDuration(durations).
inline Mat expandByDuration(const Mat& phoneme_level, const std::vector<int>& durations, int t_y) {
  Mat out(t_y, phoneme_level.cols());
  int t = 0;
  for (size_t i = 0; i < durations.size() && t < t_y; i++) {
    for (int d = 0; d < durations[i] && t < t_y; d++, t++) {
      std::copy(phoneme_level.row((int)i), phoneme_level.row((int)i) + phoneme_level.cols(), out.row(t));
    }
  }
  return out;
}

}  // namespace alignment
}  // namespace tinytts
