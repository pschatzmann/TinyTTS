#pragma once
// Linux only, needs the Milk-V Duo TPU SDK -- see CviModel.h.
#include <cstring>
#include <memory>
#include <vector>

#include "TinyTTS/BucketedFlow.h"
#include "TinyTTS/CviModel.h"
#include "TinyTTS/TinyTTSCore.h"

namespace tinytts {

/**
 * @brief Runs the flow on the TPU of the Milk-V Duo 256M / Duo S
 * (SG2002/SG2000) through the CVITEK runtime, as a BucketedFlow.
 *
 * Loads the cvimodels built by research/milkv_duo/build_flow_cvimodels.sh,
 * one per length: inputs z_p [1, inter_channels, L], y_mask [1, 1, L] and g
 * [1, gin_channels, 1], output z [1, inter_channels, L], all float32. L is
 * read from each model. Longer utterances than the largest model run on the
 * CPU flow instead.
 *
 *   CviTpuFlow flow;
 *   for (const char* f : {"flow_44100hz_l128_bf16.cvimodel", "flow_44100hz_l256_bf16.cvimodel"}) flow.add(f);
 *   tts.setFlow(flow.flowFn());
 *   ...  // tts.begin()
 *   if (!flow.checkModel(tts.core())) ...  // flow from another checkpoint
 *
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class CviTpuFlow {
 public:
  /// Loads one bucket. Returns false (and keeps the others) on failure.
  bool add(const char* cvimodel_path) {
    std::unique_ptr<Bucket> b(new Bucket());
    CviModel& m = b->model;
    if (!m.begin(cvimodel_path)) return false;
    b->z_p = m.input("z_p");
    b->y_mask = m.input("y_mask");
    b->g = m.input("g");
    b->z = m.output();
    int out_channels = 0, out_length = 0;
    if (!b->z_p || !b->y_mask || !b->g || !b->z || !m.shape(b->z_p, b->channels, b->length) ||
        !m.shape(b->z, out_channels, out_length))
      return false;
    if (out_channels != b->channels || out_length != b->length) return m.fail("z doesn't match z_p");
    if (CVI_NN_TensorCount(b->y_mask) != (size_t)b->length) return m.fail("y_mask doesn't match z_p");
    Bucket* raw = b.get();
    flow_.addBucket(b->length,
                    [raw](const Mat& z_p, int valid, const Mat& g, Mat& z) { return run(*raw, z_p, valid, g, z); });
    buckets_.push_back(std::move(b));
    return true;
  }

  void end() {
    flow_.clear();
    buckets_.clear();
  }

  /// Longest utterance (in frames) that runs on the TPU; 0 without models.
  int maxLength() const { return flow_.maxLength(); }

  /// After core.begin(): false (and an error message) if a model's file name
  /// gives another sample rate than the loaded weights, i.e. the flow models
  /// come from another checkpoint. The flow's shapes don't depend on the
  /// sample rate, so only the name tells.
  bool checkModel(const TinyTTSCore& core) const {
    for (const auto& b : buckets_)
      if (!b->model.checkSampleRate(core.sampleRate())) return false;
    return true;
  }

  /// For TinyTTS::setFlow() / TinyTTSCore::setFlow(). This object must
  /// outlive its use there.
  TinyTTSCore::FlowFn flowFn() {
    return [this](const Mat& z_p, const Mat& g, Mat& z) { return flow_.reverse(z_p, g, z); };
  }

 private:
  struct Bucket {
    CviModel model;
    CVI_TENSOR *z_p = nullptr, *y_mask = nullptr, *g = nullptr, *z = nullptr;
    int channels = 0;
    int length = 0;
  };

  static bool run(Bucket& b, const Mat& z_p, int valid, const Mat& g, Mat& z) {
    if (z_p.cols() != b.channels || (size_t)g.cols() != CVI_NN_TensorCount(b.g))
      return b.model.fail("z_p/g don't match the model (%d/%d channels)", z_p.cols(), g.cols());
    CviModel::setChannelFirst(b.z_p, z_p, valid, b.length);
    CviModel::setMask(b.y_mask, 0, valid);
    std::memcpy(CVI_NN_TensorPtr(b.g), g.row(0), sizeof(float) * g.cols());
    if (!b.model.forward()) return false;
    // [1, C, L] channel-first back to [valid, C].
    const float* p = (const float*)CVI_NN_TensorPtr(b.z);
    z = Mat(valid, b.channels);
    for (int t = 0; t < valid; t++)
      for (int c = 0; c < b.channels; c++) z.at(t, c) = p[(size_t)c * b.length + t];
    return true;
  }

  BucketedFlow flow_;
  std::vector<std::unique_ptr<Bucket>> buckets_;
};

}  // namespace tinytts
