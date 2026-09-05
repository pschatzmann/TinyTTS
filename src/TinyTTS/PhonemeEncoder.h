#pragma once
#include <cmath>
#include <vector>

#include "TinyTTS/Mat.h"
#include "TinyTTS/Ops.h"
#include "TinyTTS/TransformerBlock.h"
#include "TinyTTS/WeightStore.h"

namespace tinytts {

/// x: post-encoder hidden state [T, hidden]. m_p/logs_p: [T, inter_channels]
/// (the normalizing-flow prior distribution parameters).
struct PhonemeEncoderOutput {
  Mat x;
  Mat m_p;
  Mat logs_p;
};

/**
 * @brief Phoneme/tone/language embeddings + windowed relative-position
 * transformer + projection to the flow's prior distribution parameters.
 * Ported from tiny_tts.models.synthesizer.PhonemeEncoder.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class PhonemeEncoder {
 public:
  /// `ws` must outlive this object (weights are referenced, not copied, for
  /// the kernel_size>1 conv layers).
  void begin(const WeightStore& ws, int n_heads, int window_size) {
    emb_ = ws.embedding("enc_p.emb.weight");
    tone_emb_ = ws.embedding("enc_p.tone_emb.weight");
    language_emb_ = ws.embedding("enc_p.language_emb.weight");
    bert_proj_b_ = ws.vec("enc_p.bert_proj.bias");
    ja_bert_proj_b_ = ws.vec("enc_p.ja_bert_proj.bias");
    encoder_.begin(ws, "enc_p.encoder", /*n_layers=*/3, n_heads, window_size);
    proj_w_ = ws.convWeight("enc_p.proj.weight");
    proj_b_ = ws.vec("enc_p.proj.bias");
    hidden_channels_ = emb_.cols();
    inter_channels_ = (int)proj_b_.size() / 2;
  }

  /// phone_ids/tone_ids/language_ids: length-T int arrays. g: [1,
  /// gin_channels] speaker embedding.
  ///
  /// The reference model also takes bert/ja_bert (multi-lingual BERT
  /// conditioning) as inputs, but that feature is disabled by default
  /// (always a zero tensor in every real call path this project has) --
  /// with a zero input, Conv1d(kernel_size=1)'s output is just its bias
  /// term (see Ops.h's linear()), so bert_proj/ja_bert_proj's weight
  /// matrices reduce to a constant per-channel add and were dropped from
  /// weights.bin entirely (only their small bias vectors are still
  /// loaded) -- no bert/ja_bert parameters needed here as a result.
  PhonemeEncoderOutput forward(const std::vector<int>& phone_ids, const std::vector<int>& tone_ids,
                                const std::vector<int>& language_ids, const Mat& g) const {
    int t_len = (int)phone_ids.size();
    int h = hidden_channels_;

    Mat x(t_len, h);
    float scale = std::sqrt((float)h);
    for (int t = 0; t < t_len; t++) {
      for (int c = 0; c < h; c++) {
        float v = emb_.at(phone_ids[t], c) + tone_emb_.at(tone_ids[t], c) + language_emb_.at(language_ids[t], c) +
                   bert_proj_b_[c] + ja_bert_proj_b_[c];
        x.at(t, c) = v * scale;
      }
    }

    x = encoder_.forward(x, &g);

    Mat stats = ops::conv1d(x, *proj_w_, proj_b_);  // [T, 2*inter_channels]
    PhonemeEncoderOutput out;
    out.x = x;
    out.m_p = Mat(t_len, inter_channels_);
    out.logs_p = Mat(t_len, inter_channels_);
    for (int t = 0; t < t_len; t++) {
      for (int c = 0; c < inter_channels_; c++) {
        out.m_p.at(t, c) = stats.at(t, c);
        out.logs_p.at(t, c) = stats.at(t, c + inter_channels_);
      }
    }
    return out;
  }

  int hiddenChannels() const { return hidden_channels_; }
  int interChannels() const { return inter_channels_; }

 private:
  Mat emb_, tone_emb_, language_emb_;
  std::vector<float> bert_proj_b_;
  std::vector<float> ja_bert_proj_b_;
  TransformerBlock encoder_;
  const WeightStore::Entry* proj_w_ = nullptr;
  std::vector<float> proj_b_;
  int hidden_channels_ = 32;
  int inter_channels_ = 32;
};

}  // namespace tinytts
