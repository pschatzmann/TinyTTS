#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace tinytts {

/**
 * @brief Neural grapheme-to-phoneme fallback for words not in CmuDict: a
 * from-scratch C++ port of the upstream project's own g2p_predict.js
 * (itself a port of the Python `g2p_en` package's `G2p.predict()`) -- a
 * small (~833K-param) single-layer GRU encoder-decoder, greedy-decoded.
 * Verified bit-exact against both g2p_predict.js's algorithm and the
 * actual `g2p_en` Python package's output before being ported (see
 * research/g2p_reference_numpy.py, docs/research.md).
 *
 * Weights come from research/export_dictionary_model.py's
 * dictionary_model.bin, loaded via begin(const uint8_t*, size_t) from an
 * in-memory buffer (a flash-embedded const array, or one read from
 * LittleFS/SD at startup) the caller keeps alive for as long as this
 * object is used -- same borrowing convention as CmuDict.h, no copy.
 *
 * The four GRU weight matrices (enc_w_ih/enc_w_hh/dec_w_ih/dec_w_hh, ~94%
 * of the model's params) are INT8-quantized (symmetric, per output row --
 * see the export script for the exact scheme and the accuracy check run
 * before trusting it); everything else (the two small embedding tables,
 * biases, the output projection) stays float32. Not routed through TFLite
 * Micro, so none of its "hybrid model" restrictions apply here -- this is
 * plain hand-written dequantize-on-the-fly arithmetic.
 *
 * The input grapheme vocabulary ('<pad>','<unk>','</s>','a'..'z') is fixed
 * and small enough to compute the index arithmetically (graphemeIndex())
 * instead of shipping a lookup table. The output phoneme vocabulary is
 * exported as two parallel (symbol_id, tone) arrays matching the project's
 * shared symbol table (the same one CmuDict entries use), so predict()'s
 * output is directly interchangeable with a CmuDict lookup's -- TextG2P.h
 * doesn't need to care which one produced a word's pronunciation.
 *
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class DictionaryModel {
 public:
  struct Phoneme {
    uint8_t symbol_id;
    uint8_t tone;
  };

  /// Parses `buf` (research/export_dictionary_model.py's dictionary_model.bin
  /// format) without copying -- `buf` must outlive this object. Returns
  /// false on a malformed/truncated buffer.
  bool begin(const uint8_t* buf, size_t len) {
    size_t pos = 0;
    if (!readHeader(buf, len, pos)) return false;

    enc_emb_ = readFloats(buf, len, pos, (size_t)num_graphemes_ * hidden_dim_);
    dec_emb_ = readFloats(buf, len, pos, (size_t)num_dec_symbols_ * hidden_dim_);
    if (!enc_emb_ || !dec_emb_) return false;

    if (!readQuantized(buf, len, pos, &enc_w_ih_)) return false;
    if (!readQuantized(buf, len, pos, &enc_w_hh_)) return false;
    if (!readQuantized(buf, len, pos, &dec_w_ih_)) return false;
    if (!readQuantized(buf, len, pos, &dec_w_hh_)) return false;

    enc_b_ih_ = readFloats(buf, len, pos, 3 * (size_t)hidden_dim_);
    enc_b_hh_ = readFloats(buf, len, pos, 3 * (size_t)hidden_dim_);
    dec_b_ih_ = readFloats(buf, len, pos, 3 * (size_t)hidden_dim_);
    dec_b_hh_ = readFloats(buf, len, pos, 3 * (size_t)hidden_dim_);
    fc_w_ = readFloats(buf, len, pos, (size_t)num_phonemes_ * hidden_dim_);
    fc_b_ = readFloats(buf, len, pos, (size_t)num_phonemes_);
    if (!enc_b_ih_ || !enc_b_hh_ || !dec_b_ih_ || !dec_b_hh_ || !fc_w_ || !fc_b_) return false;

    phoneme_symbol_id_ = readBytes(buf, len, pos, (size_t)num_phonemes_);
    phoneme_tone_ = readBytes(buf, len, pos, (size_t)num_phonemes_);
    return phoneme_symbol_id_ != nullptr && phoneme_tone_ != nullptr;
  }

  /// `word` should be lowercase ASCII; anything else falls through to
  /// graphemeIndex()'s <unk> handling. Returns the greedy-decoded phoneme
  /// sequence, empty if decoding somehow ran the full 20-step cap without
  /// emitting </s> (not observed in practice, but not assumed impossible).
  std::vector<Phoneme> predict(const std::string& word) const {
    std::vector<float> h(hidden_dim_, 0.0f);
    for (char c : word) {
      gruStep(embRow(enc_emb_, graphemeIndex(c)), h, enc_w_ih_, enc_w_hh_, enc_b_ih_, enc_b_hh_);
    }
    gruStep(embRow(enc_emb_, 2 /* </s> */), h, enc_w_ih_, enc_w_hh_, enc_b_ih_, enc_b_hh_);

    std::vector<Phoneme> out;
    const float* dec = embRow(dec_emb_, 2 /* <s> */);
    for (int step = 0; step < 20; step++) {
      gruStep(dec, h, dec_w_ih_, dec_w_hh_, dec_b_ih_, dec_b_hh_);
      int idx = argmaxLogits(h);
      if (idx == 3 /* </s> */) break;
      out.push_back({phoneme_symbol_id_[idx], phoneme_tone_[idx]});
      dec = embRow(dec_emb_, idx);
    }
    return out;
  }

 private:
  // One [rows, hidden_dim_] weight matrix, symmetric per-row int8
  // (dequantized value = data[r*cols+c] * row_scale[r]).
  struct QuantizedWeight {
    const int8_t* data = nullptr;
    const float* row_scale = nullptr;
    int rows = 0;
  };

  bool readHeader(const uint8_t* buf, size_t len, size_t& pos) {
    if (pos + 16 > len) return false;
    int32_t hidden_dim, num_graphemes, num_dec_symbols, num_phonemes;
    std::memcpy(&hidden_dim, buf + pos, 4);
    std::memcpy(&num_graphemes, buf + pos + 4, 4);
    std::memcpy(&num_dec_symbols, buf + pos + 8, 4);
    std::memcpy(&num_phonemes, buf + pos + 12, 4);
    pos += 16;
    if (hidden_dim <= 0 || num_graphemes <= 0 || num_dec_symbols <= 0 || num_phonemes <= 0) return false;
    hidden_dim_ = hidden_dim;
    num_graphemes_ = num_graphemes;
    num_dec_symbols_ = num_dec_symbols;
    num_phonemes_ = num_phonemes;
    return true;
  }

  static const float* readFloats(const uint8_t* buf, size_t len, size_t& pos, size_t count) {
    size_t bytes = count * sizeof(float);
    if (pos + bytes > len) return nullptr;
    const float* p = reinterpret_cast<const float*>(buf + pos);
    pos += bytes;
    return p;
  }

  static const uint8_t* readBytes(const uint8_t* buf, size_t len, size_t& pos, size_t count) {
    if (pos + count > len) return nullptr;
    const uint8_t* p = buf + pos;
    pos += count;
    return p;
  }

  bool readQuantized(const uint8_t* buf, size_t len, size_t& pos, QuantizedWeight* w) const {
    int rows = 3 * hidden_dim_;
    size_t data_bytes = (size_t)rows * hidden_dim_;
    if (pos + data_bytes > len) return false;
    w->data = reinterpret_cast<const int8_t*>(buf + pos);
    pos += data_bytes;
    w->row_scale = readFloats(buf, len, pos, (size_t)rows);
    w->rows = rows;
    return w->row_scale != nullptr;
  }

  // Fixed vocab order from the upstream g2p_model.json: ['<pad>','<unk>','</s>','a'..'z'].
  static int graphemeIndex(char c) {
    if (c >= 'a' && c <= 'z') return 3 + (c - 'a');
    return 1;  // <unk>
  }

  const float* embRow(const float* emb, int idx) const { return emb + (size_t)idx * hidden_dim_; }

  // in-place GRU cell: h is both the input hidden state and the output.
  void gruStep(const float* x, std::vector<float>& h, const QuantizedWeight& w_ih, const QuantizedWeight& w_hh,
               const float* b_ih, const float* b_hh) const {
    int hidden = hidden_dim_;
    std::vector<float> rzn_ih(3 * hidden), rzn_hh(3 * hidden);
    // dequantized_weight[i][j] == row[j] * row_scale[i], so
    // dot(x, dequantized_row) == row_scale[i] * dot(x, row) -- scale the
    // int8 dot product once at the end rather than dequantizing every
    // element first.
    for (int i = 0; i < 3 * hidden; i++) {
      const int8_t* row = w_ih.data + (size_t)i * hidden;
      float dot = 0.0f;
      for (int j = 0; j < hidden; j++) dot += x[j] * (float)row[j];
      rzn_ih[i] = b_ih[i] + dot * w_ih.row_scale[i];
    }
    for (int i = 0; i < 3 * hidden; i++) {
      const int8_t* row = w_hh.data + (size_t)i * hidden;
      float dot = 0.0f;
      for (int j = 0; j < hidden; j++) dot += h[j] * (float)row[j];
      rzn_hh[i] = b_hh[i] + dot * w_hh.row_scale[i];
    }
    for (int i = 0; i < hidden; i++) {
      float r = 1.0f / (1.0f + std::exp(-(rzn_ih[i] + rzn_hh[i])));
      float z = 1.0f / (1.0f + std::exp(-(rzn_ih[hidden + i] + rzn_hh[hidden + i])));
      float n = std::tanh(rzn_ih[2 * hidden + i] + r * rzn_hh[2 * hidden + i]);
      h[i] = (1 - z) * n + z * h[i];
    }
  }

  int argmaxLogits(const std::vector<float>& h) const {
    int best = 0;
    float best_val = -1e30f;
    for (int j = 0; j < num_phonemes_; j++) {
      float logit = fc_b_[j];
      const float* row = fc_w_ + (size_t)j * hidden_dim_;
      for (int k = 0; k < hidden_dim_; k++) logit += h[k] * row[k];
      if (logit > best_val) {
        best_val = logit;
        best = j;
      }
    }
    return best;
  }

  const float* enc_emb_ = nullptr;
  const float* dec_emb_ = nullptr;
  QuantizedWeight enc_w_ih_, enc_w_hh_, dec_w_ih_, dec_w_hh_;
  const float* enc_b_ih_ = nullptr;
  const float* enc_b_hh_ = nullptr;
  const float* dec_b_ih_ = nullptr;
  const float* dec_b_hh_ = nullptr;
  const float* fc_w_ = nullptr;
  const float* fc_b_ = nullptr;
  const uint8_t* phoneme_symbol_id_ = nullptr;
  const uint8_t* phoneme_tone_ = nullptr;
  int hidden_dim_ = 0;
  int num_graphemes_ = 0;
  int num_dec_symbols_ = 0;
  int num_phonemes_ = 0;
};

}  // namespace tinytts
