#pragma once
#include <cctype>
#include <string>
#include <vector>

#include "TinyTTS/CmuDict.h"
#include "TinyTTS/DictionaryModel.h"

namespace tinytts {

/// phone_ids/tone_ids/language_ids are aligned, blank-interleaved sequences
/// ready for PhonemeEncoder::forward().
struct G2POutput {
  std::vector<int> phone_ids;
  std::vector<int> tone_ids;
  std::vector<int> language_ids;
};

/**
 * @brief English text -> (phone_ids, tone_ids, language_ids). Follows the
 * TinyTTS project's own Node.js reference implementation
 * (npm-package/index.js's graphemeToPhoneme) -- a from-scratch,
 * dependency-free reimplementation, deliberately simpler than the Python
 * package's NLTK/BERT-tokenizer path and the right one to port for an
 * embedded target.
 *
 * Out-of-CMU-dictionary words fall back to DictionaryModel (a from-scratch
 * C++ port of the reference's own neural GRU G2P, g2p_predict.js/the
 * upstream g2p_model.json -- see DictionaryModel.h), matching the
 * reference's own fallback order. If no DictionaryModel is wired in (or it
 * somehow produces nothing), falls back further to crude character-level
 * phonemes, matching the reference's own "ultimate fallback" path.
 * @author Phil Schatzmann
 * @copyright Apache-2.0
 */
class TextG2P {
 public:
  static constexpr int kSymPad = 0;    // '_'
  static constexpr int kSymUnk = 218;  // see research/export_cmudict.py's symbol table dump
  static constexpr int kLangIdEn = 2;
  static constexpr int kToneOffsetEn = 7;

  /// `dict` must outlive this object (and the buffer passed to
  /// dict.begin() must outlive `dict`). `dictionary_model` is optional (may
  /// be nullptr to skip straight to the character-level fallback for
  /// out-of-dictionary words) and, if given, must also outlive this object.
  void begin(const CmuDict& dict, const DictionaryModel* dictionary_model = nullptr) {
    dict_ = &dict;
    dictionary_model_ = dictionary_model;
  }

  G2POutput process(const std::string& text) const {
    std::string lower = toLower(text);
    std::vector<int> phone_ids, tone_ids, language_ids;

    size_t i = 0, n = lower.size();
    while (i < n) {
      while (i < n && std::isspace((unsigned char)lower[i])) i++;
      if (i >= n) break;
      size_t word_start = i;
      while (i < n && !std::isspace((unsigned char)lower[i])) i++;
      std::string word = lower.substr(word_start, i - word_start);

      size_t lead_end = 0;
      while (lead_end < word.size() && !isWordChar(word[lead_end])) lead_end++;
      size_t trail_start = word.size();
      while (trail_start > lead_end && !isWordChar(word[trail_start - 1]) && word[trail_start - 1] != '\'')
        trail_start--;

      std::string lead = word.substr(0, lead_end);
      std::string trail = word.substr(trail_start);
      std::string core = word.substr(lead_end, trail_start - lead_end);

      for (char c : lead) {
        phone_ids.push_back(punctuationSymbolId(c));
        tone_ids.push_back(kToneOffsetEn);
      }

      if (!core.empty()) {
        size_t apos = core.find('\'');
        if (apos != std::string::npos) {
          size_t start = 0;
          while (start <= core.size()) {
            size_t next = core.find('\'', start);
            std::string part = core.substr(start, next == std::string::npos ? std::string::npos : next - start);
            if (start > 0) {
              phone_ids.push_back(punctuationSymbolId('\''));
              tone_ids.push_back(kToneOffsetEn);
            }
            if (!part.empty()) resolveWord(part, &phone_ids, &tone_ids);
            if (next == std::string::npos) break;
            start = next + 1;
          }
        } else {
          resolveWord(core, &phone_ids, &tone_ids);
        }
      }

      for (char c : trail) {
        phone_ids.push_back(punctuationSymbolId(c));
        tone_ids.push_back(kToneOffsetEn);
      }
    }

    // pad start/end with '_' (matches pad_start_end=True in the reference)
    phone_ids.insert(phone_ids.begin(), kSymPad);
    phone_ids.push_back(kSymPad);
    tone_ids.insert(tone_ids.begin(), 0);
    tone_ids.push_back(0);
    language_ids.assign(phone_ids.size(), kLangIdEn);

    G2POutput out;
    out.phone_ids = std::move(phone_ids);
    out.tone_ids = std::move(tone_ids);
    out.language_ids = std::move(language_ids);
    return out;
  }

  /// Matches commons.insert_blanks: interleaves a pad (0) between every
  /// symbol, including before the first and after the last.
  /// result.size() == 2*ids.size()+1.
  static std::vector<int> insertBlanks(const std::vector<int>& ids) {
    std::vector<int> out(ids.size() * 2 + 1, 0);
    for (size_t i = 0; i < ids.size(); i++) out[1 + i * 2] = ids[i];
    return out;
  }

 private:
  static bool isWordChar(char c) { return std::isalnum((unsigned char)c); }

  static std::string toUpper(const std::string& s) {
    std::string out = s;
    for (auto& c : out) c = std::toupper((unsigned char)c);
    return out;
  }

  static std::string toLower(const std::string& s) {
    std::string out = s;
    for (auto& c : out) c = std::tolower((unsigned char)c);
    return out;
  }

  // Minimal symbol lookup for ASCII punctuation actually reachable from
  // lead/trail splitting (word-internal alpha content never reaches this path).
  static int punctuationSymbolId(char ch) {
    switch (ch) {
      case '!': return 208;
      case '?': return 209;
      case ',': return 211;
      case '.': return 212;
      case '\'': return 213;
      case '-': return 214;
      default: return kSymUnk;
    }
  }

  // Character-level fallback -- only reached if there's no DictionaryModel
  // wired in, or it somehow produced nothing (not observed in practice; see
  // DictionaryModel.h's predict() doc). Each letter becomes its own phone symbol,
  // tone 0; only correct for the small set of Latin letters that happen to
  // coincide with the shared multi-lingual symbol table's raw-letter
  // entries (used for Chinese pinyin) -- a deliberately crude last resort.
  static int letterSymbolId(char lower_ch) {
    static const std::pair<char, int> kLetterIds[] = {
        {'a', 19}, {'b', 30}, {'c', 32}, {'d', 34}, {'e', 37}, {'f', 45}, {'g', 46},
        {'h', 48}, {'i', 51}, {'j', 66}, {'k', 68}, {'l', 70}, {'m', 71}, {'n', 73},
        {'o', 76}, {'p', 82}, {'q', 84}, {'r', 85}, {'s', 87}, {'t', 89}, {'u', 93},
        {'v', 104}, {'w', 108}, {'x', 109}, {'y', 110}, {'z', 111},
    };
    for (auto& p : kLetterIds)
      if (p.first == lower_ch) return p.second;
    return kSymUnk;
  }

  void resolveWord(const std::string& core, std::vector<int>* phone_ids, std::vector<int>* tone_ids) const {
    CmuDict::Entry entry;
    if (dict_ != nullptr && dict_->lookup(toUpper(core), &entry)) {
      // entry tones are the raw (digit+1, or 0 for unstressed) values
      // export_cmudict.py stored -- matches the reference's phonemes_to_ids,
      // which adds language_tone_start_map[lang] (7 for EN) on top of
      // exactly this raw value, nothing else.
      for (int i = 0; i < entry.count; i++) {
        phone_ids->push_back(entry.symbol_tone_pairs[i * 2]);
        tone_ids->push_back(entry.symbol_tone_pairs[i * 2 + 1] + kToneOffsetEn);
      }
      return;
    }
    if (dictionary_model_ != nullptr) {
      std::vector<DictionaryModel::Phoneme> predicted = dictionary_model_->predict(core);
      if (!predicted.empty()) {
        for (const auto& p : predicted) {
          phone_ids->push_back(p.symbol_id);
          tone_ids->push_back(p.tone + kToneOffsetEn);
        }
        return;
      }
    }
    for (char c : core) {
      if (c == '\'') continue;
      phone_ids->push_back(letterSymbolId(std::tolower((unsigned char)c)));
      tone_ids->push_back(kToneOffsetEn);
    }
  }

  const CmuDict* dict_ = nullptr;
  const DictionaryModel* dictionary_model_ = nullptr;
};

}  // namespace tinytts
