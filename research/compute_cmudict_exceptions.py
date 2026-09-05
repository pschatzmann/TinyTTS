"""
Runs the neural G2P model (g2p_reference_numpy.py) AND TextG2P.h's own
crude character-level fallback (replicated here exactly -- see
LETTER_IDS/char_fallback() below, matching TextG2P.h's letterSymbolId()) over
every word in npm-package/cmudict.json, and reports how many either one
already predicts correctly (exact (symbol_id, tone) sequence match against
the dictionary's own entry, first pronunciation only -- cmudict.json's
alternates aren't modeled here).

The point: TextG2P currently needs the FULL 123,463-word dictionary on
flash (3.3MB) as its only way to pronounce any word. Once the neural G2P
model (export_dictionary_model.py) is wired in as the real fallback, the
dictionary only needs to store the words NEITHER fallback gets right --
everything else, one of the two already reproduces at runtime, so shipping
it in the dictionary too is pure waste. This script measures how big that
win actually is before committing to the integration work. (The
character-level fallback essentially never coincides with a real word's
correct ARPAbet pronunciation -- it maps letters to unrelated "spell it
out" symbols with a flat tone, not real phonemes -- but it's cheap to check
properly rather than assume, and it does catch the rare degenerate case,
e.g. two-letter initialisms.)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tiny-tts"))

from export_cmudict import SYM_TO_ID, map_phoneme, parse_phone
from g2p_reference_numpy import DictionaryModel

HERE = os.path.dirname(os.path.abspath(__file__))
UNK_ID = SYM_TO_ID["UNK"]
TONE_OFFSET_EN = 7  # TextG2P::kToneOffsetEn

# Matches TextG2P.h's letterSymbolId() table exactly.
LETTER_IDS = {
    "a": 19, "b": 30, "c": 32, "d": 34, "e": 37, "f": 45, "g": 46,
    "h": 48, "i": 51, "j": 66, "k": 68, "l": 70, "m": 71, "n": 73,
    "o": 76, "p": 82, "q": 84, "r": 85, "s": 87, "t": 89, "u": 93,
    "v": 104, "w": 108, "x": 109, "y": 110, "z": 111,
}


def expected_pairs(raw_phones):
    """cmudict.json's own phones -> (symbol_id, tone) pairs in TextG2P.h's
    representation (tone already includes kToneOffsetEn), for comparison
    against both fallbacks' output."""
    pairs = []
    for raw in raw_phones:
        phone, tone = parse_phone(raw)
        sym = map_phoneme(phone)
        pairs.append((SYM_TO_ID[sym], tone + TONE_OFFSET_EN))
    return pairs


def model_pairs(model, word):
    phones = model.predict(word.lower())
    pairs = []
    for ph in phones:
        phone, tone = parse_phone(ph)
        sym = map_phoneme(phone)
        pairs.append((SYM_TO_ID[sym], tone + TONE_OFFSET_EN))
    return pairs


def char_fallback_pairs(word):
    """Matches TextG2P.h's resolveWord() character-level fallback exactly:
    each letter (apostrophes skipped) -> (letterSymbolId, kToneOffsetEn)."""
    pairs = []
    for c in word.lower():
        if c == "'":
            continue
        pairs.append((LETTER_IDS.get(c, UNK_ID), TONE_OFFSET_EN))
    return pairs


def main():
    cmu_path = os.path.join(HERE, "tiny-tts", "npm-package", "cmudict.json")
    with open(cmu_path, "r", encoding="utf-8") as f:
        cmu = json.load(f)

    model = DictionaryModel()

    words = sorted(cmu.keys())
    total = len(words)
    model_matches = 0
    char_matches = 0
    exceptions = {}

    start = time.time()
    for i, w in enumerate(words):
        expected = expected_pairs(cmu[w])
        model_pred = model_pairs(model, w)
        model_ok = model_pred == expected
        char_ok = char_fallback_pairs(w) == expected
        if model_ok:
            model_matches += 1
        if char_ok:
            char_matches += 1
        if not (model_ok or char_ok):
            exceptions[w] = cmu[w]
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"{i + 1}/{total} ({rate:.0f} words/s, eta {eta:.0f}s)", file=sys.stderr)

    print(f"\ntotal words: {total}")
    print(f"neural G2P model already correct: {model_matches} ({100 * model_matches / total:.1f}%)")
    print(f"character-level fallback already correct: {char_matches} ({100 * char_matches / total:.1f}%)")
    print(f"exceptions (neither fallback gets it right, must stay in shipped dictionary): "
          f"{len(exceptions)} ({100 * len(exceptions) / total:.1f}%)")

    out_path = os.path.join(HERE, "cmudict_exceptions.json")
    with open(out_path, "w") as f:
        json.dump(exceptions, f)
    print(f"wrote {out_path}: {os.path.getsize(out_path) / 1024 / 1024:.2f} MB (json, uncompressed)")


if __name__ == "__main__":
    main()
