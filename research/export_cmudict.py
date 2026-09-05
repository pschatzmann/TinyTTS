"""
M2 G2P step: converts a word -> ARPAbet-phones JSON dictionary into a
compact binary dictionary for the C++ port, with symbol IDs and tones
already resolved against tiny_tts.text.symbols -- the C++ side does a pure
binary-search lookup, no string/symbol-table logic needed at runtime.

Produces TWO outputs, both real, shipped options (see export_headers.py and
the README's "Model data" section for the tradeoff):
  - cmudict.bin: the FULL dictionary (npm-package/cmudict.json, 123,463
    words) -- works standalone, no DictionaryModel required.
  - cmudict_slim.bin: only the ~30% of words neither the neural G2P
    fallback (DictionaryModel.h, see export_dictionary_model.py) nor
    TextG2P.h's character-level fallback gets right
    (compute_cmudict_exceptions.py's cmudict_exceptions.json) -- much
    smaller (~1.1MB vs ~3.3MB), but REQUIRES DictionaryModel to be wired in
    too (via setDictionaryModel()) for words outside this slimmed set to
    still work; used alone (without the model) it would silently
    mispronounce ~70% of real English words. Run
    compute_cmudict_exceptions.py first to (re)generate
    cmudict_exceptions.json if you've changed the model or the source
    dictionary.
  Total data size, full-dictionary-alone vs. slim-dictionary-plus-model:
  3.3MB vs. ~1.1MB + ~3.2MB = ~4.3MB -- the slim option is actually LARGER
  in this project's numbers (see docs/research.md), so it's offered for
  flexibility (e.g. if you already need DictionaryModel for its coverage
  and are loading the dictionary from external storage rather than flash
  anyway), not as a universally-better default.

Phoneme resolution matches npm-package/index.js's _parsePhone + _mapPhoneme
exactly:
  - strip a trailing stress digit (0/1/2) -> tone = digit + 1, else tone = 0
  - lowercase the phone
  - apply the punctuation/symbol replacement map (critically: 'v' -> 'V',
    since English ARPAbet V would otherwise collide with the Chinese pinyin
    'v' symbol already in the shared multi-lingual symbol table)
  - map to symbol id via tiny_tts.text.symbols; UNK if not found

Binary format (both cmudict.bin and cmudict_slim.bin):
  int32 word_count
  int32 word_offsets[word_count]     -- offset into words_blob for word i
  int32 word_offsets[word_count]     -- (sentinel) end offset == len(words_blob)
  int32 phoneme_offsets[word_count+1]  -- byte offset into phoneme_data
  words_blob: concatenated words, NOT null-terminated (lengths derived from offsets)
  phoneme_data: repeated (uint8 symbol_id, uint8 tone) pairs

Words are sorted ascending (uppercase ASCII) so the C++ side can binary search.
"""
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tiny-tts"))

from tiny_tts.text.symbols import symbols

HERE = os.path.dirname(os.path.abspath(__file__))

SYM_TO_ID = {s: i for i, s in enumerate(symbols)}
UNK_ID = SYM_TO_ID["UNK"]

# matches npm-package/index.js's _mapPhoneme replacement table
REPLACEMENTS = {
    "：": ",", "；": ",", "，": ",", "。": ".",
    "！": "!", "？": "?", "\n": ".", "\xb7": ",",
    "、": ",", "...": "…", "v": "V",
}


def parse_phone(phn: str):
    if phn and phn[-1].isdigit():
        return phn[:-1].lower(), int(phn[-1]) + 1
    return phn.lower(), 0


def map_phoneme(ph: str) -> str:
    if ph in REPLACEMENTS:
        return REPLACEMENTS[ph]
    if ph in SYM_TO_ID:
        return ph
    return "UNK"


def export_dict(cmu: dict, out_path: str):
    words = sorted(cmu.keys())

    word_offsets = []
    words_blob = bytearray()
    phoneme_offsets = [0]
    phoneme_data = bytearray()

    unk_count = 0
    for w in words:
        word_offsets.append(len(words_blob))
        words_blob += w.encode("ascii", errors="replace")

        for raw_phone in cmu[w]:
            phone, tone = parse_phone(raw_phone)
            sym = map_phoneme(phone)
            sym_id = SYM_TO_ID[sym]
            if sym_id == UNK_ID:
                unk_count += 1
            phoneme_data.append(sym_id)
            phoneme_data.append(tone)
        phoneme_offsets.append(len(phoneme_data))

    word_offsets.append(len(words_blob))  # sentinel

    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", len(words)))
        for off in word_offsets:
            f.write(struct.pack("<i", off))
        for off in phoneme_offsets:
            f.write(struct.pack("<i", off))
        f.write(bytes(words_blob))
        f.write(bytes(phoneme_data))

    total = os.path.getsize(out_path)
    print(f"{len(words)} words, words_blob={len(words_blob)}B, phoneme_data={len(phoneme_data)}B, "
          f"unmapped={unk_count}")
    print(f"wrote {out_path}: {total} bytes ({total / 1024 / 1024:.2f} MB)")


def main():
    full_path = os.path.join(HERE, "tiny-tts", "npm-package", "cmudict.json")
    with open(full_path, "r", encoding="utf-8") as f:
        full_cmu = json.load(f)
    export_dict(full_cmu, os.path.join(HERE, "cmudict.bin"))

    slim_path = os.path.join(HERE, "cmudict_exceptions.json")
    if os.path.exists(slim_path):
        with open(slim_path, "r", encoding="utf-8") as f:
            slim_cmu = json.load(f)
        export_dict(slim_cmu, os.path.join(HERE, "cmudict_slim.bin"))
    else:
        print(f"skipping cmudict_slim.bin -- {slim_path} not found "
              "(run compute_cmudict_exceptions.py first)")


if __name__ == "__main__":
    main()
