"""
M2 neural G2P fallback: converts g2p_model.json (the GRU encoder-decoder
g2p_predict.js/g2p_en.G2p.predict() ports, see g2p_reference_numpy.py and
docs/research.md) into a compact binary (dictionary_model.bin) for
DictionaryModel.h to load on-device.

The four GRU weight matrices (enc_w_ih, enc_w_hh, dec_w_ih, dec_w_hh) are
~94% of the model's ~833K params -- those are quantized to INT8 (symmetric,
per-row: each of a matrix's 768 rows gets its own scale = max(abs(row))/127,
zero_point implicitly 0), shrinking the model from ~3.19MB to ~0.95MB.
Everything else (the two small embedding tables, biases, the output
projection) stays float32 -- they're a small fraction of the total size, not
worth the added complexity/risk. Since DictionaryModel is hand-written C++
(not routed through TFLite Micro), none of TFLM's "hybrid model" restrictions
apply here -- full freedom over the quantization scheme.

Checked before trusting (see docs/research.md): a NumPy simulation of this
exact quantization scheme against a 5,000-word random sample of
cmudict.json scored 70.8% correct vs. the float32 reference's 70.7% --
statistically identical, with only 0.96% of individual predictions
differing at all. Not run through validate_int8.py (that script is for the
TFLite duration_predictor/decoder stages) -- see g2p_reference_numpy.py for
how to reproduce this check.

Binary format (dictionary_model.bin):
  int32 hidden_dim, int32 num_graphemes, int32 num_dec_symbols, int32 num_phonemes
  float32 enc_emb[num_graphemes * hidden_dim]
  float32 dec_emb[num_dec_symbols * hidden_dim]
  for each of enc_w_ih, enc_w_hh, dec_w_ih, dec_w_hh (shape [3*hidden_dim, hidden_dim]):
    int8    weight[3*hidden_dim * hidden_dim]   -- row-major, per-row scale below
    float32 row_scale[3*hidden_dim]
  float32 enc_b_ih[3*hidden_dim], float32 enc_b_hh[3*hidden_dim]
  float32 dec_b_ih[3*hidden_dim], float32 dec_b_hh[3*hidden_dim]
  float32 fc_w[num_phonemes * hidden_dim], float32 fc_b[num_phonemes]
  uint8 phoneme_symbol_id[num_phonemes], uint8 phoneme_tone[num_phonemes]

Grapheme (input character) vocabulary is NOT exported: it's fixed
('<pad>','<unk>','</s>','a'..'z', in that order) and small enough that
DictionaryModel.h computes the index arithmetically instead of needing a
table. Phoneme (output) vocabulary IS exported, but not as strings -- each
entry is mapped, using the exact same parse_phone()/map_phoneme()/SYM_TO_ID
logic export_cmudict.py uses, to a (symbol_id, tone) pair in the project's
shared symbol table, so DictionaryModel's output is directly usable
anywhere a CmuDict::Entry's phonemes are.
"""
import base64
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny-tts"))

from export_cmudict import SYM_TO_ID, UNK_ID, map_phoneme, parse_phone  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "tiny-tts", "npm-package", "g2p_model.json")
OUT_PATH = os.path.join(HERE, "dictionary_model.bin")

QUANTIZED_WEIGHTS = ["enc_w_ih", "enc_w_hh", "dec_w_ih", "dec_w_hh"]


def b64_to_f32(b64str, shape):
    raw = base64.b64decode(b64str)
    arr = np.frombuffer(raw, dtype="<f4")
    return arr.reshape(shape) if len(shape) > 1 else arr.copy()


def quantize_rows(w):
    """Symmetric per-row int8: scale = max(abs(row))/127, zero_point 0."""
    scales = np.abs(w).max(axis=1) / 127.0
    scales[scales == 0] = 1.0  # an all-zero row would divide by zero otherwise
    q = np.round(w / scales[:, None]).clip(-127, 127).astype(np.int8)
    return q, scales.astype(np.float32)


def main():
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    tensors = {}
    for name in [
        "enc_emb", "enc_w_ih", "enc_w_hh", "enc_b_ih", "enc_b_hh",
        "dec_emb", "dec_w_ih", "dec_w_hh", "dec_b_ih", "dec_b_hh",
        "fc_w", "fc_b",
    ]:
        tensors[name] = b64_to_f32(raw[name]["data"], raw[name]["shape"])

    hidden_dim = tensors["enc_w_hh"].shape[1]
    num_graphemes = tensors["enc_emb"].shape[0]
    num_dec_symbols = tensors["dec_emb"].shape[0]
    num_phonemes = tensors["fc_w"].shape[0]

    phonemes = raw["phonemes"]
    symbol_ids = np.zeros(len(phonemes), dtype=np.uint8)
    tones = np.zeros(len(phonemes), dtype=np.uint8)
    unmapped = 0
    for i, ph in enumerate(phonemes):
        if ph in ("<pad>", "<unk>", "<s>", "</s>"):
            symbol_ids[i] = UNK_ID
            continue
        phone, tone = parse_phone(ph)
        sym = map_phoneme(phone)
        sym_id = SYM_TO_ID[sym]
        if sym_id == UNK_ID:
            unmapped += 1
        symbol_ids[i] = sym_id
        tones[i] = tone
    if unmapped:
        print(f"WARNING: {unmapped} phoneme(s) in g2p_model.json's vocab did not map to a known symbol")

    with open(OUT_PATH, "wb") as f:
        f.write(struct.pack("<iiii", hidden_dim, num_graphemes, num_dec_symbols, num_phonemes))
        f.write(np.ascontiguousarray(tensors["enc_emb"], dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(tensors["dec_emb"], dtype=np.float32).tobytes())
        for name in QUANTIZED_WEIGHTS:
            q, scale = quantize_rows(tensors[name])
            f.write(np.ascontiguousarray(q).tobytes())
            f.write(np.ascontiguousarray(scale).tobytes())
        for name in ["enc_b_ih", "enc_b_hh", "dec_b_ih", "dec_b_hh"]:
            f.write(np.ascontiguousarray(tensors[name], dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(tensors["fc_w"], dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(tensors["fc_b"], dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(symbol_ids).tobytes())
        f.write(np.ascontiguousarray(tones).tobytes())

    total = os.path.getsize(OUT_PATH)
    print(f"wrote {OUT_PATH}: {total} bytes ({total / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
