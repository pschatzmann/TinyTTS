"""
NumPy reference implementation of g2p_model.json's GRU encoder-decoder,
matching npm-package/g2p_predict.js exactly (same GRU cell math, same
greedy-decode loop, same 20-step cap and </s>-index-3 stop condition).

Cross-checked against the actual Python `g2p_en` package's own output
(`pip install g2p_en`) before being trusted -- bit-exact for every word
tried, same "verify against ground truth before porting" approach as every
other model in this project (see docs/research.md).

Used for two things:
  1. test/main.cpp's testDictionaryModel() checks the C++ DictionaryModel.h
     port's output against reference sequences computed this way.
  2. compute_cmudict_exceptions.py runs this over all 123,463 cmudict.json
     words and keeps only the ones this model gets wrong -- see that
     script's docstring for why.
"""
import base64
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "tiny-tts", "npm-package", "g2p_model.json")


def _b64_to_f32(b64str, shape):
    raw = base64.b64decode(b64str)
    arr = np.frombuffer(raw, dtype="<f4")
    return arr.reshape(shape) if len(shape) > 1 else arr.copy()


class DictionaryModel:
    def __init__(self, model_path=MODEL_PATH):
        with open(model_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.tensors = {}
        for name in [
            "enc_emb", "enc_w_ih", "enc_w_hh", "enc_b_ih", "enc_b_hh",
            "dec_emb", "dec_w_ih", "dec_w_hh", "dec_b_ih", "dec_b_hh",
            "fc_w", "fc_b",
        ]:
            self.tensors[name] = _b64_to_f32(raw[name]["data"], raw[name]["shape"])

        self.graphemes = raw["graphemes"]
        self.phonemes = raw["phonemes"]
        self.g2idx = {g: i for i, g in enumerate(self.graphemes)}
        self.hidden_dim = self.tensors["enc_w_hh"].shape[1]

    def _gru_cell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        h_dim = self.hidden_dim
        rzn_ih = x @ w_ih.T + b_ih
        rzn_hh = h @ w_hh.T + b_hh
        rz = 1.0 / (1.0 + np.exp(-(rzn_ih[: 2 * h_dim] + rzn_hh[: 2 * h_dim])))
        r, z = rz[:h_dim], rz[h_dim:]
        n = np.tanh(rzn_ih[2 * h_dim :] + r * rzn_hh[2 * h_dim :])
        return (1 - z) * n + z * h

    def predict(self, word):
        """Returns a list of phoneme strings (with stress digits), e.g.
        ['HH', 'AH0', 'L', 'OW1'] for 'hello' -- same shape as a cmudict.json
        entry, so it can be diffed against one directly."""
        t = self.tensors
        chars = list(word) + ["</s>"]
        unk = self.g2idx["<unk>"]
        enc_inputs = [t["enc_emb"][self.g2idx.get(ch, unk)] for ch in chars]

        h = np.zeros(self.hidden_dim, dtype=np.float32)
        for x in enc_inputs:
            h = self._gru_cell(x, h, t["enc_w_ih"], t["enc_w_hh"], t["enc_b_ih"], t["enc_b_hh"])

        dec = t["dec_emb"][2]  # <s>
        preds = []
        for _ in range(20):
            h = self._gru_cell(dec, h, t["dec_w_ih"], t["dec_w_hh"], t["dec_b_ih"], t["dec_b_hh"])
            logits = h @ t["fc_w"].T + t["fc_b"]
            idx = int(np.argmax(logits))
            if idx == 3:  # </s>
                break
            preds.append(self.phonemes[idx])
            dec = t["dec_emb"][idx]
        return preds


if __name__ == "__main__":
    m = DictionaryModel()
    for w in ["hello", "world", "arduino", "esp32", "tinytts"]:
        print(w, "->", m.predict(w))
