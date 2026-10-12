"""
Runs build_cvimodel.sh's cvimodels in TPU-MLIR's simulator over whole
utterances -- chunked the same way as ChunkedVocoder on the device -- and
compares them with the float32 PyTorch decoder. Each model is compared with
the utterances export_decoder_tpu.py wrote for it -- out/<name>_<precision>.cvimodel
with out/<name>_utterances.npz, which also holds the model's sample rate and
hop. Writes out/<model>_<n>.wav to listen to.

Needs TPU-MLIR's Python runtime (pyruntime_cvi), so run it in the same
environment as build_cvimodel.sh, from research/:

  python3 -I milkv_duo/validate_cvimodel.py [out/decoder_44100hz_w128_bf16.cvimodel ...]
"""
import glob
import os
import sys

import numpy as np
import tpu_mlir.entry  # noqa: F401 -- puts TPU-MLIR's runtime modules on sys.path, as its own tools do
from scipy.io import wavfile

try:
    import pyruntime_cvi
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(tpu_mlir.entry.__file__), "python"))
    import pyruntime_cvi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
WAV_UTTERANCES = (0, 1)  # "Hello world!", "The quick brown fox ..."


def snr_db(ref, test):
    noise = np.sum((ref - test) ** 2)
    return float("inf") if noise == 0 else 10 * np.log10(np.sum(ref**2) / noise)


class CviDecoder:
    def __init__(self, path):
        self.model = pyruntime_cvi.Model(path, output_all_tensors=False)
        self.inputs = {t.name: t for t in self.model.inputs}
        for name in ("z", "g", "mask"):
            t = self.inputs[name]
            assert t.data.dtype == np.float32, f"input {name} is {t.data.dtype}, CviTpuVocoder needs float32"
        assert self.model.outputs[0].data.dtype == np.float32, "output must be float32"
        self.window = self.inputs["z"].data.shape[2]
        self.hop = self.model.outputs[0].data.size // self.window  # output samples per frame

    def __call__(self, z, g, mask):
        # The cvimodel's tensors are [1, C, T, 1] (1-D convs run as 2-D ones).
        for name, value in (("z", z), ("g", g), ("mask", mask)):
            t = self.inputs[name].data
            t[:] = value.reshape(t.shape)
        self.model.forward()
        return np.array(self.model.outputs[0].data).reshape(-1)


def utterances_for(model_path):
    """export_decoder_tpu.py's <name>_utterances.npz for out/<name>_<precision>.cvimodel."""
    base = os.path.splitext(os.path.basename(model_path))[0].rsplit("_", 1)[0]
    return np.load(os.path.join(os.path.dirname(os.path.abspath(model_path)), f"{base}_utterances.npz"))


def chunked_decode(dec, z, g, context, hop):
    """Same schedule as ChunkedVocoder::forward()."""
    W, (_, C, T) = dec.window, z.shape
    core = W - 2 * context
    out = np.zeros(T * hop, dtype=np.float32)
    for s in range(0, T, core):
        win = np.zeros((1, C, W), dtype=np.float32)
        mask = np.zeros((1, 1, W), dtype=np.float32)
        lo = s - context
        src_lo, src_hi = max(lo, 0), min(lo + W, T)
        win[:, :, src_lo - lo : src_hi - lo] = z[:, :, src_lo:src_hi]
        mask[:, :, src_lo - lo : src_hi - lo] = 1.0
        n = min(core, T - s)
        out[s * hop : (s + n) * hop] = dec(win, g, mask)[context * hop : (context + n) * hop]
    return out


def main():
    models = sys.argv[1:] or sorted(glob.glob(os.path.join(OUT, "decoder_*.cvimodel")))
    print("SNR / cosine vs. the float32 one-pass decoder")
    for path in models:
        data = utterances_for(path)
        context, hop, rate, g = int(data["context"]), int(data["hop"]), int(data["sample_rate"]), data["g"]
        n_utt = len([k for k in data.files if k.startswith("z")])
        dec = CviDecoder(path)
        assert dec.hop == hop, f"{path}: {dec.hop} samples per frame, its utterances {hop}"
        for i in WAV_UTTERANCES:
            wavfile.write(os.path.join(OUT, f"float32_{rate}hz_{i}.wav"), rate, data[f"audio{i}"])
        snrs, coss = [], []
        for i in range(n_utt):
            ref = data[f"audio{i}"]
            out = chunked_decode(dec, data[f"z{i}"], g, context, hop)
            snrs.append(snr_db(ref, out))
            coss.append(float(np.dot(ref, out) / (np.linalg.norm(ref) * np.linalg.norm(out) + 1e-12)))
            if i in WAV_UTTERANCES:
                name = os.path.splitext(os.path.basename(path))[0]
                wavfile.write(os.path.join(OUT, f"{name}_{i}.wav"), rate, out)
        print(
            f"  {os.path.basename(path):40s} {n_utt} utterances, context {context}: SNR median {np.median(snrs):5.1f} dB, worst {min(snrs):5.1f} dB;"
            f" cosine worst {min(coss):.4f}"
        )


if __name__ == "__main__":
    main()
