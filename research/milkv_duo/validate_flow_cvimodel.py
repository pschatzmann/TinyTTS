"""
Runs build_flow_cvimodels.sh's flow models in TPU-MLIR's simulator over the
20 sentences of export_flow_tpu.py (out/flow_<rate>hz_utterances.npz), each padded to
the smallest bucket that fits -- as CviTpuFlow does on the device -- and
compares z with the float32 flow. With a decoder cvimodel it also decodes
the TPU flow's z on the (simulated) TPU and compares the audio with the
float32 pipeline, and writes out/tpu_flow_decoder_<rate>hz_<n>.wav to listen to.
The decoder must come from the same checkpoint (same sample rate).

Run in the same environment as build_flow_cvimodels.sh, from research/:

  python3 -I milkv_duo/validate_flow_cvimodel.py [--sample-rate 44100] [out/decoder_44100hz_w128_bf16.cvimodel]
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from scipy.io import wavfile

from validate_cvimodel import CviDecoder, chunked_decode, pyruntime_cvi, snr_db, utterances_for

OUT = os.path.join(HERE, "out")
WAV_UTTERANCES = (0, 1)


def log_spectrogram(a, hop):
    n = 4 * hop  # training/audio_config.py's n_fft
    frames = [np.abs(np.fft.rfft(a[i : i + n] * np.hanning(n))) for i in range(0, len(a) - n, hop)]
    return np.log(np.array(frames) + 1e-5)


class CviFlow:
    def __init__(self, path):
        self.model = pyruntime_cvi.Model(path, output_all_tensors=False)
        self.inputs = {t.name: t for t in self.model.inputs}
        self.length = self.inputs["z_p"].data.shape[2]

    def __call__(self, z_p, g):
        C, T = z_p.shape[1], z_p.shape[2]
        zp = np.zeros((1, C, self.length), np.float32)
        zp[:, :, :T] = z_p
        mask = np.zeros((1, 1, self.length), np.float32)
        mask[..., :T] = 1.0
        for name, value in (("z_p", zp), ("y_mask", mask), ("g", g)):
            t = self.inputs[name].data
            t[:] = value.reshape(t.shape)
        self.model.forward()
        return np.array(self.model.outputs[0].data).reshape(1, C, self.length)[:, :, :T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-rate", type=int, default=44100, help="which model's flow_<rate>hz_* files to use")
    ap.add_argument("decoder", nargs="?", help="a decoder cvimodel of the same model, to also check the audio")
    args = ap.parse_args()
    rate = args.sample_rate

    flows = sorted(
        (CviFlow(p) for p in glob.glob(os.path.join(OUT, f"flow_{rate}hz_l*_bf16.cvimodel"))), key=lambda f: f.length
    )
    print("flow buckets:", [f.length for f in flows])
    u = np.load(os.path.join(OUT, f"flow_{rate}hz_utterances.npz"))
    assert int(u["sample_rate"]) == rate
    g = u["g"]
    decoder = CviDecoder(args.decoder) if args.decoder else None
    dec_data = utterances_for(args.decoder) if decoder else None
    if decoder:
        assert int(dec_data["sample_rate"]) == rate, f"{args.decoder} is a {int(dec_data['sample_rate'])} Hz model"
        assert decoder.hop == int(dec_data["hop"])
    context = int(dec_data["context"]) if decoder else 0
    hop = int(dec_data["hop"]) if decoder else 0
    n = len([k for k in u.files if k.startswith("z_p")])

    z_snrs, audio_snrs, spec_corrs = [], [], []
    for i in range(n):
        z_p, z_ref = u[f"z_p{i}"], u[f"z{i}"]
        T = z_p.shape[2]
        bucket = next((f for f in flows if f.length >= T), None)
        if bucket is None:
            print(f"  utterance {i}: {T} frames, longer than the largest bucket (CPU flow on the device)")
            continue
        z = bucket(z_p, g)
        z_snrs.append(snr_db(z_ref, z))
        line = f"  utterance {i:2d}: {T:3d} frames in bucket {bucket.length:4d}: z SNR {z_snrs[-1]:5.1f} dB"
        if decoder:
            # <name>_utterances.npz's audio is the float32 decoder applied to the float32 flow's z.
            ref = dec_data[f"audio{i}"]
            audio = chunked_decode(decoder, z.astype(np.float32), g, context, hop)
            audio_snrs.append(snr_db(ref, audio))
            spec_corrs.append(np.corrcoef(log_spectrogram(ref, hop).ravel(), log_spectrogram(audio, hop).ravel())[0, 1])
            line += f"; flow+decoder audio SNR {audio_snrs[-1]:5.1f} dB, log-spectrogram corr {spec_corrs[-1]:.4f}"
            if i in WAV_UTTERANCES:
                wavfile.write(os.path.join(OUT, f"tpu_flow_decoder_{rate}hz_{i}.wav"), rate, audio)
        print(line, flush=True)

    print(f"flow z SNR: median {np.median(z_snrs):.1f} dB, worst {min(z_snrs):.1f} dB ({len(z_snrs)} utterances)")
    if decoder:
        print(
            f"flow + decoder audio SNR: median {np.median(audio_snrs):.1f} dB, worst {min(audio_snrs):.1f} dB; "
            f"log-spectrogram corr worst {min(spec_corrs):.4f}"
        )


if __name__ == "__main__":
    main()
