"""
Exports the vocoder (`net_g.dec`) as a fixed-shape ONNX window for the
Milk-V Duo 256M / Duo S TPU (SG2002/SG2000, TPU-MLIR chip `cv181x`), and
produces everything TPU-MLIR needs to compile and check it:

  out/<name>.onnx             z [1, 32, W], g [1, 128, 1], mask [1, 1, W]
                              -> audio [1, 1, W*hop]
  out/<name>_calib/*.npz      real decoder input windows, for run_calibration
  out/<name>_test_input.npz   one window, for model_deploy --test_input
  out/<name>_test_ref.npz     its float32 PyTorch output, for --test_reference
  out/<name>_utterances.npz   whole-utterance z/g/audio plus hop, sample rate,
                              window and context, for validate_cvimodel.py

<name> is decoder_<sample rate>hz_w<W>, e.g. decoder_44100hz_w128. The sample
rate is in the name because a decoder only fits the weights.bin of the same
checkpoint (CviTpuVocoder::checkModel() compares it with the loaded weights).

--checkpoint takes a G_*.pth from research/training/ (default: the released
44.1 kHz model). Its sample rate, hop (output samples per frame) and vocoder
upsample stack come from the checkpoint (training/audio_config.py), as in
export_weights_and_vectors.py -- export the weights.bin and the TPU models from
the same checkpoint.

TPU-MLIR's CV18xx backend only compiles static shapes, so the C++ side
(src/TinyTTS/ChunkedVocoder.h) runs an utterance as a series of W-frame
windows that overlap by `context` frames on each side and keeps only the
middle `W - 2*context` frames of each. The decoder is fully convolutional,
so with enough context that's the same as decoding the whole utterance at
once; this script measures how much context is enough (--sweep) before
exporting.

`mask` is 1 for frames inside the utterance and 0 for the zero padding past
either end. It's multiplied in before every convolution (upsampled to that
layer's rate), which is exactly the zero padding the one-pass decoder sees at
the utterance's ends -- without it, the edge windows differ audibly from the
one-pass result whenever the utterance doesn't start/end in silence.

Run from research/:  source venv/bin/activate && python milkv_duo/export_decoder_tpu.py
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tiny-tts"))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import export_weights_and_vectors
from tiny_tts.nn import commons, modules
from tiny_tts.text import phonemes_to_ids
from tiny_tts.text.english import grapheme_to_phoneme, normalize_text

# Varied length, punctuation and phoneme content: the calibration set decides
# the INT8 activation ranges, so it should look like real use.
SENTENCES = [
    "Hello world!",
    "The quick brown fox jumps over the lazy dog.",
    "How are you doing today?",
    "Please turn on the kitchen lights.",
    "It is twenty three degrees and sunny outside.",
    "Warning: the battery level is low.",
    "She sells seashells by the seashore.",
    "Would you like me to set a timer for ten minutes?",
    "The meeting has been moved to Thursday afternoon.",
    "Thank you, goodbye.",
    "Rain is expected later this evening, so take an umbrella.",
    "One, two, three, four, five, six, seven, eight, nine, ten.",
    "This is a small neural text to speech model running on a tiny board.",
    "Yes.",
    "No, I don't think that's right at all!",
    "Measuring the temperature, humidity and air pressure every minute.",
    "The red sign said stop, but nobody listened.",
    "Can you hear me now?",
    "A journey of a thousand miles begins with a single step.",
    "Error: could not connect to the network.",
]


def load_checkpoint(path=None):
    """(net_g, AudioConfig) for a G_*.pth, or the released 44.1 kHz model if `path` is None."""
    return export_weights_and_vectors.load_model(path)


def text_to_inputs(text):
    phones, tones, _ = grapheme_to_phoneme(normalize_text(text))
    phone_ids, tone_ids, lang_ids = phonemes_to_ids(phones, tones, "EN")
    phone_ids = commons.insert_blanks(phone_ids, 0)
    tone_ids = commons.insert_blanks(tone_ids, 0)
    lang_ids = commons.insert_blanks(lang_ids, 0)
    n = len(phone_ids)
    return (
        torch.LongTensor(phone_ids).unsqueeze(0),
        torch.LongTensor([n]),
        torch.LongTensor([0]),
        torch.LongTensor(tone_ids).unsqueeze(0),
        torch.LongTensor(lang_ids).unsqueeze(0),
        torch.zeros(1, 1024, n),
        torch.zeros(1, 768, n),
    )


def utterance_latents(net_g, text, seed):
    """Returns (z [1,32,T], g [1,128,1]) exactly as the decoder sees them in infer()."""
    torch.manual_seed(seed)
    x, x_len, sid, tone, lang, bert, ja_bert = text_to_inputs(text)
    with torch.no_grad():
        _, _, y_mask, (z, _, _, _) = net_g.infer(
            x, x_len, sid, tone, lang, bert, ja_bert, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0
        )
        g = net_g.emb_g(sid).unsqueeze(-1)
    return z * y_mask, g


def windows(z, W, context):
    """Same schedule as ChunkedVocoder::forward(): yields (start_frame, window [1,C,W], mask [1,1,W])."""
    C, T = z.shape[1], z.shape[2]
    core = W - 2 * context
    for s in range(0, T, core):
        win = torch.zeros(1, C, W)
        mask = torch.zeros(1, 1, W)
        lo, hi = s - context, s - context + W
        src_lo, src_hi = max(lo, 0), min(hi, T)
        win[:, :, src_lo - lo : src_hi - lo] = z[:, :, src_lo:src_hi]
        mask[:, :, src_lo - lo : src_hi - lo] = 1.0
        yield s, win, mask


def chunked_decode(dec_fn, z, g, W, context, hop):
    T = z.shape[2]
    core = W - 2 * context
    out = np.zeros(T * hop, dtype=np.float32)
    for s, win, mask in windows(z, W, context):
        audio = dec_fn(win, g, mask).reshape(-1)
        n = min(core, T - s)
        out[s * hop : (s + n) * hop] = audio[context * hop : (context + n) * hop]
    return out


def random_latents(net_g, T):
    """Like utterance_latents(), from random phonemes (test/main.cpp's test vectors do the same)."""
    from tiny_tts.text.symbols import symbols

    x = torch.randint(0, len(symbols), (1, T))
    tone = torch.randint(0, 16, (1, T))
    lang = torch.full((1, T), 2, dtype=torch.long)
    sid = torch.LongTensor([0])
    with torch.no_grad():
        _, _, y_mask, (z, _, _, _) = net_g.infer(
            x, torch.LongTensor([T]), sid, tone, lang, torch.zeros(1, 1024, T), torch.zeros(1, 768, T),
            noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0,
        )
        g = net_g.emb_g(sid).unsqueeze(-1)
    return z * y_mask, g


def snr_db(ref, test):
    noise = np.sum((ref - test) ** 2)
    return float("inf") if noise == 0 else 10 * np.log10(np.sum(ref**2) / noise)


def repeat_frames(mask, r):
    """[1,1,L] -> [1,1,L*r], each value repeated r times. Not F.interpolate():
    TPU-MLIR 1.30 crashes compiling Resize for cv181x (std::length_error)."""
    L = mask.shape[2]
    return mask.unsqueeze(-1).expand(1, 1, L, r).reshape(1, 1, L * r)


class DecoderWindow(nn.Module):
    """WaveformDecoder.forward() with `mask` applied before every convolution."""

    def __init__(self, dec, rates):
        super().__init__()
        self.dec = dec
        self.rates = rates

    def forward(self, z, g, mask):
        dec = self.dec
        x = dec.conv_pre(z * mask) + dec.cond(g)
        for i in range(dec.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = dec.ups[i](x * mask)
            mask = repeat_frames(mask, self.rates[i])
            xs = None
            for j in range(dec.num_kernels):
                r = dec.resblocks[i * dec.num_kernels + j](x, mask)
                xs = r if xs is None else xs + r
            x = xs / dec.num_kernels
        x = F.leaky_relu(x)
        return torch.tanh(dec.conv_post(x * mask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=128, help="frames per TPU call (W)")
    ap.add_argument("--context", type=int, default=12, help="overlap frames on each side")
    ap.add_argument("--sweep", action="store_true", help="only report SNR vs. context size, export nothing")
    ap.add_argument("--checkpoint", help="G_*.pth from research/training/ (default: the released model)")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()

    net_g, audio_cfg = load_checkpoint(args.checkpoint)
    hop, rate = audio_cfg.hop_length, audio_cfg.sample_rate
    dec = net_g.dec
    assert dec.num_upsamples == len(audio_cfg.upsample_rates) and int(np.prod(audio_cfg.upsample_rates)) == hop
    window_model = DecoderWindow(dec, audio_cfg.upsample_rates).eval()
    print(f"model: {rate} Hz, {hop} samples per frame, upsample rates {audio_cfg.upsample_rates}")

    def torch_dec(z, g, mask):
        with torch.no_grad():
            return window_model(z, g, mask).numpy()

    utts = [utterance_latents(net_g, s, 100 + i) for i, s in enumerate(SENTENCES)]
    # Random phonemes don't start/end in silence -- the case the mask is for.
    for i in range(4):
        torch.manual_seed(500 + i)
        utts.append(random_latents(net_g, 40 + 25 * i))
    with torch.no_grad():
        refs = [dec(z, g=g).numpy().reshape(-1) for z, g in utts]
    frames = [z.shape[2] for z, _ in utts]
    print(f"{len(utts)} utterances, {min(frames)}-{max(frames)} frames, {sum(frames) * hop / rate:.1f}s audio")

    if args.sweep:
        print("\nchunked vs. whole-utterance decode, worst utterance SNR (float32, PyTorch):")
        for W in (32, 64, 128):
            for context in (0, 2, 4, 6, 8, 10, 12, 16):
                if W - 2 * context < 8:
                    continue
                worst = min(
                    snr_db(r, chunked_decode(torch_dec, z, g, W, context, hop)) for (z, g), r in zip(utts, refs)
                )
                cost = W / (W - 2 * context)
                print(f"  W={W:3d} context={context:2d}  worst SNR {worst:6.1f} dB   compute x{cost:.2f}")
        return

    W, context = args.window, args.context
    name = f"decoder_{rate}hz_w{W}"
    calib_dir = os.path.join(args.out, f"{name}_calib")
    os.makedirs(calib_dir, exist_ok=True)

    worst = min(snr_db(r, chunked_decode(torch_dec, z, g, W, context, hop)) for (z, g), r in zip(utts, refs))
    print(f"W={W} context={context}: chunked float32 decode, worst SNR vs. whole utterance {worst:.1f} dB")

    onnx_path = os.path.join(args.out, f"{name}.onnx")
    z0, g0 = utts[1]
    _, win0, mask0 = next(windows(z0, W, context))
    torch.onnx.export(
        window_model,
        (win0, g0, mask0),
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
        input_names=["z", "g", "mask"],
        output_names=["audio"],
    )
    print("wrote", onnx_path)

    # Every window of every utterance is a calibration sample; TPU-MLIR's
    # run_calibration reads a directory of npz files keyed by input name.
    n_cal = 0
    for z, g in utts[: len(SENTENCES)]:
        for _, win, mask in windows(z, W, context):
            np.savez(
                os.path.join(calib_dir, f"win_{n_cal:04d}.npz"),
                z=win.numpy(),
                g=g.numpy(),
                mask=mask.numpy(),
            )
            n_cal += 1
    print(f"wrote {n_cal} calibration windows")

    np.savez(os.path.join(args.out, f"{name}_test_input.npz"), z=win0.numpy(), g=g0.numpy(), mask=mask0.numpy())
    np.savez(os.path.join(args.out, f"{name}_test_ref.npz"), audio=torch_dec(win0, g0, mask0))
    np.savez(
        os.path.join(args.out, f"{name}_utterances.npz"),
        **{f"z{i}": z.numpy() for i, (z, _) in enumerate(utts[: len(SENTENCES)])},
        **{f"audio{i}": r for i, r in enumerate(refs[: len(SENTENCES)])},
        g=utts[0][1].numpy(),
        window=W,
        context=context,
        hop=hop,
        sample_rate=rate,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_dec = lambda z, g, m: sess.run(None, {"z": z.numpy(), "g": g.numpy(), "mask": m.numpy()})[0]  # noqa: E731
    worst = min(snr_db(r, chunked_decode(ort_dec, z, g, W, context, hop)) for (z, g), r in zip(utts, refs))
    print(f"onnxruntime, chunked: worst SNR vs. PyTorch whole utterance {worst:.1f} dB")


if __name__ == "__main__":
    main()
