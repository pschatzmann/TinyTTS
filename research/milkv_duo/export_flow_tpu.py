"""
Exports the flow (`net_g.flow`, run in reverse) as fixed-length ONNX models
for the Milk-V Duo TPU, one per length bucket:

  out/flow_<rate>hz_l<L>.onnx   z_p [1, 32, L], y_mask [1, 1, L], g [1, 128, 1] -> z [1, 32, L]
  out/flow_<rate>hz_l<L>_test_input.npz / _test_ref.npz   for model_deploy --test_input/--test_reference
  out/flow_<rate>hz_utterances.npz   real z_p/z pairs, for validate_flow_cvimodel.py

--checkpoint takes a G_*.pth from research/training/ (default: the released
44.1 kHz model), as for export_decoder_tpu.py. The flow itself doesn't depend
on the sample rate, but its weights come from the checkpoint, so the rate in
the names keeps the flow models together with that checkpoint's decoder and
weights.bin. The bucket lengths are frames: at 16 kHz (256 samples per frame)
a 1024-frame bucket holds 16.4s of speech, at 44.1 kHz 11.9s.

TPU-MLIR's CV18xx backend only compiles static shapes. Unlike the decoder, the
flow needs no overlapping windows: its convolutions and attention all take
y_mask, so padding an utterance to the next bucket length (y_mask 0 on the
padding) gives the unpadded result -- measured here, against PyTorch and
onnxruntime. Utterances longer than the largest bucket fall back to the CPU
flow (see src/TinyTTS/CviTpuFlow.h).

The attention is exported in a TPU-friendly but equivalent form
(band_attention()): the upstream one builds its relative-position terms with
gather over [L, 2L-1] indices (24 GatherElements, which TPU-MLIR runs on the
CPU) and masks with masked_fill(mask == 0) (a compare the CV18xx backend can't
compile). The relative embeddings only cover +-window_size frames, so the
relative terms are a band of 2*window_size+1 diagonals, moved between band and
[L, L] layout with pad/reshape/slice; the mask becomes an additive -1e4.
The keys are centered (k - mean(k)) before q.k: that changes every score in
a row by the same q.mean(k), which softmax ignores, but removes a per-row
offset of up to +-90 that otherwise costs the scores most of BF16's 8-bit
mantissa.
The attention mask is the key mask only, [1,1,1,L] broadcast over the rows
(key_mask_encoder_forward()), not upstream's [1,1,L,L] outer product: TPU-MLIR
1.30 compiles that two-sided broadcast multiply wrongly for cv181x (0dB SNR
even for an all-ones mask). Rows of padded queries then differ, but those are
masked out before they reach any valid frame.
The channel flips between coupling layers become 1x1 convolutions with an
anti-diagonal 0/1 weight (FlipAsConv): CV18xx only reverses along axis 0 or 3.

Run from research/:  source venv/bin/activate && python milkv_duo/export_flow_tpu.py
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tiny-tts"))

import math
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from export_decoder_tpu import SENTENCES, load_checkpoint, snr_db, text_to_inputs


def band_to_abs(band, w):
    """[b,h,L,2w+1] band (column e = relative position e-w) -> [b,h,L,L], zero outside the band."""
    b, h, L, _ = band.shape
    a = F.pad(band, [0, L])  # [b,h,L,L+2w+1]
    flat = a.reshape(b, h, L * (L + 2 * w + 1))[:, :, : L * (L + 2 * w)]
    return flat.reshape(b, h, L, L + 2 * w)[:, :, :, w : w + L]  # row i shifted right by i


def abs_to_band(x, w):
    """[b,h,L,L] -> [b,h,L,2w+1]: column e holds x[i, i+e-w] (0 outside the matrix)."""
    b, h, L, _ = x.shape
    p = F.pad(x, [w, w])  # [b,h,L,L+2w]
    flat = F.pad(p.reshape(b, h, L * (L + 2 * w)), [0, L])
    return flat.reshape(b, h, L, L + 2 * w + 1)[:, :, :, : 2 * w + 1]  # row i shifted left by i


def band_attention(self, query, key, value, mask=None):
    """MultiHeadSelfAttention.attention() without gather and masked_fill, same result."""
    b, d, L = key.size()
    h, k, w = self.n_heads, self.k_channels, self.window_size
    query = query.view(b, h, k, L).transpose(2, 3) / math.sqrt(k)
    key = key.view(b, h, k, L).transpose(2, 3)
    value = value.view(b, h, k, L).transpose(2, 3)
    key = key - key.mean(2, keepdim=True)  # per-row constant for softmax, see the module doc
    scores = torch.matmul(query, key.transpose(-2, -1))
    rel_k = self.emb_rel_k.unsqueeze(0)  # [1,1|h,2w+1,k]
    scores = scores + band_to_abs(torch.matmul(query, rel_k.transpose(-2, -1)), w)
    if mask is not None:  # [1,1,1,L] key mask: masked_fill(mask == 0, -1e4) for the rows that matter
        scores = scores + (mask - 1.0) * 1e4
    p_attn = F.softmax(scores, dim=-1)
    output = torch.matmul(p_attn, value)
    output = output + torch.matmul(abs_to_band(p_attn, w), self.emb_rel_v.unsqueeze(0))
    return output, p_attn


class FlipAsConv(nn.Module):
    """FlipTransform(reverse=True), i.e. torch.flip(x, [1]), as a 1x1 convolution."""

    def __init__(self, channels):
        super().__init__()
        self.register_buffer("weight", torch.eye(channels).flip(0).unsqueeze(-1))

    def forward(self, x, *args, reverse=False, **kwargs):
        assert reverse
        return F.conv1d(x, self.weight)


def use_flip_as_conv(flow, channels):
    n = 0
    for i, f in enumerate(flow.flows):
        if type(f).__name__ == "FlipTransform":
            flow.flows[i] = FlipAsConv(channels)
            n += 1
    return n


def key_mask_encoder_forward(self, x, x_mask, g=None):
    """TransformerBlock.forward() with a [1,1,1,L] key mask instead of the [1,1,L,L] outer product."""
    attn_mask = x_mask.unsqueeze(2)
    x = x * x_mask
    for i in range(self.n_layers):
        if i == self.cond_layer_idx and g is not None:
            x = (x + self.spk_emb_linear(g.transpose(1, 2)).transpose(1, 2)) * x_mask
        x = self.norm_layers_1[i](x + self.attn_layers[i](x, x, attn_mask))
        x = self.norm_layers_2[i](x + self.ffn_layers[i](x, x_mask))
    return x * x_mask


def use_band_attention(module):
    n = 0
    for m in module.modules():
        if type(m).__name__ == "MultiHeadSelfAttention":
            assert m.window_size is not None and not m.proximal_bias and m.block_length is None
            m.attention = types.MethodType(band_attention, m)
            n += 1
        elif type(m).__name__ == "TransformerBlock":
            m.forward = types.MethodType(key_mask_encoder_forward, m)
    return n


class FlowReverse(nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, z_p, y_mask, g):
        return self.flow(z_p, y_mask, g=g, reverse=True) * y_mask


def utterance_prior(net_g, text, seed):
    """Returns (z_p [1,32,T], g [1,128,1]) exactly as the flow sees them in infer()."""
    torch.manual_seed(seed)
    x, x_len, sid, tone, lang, bert, ja_bert = text_to_inputs(text)
    with torch.no_grad():
        _, _, y_mask, (_, z_p, _, _) = net_g.infer(
            x, x_len, sid, tone, lang, bert, ja_bert, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0
        )
        g = net_g.emb_g(sid).unsqueeze(-1)
    return z_p * y_mask, g


def pad(z_p, L):
    C, T = z_p.shape[1], z_p.shape[2]
    zp = torch.zeros(1, C, L)
    mask = torch.zeros(1, 1, L)
    zp[:, :, :T] = z_p
    mask[:, :, :T] = 1.0
    return zp, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024], help="bucket lengths (frames)")
    ap.add_argument("--checkpoint", help="G_*.pth from research/training/ (default: the released model)")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    net_g, audio_cfg = load_checkpoint(args.checkpoint)
    rate = audio_cfg.sample_rate
    print(f"model: {rate} Hz, {audio_cfg.hop_length} samples per frame")
    model = FlowReverse(net_g.flow).eval()
    utts = [utterance_prior(net_g, s, 100 + i) for i, s in enumerate(SENTENCES)]
    with torch.no_grad():
        refs = [model(z_p, torch.ones(1, 1, z_p.shape[2]), g) for z_p, g in utts]  # upstream attention
    print(f"band attention in {use_band_attention(model)} attention layers")
    print(f"{use_flip_as_conv(model.flow, utts[0][0].shape[1])} channel flips as 1x1 convolutions")
    frames = [z_p.shape[2] for z_p, _ in utts]
    print(f"{len(utts)} utterances, {min(frames)}-{max(frames)} frames")

    import onnxruntime as ort

    for L in sorted(args.lengths):
        fits = [i for i, n in enumerate(frames) if n <= L]
        name = f"flow_{rate}hz_l{L}"
        path = os.path.join(args.out, f"{name}.onnx")
        # Export with a real utterance padded to L, the shortest one that fits.
        i0 = fits[-1] if fits else 0
        z0, m0 = pad(utts[i0][0][:, :, :L], L)
        g0 = utts[i0][1]
        torch.onnx.export(
            model,
            (z0, m0, g0),
            path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            dynamo=False,
            input_names=["z_p", "y_mask", "g"],
            output_names=["z"],
        )
        with torch.no_grad():
            ref0 = model(z0, m0, g0)
        np.savez(os.path.join(args.out, f"{name}_test_input.npz"), z_p=z0.numpy(), y_mask=m0.numpy(), g=g0.numpy())
        np.savez(os.path.join(args.out, f"{name}_test_ref.npz"), z=ref0.numpy())

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        worst_pt, worst_ort = float("inf"), float("inf")
        for i in fits:
            z_p, g = utts[i]
            T = z_p.shape[2]
            zp, m = pad(z_p, L)
            with torch.no_grad():
                out_pt = model(zp, m, g)[:, :, :T].numpy()
            out_ort = sess.run(None, {"z_p": zp.numpy(), "y_mask": m.numpy(), "g": g.numpy()})[0][:, :, :T]
            ref = refs[i].numpy()
            worst_pt = min(worst_pt, snr_db(ref, out_pt))
            worst_ort = min(worst_ort, snr_db(ref, out_ort))
        print(
            f"{name}.onnx: {len(fits)} utterances fit; padded vs. unpadded worst SNR "
            f"PyTorch {worst_pt:.1f} dB, onnxruntime {worst_ort:.1f} dB"
        )

    np.savez(
        os.path.join(args.out, f"flow_{rate}hz_utterances.npz"),
        **{f"z_p{i}": z_p.numpy() for i, (z_p, _) in enumerate(utts)},
        **{f"z{i}": r.numpy() for i, r in enumerate(refs)},
        g=utts[0][1].numpy(),
        sample_rate=rate,
    )


if __name__ == "__main__":
    main()
