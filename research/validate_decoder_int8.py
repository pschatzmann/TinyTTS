"""
Accuracy check for weights-only (dynamic-range) INT8 quantization of
`decoder` (tiny_tts.models.synthesizer.WaveformDecoder, `net_g.dec`), BEFORE
committing to porting it into the hand-written C++ Decoder.h.

Scheme: symmetric, per-output-row INT8 (scale = max(abs(row))/127, same as
DictionaryModel's already-shipped GRU weight quantization) applied to every
Conv1d/ConvTranspose1d weight tensor in the decoder. Biases stay float32
(tiny, not worth it). Activations stay float32 throughout -- this is
"weights-only" quantization, not the full-INT8-with-INT16-activations
scheme the old TFLite Micro decoder used. That scheme was forced by a TFLite
Micro constraint (it rejects "hybrid" weights-only-quantized models
outright); a hand-written decoder has no such constraint, and the earlier
quantization study (see docs/research.md) found dynamic-range/weights-only
quantization scored *better* audio quality than full-INT8-with-INT16-act
anyway (22.1dB SNR, no outlier spike, vs 22.6dB with a persistent outlier) --
so this is the better scheme now that nothing forces the worse one.

This script fake-quantizes (quantize then immediately dequantize) the real
PyTorch decoder's weights in place, runs the same real z/g inputs through
both the original and fake-quantized decoder, and compares outputs via
cosine similarity + RMS-based SNR (the metric that matters for audio
quality, not just cosine similarity) -- go/no-go before writing any C++.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tiny-tts"))

import numpy as np
import torch

from tiny_tts.models import VoiceSynthesizer
from tiny_tts.text.symbols import symbols
from tiny_tts.utils import SPEC_CHANNELS, SEGMENT_FRAMES, N_SPEAKERS, MODEL_PARAMS


def load_model():
    net_g = VoiceSynthesizer(len(symbols), SPEC_CHANNELS, SEGMENT_FRAMES, n_speakers=N_SPEAKERS, **MODEL_PARAMS)
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download("backtracking/tiny-tts", "G.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net_g.load_state_dict(ckpt["model"], strict=False)
    net_g.eval()
    net_g.dec.remove_weight_norm()
    return net_g


def quantize_dequantize(weight: torch.Tensor) -> torch.Tensor:
    """Symmetric per-output-row INT8 fake-quantization (quantize then
    immediately dequantize, simulating the precision loss of storing this
    as int8 -- same scheme as DictionaryModel's GRU weights)."""
    w = weight.detach().numpy()
    out = np.empty_like(w)
    rows = w.reshape(w.shape[0], -1)
    out_rows = out.reshape(w.shape[0], -1)
    for i in range(rows.shape[0]):
        row = rows[i]
        maxabs = np.max(np.abs(row))
        scale = maxabs / 127.0 if maxabs > 0 else 1.0
        q = np.clip(np.round(row / scale), -127, 127).astype(np.int8)
        out_rows[i] = q.astype(np.float32) * scale
    return torch.from_numpy(out).float()


def quantize_decoder_(dec: torch.nn.Module) -> int:
    """Fake-quantizes every Conv1d/ConvTranspose1d weight in `dec` in
    place. Returns how many tensors were touched."""
    n = 0
    for module in dec.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            with torch.no_grad():
                module.weight.copy_(quantize_dequantize(module.weight))
            n += 1
    return n


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    ref, test = reference.flatten().astype(np.float64), test.flatten().astype(np.float64)
    signal_power = np.mean(ref**2)
    noise_power = np.mean((ref - test) ** 2)
    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def main():
    net_g = load_model()

    # Same deterministic scenario export_weights_and_vectors.py uses, so
    # this is testing on a realistic z/g pair, not synthetic noise.
    torch.manual_seed(123)
    T = 17
    phone_ids = torch.randint(0, len(symbols), (1, T), dtype=torch.long)
    phone_lengths = torch.tensor([T], dtype=torch.long)
    tone_ids = torch.randint(0, 16, (1, T), dtype=torch.long)
    language_ids = torch.full((1, T), 2, dtype=torch.long)
    bert = torch.zeros(1, 1024, T, dtype=torch.float32)
    ja_bert = torch.zeros(1, 768, T, dtype=torch.float32)
    speaker_id = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        g = net_g.emb_g(speaker_id).unsqueeze(-1)
        x_pt, m_p_pt, logs_p_pt, x_mask_pt = net_g.enc_p(
            phone_ids, phone_lengths, tone_ids, language_ids, bert, ja_bert, g=g
        )
        logw_pt = net_g.dp(x_pt, x_mask_pt, g=g)
        w = torch.exp(logw_pt) * x_mask_pt
        w_ceil = torch.ceil(w)
        y_len = max(1, int(w_ceil.sum().item()))

        torch.manual_seed(456)
        z_p_pt = torch.randn(1, MODEL_PARAMS["inter_channels"], y_len)
        y_mask_pt = torch.ones(1, 1, y_len)
        z_pt = net_g.flow(z_p_pt, y_mask_pt, g=g, reverse=True)

        audio_ref = net_g.dec(z_pt, g=g).numpy()

        dec_q = copy.deepcopy(net_g.dec)
        n_quantized = quantize_decoder_(dec_q)
        audio_q = dec_q(z_pt, g=g).numpy()

    # Also run on a second, longer utterance for a bit more coverage --
    # "Hello world!"-scale (T=17) alone isn't a strong signal either way.
    torch.manual_seed(789)
    T2 = 60
    phone_ids2 = torch.randint(0, len(symbols), (1, T2), dtype=torch.long)
    phone_lengths2 = torch.tensor([T2], dtype=torch.long)
    tone_ids2 = torch.randint(0, 16, (1, T2), dtype=torch.long)
    language_ids2 = torch.full((1, T2), 2, dtype=torch.long)
    bert2 = torch.zeros(1, 1024, T2, dtype=torch.float32)
    ja_bert2 = torch.zeros(1, 768, T2, dtype=torch.float32)
    with torch.no_grad():
        x2, m_p2, logs_p2, x_mask2 = net_g.enc_p(
            phone_ids2, phone_lengths2, tone_ids2, language_ids2, bert2, ja_bert2, g=g
        )
        logw2 = net_g.dp(x2, x_mask2, g=g)
        w2 = torch.exp(logw2) * x_mask2
        w_ceil2 = torch.ceil(w2)
        y_len2 = max(1, int(w_ceil2.sum().item()))
        torch.manual_seed(321)
        z_p2 = torch.randn(1, MODEL_PARAMS["inter_channels"], y_len2)
        y_mask2 = torch.ones(1, 1, y_len2)
        z2 = net_g.flow(z_p2, y_mask2, g=g, reverse=True)
        audio_ref2 = net_g.dec(z2, g=g).numpy()
        audio_q2 = dec_q(z2, g=g).numpy()

    print(f"quantized {n_quantized} Conv1d/ConvTranspose1d weight tensors (weights-only INT8, per-row scale)")
    print()
    print(f"utterance 1 (T={T}, y_len={y_len}):")
    print(f"  cos_sim = {cos_sim(audio_ref, audio_q):.6f}")
    print(f"  SNR     = {snr_db(audio_ref, audio_q):.1f} dB")
    print(f"  max_abs_diff = {np.max(np.abs(audio_ref - audio_q)):.6f}")
    print()
    print(f"utterance 2 (T={T2}, y_len={y_len2}):")
    print(f"  cos_sim = {cos_sim(audio_ref2, audio_q2):.6f}")
    print(f"  SNR     = {snr_db(audio_ref2, audio_q2):.1f} dB")
    print(f"  max_abs_diff = {np.max(np.abs(audio_ref2 - audio_q2)):.6f}")


if __name__ == "__main__":
    main()
