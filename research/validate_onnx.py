"""
M0 step 2: Validate the split ONNX graphs against direct PyTorch inference.

Critically tests generalization to sequence lengths OTHER than the one used
during tracing (T=17), since the relative-position attention code
(nn/attentions.py _get_relative_embeddings) uses Python-level int control
flow that torch.onnx's tracer warned about baking in as constants.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tiny-tts"))

import numpy as np
import onnxruntime as ort
import torch

from tiny_tts.models import VoiceSynthesizer
from tiny_tts.text.symbols import symbols
from tiny_tts.utils import SPEC_CHANNELS, SEGMENT_FRAMES, N_SPEAKERS, MODEL_PARAMS

ONNX_DIR = os.path.join(os.path.dirname(__file__), "onnx_out")


def load_model():
    net_g = VoiceSynthesizer(
        len(symbols), SPEC_CHANNELS, SEGMENT_FRAMES, n_speakers=N_SPEAKERS, **MODEL_PARAMS
    )
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download("backtracking/tiny-tts", "G.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net_g.load_state_dict(ckpt["model"], strict=False)
    net_g.eval()
    net_g.dec.remove_weight_norm()
    return net_g


def cos_sim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def run_case(net_g, sess_enc, sess_dp, sess_flow, sess_dec, T, label):
    torch.manual_seed(123)
    phone_ids = torch.randint(0, len(symbols), (1, T), dtype=torch.long)
    phone_lengths = torch.tensor([T], dtype=torch.long)
    tone_ids = torch.randint(0, 16, (1, T), dtype=torch.long)
    language_ids = torch.full((1, T), 2, dtype=torch.long)
    bert = torch.zeros(1, 1024, T, dtype=torch.float32)
    ja_bert = torch.zeros(1, 768, T, dtype=torch.float32)
    speaker_id = torch.tensor([0], dtype=torch.long)

    # ---- PyTorch reference ----
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
        audio_pt = net_g.dec(z_pt * y_mask_pt, g=g)

    # ---- ONNX Runtime ----
    np_kwargs = dict(
        phone_ids=phone_ids.numpy(),
        phone_lengths=phone_lengths.numpy(),
        tone_ids=tone_ids.numpy(),
        language_ids=language_ids.numpy(),
        bert=bert.numpy(),
        ja_bert=ja_bert.numpy(),
        speaker_id=speaker_id.numpy(),
    )
    x_o, m_p_o, logs_p_o, x_mask_o, g_o = sess_enc.run(None, np_kwargs)
    logw_o = sess_dp.run(None, {"x": x_o, "x_mask": x_mask_o, "g": g_o})[0]

    z_p_o = z_p_pt.numpy()
    y_mask_o = y_mask_pt.numpy()
    z_o = sess_flow.run(None, {"z_p": z_p_o, "y_mask": y_mask_o, "g": g_o})[0]
    audio_o = sess_dec.run(None, {"z": z_o * y_mask_o, "g": g_o})[0]

    print(f"\n=== {label} (T={T}, y_len={y_len}) ===")
    print(f"  x        cos_sim={cos_sim(x_pt.numpy(), x_o):.6f}")
    print(f"  m_p      cos_sim={cos_sim(m_p_pt.numpy(), m_p_o):.6f}")
    print(f"  logs_p   cos_sim={cos_sim(logs_p_pt.numpy(), logs_p_o):.6f}")
    print(f"  logw     cos_sim={cos_sim(logw_pt.numpy(), logw_o):.6f}  max_abs_diff={np.max(np.abs(logw_pt.numpy()-logw_o)):.6f}")
    print(f"  z        cos_sim={cos_sim(z_pt.numpy(), z_o):.6f}")
    print(f"  audio    cos_sim={cos_sim(audio_pt.numpy(), audio_o):.6f}  max_abs_diff={np.max(np.abs(audio_pt.numpy()-audio_o)):.6f}")


def main():
    net_g = load_model()

    sess_enc = ort.InferenceSession(os.path.join(ONNX_DIR, "text_encoder.onnx"), providers=["CPUExecutionProvider"])
    sess_dp = ort.InferenceSession(os.path.join(ONNX_DIR, "duration_predictor.onnx"), providers=["CPUExecutionProvider"])
    sess_flow = ort.InferenceSession(os.path.join(ONNX_DIR, "flow.onnx"), providers=["CPUExecutionProvider"])
    sess_dec = ort.InferenceSession(os.path.join(ONNX_DIR, "decoder.onnx"), providers=["CPUExecutionProvider"])

    # T=17 matches the tracing shape used in export_split_onnx.py -- should be near-perfect.
    run_case(net_g, sess_enc, sess_dp, sess_flow, sess_dec, 17, "SAME length as tracing")

    # Different lengths exercise the dynamic_axes path + the relative-position
    # attention slicing that torch.onnx's tracer warned about.
    for T in [5, 11, 29, 41]:
        run_case(net_g, sess_enc, sess_dp, sess_flow, sess_dec, T, "DIFFERENT length than tracing")


if __name__ == "__main__":
    main()
