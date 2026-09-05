"""
M2 step: generate representative calibration data (real model activations,
not random noise) for onnx2tf's -oiqt full-int8 quantization of
duration_predictor.onnx and decoder.onnx.

Calibration arrays are saved in TF/channel-last layout (per onnx2tf's -cind
requirement: "must be in dimension order after conversion to TF"), leading
dim = number of calibration samples.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tiny-tts"))

import numpy as np
import torch

from tiny_tts.models import VoiceSynthesizer
from tiny_tts.text.symbols import symbols
from tiny_tts.utils import SPEC_CHANNELS, SEGMENT_FRAMES, N_SPEAKERS, MODEL_PARAMS

OUT_DIR = os.path.join(os.path.dirname(__file__), "calib")
os.makedirs(OUT_DIR, exist_ok=True)


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


def ncw_to_nwc(t: torch.Tensor) -> np.ndarray:
    return t.detach().numpy().transpose(0, 2, 1)


def main():
    net_g = load_model()

    N = 16
    T = 32  # fixed phoneme length for duration_predictor calibration samples
    FRAMES = 96  # fixed audio-frame length for decoder calibration samples (pad/crop)

    xs, x_masks, gs_dp = [], [], []
    zs, gs_dec = [], []

    for i in range(N):
        torch.manual_seed(1000 + i)
        phone_ids = torch.randint(0, len(symbols), (1, T), dtype=torch.long)
        phone_lengths = torch.tensor([T], dtype=torch.long)
        tone_ids = torch.randint(0, 16, (1, T), dtype=torch.long)
        language_ids = torch.full((1, T), 2, dtype=torch.long)
        bert = torch.zeros(1, 1024, T, dtype=torch.float32)
        ja_bert = torch.zeros(1, 768, T, dtype=torch.float32)
        speaker_id = torch.tensor([0], dtype=torch.long)

        with torch.no_grad():
            g = net_g.emb_g(speaker_id).unsqueeze(-1)
            x, m_p, logs_p, x_mask = net_g.enc_p(
                phone_ids, phone_lengths, tone_ids, language_ids, bert, ja_bert, g=g
            )
            logw = net_g.dp(x, x_mask, g=g)

            xs.append(ncw_to_nwc(x)[0])
            x_masks.append(ncw_to_nwc(x_mask)[0])
            gs_dp.append(ncw_to_nwc(g)[0])

            w = torch.exp(logw) * x_mask
            w_ceil = torch.ceil(w)
            y_len = max(1, int(w_ceil.sum().item()))
            torch.manual_seed(2000 + i)
            z_p = torch.randn(1, MODEL_PARAMS["inter_channels"], y_len)
            y_mask = torch.ones(1, 1, y_len)
            z = net_g.flow(z_p, y_mask, g=g, reverse=True)
            z_masked = (z * y_mask)[0].detach().numpy().transpose(1, 0)  # [T,C]

            # pad/crop to fixed FRAMES for a uniform calibration array
            C = z_masked.shape[1]
            fixed = np.zeros((FRAMES, C), dtype=np.float32)
            n = min(FRAMES, z_masked.shape[0])
            fixed[:n] = z_masked[:n]
            zs.append(fixed)
            gs_dec.append(ncw_to_nwc(g)[0])

    np.save(os.path.join(OUT_DIR, "dp_x.npy"), np.stack(xs))
    np.save(os.path.join(OUT_DIR, "dp_x_mask.npy"), np.stack(x_masks))
    np.save(os.path.join(OUT_DIR, "dp_g.npy"), np.stack(gs_dp))
    np.save(os.path.join(OUT_DIR, "dec_z.npy"), np.stack(zs))
    np.save(os.path.join(OUT_DIR, "dec_g.npy"), np.stack(gs_dec))

    print("dp_x", np.stack(xs).shape)
    print("dp_x_mask", np.stack(x_masks).shape)
    print("dp_g", np.stack(gs_dp).shape)
    print("dec_z", np.stack(zs).shape)
    print("dec_g", np.stack(gs_dec).shape)
    print("wrote calibration data to", OUT_DIR)


if __name__ == "__main__":
    main()
