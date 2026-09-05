"""
M0 step 1: Load G.pth, fold weight-norm, and re-export the 4-stage split
(text_encoder / duration_predictor / flow / decoder) to ONNX, matching the
input/output contract used by tiny_tts/infer_onnx.py's OnnxTinyTTS class.

Run from research/ with: source venv/bin/activate && python export_split_onnx.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tiny-tts"))

import torch
from torch import nn

from tiny_tts.models import VoiceSynthesizer
from tiny_tts.text.symbols import symbols
from tiny_tts.utils import SPEC_CHANNELS, SEGMENT_FRAMES, N_SPEAKERS, MODEL_PARAMS

OUT_DIR = os.path.join(os.path.dirname(__file__), "onnx_out")
os.makedirs(OUT_DIR, exist_ok=True)

OPSET = 14


def load_model():
    net_g = VoiceSynthesizer(
        len(symbols), SPEC_CHANNELS, SEGMENT_FRAMES, n_speakers=N_SPEAKERS, **MODEL_PARAMS
    )
    # NOTE: the repo's committed checkpoints/G.pth is a mismatched 9.8M-param dev
    # checkpoint (hidden_channels=80) that does NOT match tiny_tts/utils/config.py's
    # MODEL_PARAMS (hidden_channels=32). The real README-matching 1.6M-param model
    # only exists on the HF Hub and is what TinyTTS() auto-downloads by default.
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download("backtracking/tiny-tts", "G.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"]
    model_state = net_g.state_dict()
    new_state_dict = {}
    skipped = []
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        if key in model_state:
            if v.shape == model_state[key].shape:
                new_state_dict[key] = v
            else:
                skipped.append(key)
        else:
            new_state_dict[key] = v
    net_g.load_state_dict(new_state_dict, strict=False)
    net_g.eval()

    # Fold weight_norm -> plain conv weights (only WaveformDecoder uses weight_norm
    # on the inference path; TransformerCouplingLayer/TransformerBlock use plain
    # nn.Conv1d, nothing else to fold).
    net_g.dec.remove_weight_norm()

    if skipped:
        print(f"NOTE: {len(skipped)} checkpoint keys skipped (shape mismatch): {skipped[:5]}")

    return net_g


class TextEncoderWrapper(nn.Module):
    """phone/tone/lang ids + bert placeholders -> (x, m_p, logs_p, x_mask, g)"""

    def __init__(self, model: VoiceSynthesizer):
        super().__init__()
        self.model = model

    def forward(self, phone_ids, phone_lengths, tone_ids, language_ids, bert, ja_bert, speaker_id):
        m = self.model
        g = m.emb_g(speaker_id).unsqueeze(-1)
        g_p = None if m.use_vc else g
        x, m_p, logs_p, x_mask = m.enc_p(
            phone_ids, phone_lengths, tone_ids, language_ids, bert, ja_bert, g=g_p
        )
        return x, m_p, logs_p, x_mask, g


class DurationPredictorWrapper(nn.Module):
    """x, x_mask, g -> logw  (deterministic DurationEstimator only, sdp_ratio=0)"""

    def __init__(self, model: VoiceSynthesizer):
        super().__init__()
        self.model = model

    def forward(self, x, x_mask, g):
        return self.model.dp(x, x_mask, g=g)


class FlowWrapper(nn.Module):
    """z_p, y_mask, g -> z  (reverse flow)"""

    def __init__(self, model: VoiceSynthesizer):
        super().__init__()
        self.model = model

    def forward(self, z_p, y_mask, g):
        return self.model.flow(z_p, y_mask, g=g, reverse=True)


class DecoderWrapper(nn.Module):
    """z (already masked), g -> audio"""

    def __init__(self, model: VoiceSynthesizer):
        super().__init__()
        self.model = model

    def forward(self, z, g):
        return self.model.dec(z, g=g)


def main():
    torch.manual_seed(0)
    net_g = load_model()

    T = 17  # dummy phoneme sequence length (odd, arbitrary but realistic-ish)
    hidden_channels = MODEL_PARAMS["hidden_channels"]
    inter_channels = MODEL_PARAMS["inter_channels"]
    gin_channels = MODEL_PARAMS["gin_channels"]

    phone_ids = torch.randint(0, len(symbols), (1, T), dtype=torch.long)
    phone_lengths = torch.tensor([T], dtype=torch.long)
    tone_ids = torch.randint(0, 16, (1, T), dtype=torch.long)
    language_ids = torch.full((1, T), 2, dtype=torch.long)  # EN = 2
    bert = torch.zeros(1, 1024, T, dtype=torch.float32)
    ja_bert = torch.zeros(1, 768, T, dtype=torch.float32)
    speaker_id = torch.tensor([0], dtype=torch.long)

    # ---- 1. text_encoder ----
    enc_wrapper = TextEncoderWrapper(net_g).eval()
    enc_inputs = (phone_ids, phone_lengths, tone_ids, language_ids, bert, ja_bert, speaker_id)
    with torch.no_grad():
        x_enc, m_p, logs_p, x_mask, g = enc_wrapper(*enc_inputs)
    print("text_encoder outputs:", x_enc.shape, m_p.shape, logs_p.shape, x_mask.shape, g.shape)

    torch.onnx.export(
        enc_wrapper,
        enc_inputs,
        os.path.join(OUT_DIR, "text_encoder.onnx"),
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
        input_names=["phone_ids", "phone_lengths", "tone_ids", "language_ids", "bert", "ja_bert", "speaker_id"],
        output_names=["x", "m_p", "logs_p", "x_mask", "g"],
        dynamic_axes={
            "phone_ids": {1: "text_length"},
            "tone_ids": {1: "text_length"},
            "language_ids": {1: "text_length"},
            "bert": {2: "text_length"},
            "ja_bert": {2: "text_length"},
            "x": {2: "text_length"},
            "m_p": {2: "text_length"},
            "logs_p": {2: "text_length"},
            "x_mask": {2: "text_length"},
        },
    )

    # ---- 2. duration_predictor ----
    dp_wrapper = DurationPredictorWrapper(net_g).eval()
    dp_inputs = (x_enc, x_mask, g)
    with torch.no_grad():
        logw = dp_wrapper(*dp_inputs)
    print("duration_predictor output logw:", logw.shape)

    torch.onnx.export(
        dp_wrapper,
        dp_inputs,
        os.path.join(OUT_DIR, "duration_predictor.onnx"),
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
        input_names=["x", "x_mask", "g"],
        output_names=["logw"],
        dynamic_axes={"x": {2: "text_length"}, "x_mask": {2: "text_length"}, "logw": {2: "text_length"}},
    )

    # ---- derive realistic y_len for flow/decoder dummies, exactly like infer_onnx.py ----
    import numpy as np

    w = np.exp(logw.detach().numpy()) * x_mask.detach().numpy() * 1.0
    w_ceil = np.ceil(w)
    y_len = max(1, int(w_ceil.sum()))
    print("derived y_len:", y_len)

    z_p = torch.zeros(1, inter_channels, y_len, dtype=torch.float32)
    z_p.normal_(0, 1)
    y_mask = torch.ones(1, 1, y_len, dtype=torch.float32)

    # ---- 3. flow (reverse) ----
    flow_wrapper = FlowWrapper(net_g).eval()
    flow_inputs = (z_p, y_mask, g)
    with torch.no_grad():
        z = flow_wrapper(*flow_inputs)
    print("flow output z:", z.shape)

    torch.onnx.export(
        flow_wrapper,
        flow_inputs,
        os.path.join(OUT_DIR, "flow.onnx"),
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
        input_names=["z_p", "y_mask", "g"],
        output_names=["z"],
        dynamic_axes={"z_p": {2: "audio_frames"}, "y_mask": {2: "audio_frames"}, "z": {2: "audio_frames"}},
    )

    # ---- 4. decoder ----
    dec_wrapper = DecoderWrapper(net_g).eval()
    z_masked = z * y_mask
    dec_inputs = (z_masked, g)
    with torch.no_grad():
        audio = dec_wrapper(*dec_inputs)
    print("decoder output audio:", audio.shape)

    torch.onnx.export(
        dec_wrapper,
        dec_inputs,
        os.path.join(OUT_DIR, "decoder.onnx"),
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
        input_names=["z", "g"],
        output_names=["audio"],
        dynamic_axes={"z": {2: "audio_frames"}, "audio": {2: "audio_samples"}},
    )

    print("\nDone. Files in", OUT_DIR)
    for f in sorted(os.listdir(OUT_DIR)):
        path = os.path.join(OUT_DIR, f)
        print(f"  {f}: {os.path.getsize(path)/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
