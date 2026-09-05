"""
M2 step: validate int8-quantized .tflite graphs against the PyTorch reference.

Both int8 graphs are FIXED-SHAPE (built via onnx2tf's -ois static-shape
override to work around a tf_converter dynamic-shape tracing bug -- see plan
doc). This is an accepted, deliberate embedded design: TFLite Micro's static
tensor-arena model favors fixed shapes anyway, so duration_predictor runs on
a fixed 32-phoneme window and decoder on a fixed 96-frame audio window
(chunked/streamed for longer utterances), both using the model's existing
x_mask/y_mask masking for the valid-length semantics within that window.

Uses the low-level tensor API (not get_signature_runner()) because int8
graphs require the caller to pre-quantize inputs and dequantize outputs
using each tensor's (scale, zero_point) -- the signature runner does not do
this automatically.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tiny-tts"))

import numpy as np
import tensorflow as tf
import torch

from tiny_tts.models import VoiceSynthesizer
from tiny_tts.text.symbols import symbols
from tiny_tts.utils import SPEC_CHANNELS, SEGMENT_FRAMES, N_SPEAKERS, MODEL_PARAMS

INT8_DIR = os.path.join(os.path.dirname(__file__), "tflite_int8")


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


def ncw_to_nwc(a):
    return np.ascontiguousarray(np.transpose(a, (0, 2, 1)).astype(np.float32))


def nwc_to_ncw(a):
    return np.ascontiguousarray(np.transpose(a, (0, 2, 1)))


def make_interp(path):
    interp = tf.lite.Interpreter(
        model_path=path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()
    return interp


def run_quantized(interp, feed_nwc_by_name):
    in_details = {d["name"].split(":")[0].replace("serving_default_", ""): d
                  for d in interp.get_input_details()}
    for short_name, arr in feed_nwc_by_name.items():
        d = in_details[short_name]
        scale, zp = d["quantization"]
        q = np.round(arr / scale + zp)
        q = np.clip(q, np.iinfo(d["dtype"]).min, np.iinfo(d["dtype"]).max).astype(d["dtype"])
        interp.set_tensor(d["index"], q)
    interp.invoke()
    d = interp.get_output_details()[0]
    out_q = interp.get_tensor(d["index"])
    scale, zp = d["quantization"]
    return (out_q.astype(np.float32) - zp) * scale


def main():
    net_g = load_model()

    # ---- duration_predictor: fixed T=32 window ----
    T = 32
    torch.manual_seed(123)
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

    dp_path = os.path.join(
        INT8_DIR, "duration_predictor",
        os.environ.get("DP_MODEL", "duration_predictor_full_integer_quant.tflite"),
    )
    interp_dp = make_interp(dp_path)
    for d in interp_dp.get_input_details():
        print("dp input:", d["name"], d["shape"], d["dtype"], d["quantization"])
    logw_o_nwc = run_quantized(
        interp_dp,
        {"x": ncw_to_nwc(x_pt.numpy()), "x_mask": ncw_to_nwc(x_mask_pt.numpy()), "g": ncw_to_nwc(g.numpy())},
    )
    logw_o = nwc_to_ncw(logw_o_nwc)
    print(f"duration_predictor INT8 (T={T}): cos_sim logw = {cos_sim(logw_pt.numpy(), logw_o):.6f}  "
          f"max_abs_diff={np.max(np.abs(logw_pt.numpy() - logw_o)):.6f}")

    # ---- decoder: fixed 96-frame window ----
    with torch.no_grad():
        w = torch.exp(logw_pt) * x_mask_pt
        w_ceil = torch.ceil(w)
        y_len = max(1, int(w_ceil.sum().item()))
        torch.manual_seed(456)
        z_p = torch.randn(1, MODEL_PARAMS["inter_channels"], y_len)
        y_mask = torch.ones(1, 1, y_len)
        z = net_g.flow(z_p, y_mask, g=g, reverse=True)
        z_masked = (z * y_mask).numpy()

    FRAMES = 96
    C = z_masked.shape[1]
    z_fixed = np.zeros((1, C, FRAMES), dtype=np.float32)
    n = min(FRAMES, y_len)
    z_fixed[:, :, :n] = z_masked[:, :, :n]

    with torch.no_grad():
        audio_pt = net_g.dec(torch.from_numpy(z_fixed), g=g)

    dec_path = os.path.join(
        INT8_DIR, "decoder", os.environ.get("DEC_MODEL", "decoder_full_integer_quant.tflite")
    )
    interp_dec = make_interp(dec_path)
    for d in interp_dec.get_input_details():
        print("dec input:", d["name"], d["shape"], d["dtype"], d["quantization"])
    audio_o_nwc = run_quantized(interp_dec, {"z": ncw_to_nwc(z_fixed), "g": ncw_to_nwc(g.numpy())})
    audio_o = nwc_to_ncw(audio_o_nwc)

    print(f"decoder INT8 (96-frame window, real y_len={y_len}): "
          f"cos_sim audio = {cos_sim(audio_pt.numpy(), audio_o):.6f}  "
          f"max_abs_diff={np.max(np.abs(audio_pt.numpy() - audio_o)):.6f}")


if __name__ == "__main__":
    main()
