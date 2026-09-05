"""
M0 step 3: Validate TFLite (float32) graphs against PyTorch, end-to-end, and
list the concrete builtin ops each .tflite uses (checked against TFLite Micro's
supported op list before writing any Arduino code).

Important layout note: onnx2tf converts PyTorch/ONNX's channel-first (N,C,T)
Conv1d tensors to TFLite's native channel-last (N,T,C) layout. Every tensor
crossing a graph boundary here is transposed accordingly -- the eventual C++
runtime needs to do the same when feeding/reading TFLM tensors.
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

TFLITE_DIR = os.path.join(os.path.dirname(__file__), "tflite_out")


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


def ncw_to_nwc(x):
    return np.transpose(x, (0, 2, 1))


def nwc_to_ncw(x):
    return np.transpose(x, (0, 2, 1))


def make_interp(name):
    path = os.path.join(TFLITE_DIR, name, f"{name}_float32.tflite")
    interp = tf.lite.Interpreter(
        model_path=path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    return interp


def run_tflite(interp, feed_ncw_by_name, channel_last_names):
    """feed_ncw_by_name: dict input_name -> np.array in PyTorch (N,C,T) layout
    (or plain shape for non-conv tensors like phone_ids). channel_last_names is
    the set of input names that need transposing to (N,T,C) for TFLite."""
    in_details = {d["name"]: d for d in interp.get_input_details()}
    feed = {}
    for name, arr in feed_ncw_by_name.items():
        d = in_details[name]
        a = ncw_to_nwc(arr) if name in channel_last_names else arr
        feed[name] = a
        if list(d["shape"]) != list(a.shape):
            interp.resize_tensor_input(d["index"], a.shape, strict=False)
    interp.allocate_tensors()
    in_details = {d["name"]: d for d in interp.get_input_details()}
    for name, a in feed.items():
        d = in_details[name]
        interp.set_tensor(d["index"], a.astype(d["dtype"]))
    interp.invoke()
    out_details = {d["name"]: d for d in interp.get_output_details()}
    result = {}
    for name, d in out_details.items():
        t = interp.get_tensor(d["index"])
        result[name] = nwc_to_ncw(t) if name in channel_last_names else t
    return result


def list_ops(name):
    path = os.path.join(TFLITE_DIR, name, f"{name}_float32.tflite")
    interp = tf.lite.Interpreter(model_path=path)
    return sorted({d["op_name"] for d in interp._get_ops_details()})


def main():
    print("### Builtin op inventory per graph (float32 .tflite) ###")
    all_ops = set()
    for name in ["text_encoder", "duration_predictor", "flow", "decoder"]:
        ops = list_ops(name)
        all_ops.update(ops)
        print(f"{name}: {ops}")
    print(f"\nUnion of all ops used: {sorted(all_ops)}\n")

    net_g = load_model()

    T = 17
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
        w = torch.exp(logw_pt) * x_mask_pt
        w_ceil = torch.ceil(w)
        y_len = max(1, int(w_ceil.sum().item()))
        torch.manual_seed(456)
        z_p_pt = torch.randn(1, MODEL_PARAMS["inter_channels"], y_len)
        y_mask_pt = torch.ones(1, 1, y_len)
        z_pt = net_g.flow(z_p_pt, y_mask_pt, g=g, reverse=True)
        audio_pt = net_g.dec(z_pt * y_mask_pt, g=g)

    # channel_last_names: inputs/outputs that are (N,C,T) in Torch/ONNX and need
    # transposing to (N,T,C) for TFLite. bert/ja_bert also (N,C,T). Non-conv
    # tensors (ids, lengths, masks used as broadcast-only) are NOT transposed --
    # x_mask/y_mask happen to have C=1 so transposing is a no-op for them but we
    # keep them out of the set for clarity since onnx2tf kept mask shape (1,1,T)->(1,T,1).
    enc_channel_last = {"bert", "ja_bert", "x", "m_p", "logs_p", "g", "x_mask"}
    interp_enc = make_interp("text_encoder")
    outs = run_tflite(
        interp_enc,
        {
            "phone_ids": phone_ids.numpy().astype(np.int64),
            "phone_lengths": phone_lengths.numpy().astype(np.int64),
            "tone_ids": tone_ids.numpy().astype(np.int64),
            "language_ids": language_ids.numpy().astype(np.int64),
            "bert": bert.numpy(),
            "ja_bert": ja_bert.numpy(),
            "speaker_id": speaker_id.numpy().astype(np.int64),
        },
        enc_channel_last,
    )
    x_o, m_p_o, logs_p_o, x_mask_o, g_o = outs["x"], outs["m_p"], outs["logs_p"], outs["x_mask"], outs["g"]
    print(f"cos_sim x:      {cos_sim(x_pt.numpy(), x_o):.6f}")
    print(f"cos_sim m_p:    {cos_sim(m_p_pt.numpy(), m_p_o):.6f}")
    print(f"cos_sim logs_p: {cos_sim(logs_p_pt.numpy(), logs_p_o):.6f}")
    print(f"cos_sim g:      {cos_sim(g.numpy(), g_o):.6f}")

    dp_channel_last = {"x", "x_mask", "g", "logw"}
    interp_dp = make_interp("duration_predictor")
    outs = run_tflite(interp_dp, {"x": x_o, "x_mask": x_mask_o, "g": g_o}, dp_channel_last)
    logw_o = outs["logw"]
    print(f"cos_sim logw:   {cos_sim(logw_pt.numpy(), logw_o):.6f}")

    flow_channel_last = {"z_p", "y_mask", "g", "z"}
    interp_flow = make_interp("flow")
    outs = run_tflite(
        interp_flow, {"z_p": z_p_pt.numpy(), "y_mask": y_mask_pt.numpy(), "g": g_o}, flow_channel_last
    )
    z_o = outs["z"]
    print(f"cos_sim z:      {cos_sim(z_pt.numpy(), z_o):.6f}")

    dec_channel_last = {"z", "g", "audio"}
    interp_dec = make_interp("decoder")
    outs = run_tflite(interp_dec, {"z": z_o * y_mask_pt.numpy(), "g": g_o}, dec_channel_last)
    audio_o = outs["audio"]
    diff = np.max(np.abs(audio_pt.numpy() - audio_o))
    print(f"cos_sim audio:  {cos_sim(audio_pt.numpy(), audio_o):.6f}  max_abs_diff={diff:.6f}")


if __name__ == "__main__":
    main()
