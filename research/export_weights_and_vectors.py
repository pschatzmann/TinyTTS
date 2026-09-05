"""
M2 step 1: export the weights needed for a hand-written C++ PhonemeEncoder +
AttentionFlowBlock (text_encoder + flow -- the two attention-bearing graphs
onnx2tf can't handle), plus reference input/output test vectors from the
patched-but-numerically-identical PyTorch model, for validating the C++ port.

Binary weight format (weights.bin and test_vectors.bin), named-tensor,
WeightStore.h parses both:
  repeated: [int32 name_len][name bytes (utf8)][int32 ndims][int32 dims...]
            [int32 dtype][data...]
  terminated by name_len == 0
  dtype 0: data is float32, one 4-byte value per element (used for
    test_vectors.bin -- reference values for numerical validation via
    cosine similarity, kept full precision).
  dtype 1: data is float16 (IEEE 754 binary16), one 2-byte value per
    element -- WeightStore.h expands each to float32 at parse time, so
    every consumer (PhonemeEncoder/Flow/Attention/...) still just sees
    plain float32 Mats; only the stored binary is smaller. Used for
    weights.bin -- halves its size for a rounding-error-level precision
    cost (see docs/research.md for the numbers checked before trusting
    this).
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tiny-tts"))

import numpy as np
import torch

from tiny_tts.models import VoiceSynthesizer
from tiny_tts.text.symbols import symbols
from tiny_tts.utils import SPEC_CHANNELS, SEGMENT_FRAMES, N_SPEAKERS, MODEL_PARAMS

OUT_DIR = os.path.dirname(__file__)


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


def write_tensor(f, name, arr, dtype="float32"):
    name_b = name.encode("utf-8")
    f.write(struct.pack("<i", len(name_b)))
    f.write(name_b)
    f.write(struct.pack("<i", arr.ndim))
    for d in arr.shape:
        f.write(struct.pack("<i", d))
    if dtype == "float16":
        f.write(struct.pack("<i", 1))
        arr = np.ascontiguousarray(arr.astype(np.float16))
    else:
        f.write(struct.pack("<i", 0))
        arr = np.ascontiguousarray(arr.astype(np.float32))
    f.write(arr.tobytes())


def write_terminator(f):
    f.write(struct.pack("<i", 0))


def export_state_dict_subset(net_g, prefixes, out_path, exclude=(), dtype="float32"):
    sd = net_g.state_dict()
    with open(out_path, "wb") as f:
        n = 0
        for k, v in sd.items():
            if k in exclude:
                continue
            if any(k.startswith(p) for p in prefixes):
                write_tensor(f, k, v.detach().numpy(), dtype=dtype)
                n += 1
        write_terminator(f)
    print(f"wrote {n} tensors to {out_path}")


# bert_proj/ja_bert_proj's weight matrices (multi-lingual BERT conditioning)
# are dead weight: that feature is disabled by default (bert/ja_bert are
# always a zero tensor in every real call path this project has), and a
# Conv1d(kernel_size=1)'s output with a zero input is just its bias term --
# see PhonemeEncoder.h's forward() doc. Their bias vectors (tiny) are still
# exported and used; only the weight matrices (229,376 of weights.bin's
# 2,935,304 bytes, ~7.8%) are dropped.
UNUSED_BERT_PROJ_WEIGHTS = {"enc_p.bert_proj.weight", "enc_p.ja_bert_proj.weight"}


def main():
    net_g = load_model()

    export_state_dict_subset(
        net_g, ["enc_p.", "flow.", "emb_g.", "dp.", "dec."], os.path.join(OUT_DIR, "weights.bin"),
        exclude=UNUSED_BERT_PROJ_WEIGHTS, dtype="float16",
    )

    # ---- reference test vectors ----
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

        # ---- alignment expansion reference (matches infer_onnx.py's
        # _compute_alignment_path_np, via the same attn-matrix formulation) ----
        durations = w_ceil[0, 0, :].long()  # [T_x]
        y_mask_full = torch.ones(1, 1, y_len)
        attn_mask = y_mask_full.unsqueeze(-1) * x_mask_pt.unsqueeze(2)  # [1,1,T_y,T_x]
        from tiny_tts.nn import commons

        attn = commons.compute_alignment_path(w_ceil, attn_mask)  # [1,1,T_y,T_x]
        m_p_exp_pt = torch.matmul(attn.squeeze(1), m_p_pt.transpose(1, 2)).transpose(1, 2)
        logs_p_exp_pt = torch.matmul(attn.squeeze(1), logs_p_pt.transpose(1, 2)).transpose(1, 2)

        torch.manual_seed(456)
        z_p_pt = torch.randn(1, MODEL_PARAMS["inter_channels"], y_len)
        y_mask_pt = torch.ones(1, 1, y_len)
        z_pt = net_g.flow(z_p_pt, y_mask_pt, g=g, reverse=True)
        audio_pt = net_g.dec(z_pt, g=g)

    with open(os.path.join(OUT_DIR, "test_vectors.bin"), "wb") as f:
        write_tensor(f, "phone_ids", phone_ids.numpy())
        write_tensor(f, "phone_lengths", phone_lengths.numpy())
        write_tensor(f, "tone_ids", tone_ids.numpy())
        write_tensor(f, "language_ids", language_ids.numpy())
        write_tensor(f, "speaker_id", speaker_id.numpy())
        write_tensor(f, "g", g.numpy())
        write_tensor(f, "x_ref", x_pt.numpy())
        write_tensor(f, "m_p_ref", m_p_pt.numpy())
        write_tensor(f, "logs_p_ref", logs_p_pt.numpy())
        write_tensor(f, "x_mask_ref", x_mask_pt.numpy())
        write_tensor(f, "z_p", z_p_pt.numpy())
        write_tensor(f, "y_mask", y_mask_pt.numpy())
        write_tensor(f, "z_ref", z_pt.numpy())
        write_tensor(f, "logw_ref", logw_pt.numpy())
        write_tensor(f, "durations_ref", durations.numpy().astype("int32").reshape(1, 1, -1))
        write_tensor(f, "m_p_exp_ref", m_p_exp_pt.numpy())
        write_tensor(f, "logs_p_exp_ref", logs_p_exp_pt.numpy())
        write_tensor(f, "audio_ref", audio_pt.numpy())
        write_terminator(f)
    print("wrote test_vectors.bin")
    print(f"T={T} y_len={y_len}")


if __name__ == "__main__":
    main()
