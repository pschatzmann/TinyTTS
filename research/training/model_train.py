"""
TrainableSynthesizer: upstream's VoiceSynthesizer plus the training forward
pass upstream doesn't ship.

Upstream is a renamed MeloTTS / Bert-VITS2 model whose training code was
removed, but every module's training path (enc_q, flow/sdp forward direction,
MAS) is still there -- only the top-level forward() wiring them together is
missing. This restores MeloTTS's SynthesizerTrn.forward() (MIT License,
Copyright (c) 2024 MyShell.ai) on top of the upstream classes. It adds no
parameters, so the state_dict has exactly G.pth's keys and a trained
checkpoint goes through export_weights_and_vectors.py unchanged.
"""
import math

import torch

import upstream  # noqa: F401

import tiny_tts.alignment as alignment
from tiny_tts.models import VoiceSynthesizer
from tiny_tts.nn import commons
from tiny_tts.text.symbols import symbols
from tiny_tts.utils import MODEL_PARAMS, N_SPEAKERS, SEGMENT_FRAMES, SPEC_CHANNELS

# BERT conditioning is disabled in this model (inference always feeds zeros,
# see tiny_tts/infer.py and PhonemeEncoder.h), so training does the same.
BERT_DIM = 1024
JA_BERT_DIM = 768


class TrainableSynthesizer(VoiceSynthesizer):
    def update_mas_noise(self, step):
        if self.use_noise_scaled_mas:
            self.current_mas_noise_scale = max(self.mas_noise_scale_initial - self.noise_scale_delta * step, 0.0)

    def forward(self, x, x_lengths, y, y_lengths, sid, tone, language):
        """
        x/tone/language: [B, T_text] ids; y: [B, SPEC_CHANNELS, T_frames]
        linear spectrogram. Returns the generated waveform slice plus
        everything the losses need.
        """
        b, t = x.shape
        bert = x.new_zeros((b, BERT_DIM, t), dtype=y.dtype)
        ja_bert = x.new_zeros((b, JA_BERT_DIM, t), dtype=y.dtype)

        g = self.emb_g(sid).unsqueeze(-1) if self.n_speakers > 0 else self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        g_p = None if self.use_vc else g

        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tone, language, bert, ja_bert, g=g_p)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        # Monotonic alignment search: log-likelihood of each frame's z_p under
        # each phone's prior N(m_p, exp(logs_p)), best monotonic path.
        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t_x]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4  # [b, t_y, t_x]
            if self.use_noise_scaled_mas:
                epsilon = torch.std(neg_cent) * torch.randn_like(neg_cent) * self.current_mas_noise_scale
                neg_cent = neg_cent + epsilon
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)  # [b, 1, t_y, t_x]
            attn = alignment.viterbi_decode(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        # Duration targets from the alignment, for both duration predictors.
        # Only `dp` is used at inference (sdp_ratio = 0 upstream and in C++),
        # but sdp is trained too so the checkpoint stays complete.
        w = attn.sum(2)  # [b, 1, t_x] frames per phone
        l_length_sdp = self.sdp(x, x_mask, w, g=g) / torch.sum(x_mask)
        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(x, x_mask, g=g)
        l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)
        l_length = l_length_dp + l_length_sdp

        # Expand the phone-level prior to frame level along the alignment.
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.random_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return dict(
            wav=o,
            l_length=l_length,
            attn=attn,
            ids_slice=ids_slice,
            x_mask=x_mask,
            y_mask=y_mask,
            z_p=z_p,
            m_p=m_p,
            logs_p=logs_p,
            logs_q=logs_q,
        )


def build_generator(segment_frames=SEGMENT_FRAMES):
    """Same constructor call as tiny_tts/infer.py and export_weights_and_vectors.py."""
    return TrainableSynthesizer(
        len(symbols), SPEC_CHANNELS, segment_frames, n_speakers=N_SPEAKERS, **MODEL_PARAMS
    )


def load_pretrained(net_g, path=None):
    """
    Loads G.pth (default: the HuggingFace backtracking/tiny-tts checkpoint the
    shipped weights come from -- not the stale one committed in the upstream
    repo, see docs/research.md). Returns (missing, unexpected) key lists.
    """
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("backtracking/tiny-tts", "G.pth")
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    own = net_g.state_dict()
    filtered = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    mismatched = [k for k in state if k in own and own[k].shape != state[k].shape]
    missing, unexpected = net_g.load_state_dict(filtered, strict=False)
    return list(missing) + mismatched, list(unexpected)
