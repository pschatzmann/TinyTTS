"""
Linear/mel spectrograms for training, with the STFT parameters taken from
upstream's tiny_tts.utils.config (44.1 kHz, n_fft = win = 2048, hop 512 ->
SPEC_CHANNELS = 1025 linear bins for enc_q, N_MEL_CHANNELS = 128 mel bins for
the reconstruction loss).

Adapted from MeloTTS / VITS mel_processing.py (MIT License,
Copyright (c) 2021 Jaehyeon Kim, Copyright (c) 2024 MyShell.ai).
"""
import torch
import torchaudio

import upstream  # noqa: F401

from tiny_tts.utils.config import FILTER_LENGTH, HOP_LENGTH, N_MEL_CHANNELS, SAMPLING_RATE

MEL_FMIN = 0.0
MEL_FMAX = None

_hann_window = {}
_mel_basis = {}


def _window(win_size, y):
    key = (win_size, y.dtype, y.device)
    if key not in _hann_window:
        _hann_window[key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    return _hann_window[key]


def _mel_fb(spec):
    key = (spec.dtype, spec.device)
    if key not in _mel_basis:
        # Slaney-style filterbank, same as librosa.filters.mel's default.
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=FILTER_LENGTH // 2 + 1,
            f_min=MEL_FMIN,
            f_max=MEL_FMAX if MEL_FMAX is not None else SAMPLING_RATE / 2,
            n_mels=N_MEL_CHANNELS,
            sample_rate=SAMPLING_RATE,
            norm="slaney",
            mel_scale="slaney",
        )
        _mel_basis[key] = fb.T.to(dtype=spec.dtype, device=spec.device)  # [n_mels, n_freqs]
    return _mel_basis[key]


def spectrogram_torch(y, n_fft=FILTER_LENGTH, hop_size=HOP_LENGTH, win_size=FILTER_LENGTH):
    """y: [B, T] audio in [-1, 1] -> [B, n_fft/2+1, frames] linear magnitude."""
    pad = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_window(win_size, y),
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)


def spec_to_mel_torch(spec):
    """Linear magnitude [B, n_freqs, frames] -> log-compressed mel [B, n_mels, frames]."""
    mel = torch.matmul(_mel_fb(spec), spec)
    return torch.log(torch.clamp(mel, min=1e-5))


def mel_spectrogram_torch(y):
    return spec_to_mel_torch(spectrogram_torch(y))
