"""
Linear/mel spectrograms for training. STFT/mel parameters come from an
AudioConfig (audio_config.py); for the default 44.1 kHz that's exactly
upstream's tiny_tts.utils.config (n_fft = win = 2048, hop 512 -> 1025 linear
bins for enc_q, 128 mel bins for the reconstruction loss).

Adapted from MeloTTS / VITS mel_processing.py (MIT License,
Copyright (c) 2021 Jaehyeon Kim, Copyright (c) 2024 MyShell.ai).
"""
import torch
import torchaudio

MEL_FMIN = 0.0

_hann_window = {}
_mel_basis = {}


def _window(win_size, y):
    key = (win_size, y.dtype, y.device)
    if key not in _hann_window:
        _hann_window[key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
    return _hann_window[key]


def _mel_fb(cfg, spec):
    key = (cfg.sample_rate, cfg.n_fft, cfg.n_mels, spec.dtype, spec.device)
    if key not in _mel_basis:
        # Slaney-style filterbank, same as librosa.filters.mel's default.
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=cfg.n_fft // 2 + 1,
            f_min=MEL_FMIN,
            f_max=cfg.sample_rate / 2,
            n_mels=cfg.n_mels,
            sample_rate=cfg.sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        _mel_basis[key] = fb.T.to(dtype=spec.dtype, device=spec.device)  # [n_mels, n_freqs]
    return _mel_basis[key]


def spectrogram_torch(cfg, y):
    """y: [B, T] audio in [-1, 1] -> [B, n_fft/2+1, T/hop] linear magnitude."""
    pad = (cfg.n_fft - cfg.hop_length) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=_window(cfg.win_length, y),
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)


def spec_to_mel_torch(cfg, spec):
    """Linear magnitude [B, n_freqs, frames] -> log-compressed mel [B, n_mels, frames]."""
    mel = torch.matmul(_mel_fb(cfg, spec), spec)
    return torch.log(torch.clamp(mel, min=1e-5))


def mel_spectrogram_torch(cfg, y):
    return spec_to_mel_torch(cfg, spectrogram_torch(cfg, y))
