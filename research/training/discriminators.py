"""
HiFi-GAN multi-period + scale discriminator, as used by VITS / MeloTTS
(MIT License, Copyright (c) 2021 Jaehyeon Kim, Copyright (c) 2024 MyShell.ai).

Training-only: none of this is exported or needed at inference. `width`
scales every channel count -- the full-width discriminator (~46M params) is
far bigger than the 1.6M-param generator and dominates a CPU training step;
width=0.25 cuts it ~16x and is plenty to fine-tune against.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

LRELU_SLOPE = 0.1


def _ch(c, width):
    return max(1, int(round(c * width)))


class DiscriminatorP(nn.Module):
    def __init__(self, period, width=1.0, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        chs = [1] + [_ch(c, width) for c in (32, 128, 512, 1024, 1024)]
        pad = ((kernel_size - 1) // 2, 0)
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(chs[i], chs[i + 1], (kernel_size, 1), (stride if i < 4 else 1, 1), padding=pad))
                for i in range(5)
            ]
        )
        self.conv_post = norm_f(nn.Conv2d(chs[-1], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for layer in self.convs:
            x = F.leaky_relu(layer(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(nn.Module):
    def __init__(self, width=1.0, use_spectral_norm=False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        # (out_channels, kernel, stride, groups, padding)
        spec = [
            (16, 15, 1, 1, 7),
            (64, 41, 4, 4, 20),
            (256, 41, 4, 16, 20),
            (1024, 41, 4, 64, 20),
            (1024, 41, 4, 256, 20),
            (1024, 5, 1, 1, 2),
        ]
        layers, in_ch = [], 1
        for out, k, s, groups, pad in spec:
            out = _ch(out, width)
            # Keep the grouped convs valid at reduced width.
            groups = max(1, min(groups, in_ch, out))
            while in_ch % groups or out % groups:
                groups -= 1
            layers.append(norm_f(nn.Conv1d(in_ch, out, k, s, groups=groups, padding=pad)))
            in_ch = out
        self.convs = nn.ModuleList(layers)
        self.conv_post = norm_f(nn.Conv1d(in_ch, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for layer in self.convs:
            x = F.leaky_relu(layer(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, width=1.0, periods=(2, 3, 5, 7, 11), use_spectral_norm=False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorS(width, use_spectral_norm)] + [DiscriminatorP(p, width, use_spectral_norm=use_spectral_norm) for p in periods]
        )

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
