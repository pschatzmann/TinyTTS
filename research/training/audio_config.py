"""
Sample-rate-dependent audio/model settings.

The released model runs at 44.1 kHz with a hop of 512 samples, and its
vocoder upsamples each spectrogram frame by exactly that much (8*8*2*2*2).
A lower output rate (16 kHz or 8 kHz is plenty for speech on a
microcontroller speaker) therefore isn't just a resampling step: the hop,
STFT size, mel bands and the vocoder's upsample stack all scale with it.
audio_config_for() derives all of them from the sample rate:

  sample rate   hop  frame    n_fft  mels  vocoder upsample rates
  44100         512  11.6 ms  2048   128   8 8 2 2 2   (the released model)
  22050/24000   256  11.6 ms  1024    80   8 8 2 2
  16000         256  16.0 ms  1024    80   8 8 2 2
  8000          128  16.0 ms   512    64   8 8 2

Dropping upsample stages (from the end, keeping upstream's channel halving)
is what makes a lower rate cheaper at inference: fewer output samples AND
fewer vocoder layers. Everything else (text encoder, flow, duration
predictor) is unchanged, so fine-tuning from the 44.1 kHz checkpoint still
reuses all of it plus the shared front of the vocoder; only the parts whose
shape changes (enc_q's input layer, the vocoder's final conv) start fresh --
load_pretrained() reports them.

Checkpoints store AudioConfig.to_dict() under "audio", which is how
synthesize.py and export_weights_and_vectors.py rebuild the right model.
"""
import math
from dataclasses import asdict, dataclass, field

import upstream  # noqa: F401

from tiny_tts.utils.config import MODEL_PARAMS, SAMPLING_RATE

DEFAULT_SAMPLE_RATE = SAMPLING_RATE  # 44100


@dataclass
class AudioConfig:
    sample_rate: int
    hop_length: int
    n_fft: int
    win_length: int
    n_mels: int
    upsample_rates: list = field(default_factory=list)
    upsample_kernel_sizes: list = field(default_factory=list)

    @property
    def spec_channels(self):
        return self.n_fft // 2 + 1

    def model_params(self):
        """upstream MODEL_PARAMS with this config's vocoder upsample stack."""
        params = dict(MODEL_PARAMS)
        params["upsample_rates"] = list(self.upsample_rates)
        params["upsample_kernel_sizes"] = list(self.upsample_kernel_sizes)
        return params

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def audio_config_for(sample_rate):
    sample_rate = int(sample_rate)
    if sample_rate == DEFAULT_SAMPLE_RATE:
        # Exactly the released model's settings (tiny_tts.utils.config).
        return AudioConfig(
            sample_rate=sample_rate,
            hop_length=512,
            n_fft=2048,
            win_length=2048,
            n_mels=128,
            upsample_rates=list(MODEL_PARAMS["upsample_rates"]),
            upsample_kernel_sizes=list(MODEL_PARAMS["upsample_kernel_sizes"]),
        )
    if not 4000 <= sample_rate <= 48000:
        raise ValueError(f"sample rate {sample_rate} outside the supported 4000..48000 Hz")
    # Power-of-two hop giving ~12-16 ms frames (the released model: 11.6 ms).
    hop = 2 ** max(7, min(9, round(math.log2(sample_rate * 0.0128))))
    n_stages = int(math.log2(hop)) - 4  # 8*8 = 2**6, then one x2 stage per extra power of two
    rates = [8, 8] + [2] * (n_stages - 2)
    # Upstream's kernel pattern: 2*rate for the x8 stages, 8 for the first
    # x2 stage, 2 for the rest.
    kernels = [16, 16] + [8, 2, 2, 2][: n_stages - 2]
    n_mels = 128 if sample_rate >= 44100 else (80 if sample_rate >= 16000 else 64)
    return AudioConfig(
        sample_rate=sample_rate,
        hop_length=hop,
        n_fft=4 * hop,
        win_length=4 * hop,
        n_mels=n_mels,
        upsample_rates=rates,
        upsample_kernel_sizes=kernels,
    )


def audio_config_from_checkpoint(ckpt):
    """AudioConfig a checkpoint was trained with (pre-sample-rate-option checkpoints and the released G.pth: 44.1 kHz)."""
    if isinstance(ckpt, dict) and "audio" in ckpt:
        return AudioConfig.from_dict(ckpt["audio"])
    return audio_config_for(DEFAULT_SAMPLE_RATE)
