"""
Training data loading: prepare_dataset.py filelists -> padded batches.

The linear spectrogram enc_q consumes is computed on the fly (in the
DataLoader workers) rather than cached: at 44.1 kHz with 1025 bins, a cache
for all of LJSpeech would be ~30 GB, while an STFT per clip is cheap next to a
training step.
"""
import random

import soundfile as sf
import torch

from audio_config import audio_config_for
from mel import spectrogram_torch


def read_filelist(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            wav, spk, lang, phones, tones, langs = line.split("|")
            items.append(
                dict(
                    wav=wav,
                    spk=int(spk),
                    lang=lang,
                    phones=[int(x) for x in phones.split()],
                    tones=[int(x) for x in tones.split()],
                    langs=[int(x) for x in langs.split()],
                )
            )
    return items


class TextAudioDataset(torch.utils.data.Dataset):
    """
    The sample rate (and with it hop/STFT/vocoder settings, see
    audio_config.py) comes from the prepared wavs themselves, i.e. from
    prepare_dataset.py --sample-rate; mixed rates in one filelist are an error.
    """

    def __init__(self, filelist):
        self.items = read_filelist(filelist)
        infos = [sf.info(it["wav"]) for it in self.items]  # headers only, no decoding
        rates = sorted({info.samplerate for info in infos})
        if len(rates) != 1:
            raise ValueError(f"{filelist}: wavs have mixed sample rates {rates} -- re-run prepare_dataset.py")
        self.audio = audio_config_for(rates[0])
        # Frame count per item, for length bucketing.
        self.frames = [info.frames // self.audio.hop_length for info in infos]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        audio, _ = sf.read(it["wav"], dtype="float32")
        audio = torch.from_numpy(audio)
        # Trim to a whole number of hops so frames * hop == samples exactly
        # (the decoder's total upsampling factor is the hop length).
        hop = self.audio.hop_length
        audio = audio[: (audio.shape[0] // hop) * hop]
        spec = spectrogram_torch(self.audio, audio.unsqueeze(0)).squeeze(0)  # [n_fft/2+1, frames]
        return dict(
            phones=torch.LongTensor(it["phones"]),
            tones=torch.LongTensor(it["tones"]),
            langs=torch.LongTensor(it["langs"]),
            spec=spec,
            wav=audio,
            spk=it["spk"],
        )


def collate(batch):
    b = len(batch)
    max_text = max(len(x["phones"]) for x in batch)
    max_spec = max(x["spec"].shape[1] for x in batch)
    max_wav = max(x["wav"].shape[0] for x in batch)
    n_freq = batch[0]["spec"].shape[0]

    phones = torch.zeros(b, max_text, dtype=torch.long)
    tones = torch.zeros(b, max_text, dtype=torch.long)
    langs = torch.zeros(b, max_text, dtype=torch.long)
    text_lengths = torch.zeros(b, dtype=torch.long)
    spec = torch.zeros(b, n_freq, max_spec)
    spec_lengths = torch.zeros(b, dtype=torch.long)
    wav = torch.zeros(b, 1, max_wav)
    wav_lengths = torch.zeros(b, dtype=torch.long)
    sid = torch.zeros(b, dtype=torch.long)

    for i, x in enumerate(batch):
        t = len(x["phones"])
        phones[i, :t] = x["phones"]
        tones[i, :t] = x["tones"]
        langs[i, :t] = x["langs"]
        text_lengths[i] = t
        f = x["spec"].shape[1]
        spec[i, :, :f] = x["spec"]
        spec_lengths[i] = f
        w = x["wav"].shape[0]
        wav[i, 0, :w] = x["wav"]
        wav_lengths[i] = w
        sid[i] = x["spk"]
    return dict(
        phones=phones,
        tones=tones,
        langs=langs,
        text_lengths=text_lengths,
        spec=spec,
        spec_lengths=spec_lengths,
        wav=wav,
        wav_lengths=wav_lengths,
        sid=sid,
    )


class BucketBatchSampler(torch.utils.data.Sampler):
    """
    Batches of similar-length utterances (less padding = less wasted compute,
    which matters a lot on CPU), in a different random order every epoch.
    """

    def __init__(self, lengths, batch_size, pool_batches=50, seed=1234):
        self.lengths = lengths
        self.batch_size = batch_size
        self.pool = batch_size * pool_batches
        self.rng = random.Random(seed)

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        self.rng.shuffle(idx)
        batches = []
        for s in range(0, len(idx), self.pool):
            chunk = sorted(idx[s : s + self.pool], key=lambda i: self.lengths[i])
            batches += [chunk[j : j + self.batch_size] for j in range(0, len(chunk), self.batch_size)]
        self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
