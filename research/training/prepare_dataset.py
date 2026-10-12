"""
Turns a speech dataset into training filelists for train.py.

Input: the LJSpeech layout -- `<input>/metadata.csv` with
`id|text[|normalized text]` lines and `<input>/wavs/<id>.wav`. That's LJSpeech
itself, and also the simplest way to bring your own recordings (one speaker,
any sample rate, mono or stereo).

Output (`<out>/`):
  wavs/<id>.wav      audio resampled to --sample-rate (default: the released
                     model's 44.1 kHz), mono, int16. The rate chosen here
                     defines the trained model's output rate -- see
                     audio_config.py; train.py picks it up from the wavs.
  train.txt, val.txt one utterance per line:
                     wav_path|speaker_id|LANG|phone_ids|tone_ids|lang_ids
                     (ids space-separated, already blank-interleaved)

Phonemization runs once here, not per training step, through the language's
frontend (frontends/), so training needs no G2P dependencies at all.

Example (a CPU-sized 500-utterance subset):
  python prepare_dataset.py --input data/LJSpeech-1.1 --out data/ljspeech --max-items 500
"""
import argparse
import csv
import os
import random
import sys
from multiprocessing import Pool

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upstream  # noqa: F401,E402

from audio_config import DEFAULT_SAMPLE_RATE, audio_config_for  # noqa: E402
from frontends import get_frontend  # noqa: E402

_frontend = None
_args = None
_audio = None


def _init_worker(args):
    global _frontend, _args, _audio
    _args = args
    _audio = audio_config_for(args.sample_rate)
    torch.set_num_threads(1)
    _frontend = get_frontend(args.lang)


def read_metadata(input_dir):
    rows = []
    with open(os.path.join(input_dir, "metadata.csv"), encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
            if not row:
                continue
            utt_id = row[0]
            # LJSpeech's 3rd column has numbers/abbreviations already spelled
            # out; fall back to the raw text for 2-column metadata.
            text = row[2] if len(row) > 2 and row[2].strip() else row[1]
            rows.append((utt_id, text.strip()))
    return rows


def process(item):
    utt_id, text = item
    src = os.path.join(_args.input, "wavs", utt_id + ".wav")
    dst = os.path.join(_args.out, "wavs", utt_id + ".wav")
    try:
        if not os.path.exists(dst):
            audio, sr = sf.read(src, dtype="float32", always_2d=True)
            audio = torch.from_numpy(audio.mean(axis=1))
            if sr != _audio.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, _audio.sample_rate)
            audio = audio.clamp(-1.0, 1.0).numpy()
            sf.write(dst, (audio * 32767.0).astype(np.int16), _audio.sample_rate, subtype="PCM_16")
            n_samples = len(audio)
        else:
            info = sf.info(dst)
            if info.samplerate != _audio.sample_rate:
                return None, f"{utt_id}: existing {dst} is {info.samplerate} Hz, not {_audio.sample_rate} -- use a new --out dir"
            n_samples = info.frames
        phone_ids, tone_ids, lang_ids = _frontend.text_to_ids(text)
    except Exception as e:  # keep going -- one bad clip shouldn't kill a 13k-clip run
        return None, f"{utt_id}: {e}"

    seconds = n_samples / _audio.sample_rate
    if not (_args.min_seconds <= seconds <= _args.max_seconds):
        return None, f"{utt_id}: {seconds:.2f}s outside [{_args.min_seconds}, {_args.max_seconds}]"
    if len(phone_ids) > _args.max_phones:
        return None, f"{utt_id}: {len(phone_ids)} phones > {_args.max_phones}"
    # Every spectrogram frame must be able to align to a phone (MAS needs
    # frames >= phones).
    if n_samples // _audio.hop_length < len(phone_ids):
        return None, f"{utt_id}: fewer frames than phones"
    line = "|".join(
        [
            os.path.abspath(dst),
            str(_args.speaker_id),
            _args.lang.upper(),
            " ".join(map(str, phone_ids)),
            " ".join(map(str, tone_ids)),
            " ".join(map(str, lang_ids)),
        ]
    )
    return line, None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="dataset dir containing metadata.csv and wavs/")
    p.add_argument("--out", required=True, help="output dir for resampled wavs + filelists")
    p.add_argument("--lang", default="EN", help="upstream language code (default EN)")
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                   help=f"output sample rate of the model to train (default {DEFAULT_SAMPLE_RATE}, the released model's; "
                        "e.g. 16000 or 8000 for a cheaper microcontroller model -- see audio_config.py)")
    p.add_argument("--speaker-id", type=int, default=0, help="speaker id written to the filelist (default 0)")
    p.add_argument("--max-items", type=int, default=0, help="use only N randomly chosen utterances (0 = all)")
    p.add_argument("--val-items", type=int, default=20, help="utterances held out for val.txt (default 20)")
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=12.0)
    p.add_argument("--max-phones", type=int, default=400, help="max blank-interleaved phone ids")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    audio_config_for(args.sample_rate)  # validate before spawning workers
    os.makedirs(os.path.join(args.out, "wavs"), exist_ok=True)
    rows = read_metadata(args.input)
    random.Random(args.seed).shuffle(rows)
    if args.max_items:
        rows = rows[: args.max_items]
    print(f"processing {len(rows)} utterances at {args.sample_rate} Hz with {args.workers} workers ...")

    lines, skipped = [], []
    with Pool(args.workers, initializer=_init_worker, initargs=(args,)) as pool:
        for i, (line, err) in enumerate(pool.imap(process, rows, chunksize=8), 1):
            if line:
                lines.append(line)
            else:
                skipped.append(err)
            if i % 500 == 0:
                print(f"  {i}/{len(rows)}")

    for err in skipped[:10]:
        print("  skipped", err)
    if len(skipped) > 10:
        print(f"  ... and {len(skipped) - 10} more skipped")
    if len(lines) <= args.val_items:
        raise SystemExit(f"only {len(lines)} usable utterances -- need more than --val-items={args.val_items}")

    val, train = lines[: args.val_items], lines[args.val_items :]
    for name, subset in (("train.txt", train), ("val.txt", val)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            f.write("\n".join(subset) + "\n")
    print(f"wrote {len(train)} train / {len(val)} val utterances to {args.out} ({len(skipped)} skipped)")


if __name__ == "__main__":
    main()
