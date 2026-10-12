"""
Turns a speech dataset into training filelists for train.py.

Input: the LJSpeech layout -- `<input>/metadata.csv` with
`id|text[|normalized text]` lines and `<input>/wavs/<id>.wav`. That's LJSpeech
itself, and also the simplest way to bring your own recordings (one speaker,
any sample rate, mono or stereo).

Output (`<out>/`):
  wavs/<id>.wav      audio resampled to the model's 44.1 kHz, mono, int16
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

from frontends import get_frontend  # noqa: E402
from tiny_tts.utils.config import HOP_LENGTH, SAMPLING_RATE  # noqa: E402

_frontend = None
_args = None


def _init_worker(args):
    global _frontend, _args
    _args = args
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
            if sr != SAMPLING_RATE:
                audio = torchaudio.functional.resample(audio, sr, SAMPLING_RATE)
            audio = audio.clamp(-1.0, 1.0).numpy()
            sf.write(dst, (audio * 32767.0).astype(np.int16), SAMPLING_RATE, subtype="PCM_16")
            n_samples = len(audio)
        else:
            n_samples = sf.info(dst).frames
        phone_ids, tone_ids, lang_ids = _frontend.text_to_ids(text)
    except Exception as e:  # keep going -- one bad clip shouldn't kill a 13k-clip run
        return None, f"{utt_id}: {e}"

    seconds = n_samples / SAMPLING_RATE
    if not (_args.min_seconds <= seconds <= _args.max_seconds):
        return None, f"{utt_id}: {seconds:.2f}s outside [{_args.min_seconds}, {_args.max_seconds}]"
    if len(phone_ids) > _args.max_phones:
        return None, f"{utt_id}: {len(phone_ids)} phones > {_args.max_phones}"
    # Every spectrogram frame must be able to align to a phone (MAS needs
    # frames >= phones).
    if n_samples // HOP_LENGTH < len(phone_ids):
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
    p.add_argument("--speaker-id", type=int, default=0, help="speaker id written to the filelist (default 0)")
    p.add_argument("--max-items", type=int, default=0, help="use only N randomly chosen utterances (0 = all)")
    p.add_argument("--val-items", type=int, default=20, help="utterances held out for val.txt (default 20)")
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=12.0)
    p.add_argument("--max-phones", type=int, default=400, help="max blank-interleaved phone ids")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    os.makedirs(os.path.join(args.out, "wavs"), exist_ok=True)
    rows = read_metadata(args.input)
    random.Random(args.seed).shuffle(rows)
    if args.max_items:
        rows = rows[: args.max_items]
    print(f"processing {len(rows)} utterances with {args.workers} workers ...")

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
