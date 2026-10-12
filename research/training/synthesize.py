"""
Quick listening check for a training checkpoint, in Python: text -> WAV via
the upstream model's infer(), with sdp_ratio=0 like the C++ runtime (which
only implements the deterministic duration predictor).

  python synthesize.py --checkpoint runs/ljspeech/G_5000.pth --text "Hello world." -o hello.wav
"""
import argparse
import os
import sys

import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upstream  # noqa: F401,E402

from frontends import get_frontend  # noqa: E402
from model_train import build_generator, load_pretrained  # noqa: E402
from tiny_tts.utils.config import SAMPLING_RATE  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", help="G_*.pth (default: the released HF checkpoint)")
    p.add_argument("--text", required=True)
    p.add_argument("--lang", default="EN")
    p.add_argument("--speaker-id", type=int, default=0)
    p.add_argument("--noise-scale", type=float, default=0.667)
    p.add_argument("--length-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--output", default="out.wav")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    net_g = build_generator()
    missing, _ = load_pretrained(net_g, args.checkpoint)
    if missing:
        print(f"warning: {len(missing)} tensors missing from checkpoint")
    net_g.eval()
    net_g.dec.remove_weight_norm()

    phone_ids, tone_ids, lang_ids = get_frontend(args.lang).text_to_ids(args.text)
    t = len(phone_ids)
    with torch.no_grad():
        audio, *_ = net_g.infer(
            torch.LongTensor([phone_ids]),
            torch.LongTensor([t]),
            torch.LongTensor([args.speaker_id]),
            torch.LongTensor([tone_ids]),
            torch.LongTensor([lang_ids]),
            torch.zeros(1, 1024, t),
            torch.zeros(1, 768, t),
            noise_scale=args.noise_scale,
            length_scale=args.length_scale,
            sdp_ratio=0.0,
        )
    sf.write(args.output, audio[0, 0].numpy(), SAMPLING_RATE)
    print(f"wrote {args.output} ({audio.shape[-1] / SAMPLING_RATE:.2f}s)")


if __name__ == "__main__":
    main()
