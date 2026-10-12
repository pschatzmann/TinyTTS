"""
Trains / fine-tunes the TinyTTS (VITS-style) model on a prepare_dataset.py
filelist. Single process, runs on CPU or CUDA (auto-detected).

Checkpoints (`<run_dir>/G_<step>.pth`) use upstream's `{"model": state_dict}`
layout with exactly G.pth's keys, so they go straight into
`../export_weights_and_vectors.py --checkpoint` and from there into the C++
library.

Typical CPU fine-tune of the released English model on a dataset subset:
  python train.py --config configs/en_ljspeech.json \
      --train data/ljspeech/train.txt --val data/ljspeech/val.txt --run-dir runs/ljspeech

Resume: same command plus --resume (continues from the newest G_/D_ in --run-dir).
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upstream  # noqa: F401,E402

from data import BucketBatchSampler, TextAudioDataset, collate  # noqa: E402
from discriminators import MultiPeriodDiscriminator  # noqa: E402
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss  # noqa: E402
from mel import mel_spectrogram_torch, spec_to_mel_torch  # noqa: E402
from model_train import build_generator, load_pretrained  # noqa: E402
from tiny_tts.nn import commons  # noqa: E402
from tiny_tts.utils.config import HOP_LENGTH, SAMPLING_RATE, SEGMENT_FRAMES  # noqa: E402

DEFAULTS = dict(
    train=None,
    val=None,
    run_dir="runs/default",
    init="pretrained",
    init_checkpoint=None,
    resume=False,
    device="auto",
    threads=0,
    workers=2,
    seed=1234,
    batch_size=4,
    segment_frames=SEGMENT_FRAMES,
    max_steps=10000,
    learning_rate=2e-4,
    betas=[0.8, 0.99],
    eps=1e-9,
    lr_decay=0.999875,
    c_mel=45.0,
    c_kl=1.0,
    grad_clip=None,
    gan_start_step=0,
    disc_width=1.0,
    freeze=[],
    log_every=10,
    eval_every=500,
    save_every=1000,
    keep_checkpoints=3,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON file of defaults (keys = option names with underscores); CLI flags override it")
    p.add_argument("--train", help="train.txt filelist from prepare_dataset.py")
    p.add_argument("--val", help="val.txt filelist (used for eval samples)")
    p.add_argument("--run-dir", help="where checkpoints, samples and logs go")
    p.add_argument("--init", choices=["pretrained", "scratch"], help="start from the released checkpoint (default) or random weights")
    p.add_argument("--init-checkpoint", help="G_*.pth to start from instead of the HuggingFace backtracking/tiny-tts G.pth")
    p.add_argument("--resume", action="store_true", default=None, help="continue from the newest checkpoint in --run-dir")
    p.add_argument("--device", help="auto | cpu | cuda")
    p.add_argument("--threads", type=int, help="torch CPU threads (0 = all cores)")
    p.add_argument("--workers", type=int, help="DataLoader worker processes")
    p.add_argument("--seed", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--segment-frames", type=int, help=f"decoder training slice in spectrogram frames (x{HOP_LENGTH} samples)")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--lr-decay", type=float, help="ExponentialLR gamma, applied per epoch")
    p.add_argument("--c-mel", type=float)
    p.add_argument("--c-kl", type=float)
    p.add_argument("--gan-start-step", type=int, help="train without the discriminator (mel/KL/duration only) until this step")
    p.add_argument("--disc-width", type=float, help="discriminator channel-width multiplier (1.0 = standard HiFi-GAN; 0.25 is much faster on CPU)")
    p.add_argument("--freeze", nargs="*", help="generator parameter-name prefixes to keep fixed, e.g. dec enc_q")
    p.add_argument("--log-every", type=int)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--save-every", type=int)
    p.add_argument("--keep-checkpoints", type=int, help="how many G_/D_ pairs to keep (0 = all)")
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    for k, v in vars(args).items():
        if k != "config" and v is not None:
            cfg[k] = v
    if not cfg["train"]:
        p.error("--train is required (directly or via --config)")
    return argparse.Namespace(**cfg)


def checkpoint_steps(run_dir, prefix):
    steps = []
    for path in glob.glob(os.path.join(run_dir, f"{prefix}_*.pth")):
        m = re.search(rf"{prefix}_(\d+)\.pth$", path)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def save_checkpoint(args, step, epoch, net_g, net_d, optim_g, optim_d, sched_g, sched_d):
    g_path = os.path.join(args.run_dir, f"G_{step}.pth")
    torch.save(
        {"model": net_g.state_dict(), "optimizer": optim_g.state_dict(), "scheduler": sched_g.state_dict(), "step": step, "epoch": epoch},
        g_path,
    )
    torch.save(
        {"model": net_d.state_dict(), "optimizer": optim_d.state_dict(), "scheduler": sched_d.state_dict(), "step": step, "epoch": epoch},
        os.path.join(args.run_dir, f"D_{step}.pth"),
    )
    if args.keep_checkpoints:
        for old in checkpoint_steps(args.run_dir, "G")[: -args.keep_checkpoints]:
            for prefix in ("G", "D"):
                path = os.path.join(args.run_dir, f"{prefix}_{old}.pth")
                if os.path.exists(path):
                    os.remove(path)
    print(f"saved {g_path}", flush=True)


@torch.no_grad()
def write_sample(args, net_g, val_items, step, device):
    if not val_items:
        return
    net_g.eval()
    it = val_items[0]
    x = torch.LongTensor(it["phones"]).unsqueeze(0).to(device)
    tone = torch.LongTensor(it["tones"]).unsqueeze(0).to(device)
    lang = torch.LongTensor(it["langs"]).unsqueeze(0).to(device)
    t = x.shape[1]
    audio, *_ = net_g.infer(
        x,
        torch.LongTensor([t]).to(device),
        torch.LongTensor([it["spk"]]).to(device),
        tone,
        lang,
        torch.zeros(1, 1024, t, device=device),
        torch.zeros(1, 768, t, device=device),
        noise_scale=0.667,
        noise_scale_w=0.8,
        sdp_ratio=0.0,  # the C++ runtime only implements dp
    )
    os.makedirs(os.path.join(args.run_dir, "samples"), exist_ok=True)
    path = os.path.join(args.run_dir, "samples", f"step_{step}.wav")
    sf.write(path, audio[0, 0].cpu().numpy(), SAMPLING_RATE)
    print(f"wrote sample {path}", flush=True)
    net_g.train()


def main():
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    print(f"device {device}, {torch.get_num_threads()} CPU threads", flush=True)

    train_set = TextAudioDataset(args.train)
    val_items = TextAudioDataset(args.val).items if args.val else []
    loader = torch.utils.data.DataLoader(
        train_set,
        batch_sampler=BucketBatchSampler(train_set.frames, args.batch_size, seed=args.seed),
        collate_fn=collate,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    print(f"{len(train_set)} training utterances, {len(loader)} batches/epoch", flush=True)

    net_g = build_generator(args.segment_frames).to(device)
    net_d = MultiPeriodDiscriminator(width=args.disc_width).to(device)

    step, epoch = 0, 0
    resume_steps = checkpoint_steps(args.run_dir, "G") if args.resume else []
    if not resume_steps and args.init == "pretrained":
        missing, unexpected = load_pretrained(net_g, args.init_checkpoint)
        print(f"initialized from {args.init_checkpoint or 'HF backtracking/tiny-tts G.pth'}", flush=True)
        if missing:
            print(f"  {len(missing)} generator tensors not in checkpoint (trained from scratch): "
                  f"{sorted({k.split('.')[0] for k in missing})}", flush=True)
        if unexpected:
            print(f"  {len(unexpected)} checkpoint tensors ignored", flush=True)

    for name, param in net_g.named_parameters():
        if any(name.startswith(prefix) for prefix in args.freeze):
            param.requires_grad_(False)
    g_params = [p for p in net_g.parameters() if p.requires_grad]
    print(f"generator {sum(p.numel() for p in net_g.parameters()) / 1e6:.2f}M params "
          f"({sum(p.numel() for p in g_params) / 1e6:.2f}M trainable), "
          f"discriminator {sum(p.numel() for p in net_d.parameters()) / 1e6:.2f}M params", flush=True)

    optim_g = torch.optim.AdamW(g_params, args.learning_rate, betas=args.betas, eps=args.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), args.learning_rate, betas=args.betas, eps=args.eps)
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=args.lr_decay)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=args.lr_decay)

    if resume_steps:
        last = resume_steps[-1]
        for prefix, net, optim, sched in (("G", net_g, optim_g, sched_g), ("D", net_d, optim_d, sched_d)):
            ckpt = torch.load(os.path.join(args.run_dir, f"{prefix}_{last}.pth"), map_location=device)
            net.load_state_dict(ckpt["model"])
            optim.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["scheduler"])
        step, epoch = last, ckpt["epoch"]
        print(f"resumed from step {step} (epoch {epoch})", flush=True)

    seg_samples = args.segment_frames * HOP_LENGTH
    net_g.train()
    net_d.train()
    t_last = time.time()
    while step < args.max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            net_g.update_mas_noise(step)
            use_gan = step >= args.gan_start_step

            out = net_g(
                batch["phones"], batch["text_lengths"], batch["spec"], batch["spec_lengths"],
                batch["sid"], batch["tones"], batch["langs"],
            )
            y_hat = out["wav"]
            mel = spec_to_mel_torch(batch["spec"])
            y_mel = commons.extract_segments(mel, out["ids_slice"], args.segment_frames)
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1))
            y = commons.extract_segments(batch["wav"], out["ids_slice"] * HOP_LENGTH, seg_samples)

            # Discriminator step.
            loss_disc = torch.zeros((), device=device)
            if use_gan:
                y_d_r, y_d_g, _, _ = net_d(y, y_hat.detach())
                loss_disc = discriminator_loss(y_d_r, y_d_g)
                optim_d.zero_grad()
                loss_disc.backward()
                commons.clip_grad_value_(net_d.parameters(), args.grad_clip)
                optim_d.step()

            # Generator step.
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * args.c_mel
            loss_kl = kl_loss(out["z_p"], out["logs_q"], out["m_p"], out["logs_p"], out["y_mask"]) * args.c_kl
            loss_dur = torch.sum(out["l_length"].float())
            loss_g = loss_mel + loss_kl + loss_dur
            loss_fm = loss_gen = torch.zeros((), device=device)
            if use_gan:
                _, y_d_g, fmap_r, fmap_g = net_d(y, y_hat)
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen = generator_loss(y_d_g)
                loss_g = loss_g + loss_fm + loss_gen
            optim_g.zero_grad()
            loss_g.backward()
            commons.clip_grad_value_(g_params, args.grad_clip)
            optim_g.step()
            step += 1

            if not torch.isfinite(loss_g):
                raise SystemExit(f"non-finite generator loss at step {step} -- lower --learning-rate or check the data")

            if step % args.log_every == 0:
                dt = (time.time() - t_last) / args.log_every
                t_last = time.time()
                print(
                    f"step {step} epoch {epoch} | g {loss_g.item():.3f} (mel {loss_mel.item():.3f} kl {loss_kl.item():.3f} "
                    f"dur {loss_dur.item():.3f} fm {loss_fm.item():.3f} gen {loss_gen.item():.3f}) | d {loss_disc.item():.3f} "
                    f"| lr {optim_g.param_groups[0]['lr']:.2e} | {dt:.2f}s/step",
                    flush=True,
                )
            if args.eval_every and step % args.eval_every == 0:
                write_sample(args, net_g, val_items, step, device)
            if step % args.save_every == 0 or step >= args.max_steps:
                save_checkpoint(args, step, epoch, net_g, net_d, optim_g, optim_d, sched_g, sched_d)
            if step >= args.max_steps:
                break
        else:
            epoch += 1
            sched_g.step()
            sched_d.step()
    print("done", flush=True)


if __name__ == "__main__":
    main()
