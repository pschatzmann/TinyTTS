"""
Trains (or fine-tunes) the neural G2P fallback model behind
dictionary_model.bin: the single-layer 256-dim GRU encoder-decoder that
g2p_reference_numpy.py / DictionaryModel.h run.

The architecture is fixed by the C++ runtime, so this script reproduces it
exactly in PyTorch (same vocabularies, same GRU gate layout r|z|n, same
<s>=2 / </s>=3 conventions, greedy decode capped at 20 steps) and writes the
result in g2p_model.json's format. By default it starts from the shipped
weights (fine-tuning); --scratch starts from random weights.

Training data is tiny-tts/npm-package/cmudict.json (first pronunciation of each
a-z word with at most 19 phones, so the 20-step decode cap can reach </s>).
A small holdout measures generalization. Note that the slimmed dictionary is
built from the words the model gets wrong, so training on the whole
dictionary (--holdout 0) is legitimate for the final run -- every word the
model memorizes is a word the slim dictionary no longer has to carry.

Usage (from research/training/):
    python train_g2p.py                               # fine-tune, 3% holdout
    python train_g2p.py --epochs 60 --holdout 0       # final run on all words
    python train_g2p.py --scratch --epochs 80 --lr 2e-3

Output: runs/g2p/g2p_model.json (and g2p_best.pt / g2p_last.pt for --resume).
After training it automatically rebuilds everything that depends on the model
(skip with --no-rebuild): export_dictionary_model.py -> dictionary_model.bin,
compute_cmudict_exceptions.py -> cmudict_exceptions.json (a few minutes), and
export_cmudict.py -> cmudict.bin + cmudict_slim.bin, all written to
<out-dir>/export/. To ship the result, copy the .bin files to ../../data/default_*.bin
and run export_headers.py (-> src/TinyTTS/data/*.h).
"""
import argparse
import base64
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

HERE = os.path.dirname(os.path.abspath(__file__))  # research/training
RESEARCH = os.path.dirname(HERE)  # export scripts and tiny-tts live here
NPM = os.path.join(RESEARCH, "tiny-tts", "npm-package")
BASE_MODEL = os.path.join(NPM, "g2p_model.json")
CMUDICT = os.path.join(NPM, "cmudict.json")

PAD, UNK, SOS, EOS = 0, 1, 2, 3  # decoder (phoneme) ids; grapheme </s> is 2
G_EOS = 2
MAX_STEPS = 20
TENSORS = [
    "enc_emb", "enc_w_ih", "enc_w_hh", "enc_b_ih", "enc_b_hh",
    "dec_emb", "dec_w_ih", "dec_w_hh", "dec_b_ih", "dec_b_hh",
    "fc_w", "fc_b",
]


def b64_to_f32(entry):
    arr = np.frombuffer(base64.b64decode(entry["data"]), dtype="<f4")
    return arr.reshape(entry["shape"]).copy()


class G2P(nn.Module):
    def __init__(self, n_graphemes, n_phonemes, hidden=256):
        super().__init__()
        self.enc_emb = nn.Embedding(n_graphemes, hidden, padding_idx=0)
        self.enc_rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.dec_emb = nn.Embedding(n_phonemes, hidden, padding_idx=0)
        self.dec_rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_phonemes)

    def encode(self, src, src_len):
        packed = pack_padded_sequence(self.enc_emb(src), src_len.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, h = self.enc_rnn(packed)
        return h  # [1, B, H]

    def forward(self, src, src_len, dec_in):
        h = self.encode(src, src_len)
        out, _ = self.dec_rnn(self.dec_emb(dec_in), h)
        return self.fc(out)

    @torch.no_grad()
    def greedy(self, src, src_len):
        h = self.encode(src, src_len)
        tok = torch.full((src.shape[0], 1), SOS, dtype=torch.long, device=src.device)
        outs = []
        for _ in range(MAX_STEPS):
            out, h = self.dec_rnn(self.dec_emb(tok), h)
            tok = self.fc(out).argmax(-1)
            outs.append(tok)
        return torch.cat(outs, dim=1)  # [B, MAX_STEPS]

    def load_json(self, raw):
        t = {n: torch.from_numpy(b64_to_f32(raw[n])) for n in TENSORS}
        with torch.no_grad():
            self.enc_emb.weight.copy_(t["enc_emb"])
            self.dec_emb.weight.copy_(t["dec_emb"])
            for rnn, p in ((self.enc_rnn, "enc"), (self.dec_rnn, "dec")):
                rnn.weight_ih_l0.copy_(t[p + "_w_ih"])
                rnn.weight_hh_l0.copy_(t[p + "_w_hh"])
                rnn.bias_ih_l0.copy_(t[p + "_b_ih"])
                rnn.bias_hh_l0.copy_(t[p + "_b_hh"])
            self.fc.weight.copy_(t["fc_w"])
            self.fc.bias.copy_(t["fc_b"])

    def to_json(self, base):
        def enc(x):
            a = x.detach().cpu().numpy().astype("<f4")
            return {"shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode()}

        out = dict(base)  # keeps graphemes, phonemes, homograph2features
        out["enc_emb"], out["dec_emb"] = enc(self.enc_emb.weight), enc(self.dec_emb.weight)
        for rnn, p in ((self.enc_rnn, "enc"), (self.dec_rnn, "dec")):
            out[p + "_w_ih"], out[p + "_w_hh"] = enc(rnn.weight_ih_l0), enc(rnn.weight_hh_l0)
            out[p + "_b_ih"], out[p + "_b_hh"] = enc(rnn.bias_ih_l0), enc(rnn.bias_hh_l0)
        out["fc_w"], out["fc_b"] = enc(self.fc.weight), enc(self.fc.bias)
        return out


def load_data(g2idx, p2idx):
    with open(CMUDICT, "r", encoding="utf-8") as f:
        cmu = json.load(f)
    data = []
    for word, phones in cmu.items():
        w = word.lower()
        if not (w.isascii() and w.isalpha()) or len(phones) > MAX_STEPS - 1:
            continue
        if any(p not in p2idx for p in phones):
            continue
        src = [g2idx[c] for c in w] + [G_EOS]
        tgt = [p2idx[p] for p in phones] + [EOS]
        data.append((w, src, tgt))
    return data


def collate(items, device):
    B = len(items)
    ls = [len(s) for _, s, _ in items]
    lt = [len(t) for _, _, t in items]
    src = torch.zeros(B, max(ls), dtype=torch.long)
    tgt = torch.zeros(B, max(lt), dtype=torch.long)
    dec_in = torch.zeros(B, max(lt), dtype=torch.long)
    for i, (_, s, t) in enumerate(items):
        src[i, : len(s)] = torch.tensor(s)
        tgt[i, : len(t)] = torch.tensor(t)
        dec_in[i, 0] = SOS
        dec_in[i, 1 : len(t)] = torch.tensor(t[:-1])
    return src.to(device), torch.tensor(ls), dec_in.to(device), tgt.to(device)


@torch.no_grad()
def accuracy(model, items, device, bs=1024):
    """Exact-match word accuracy of the real greedy decoder (what runs on device)."""
    model.eval()
    ok = 0
    for i in range(0, len(items), bs):
        chunk = items[i : i + bs]
        src, ls, _, _ = collate(chunk, device)
        pred = model.greedy(src, ls).cpu().tolist()
        for (_, _, tgt), p in zip(chunk, pred):
            if EOS in p:
                p = p[: p.index(EOS)]
            ok += p == tgt[:-1]
    return ok / max(1, len(items))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs", "g2p"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4, help="peak LR; decays linearly to 5%%")
    ap.add_argument("--holdout", type=float, default=0.03, help="fraction of words held out (0 = none)")
    ap.add_argument("--scratch", action="store_true", help="random init instead of the shipped weights")
    ap.add_argument("--resume", action="store_true", help="continue from g2p_last.pt in --out-dir")
    ap.add_argument("--log-every", type=int, default=50, help="print running loss every N steps")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="skip rebuilding dictionary_model.bin / cmudict_exceptions.json / cmudict*.bin after training")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    with open(BASE_MODEL, "r", encoding="utf-8") as f:
        base = json.load(f)
    g2idx = {g: i for i, g in enumerate(base["graphemes"])}
    p2idx = {p: i for i, p in enumerate(base["phonemes"])}
    hidden = base["enc_w_hh"]["shape"][1]

    model = G2P(len(g2idx), len(p2idx), hidden)
    if not args.scratch:
        model.load_json(base)
    model.to(device)

    data = load_data(g2idx, p2idx)
    rng = random.Random(1234)  # fixed split so holdout is stable across runs
    rng.shuffle(data)
    n_hold = int(len(data) * args.holdout)
    hold, train = data[:n_hold], data[n_hold:]
    print(f"{len(train)} train words, {len(hold)} holdout words, device {device}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.05, 1 - s / max(1, total_steps)))

    start_epoch, best = 0, -1.0
    last_path = os.path.join(args.out_dir, "g2p_last.pt")
    best_path = os.path.join(args.out_dir, "g2p_best.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best = ck["epoch"], ck["best"]
        print(f"resumed at epoch {start_epoch}")

    eval_sub = train[:5000]
    if start_epoch == 0:
        print(f"initial: train-sample acc {accuracy(model, eval_sub, device):.4f}"
              + (f", holdout acc {accuracy(model, hold, device):.4f}" if hold else ""))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        random.shuffle(train)
        t0, loss_sum, n = time.time(), 0.0, 0
        steps_per_epoch = (len(train) + args.batch_size - 1) // args.batch_size
        win_loss, win_n = 0.0, 0
        for i in range(0, len(train), args.batch_size):
            src, ls, dec_in, tgt = collate(train[i : i + args.batch_size], device)
            logits = model(src, ls, dec_in)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=PAD)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item()
            win_loss += loss.item()
            n += 1
            win_n += 1
            if n % args.log_every == 0:
                el = time.time() - t0
                eta = el / n * (steps_per_epoch - n)
                print(f"  epoch {epoch + 1} step {n}/{steps_per_epoch} loss {win_loss / win_n:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {n / el:.1f} steps/s eta {eta:.0f}s", flush=True)
                win_loss, win_n = 0.0, 0
        tr = accuracy(model, eval_sub, device)
        ho = accuracy(model, hold, device) if hold else tr
        score = ho  # with --holdout 0 this is the train-sample accuracy
        print(f"epoch {epoch + 1}/{args.epochs} loss {loss_sum / n:.4f} train-acc {tr:.4f}"
              + (f" holdout-acc {ho:.4f}" if hold else "") + f" ({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch + 1, "best": best}, last_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    out_path = os.path.join(args.out_dir, "g2p_model.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model.to_json(base), f)
    print(f"best score {best:.4f}; wrote {out_path}")
    print("full-dictionary accuracy:", f"{accuracy(model, data, device):.4f}")

    if args.no_rebuild:
        print("skipping rebuild (--no-rebuild); run the export steps from the docstring manually")
        return
    # The slim dictionary holds exactly the words this model gets wrong, so it must be
    # regenerated from the new model; dictionary_model.bin has to match it.
    export_dir = os.path.join(args.out_dir, "export")
    env = dict(os.environ, TINYTTS_G2P_OUT=export_dir)  # keep the tracked files in research/ untouched
    for script, extra in (("export_dictionary_model.py", [out_path]),
                          ("compute_cmudict_exceptions.py", [out_path]),
                          ("export_cmudict.py", [])):
        print(f"\n=== {script} ===", flush=True)
        subprocess.run([sys.executable, os.path.join(RESEARCH, script), *extra], check=True, cwd=RESEARCH, env=env)
    print(f"\nrebuilt dictionary_model.bin, cmudict_exceptions.json, cmudict.bin, cmudict_slim.bin in {export_dir}")
    print("To ship them: copy the .bin files to data/default_*.bin, then run research/export_headers.py")


if __name__ == "__main__":
    main()
