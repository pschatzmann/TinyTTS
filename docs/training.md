# Training: fine-tuning or training the model yourself

Upstream [tronghieuit/tiny-tts](https://github.com/tronghieuit/tiny-tts) publishes the model
and inference code, but no training code. `research/training/` fills that gap. It trains (or
fine-tunes) a checkpoint with exactly the same architecture, symbol table and tensor names as
the released `G.pth`, so the result goes through the existing export path
(`export_weights_and_vectors.py` → `export_headers.py`) and runs on the unchanged C++ library.

Like the rest of `research/`, this is PC-side Python tooling and not part of the Arduino library.

## What do you want to train?

| Goal | Where to start |
|---|---|
| A new or different **voice** (LJSpeech, your own recordings) | [Setup](#setup), then [Voice model](#voice-model) |
| A voice that is **cheap to synthesize** on a microcontroller (16 kHz) | [Choosing a sample rate](#choosing-a-sample-rate) |
| Training on a **free GPU** | [Training on a Kaggle GPU](#training-on-a-kaggle-gpu) |
| A better **pronunciation fallback** for words missing from the dictionary | [Training the G2P fallback model](#training-the-g2p-fallback-model) |
| **Another language** | [Adding a language](#adding-a-language) |

Contents: [Overview](#overview) · [Setup](#setup) · [Voice model](#voice-model) ·
[G2P fallback model](#training-the-g2p-fallback-model)

## Overview

Upstream is a renamed MeloTTS / Bert-VITS2 model. Its code still contains every module's
training path: the posterior encoder `enc_q`, the forward direction of the flows and of the
stochastic duration predictor, and monotonic alignment search (`tiny_tts/alignment`). Only the
top-level training `forward()`, the discriminator, the losses and the data pipeline are missing.
`research/training/` adds those on top of the upstream classes. It imports `tiny_tts.models`,
`tiny_tts.text.symbols` and `tiny_tts.utils.config` directly instead of copying them, so the
trained model can't drift from what the C++ port implements.

| File | What it does |
|---|---|
| `prepare_dataset.py` | LJSpeech-layout dataset (`metadata.csv` + `wavs/`) → audio resampled to `--sample-rate` (default 44.1 kHz, see "Choosing a sample rate") + `train.txt`/`val.txt` filelists. Phonemization happens here, once. |
| `frontends/` | Text → phone/tone/language ids, one module per language. `en.py` is upstream's own English pipeline (CMU dictionary + g2p_en fallback). |
| `audio_config.py` | Derives hop, STFT, mel and vocoder layout from the output sample rate (see "Choosing a sample rate"). |
| `model_train.py` | `TrainableSynthesizer`: upstream `VoiceSynthesizer` plus MeloTTS's training `forward()` (MAS alignment, duration losses, random decoder slices). Adds no parameters. |
| `discriminators.py`, `losses.py` | HiFi-GAN multi-period/scale discriminator and the VITS losses (mel L1, KL, duration, adversarial, feature matching). |
| `train.py` | The training loop: CPU or CUDA, resumable, writes `G_<step>.pth` checkpoints and sample WAVs. |
XX | Trains the neural G2P fallback model (`dictionary_model.bin`), not the voice, see "Training the G2P fallback model". |
| `synthesize.py` | Text → WAV from any checkpoint, in Python, for a quick listen. |
| `configs/en_ljspeech.json` | CPU-friendly defaults for fine-tuning on LJSpeech. |
| `configs/en_ljspeech16k_gpu.json` | GPU defaults for the 16 kHz LJSpeech model (batch 16, full-size discriminator). |
| `kaggle/tinytts_ljspeech16k.ipynb` | Kaggle notebook that trains the 16 kHz LJSpeech model on a GPU, see "Training on a Kaggle GPU". |
| `kaggle/tinytts_g2p.ipynb` | Kaggle notebook for the G2P fallback model, see "Training the G2P fallback model". |
| `docker/`, `compose.yaml` | Docker image with all of the above preinstalled (CPU or CUDA), see "Alternative: Docker". |

Training only touches what the C++ runtime uses: `enc_p`, `flow`, `dp` (the deterministic
duration predictor; the C++ runtime always uses it, never `sdp`), `emb_g` and `dec`. Two more
parts are trained but never exported: `enc_q`, which is only used during training, and `sdp`,
which is trained so that checkpoints stay complete.

The G2P fallback model (`dictionary_model.bin`) is separate from the voice model. It has its own
script, `train_g2p.py`, and its own Kaggle notebook, described in
[Training the G2P fallback model](#training-the-g2p-fallback-model).

## Setup

```bash
cd research
git clone --depth 1 -b develop https://github.com/tronghieuit/tiny-tts.git tiny-tts   # if not there yet
python3.12 -m venv venv && source venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -r training/requirements.txt
```

Datasets, checkpoints and samples go under `research/training/data/` and
`research/training/runs/`. Both are gitignored.

### Alternative: Docker

`research/training/docker/` packages the whole environment into one image. The image contains
Python 3.12, a pinned torch, the upstream model code at a pinned commit, the NLTK data, the
tokenizer and the released `G.pth`. Nothing is downloaded at run time, and every machine runs
the identical setup. This is the easiest way to move a run onto a rented GPU (RunPod, vast.ai,
Lambda, …), because those services start containers directly. Colab and Kaggle don't run
Docker; there, use the plain setup above.

The scripts and their options are the same as in the plain setup. Only how you start them changes.
The image also covers the G2P fallback model (the `g2p` command, see "Training the G2P fallback model").
`./data` and `./runs` (relative to `research/training/`) are mounted into the container, so
datasets and checkpoints stay on the host. A run started in Docker can therefore be continued
outside it with `--resume`, and vice versa.

```bash
cd research/training
docker compose build train-cpu            # or train-gpu (CUDA 12.6 build)

docker compose run --rm train-cpu prepare --input data/LJSpeech-1.1 --out data/ljspeech
docker compose run --rm train-cpu train --config configs/en_ljspeech.json
docker compose run --rm train-cpu synthesize --checkpoint runs/ljspeech/G_5000.pth --text "Hello world." -o runs/hello.wav
docker compose run --rm train-cpu export --checkpoint runs/ljspeech/G_5000.pth --out-dir runs/ljspeech/export
```

The first argument picks the step (`prepare`, `train`, `synthesize`, `export`, `g2p`, or
`shell`). The remaining arguments are passed to the matching script unchanged. `g2p` trains the
G2P fallback model, see [below](#docker-and-kaggle).

- **GPU:** `train-gpu` needs an NVIDIA GPU and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  on the host. There the full-size discriminator is affordable: add `--disc-width 1.0`, and use
  a larger `--batch-size`. Docker on macOS can't use the Mac's GPU, so use `train-cpu` there.
  The CPU image also builds for arm64, e.g. Apple Silicon or a Raspberry Pi.
- **File ownership:** output files are written as UID/GID 1000 by default. If your IDs differ,
  run `export UID GID` before `docker compose run`.
- **Without compose:** build with `docker build -f research/training/docker/Dockerfile
  --build-arg TORCH_VARIANT=cpu -t tinytts-train:cpu research` from the repository root. Then
  run with `docker run --rm --shm-size=2g -v "$PWD/data:/research/training/data"
  -v "$PWD/runs:/research/training/runs" tinytts-train:cpu train ...`. Adding `--gpus all` and
  building with `TORCH_VARIANT=cu126` gives the GPU variant. The `--shm-size` matters, because
  the DataLoader workers exchange batches through shared memory.
- **Build arguments:**
  - `TORCH_VARIANT`: `cpu`, `cu126`, or `cu130` for the newest GPUs.
  - `UPSTREAM_COMMIT`: the upstream model code to use.
  - `BAKE_CHECKPOINT=0`: leaves `G.pth` out of the image; it is then downloaded on first use.

## Voice model

The steps below train the TTS model itself (`weights.bin`). The sections build on each other:
quick start, then options, then where to run, then how to use the result.

### Quick start: English LJSpeech

[LJSpeech](https://keithito.com/LJ-Speech-Dataset/) has 13,100 clips (about 24 h) from one
female speaker, at 22.05 kHz. They are resampled to the model's 44.1 kHz, so the trained voice
has no content above about 11 kHz.

```bash
cd research/training
mkdir -p data && curl -L https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 | tar -xj -C data

# Whole dataset (or e.g. --max-items 1000 for a quicker CPU experiment):
python prepare_dataset.py --input data/LJSpeech-1.1 --out data/ljspeech

python train.py --config configs/en_ljspeech.json \
    --train data/ljspeech/train.txt --val data/ljspeech/val.txt --run-dir runs/ljspeech
```

`train.py` prints the losses every `--log-every` steps. Every `--eval-every` steps it writes
`runs/ljspeech/samples/step_<N>.wav`, synthesized from the first validation sentence. Every
`--save-every` steps it saves `G_<N>.pth`/`D_<N>.pth`, keeping the newest
`--keep-checkpoints`. Interrupt the run whenever you like and continue it with the same
command plus `--resume`.

To listen to any checkpoint:

```bash
python synthesize.py --checkpoint runs/ljspeech/G_5000.pth --text "Hello world." -o hello.wav
```

### Choosing a sample rate

The released model outputs 44.1 kHz audio. For a microcontroller speaker, 16 kHz (or even
8 kHz) is usually enough, and it is cheaper to synthesize. The output rate is chosen once, when
the dataset is prepared:

```bash
python prepare_dataset.py --input data/LJSpeech-1.1 --out data/ljspeech16k --sample-rate 16000
python train.py --config configs/en_ljspeech.json --train data/ljspeech16k/train.txt \
    --val data/ljspeech16k/val.txt --run-dir runs/ljspeech16k
```

Everything after that follows automatically:
- `train.py` reads the rate from the prepared WAVs.
- Checkpoints record it.
- `synthesize.py` and `export_weights_and_vectors.py` rebuild the matching model.
- The exported `weights.bin` carries the rate, so the C++ library and `tinytts --weights` play
  it at the right speed. `TinyTTS::getAudioSampleRate()` reports it **after `begin()`**, so
  configure your I2S/audio output after `begin()`, as `examples/tts_i2s_output` does.

The rate isn't just a resampling step. The vocoder turns every spectrogram frame into exactly
one hop of samples, so the hop length, the STFT and mel settings, and the vocoder's stack of
upsampling layers all change with it (`research/training/audio_config.py`):

| Sample rate | Hop (frame) | Vocoder upsampling | Vocoder time per second of audio (desktop) |
|---|---|---|---:|
| 44100 (released model) | 512 (11.6 ms) | 8·8·2·2·2 | ~400 ms |
| 22050 / 24000 | 256 (11.6 / 10.7 ms) | 8·8·2·2 | – |
| 16000 | 256 (16 ms) | 8·8·2·2 | ~205 ms |
| 8000 | 128 (16 ms) | 8·8·2 | ~178 ms |

The time column is relative cost on the desktop CLI. A microcontroller is much slower in
absolute terms, but the ratios carry over. 16 kHz roughly halves vocoder time. 8 kHz saves
little more, because it keeps the same 62.5 frames per second, so the expensive early vocoder
stages stay identical. 8 kHz only drops the cheapest, last stage.

Fine-tuning from the released 44.1 kHz checkpoint still works at a lower rate. The text
encoder, flow and duration predictor are unaffected, and so are all vocoder stages except the
dropped ones. Only `enc_q`'s input layer and the vocoder's final conv change shape and start
fresh; `train.py` lists them at startup. Expect the mel loss to start higher than in a
44.1 kHz fine-tune, and to need more steps before the output sounds clean.

### Fine-tune vs. from scratch

- `--init pretrained` (the default) starts the generator from the released
  `backtracking/tiny-tts` `G.pth`, so text encoding, alignment and the vocoder already work.
  Training only has to move the voice to the new speaker, which takes far fewer steps.
- `--init scratch` starts from random weights. It is fully reproducible, but needs on the order of
  hundreds of thousands of steps (GPU-days) to produce intelligible speech.
- `--init-checkpoint G_x.pth` fine-tunes from one of your own checkpoints instead.

The discriminator always starts fresh, because upstream didn't publish one.
`--gan-start-step N` trains for the first N steps with the mel, KL and duration losses only.
This lets the generator adapt cheaply before the adversarial losses start.

### Training on a CPU

Measured on a 2-core/4-thread laptop CPU (Haswell-ULT, 7 GB RAM, torch 2.14 CPU build),
fine-tuning from the released checkpoint on LJSpeech:

| Setting | Discriminator | Time per step |
|---|---|---:|
| batch 2, standard discriminator (`--disc-width 1.0`) | 46.75M params | ~12.6 s |
| batch 2, `--disc-width 0.25` | 2.93M params | ~3.0 s |
| batch 4, mel/KL/duration only (before `--gan-start-step`) | – | ~3.2 s |
| batch 4, `--disc-width 0.25` (the `en_ljspeech.json` default) | 2.93M params | ~5.0 s |

At about 5 s per step, that's roughly 700 steps per hour, so `en_ljspeech.json`'s 20,000 steps
take about a day. `--threads` larger than the number of *physical* cores made things slower
(3.8 s vs. 3.0 s per step with 4 threads on 2 cores). torch's default thread count was already
the right choice. Integrated GPUs older than Intel Xe (Arc / Core Ultra) aren't supported by
PyTorch at all. On a CUDA GPU (e.g. a free Colab/Kaggle T4) the same scripts run unchanged and
much faster, and the full-width discriminator becomes affordable there.

What makes CPU training feasible:

- **Fine-tune; don't train from scratch.**
- **A smaller discriminator.** The standard HiFi-GAN discriminator has about 46M parameters
  (about 28 times the 1.6M-parameter generator) and dominates step time. `--disc-width 0.25`
  shrinks it about 16 times.
- **Small batches.** Use `--batch-size 4` and the default 32-frame (16,384-sample) decoder slice.
- **Freezing.** `--freeze dec` keeps the vocoder fixed if only the prosody or voice timbre of the
  text side should change. This is useful for a fast first pass, but a new speaker usually
  needs the decoder trained too.

### Training on a Kaggle GPU

[`research/training/kaggle/tinytts_ljspeech16k.ipynb`](../research/training/kaggle/tinytts_ljspeech16k.ipynb)
trains the 16 kHz LJSpeech model (`ljspeech16k`) on Kaggle's free GPUs, and ends with a
`weights.bin` for the C++ library. It clones this repository from GitHub (`main`), so push your
changes before running it.

1. On kaggle.com, create a notebook with *File → Import Notebook* and choose the `.ipynb` file
   (or its GitHub URL).
2. In *Settings*, set **Accelerator** to *GPU T4 x2* or *GPU P100*, and turn **Internet** *On*.
   Internet access needs a phone-verified Kaggle account.
3. Run it with *Save Version → Save & Run All (Commit)*. That runs in the background for up to
   12 hours with the browser closed, and keeps `/kaggle/working` as the version's output.

The notebook does everything else itself:
- **Setup:** installs the dependencies, but never replaces Kaggle's CUDA build of torch.
- **Upstream code:** fetches the upstream model code at the same pinned commit as the Docker image.
- **Dataset:** uses LJSpeech from an attached Kaggle dataset if there is one, and otherwise
  downloads it. It then prepares the data at 16 kHz, which takes about 20–30 minutes.
- **Training:** uses `configs/en_ljspeech16k_gpu.json`: batch 16, the full-size discriminator,
  and 100,000 steps in total.
- **Export:** writes `ljspeech16k_weights.bin` plus a test WAV to the output.

The settings at the top choose where a new run starts: the released checkpoint by default, or
one of your own `G_*.pth` files uploaded as a Kaggle dataset (`INIT_CHECKPOINT`). For example,
the laptop run's `G_40000.pth` lets the GPU continue from there. They also set the batch size and
the step and time limits.

**Sessions are limited to 12 hours, and the GPU quota is about 30 hours per week.**
`MAX_HOURS = 11` uses `train.py --max-hours`, which saves a checkpoint and stops cleanly in time
for the export. To continue, add the previous version's output to the notebook (*Add Input →
Notebook Output Files*) and run it again. The notebook finds the earlier checkpoints and the
prepared dataset, copies them back to the same paths, and resumes with `--resume`.

**Publishing as a Kaggle Model.** At the end, the notebook uploads the result as a new version
of the Kaggle Model `MODEL_HANDLE` (default `pschatzmann/tinytts/other/ljspeech16k`; change the
owner to your Kaggle user name). If the model doesn't exist yet, it is created as a private
model. Each version contains `ljspeech16k_weights.bin`, the `G_<step>.pth` checkpoint, a sample
WAV and a README. Uploading needs your Kaggle API token as a notebook secret named
`KAGGLE_API_TOKEN` (*Add-ons → Secrets*). Without it, the upload step only prints a hint, and the
files stay in the output. `PUBLISH_MODEL = False` turns publishing off. To download a version
from a shell:

```bash
kaggle models instances versions download pschatzmann/tinytts/other/ljspeech16k/<version> -p ./model --untar
```

The notebook was tested end to end on CPU with a small dataset, including resuming in a second
session and the publish step (with the upload itself stubbed out), but not yet on a Kaggle GPU.
Step times there are therefore still to be measured; check `s/step` in the log.

### Using the trained model

```bash
cd research
python export_weights_and_vectors.py --checkpoint training/runs/ljspeech/G_5000.pth --out-dir training/runs/ljspeech/export
```

Listen to it with the desktop CLI (see `docs/desktop.md`), without touching the shipped data:

```bash
./build/desktop/tinytts --weights research/training/runs/ljspeech/export/weights.bin -o hello.wav "Hello world!"
```

To compile it into a sketch alongside the default model, generate an additional header from it
(`src/TinyTTS/data/<name>_weights_data.h`, variable `<name>_weights`):

```bash
cd research
python export_headers.py --weights training/runs/ljspeech16k/export/weights.bin --name ljspeech16k
```

Then pass `tts.setWeights(ljspeech16k_weights, ljspeech16k_weights_len)` before `begin()`.
`examples/timing_benchmark` does this when `TINYTTS_BENCH_16K` is defined (uncomment it at the
top of the sketch). It prints the model's sample rate, the per-stage timing and the real-time
factor, so the 44.1 kHz and 16 kHz models can be compared on the same board.

To ship it as the library's built-in model, export into `research/` itself (omit `--out-dir`)
and regenerate the headers with `python export_headers.py`. The dictionary and the G2P
fallback model don't change, because they depend on the language and not on the voice.
Note that this also replaces `research/test_vectors.bin`, which `test/main.cpp` validates
against.

### Your own voice

Record mono WAV clips and write a `metadata.csv` with `clip_id|transcript` lines, using the
same layout as LJSpeech. Any sample rate works; 44.1 kHz or higher keeps the full bandwidth.
Clips of 2–10 s work best. Fine-tuning works from about an hour of clean audio. Then run
`prepare_dataset.py --input <your dir> --out data/myvoice` and train as above.

### Adding a language

The model is already multi-lingual by construction. It has a shared symbol table (Chinese,
Japanese, English, Korean, Spanish, French, German and Russian phoneme sets), a language
embedding with 10 slots and per-language tone ranges (`tiny_tts/text/symbols.py`). The
released checkpoint was trained on English only, so all other rows of those embeddings are
untrained. Adding a language takes four steps:

1. **Python frontend.** Add `research/training/frontends/<code>.py` with a
   `text_to_ids(text)` that returns phone ids from `tiny_tts.text.symbols` (unknown phones map
   to `UNK`), tones offset by `language_tone_start_map[<code>]`, the language id from
   `language_id_map[<code>]`, and blanks inserted exactly as `en.py` does. Then register it in
   `frontends/__init__.py`. For most European languages an IPA phonemizer such as espeak-ng
   produces symbols that already exist in the table.
2. **Dataset and training.** Run `prepare_dataset.py --lang <code>`, then fine-tune from the
   English checkpoint as above. The decoder and flow transfer well, and the new language's
   embeddings and phones get trained.
3. **C++ G2P.** `src/TinyTTS/TextG2P.h` is English-only: CMU dictionary, the neural G2P
   fallback, and EN language and tone ids. The runtime therefore needs an equivalent
   text-to-ids path for the new language that produces the same ids as the Python frontend.
   This is the part that requires C++ work. The model weights themselves export and run
   unchanged.
4. **Export.** Use the same `export_weights_and_vectors.py --checkpoint` path as for English.

## Training the G2P fallback model

`data/default_dictionary_model.bin` is not part of the voice model above. It is the small
neural grapheme-to-phoneme model (single-layer 256-dim GRU encoder-decoder, ~833K parameters)
that `DictionaryModel.h` uses for words missing from the dictionary.
`research/training/train_g2p.py` trains it with exactly the architecture, vocabularies and
greedy decode of the C++ runtime (and of `g2p_reference_numpy.py`), on the first pronunciation
of every a-z word in `cmudict.json`.

It needs the same [setup](#setup) as the voice training (torch and the upstream `tiny-tts`
checkout), but no dataset download: the training data is `cmudict.json` from that checkout.

### Run the training

```bash
cd research/training
python train_g2p.py                              # fine-tune the shipped weights, 3% holdout
python train_g2p.py --epochs 2                    # less memorization, better on unseen words
python train_g2p.py --scratch --epochs 80 --lr 2e-3
```

| Option | Meaning |
|---|---|
| (default) | Starts from the shipped weights. `--scratch` starts from random weights instead. |
| `--epochs N` | Default 5. The learning rate decays linearly over these epochs. See "How many epochs" below. |
| `--holdout F` | Fraction of words held out to measure generalization (default 0.03). The "best" checkpoint is the one with the best holdout accuracy. With `--holdout 0` there is no holdout, so the best checkpoint is chosen by training accuracy, which rewards memorizing. |
| `--patience N` | Stops after N epochs without a holdout improvement (default 3, 0 = off). |
| `--resume` | Continues an interrupted run from `g2p_last.pt`. |
| `--log-every N` | Prints the running loss every N steps (default 50). |
| `--no-rebuild` | Skips the automatic rebuild described below. |

It prints the running loss, learning rate, steps per second and ETA every `--log-every` steps.
After every epoch it prints the mean loss and the exact-match word accuracy on a training
sample and, with a holdout, on the held-out words. A CPU needs 2.5 to 5 minutes per epoch. CUDA
is used automatically if available.

### How many epochs

Fine-tuning overfits quickly, and more epochs are not better. Measured on a 3% holdout (3,466
words never used for training), starting from the shipped model at 69.6%:

| Run | Holdout accuracy | Accuracy on all dictionary words |
|---|---|---|
| 1 epoch | 72.9% | 75.8% |
| 5 epochs (default) | 71.6% | 79.3% |
| 30 epochs | 70.1% (best, in epoch 1) → 65.1% at the end | 73.2% for the best checkpoint |

Holdout accuracy measures words the model has never seen, which is what the model is for.
Accuracy on dictionary words rises with more training because the model memorizes them. That
makes the slim dictionary smaller, but it also makes the model worse on unseen words. Use 1
to 2 epochs for the best behavior on unseen words, and 5 as a compromise. The differences
between 1 and 5 epochs are small compared with the noise of a 3,466-word holdout, so don't
read too much into them. The 30-epoch run lost accuracy on both measures.

The output goes to `research/training/runs/g2p/`: `g2p_model.json` (same format as upstream's
`g2p_model.json`), plus `g2p_best.pt` and `g2p_last.pt`.

### Rebuild the dictionary

When training finishes, the script rebuilds everything that depends on the model. These are the
scripts in `research/`, run in this order:

1. `export_dictionary_model.py` → `dictionary_model.bin` (int8-quantized)
2. `compute_cmudict_exceptions.py` → `cmudict_exceptions.json` (takes about 10 minutes)
3. `export_cmudict.py` → `cmudict.bin`, `cmudict_slim.bin`

The outputs are written to `research/training/runs/g2p/export/`, so the files tracked in
`research/` stay untouched. The scripts honor the `TINYTTS_G2P_OUT` environment variable for
this; without it they write next to themselves, as before.

The slim dictionary must always be rebuilt from the model it ships with, because it contains
exactly the words that model gets wrong. To redo only the rebuild for an existing model (e.g.
after `--no-rebuild`), run the three scripts from `research/` with `TINYTTS_G2P_OUT` set, and
pass `training/runs/g2p/g2p_model.json` to the first two.

### Ship the result

Copy the files over the defaults and regenerate the C++ headers:

```bash
cd research
cp training/runs/g2p/export/dictionary_model.bin ../data/default_dictionary_model.bin
cp training/runs/g2p/export/cmudict.bin          ../data/default_cmudict.bin
cp training/runs/g2p/export/cmudict_slim.bin     ../data/default_cmudict_slim.bin
python export_headers.py                         # -> src/TinyTTS/data/*_data.h
```

Always replace `dictionary_model.bin` and `cmudict_slim.bin` together.

### Docker and Kaggle

**Docker.** The image contains the G2P data and the export scripts, so the `g2p` command does
the whole job (training and rebuild). Everything lands in the mounted `runs/` folder:

```bash
cd research/training
docker compose run --rm train-cpu g2p --epochs 5     # -> runs/g2p/export/*.bin
```

The options are the same as above. A CPU is enough for this model. `train-gpu` is faster.

**Kaggle.** [`research/training/kaggle/tinytts_g2p.ipynb`](../research/training/kaggle/tinytts_g2p.ipynb)
does the same on Kaggle. Import it like the voice notebook (see "Training on a Kaggle GPU") and
turn **Internet** on. A GPU is optional. The settings at the top choose the epochs, learning
rate, holdout and whether to start from scratch. It resumes from an earlier session's
`g2p/g2p_last.pt` when you add that output as an input. The output has the three `.bin` files in
`g2p/export/`, and the notebook can publish them as a Kaggle Model (`MODEL_HANDLE`, needs the
`KAGGLE_API_TOKEN` secret, as for the voice model).
