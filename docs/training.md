# Training: fine-tuning or training the model yourself

Upstream [tronghieuit/tiny-tts](https://github.com/tronghieuit/tiny-tts) publishes the model
and inference code, but no training code. `research/training/` fills that gap: it trains (or
fine-tunes) a checkpoint with exactly the same architecture, symbol table and tensor names as
the released `G.pth`, so the result goes through the existing export path
(`export_weights_and_vectors.py` → `export_headers.py`) and runs on the unchanged C++ library.

Like the rest of `research/`, this is PC-side Python tooling and not part of the Arduino library.

## How it works

Upstream is a renamed MeloTTS / Bert-VITS2 model. Its code still contains every module's
training path: the posterior encoder `enc_q`, the forward direction of the flows and of the
stochastic duration predictor, and monotonic alignment search (`tiny_tts/alignment`). Only the
top-level training `forward()`, the discriminator, the losses and the data pipeline are missing.
`research/training/` adds those on top of the upstream classes. It imports `tiny_tts.models`,
`tiny_tts.text.symbols` and `tiny_tts.utils.config` directly instead of copying them, so the
trained model can't drift from what the C++ port implements.

| File | What it does |
|---|---|
| `prepare_dataset.py` | LJSpeech-layout dataset (`metadata.csv` + `wavs/`) → audio resampled to `--sample-rate` (default 44.1 kHz, see "Sample rate") + `train.txt`/`val.txt` filelists. Phonemization happens here, once. |
| `frontends/` | Text → phone/tone/language ids, one module per language. `en.py` is upstream's own English pipeline (CMU dictionary + g2p_en fallback). |
| `audio_config.py` | Derives hop, STFT, mel and vocoder layout from the output sample rate (see "Sample rate"). |
| `model_train.py` | `TrainableSynthesizer`: upstream `VoiceSynthesizer` plus MeloTTS's training `forward()` (MAS alignment, duration losses, random decoder slices). Adds no parameters. |
| `discriminators.py`, `losses.py` | HiFi-GAN multi-period/scale discriminator and the VITS losses (mel L1, KL, duration, adversarial, feature matching). |
| `train.py` | The training loop: CPU or CUDA, resumable, writes `G_<step>.pth` checkpoints and sample WAVs. |
| `synthesize.py` | Text → WAV from any checkpoint, in Python, for a quick listen. |
| `configs/en_ljspeech.json` | CPU-friendly defaults for fine-tuning on LJSpeech. |
| `docker/`, `compose.yaml` | Docker image with all of the above preinstalled (CPU or CUDA), see "Alternative: Docker". |

Training only touches what the C++ runtime uses: `enc_p`, `flow`, `dp` (the deterministic
duration predictor; the C++ runtime always uses it, never `sdp`), `emb_g` and `dec`. Two more
parts are trained but never exported: `enc_q`, which is only used during training, and `sdp`,
which is trained so that checkpoints stay complete.

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

The first argument picks the step (`prepare`, `train`, `synthesize`, `export`, or `shell`). The
remaining arguments are passed to the matching script unchanged.

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

## English: LJSpeech

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

### Sample rate

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

## Your own voice

Record mono WAV clips and write a `metadata.csv` with `clip_id|transcript` lines, using the
same layout as LJSpeech. Any sample rate works; 44.1 kHz or higher keeps the full bandwidth.
Clips of 2–10 s work best. Fine-tuning works from about an hour of clean audio. Then run
`prepare_dataset.py --input <your dir> --out data/myvoice` and train as above.

## Adding a language

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
