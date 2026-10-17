# TinyTTS

[![Arduino Library](https://img.shields.io/badge/Arduino-Library-blue?logo=arduino&logoColor=white)](https://www.arduino.cc/reference/en/libraries/)
[![ESP-IDF Component](https://img.shields.io/badge/ESP--IDF-component-blue?logo=espressif&logoColor=white)](idf_component.yml)
[![CMake](https://img.shields.io/badge/CMake-supported-blue?logo=cmake&logoColor=white)](CMakeLists.txt)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)](https://opensource.org/licenses/Apache-2.0)

A header-only C++ port of [tronghieuit/tiny-tts](https://github.com/tronghieuit/tiny-tts) — a
~1.6M-parameter, VITS-style, end-to-end neural text-to-speech model — with **no external
inference-runtime dependency**, as an Arduino library. This is a proof of concept to evaluate if Machine Learning based TTS systems can be used on current Microcontrollers.

In addition you can build a command line program that you can run on your __desktop or microcomputer__.

**Status:** runs on ESP32-S3/P4 and the Sipeed Tang Nano 20K (far from real time on all of
them), and faster than real time on a desktop PC. A Milk-V Duo port that moves the flow and
the decoder to the board's TPU is complete on the PC side but hasn't been run on the board
yet. You can also train or fine-tune your own voice, including at a lower sample rate. See
[`docs/limitations.md`](docs/limitations.md) for the details.


## Sample audio

Generated `hello_world.wav` from this project:

[▶️ Play `hello_world.wav`](https://cdn.jsdelivr.net/gh/pschatzmann/TinyTTS@main/docs/assets/hello_world.wav)


## How it works

TinyTTS's model has four stages -- `text_encoder`, `flow`, `duration_predictor`, and
`decoder` -- all implemented as plain, hand-written C++. **There is no TFLite Micro, or any
other inference-runtime dependency, anywhere in this library.** Every stage runs in a single
pass with no fixed-input-shape window to hit, so `speak()` handles text of any length (see
`docs/text-input.md` for what you can feed it). Text becomes phonemes via `TextG2P`
(dictionary lookup → neural G2P fallback → character-level fallback), and an optional neural
fallback model (`DictionaryModel`) covers words the dictionary doesn't. See
`docs/architecture.md` for the mechanical breakdown of each stage.

The flow and the decoder can optionally be replaced by an external implementation
(`setFlow()`/`setVocoder()`); the Milk-V Duo build uses this to run them on its TPU
(`docs/milkv-duo.md`), and the Raspberry Pi Zero build runs both on its GPU with
hand-written QPU kernels (`docs/rpi-zero-gpu.md`). The Milk-V build is the only place an
inference runtime (the vendor's `libcviruntime`) is involved, and only in the desktop CLI
built for that board.

## Documentation

| Document | What's in it |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Why each of the four model stages is built the way it is, plus a data-flow diagram. |
| [`docs/requirements.md`](docs/requirements.md) | Board/library requirements -- flash/PSRAM sizing, board settings, the optional SIMD speedup. |
| [`data/`](data/README.md) | The model data as `.bin` files for loading from LittleFS/FFat/SD (or `tinytts --weights`), incl. the 16 kHz model. |
| [`docs/model-data.md`](docs/model-data.md) | Model data sizes, how to wire up weights/dictionary buffers, where the data comes from. |
| [`docs/esp-idf.md`](docs/esp-idf.md) | Building TinyTTS as a plain ESP-IDF component instead of an Arduino library. |
| [`docs/tangnano20k.md`](docs/tangnano20k.md) | Running on the Sipeed Tang Nano 20K FPGA board, including its on-chip INT8 accelerator. |
| [`docs/milkv-duo.md`](docs/milkv-duo.md) | Milk-V Duo 256M / Duo S: the decoder on the SG2002/SG2000 TPU, in fixed-size windows. |
| [`docs/rpi-zero-gpu.md`](docs/rpi-zero-gpu.md) | Raspberry Pi Zero: the flow and the decoder on the VideoCore IV GPU (`--qpu`) -- faster than real time. |
| [`docs/desktop.md`](docs/desktop.md) | The `desktop/` CLI (`tinytts`) -- building, installing, running, Unix piping. |
| [`docs/performance.md`](docs/performance.md) | Real-hardware optimization history: what worked, what didn't, and why. |
| [`docs/potential-improvements.md`](docs/potential-improvements.md) | What's still open for moving TinyTTS closer to real-time synthesis. |
| [`docs/phonemes.md`](docs/phonemes.md) | The phoneme symbols and stress tones the model takes, mapped to TinyTTSTools' phoneme ids. |
| [`docs/text-input.md`](docs/text-input.md) | What you can pass to `speak()`, and the one real constraint on splitting text across calls. |
| [`docs/limitations.md`](docs/limitations.md) | Status per platform and known limitations. |
| [`docs/research.md`](docs/research.md) | The offline Python tooling (`research/`) used to investigate, export, and validate the model. |
| [`docs/training.md`](docs/training.md) | Training or fine-tuning the model yourself (`research/training/`): LJSpeech, your own voice, adding languages, and the G2P fallback model (Docker and Kaggle supported). |

## Conclusions

As a proof of concept -- can a ML-based, VITS-style neural TTS model run on a current
microcontroller at all, with no external inference-runtime dependency -- three things came
out of it, in short:

- **Memory is sufficient, using a smaller, optimized model.** The recommended data set
  (weights + slimmed dictionary + neural G2P fallback model, see `docs/model-data.md`) is
  **4.21 MB total**, comfortably within a 16MB-flash module's budget, and real on-hardware
  PSRAM usage after `begin()` was only **~743 KB** -- flash/RAM was never the constraint.
- **Microcontrollers are too slow to generate audio in real time.** Even after a full
  optimization pass (INT8 weight quantization, tiled weight caching, SIMD-accelerated dot
  products), the fastest ESP32 measured (ESP32-P4) is still ~13x slower than real time
  with the default 44.1 kHz model -- ~8x with a model trained for 16 kHz output (see
  `docs/training.md`).
- **It runs perfectly well on modern desktop computers and on faster microcomputers** (e.g.
  a Raspberry Pi 4) -- both comfortably close to or faster than real time, on the exact same
  unmodified code.
- **A small GPU is enough for real time.** On a Raspberry Pi Zero W, running the flow and
  the decoder on its VideoCore IV GPU (`docs/rpi-zero-gpu.md`) takes "Hello world!" from
  ~15.7 s to ~0.58 s of synthesis -- faster than real time.
- **Open: a TPU.** Whether the Milk-V Duo's TPU makes a cheap Linux board usable
  (`docs/milkv-duo.md`) hasn't been measured on hardware yet.

Real, flashed-hardware numbers for `speak("Hello world!")`:

| Board | Model | Audio | Total time | Real-time factor |
|---|---|---:|---:|---:|
| ESP32-S3 | 44.1 kHz (default) | 1.49 s | ~33.9 s | ~22.8x slower than real time |
| ESP32-S3 | 16 kHz (trained, see `docs/training.md`) | 1.12 s | ~15.3 s | ~13.7x slower than real time |
| ESP32-P4 | 44.1 kHz (default) | 1.49 s | ~19.6 s | ~13.2x slower than real time |
| ESP32-P4 | 16 kHz (trained, see `docs/training.md`) | 1.12 s | ~9.2 s | ~8.2x slower than real time |
| Sipeed Tang Nano 20K (54MHz soft CPU, FPGA INT8 accelerator) | 44.1 kHz (default) | 1.49 s | ~34 min | ~1,400x slower than real time |
| Raspberry Pi Zero W (desktop CLI) | 44.1 kHz (default) | 1.49 s | ~15.68 s | ~11.5x slower than real time |
| Raspberry Pi Zero W (desktop CLI, flow and decoder on the GPU, `--qpu`) | 44.1 kHz (default) | 1.49 s | ~0.58 s (1.0 s with startup) | ~0.39x -- *faster* than real time |
| Raspberry Pi 4 Model B (desktop CLI, `--threads 4`) | 44.1 kHz (default) | 1.49 s | ~1.55 s | ~1.04x slower than real time |
| Desktop | 44.1 kHz (default) | 1.49 s | ~0.76 s | ~0.51x -- *faster* than real time |

The 16 kHz model is a different voice (LJSpeech) with its own speaking rate, hence the shorter
audio for the same text; compare the real-time factors. Its vocoder has one upsampling stage
fewer and produces ~2.75x fewer samples per second, which cuts the dominant vocoder stage from
~29.9 s to ~13.0 s on the ESP32-S3 (~17.5 s to ~7.9 s on the ESP32-P4). The ESP32-P4 numbers
are from a v3.2 chip (`ChipVariant=postv3`) on Arduino-ESP32 3.3.

See `docs/performance.md` for the complete optimization story and per-board/per-stage
breakdowns, including a real ESP32-S3 hardware measurement of the weight-tile-transpose
SIMD fusion (~46% faster total) and why the identical change helped far less on Pi 4's
NEON path (~6%).

## Attribution

Ported from [tronghieuit/tiny-tts](https://github.com/tronghieuit/tiny-tts), licensed
Apache-2.0. The model architecture is VITS-style (windowed relative-position multi-head
attention text encoder, normalizing-flow decoder, HiFi-GAN-style vocoder); see the upstream
repository for the original PyTorch implementation. Upstream publishes no training code;
`research/training/` adds it (see `docs/training.md`).

## License

Apache-2.0.
