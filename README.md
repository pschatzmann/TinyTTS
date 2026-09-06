# TinyTTS

[![Arduino Library](https://img.shields.io/badge/Arduino-Library-blue?logo=arduino&logoColor=white)](https://www.arduino.cc/reference/en/libraries/)
[![ESP-IDF Component](https://img.shields.io/badge/ESP--IDF-component-blue?logo=espressif&logoColor=white)](idf_component.yml)
[![CMake](https://img.shields.io/badge/CMake-supported-blue?logo=cmake&logoColor=white)](CMakeLists.txt)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)](https://opensource.org/licenses/Apache-2.0)

A header-only C++ port of [tronghieuit/tiny-tts](https://github.com/tronghieuit/tiny-tts) — a
~1.6M-parameter, VITS-style, end-to-end neural text-to-speech model — with **no external
inference-runtime dependency**, as an Arduino library. This is a proof of concept to evaluate if Machine Learning based TTS systems can be used on current Microcontrollers.

In addition you can build a command line program that you can run on your __desktop or microcomputer__.


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

## Documentation

| Document | What's in it |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Why each of the four model stages is built the way it is, plus a data-flow diagram. |
| [`docs/requirements.md`](docs/requirements.md) | Board/library requirements -- flash/PSRAM sizing, board settings, the optional SIMD speedup. |
| [`docs/model-data.md`](docs/model-data.md) | Model data sizes, how to wire up weights/dictionary buffers, where the data comes from. |
| [`docs/esp-idf.md`](docs/esp-idf.md) | Building TinyTTS as a plain ESP-IDF component instead of an Arduino library. |
| [`docs/desktop.md`](docs/desktop.md) | The `desktop/` CLI (`tinytts`) -- building, installing, running, Unix piping. |
| [`docs/performance.md`](docs/performance.md) | Real-hardware optimization history: what worked, what didn't, and why. |
| [`docs/potential-improvements.md`](docs/potential-improvements.md) | What's still open for moving TinyTTS closer to real-time synthesis. |
| [`docs/text-input.md`](docs/text-input.md) | What you can pass to `speak()`, and the one real constraint on splitting text across calls. |
| [`docs/limitations.md`](docs/limitations.md) | Status and known limitations. |
| [`docs/research.md`](docs/research.md) | The offline Python tooling (`research/`) used to investigate, export, and validate the model. |

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
  products), the fastest ESP32 measured is still ~19-32x slower than real time.
- **It runs perfectly well on modern desktop computers and on faster microcomputers** (e.g.
  a Raspberry Pi 5) -- both comfortably close to or faster than real time, on the exact same
  unmodified code.

Real, flashed-hardware numbers for `speak("Hello world!")` (1.49s of resulting audio):

| Board | Total time | vs. unoptimized baseline | Real-time factor |
|---|---:|---:|---:|
| ESP32-S3 (unoptimized baseline) | ~439.7 s | -- | ~296x slower than real time |
| ESP32-S3 (optimized) | ~47.1 s | ~9.3x faster | ~32x slower than real time |
| ESP32-P4 (optimized) | ~28.4 s | ~15.5x faster | ~19x slower than real time |
| Raspberry Pi Zero W (desktop CLI, for reference) | ~15.68 s | ~28.0x faster | ~11.5x slower than real time |
| Raspberry Pi 4 Model B (desktop CLI, for reference) | ~1.76 s | ~250x faster | ~1.29x slower than real time |
| Desktop (optimized, for reference) | ~0.76 s | ~581x faster | ~0.51x -- *faster* than real time |

See `docs/performance.md` for the complete optimization story, per-board/per-stage
breakdowns, and what was tried and didn't work (including a NEON SIMD prototype on Pi 4
that, surprisingly, didn't help).

## Attribution

Ported from [tronghieuit/tiny-tts](https://github.com/tronghieuit/tiny-tts), licensed
Apache-2.0. The model architecture is VITS-style (windowed relative-position multi-head
attention text encoder, normalizing-flow decoder, HiFi-GAN-style vocoder); see the upstream
repository for training details and the original PyTorch implementation.

## License

Apache-2.0.
