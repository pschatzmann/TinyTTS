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
| [`docs/realtime-audio-approaches.md`](docs/realtime-audio-approaches.md) | Approaches evaluated for moving TinyTTS closer to real-time synthesis. |
| [`docs/text-input.md`](docs/text-input.md) | What you can pass to `speak()`, and the one real constraint on splitting text across calls. |
| [`docs/limitations.md`](docs/limitations.md) | Status and known limitations. |
| [`docs/research.md`](docs/research.md) | The offline Python tooling (`research/`) used to investigate, export, and validate the model. |

## Conclusions

As a proof of concept -- can a ML-based, VITS-style neural TTS model run on a current
microcontroller at all, with no external inference-runtime dependency -- the answer is __yes,
but not yet in real time__. Real, flashed-hardware numbers for `speak("Hello world!")`
(1.49s of resulting audio), after a full optimization pass (INT8 weight quantization, tiled
weight caching, SIMD-accelerated dot products -- see `docs/performance.md` for the complete
story, including what was tried and didn't work):

| Board | Total time | vs. unoptimized baseline | Real-time factor |
|---|---:|---:|---:|
| ESP32-S3 (unoptimized baseline) | ~439.7 s | -- | ~296x slower than real time |
| ESP32-S3 (optimized) | ~47.1 s | ~9.3x faster | ~32x slower than real time |
| ESP32-P4 (optimized) | ~28.4 s | ~15.5x faster | ~19x slower than real time |
| Desktop (optimized, for reference) | ~0.76 s | ~581x faster | ~0.51x -- *faster* than real time |

The desktop row is the same optimized code, unmodified, running the host build (an
Intel Core i7-4650U laptop CPU @ 1.7GHz, nothing exotic) instead of an ESP32 -- included as a
reference point, not a target: no PSRAM/flash-fetch latency, no weight tiling/caching
pressure, no 32-bit-only FPU-bound microcontroller core to work around. It shows how much of
the *original* ~7.3-minute number was specifically an embedded-hardware problem (memory
latency, a comparatively weak FPU) rather than the model architecture being inherently slow.

That's a genuine ~9-16x improvement from optimization work alone, on the same hardware
class, with no change to model quality (every step verified against the reference PyTorch
model via cosine similarity/SNR, plus a runtime correctness check on real hardware for every
change). It's also still roughly 20-30x slower than real time on-device -- clearly usable for
short, pre-triggered utterances (a voice assistant's occasional spoken response, not live
conversation), not yet for anything latency-sensitive. The single biggest remaining lever
identified but not yet attempted is retraining the model at a lower native sample rate
(16kHz instead of 44.1kHz), projected at a further ~2.6x on top of the numbers above --
see `docs/performance.md` for the reasoning and what else was considered.

The on-device numbers above already include one free win worth calling out: the Arduino
ESP32 core compiles at `-Os` (size) by default, and switching to `-O2` measured ~15% faster
on real hardware for no source change and negligible flash cost -- pass
`--build-property "compiler.optimization_flags=-O2"` to `arduino-cli compile` (or your
build system's equivalent). See `docs/performance.md` for the measurement.

__Memory footprint__ is not the constraint here, which is itself a notable part of this proof
of concept: current microcontrollers have enough flash and RAM to hold a small, optimized
TTS model comfortably. The recommended data set (weights + slimmed dictionary + neural G2P
fallback model, see `docs/model-data.md`) is **4.21 MB total**, well within a 16MB-flash
module's PROGMEM/flash budget with plenty of room left for application code -- and on real
hardware, PSRAM actually used after `begin()` was only **~743 KB**, whether on an 8MB-PSRAM
ESP32-S3 module or a 32MB-PSRAM ESP32-P4 module (the rest of PSRAM stays free for the rest
of the application). The bottleneck this project ran into was entirely compute time, never
memory.

## Attribution

Ported from [tronghieuit/tiny-tts](https://github.com/tronghieuit/tiny-tts), licensed
Apache-2.0. The model architecture is VITS-style (windowed relative-position multi-head
attention text encoder, normalizing-flow decoder, HiFi-GAN-style vocoder); see the upstream
repository for training details and the original PyTorch implementation.

## License

Apache-2.0.
