# TinyTTS

[![Arduino Library](https://img.shields.io/badge/Arduino-Library-blue?logo=arduino&logoColor=white)](https://www.arduino.cc/reference/en/libraries/)
[![ESP-IDF Component](https://img.shields.io/badge/ESP--IDF-component-blue?logo=espressif&logoColor=white)](idf_component.yml)
[![CMake](https://img.shields.io/badge/CMake-supported-blue?logo=cmake&logoColor=white)](CMakeLists.txt)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)](https://opensource.org/licenses/Apache-2.0)

A header-only C++ port of [tronghieuit/tiny-tts](https://github.com/tronghieuit/tiny-tts) — a
~1.6M-parameter, VITS-style, end-to-end neural text-to-speech model — with **no external
inference-runtime dependency**, as an Arduino library. This is a proof of concept to evaluate if Machine Learning based TTS systems can be used on current Microcontrollers.

```cpp
#include <TinyTTS.h>
#include "TinyTTS/data/default_weights_data.h"
#include "TinyTTS/data/default_cmudict_slim_data.h"
#include "TinyTTS/data/default_dictionary_model_data.h"
#include "AudioTools.h"

TinyTTS tts;
I2SStream i2s_out;

void setup() {
  // TinyTTS doesn't embed a model itself -- wire one in before begin().
  // See "Model data" below for loading from a File instead.
  tts.setWeights(default_weights, default_weights_len);
  tts.setDictionary(default_cmudict_slim, default_cmudict_slim_len);
  tts.setDictionaryModel(default_dictionary_model, default_dictionary_model_len);

  auto cfg = i2s_out.defaultConfig(TX_MODE);
  cfg.sample_rate = tts.getAudioSampleRate();
  cfg.channels = tts.getAudioChannels();
  cfg.bits_per_sample = tts.getAudioBitsPerSample();
  i2s_out.begin(cfg);

  tts.begin(i2s_out);
  tts.speak("Hello world!");
}

void loop() {}
```

See `examples/tts_i2s_output/` for the complete, board-setting-annotated version of this
sketch (it needs specific board settings to compile — see "Requirements" below).

## How it works

TinyTTS's model has four stages -- `text_encoder`, `flow`, `duration_predictor`, and
`decoder` -- all implemented as plain, hand-written C++. **There is no TFLite Micro, or any
other inference-runtime dependency, anywhere in this library.** Every stage runs in a single
pass with no fixed-input-shape window to hit, so `speak()` handles text of any length (see
`docs/text-input.md` for what you can feed it). Text becomes phonemes via `TextG2P`
(dictionary lookup → neural G2P fallback → character-level fallback), and an optional neural
fallback model (`DictionaryModel`) covers words the dictionary doesn't. See
`docs/architecture.md` for the mechanical breakdown of each stage.

## Requirements

- **Board**: a module with enough external flash and PSRAM for the model data (see "Model
  data sizes" below) — the chip die itself typically has neither; both come from the
  specific module, so check your module's actual flash/PSRAM size rather than assuming from
  the chip name alone.
- **Board settings**: whichever PSRAM mode matches your module (e.g. `PSRAM=opi`) is
  required regardless of how you load model data. If native-USB Serial doesn't work on your
  board out of the box, check whether it needs a USB-mode board setting. If you compile the
  example model data into flash as shown below, size the partition scheme to fit it (see the
  `partitions.csv` shipped alongside `examples/tts_i2s_output/` for a worked example, and
  adapt it for your own flash size).
- **Libraries**: just [`arduino-audio-tools`](https://github.com/pschatzmann/arduino-audio-tools)
  (for `I2SStream`/audio output — `TinyTTS` itself only needs a plain Arduino `Print`, so any
  audio-tools output class works, or your own `Print` implementation). No inference-runtime
  library (TFLite Micro or otherwise) is needed -- every model stage is hand-written C++.
- **No extra step needed for the ESP32-S3/P4 SIMD speedup.** `src/esp-dsp-dotprod/` ships a
  minimal, flattened vendored copy of three modules from
  [espressif/esp-dsp](https://github.com/espressif/esp-dsp) (upstream is an ESP-IDF
  component `arduino-cli` can't load directly) directly inside this library's own `src/`
  tree, so Arduino picks it up automatically like any other TinyTTS header — nothing to
  install separately. `Ops.h` detects it via `__has_include` and falls back to a plain
  scalar loop on non-ESP32/host builds; see `src/esp-dsp-dotprod/NOTICE.md` for what it is
  and `docs/performance.md` for the measured difference (a real ~1.5-2x on the affected ops).

### ESP-IDF

TinyTTS also builds as a plain ESP-IDF component -- point `EXTRA_COMPONENT_DIRS` (or an
`idf_component.yml` dependency) at this repo, `#include <TinyTTS.h>`, and link against it
like any other component. There's no Arduino dependency in that path, so use the portable
API instead of the `Print`/`File` conveniences: `setWeights(data, len)`/
`setDictionary(data, len)` and `begin(const AudioChunkFn&)` (see `TinyTTS.h`) -- the same
calls this project's own host tests use.

### Model data sizes

TinyTTS needs at least two data buffers, provided via setters (see "Model data" below) --
each can be compiled into flash (`setWeights(data, len)`, ...) or loaded at runtime from
LittleFS/FFat/SD into PSRAM (`setWeights(file)`, ...). These are the sizes for the model
this project ships example headers for (`src/TinyTTS/data/*_data.h`); use them to decide
what fits in flash on your module and what should come from storage instead.
`duration_predictor` and `decoder` have no data buffers of their own -- both are hand-written
C++, and their weights are folded into the attention weights buffer below, same as
`text_encoder`/`flow`.

**Recommended: slimmed dictionary + neural G2P fallback model** (what
`examples/tts_i2s_output/` actually uses) -- smaller *and* better out-of-dictionary coverage
than the full dictionary alone:

| Data | Setter | Size |
|---|---|---:|
| Weights (all four model stages) (`default_weights_data.h`) | `setWeights` | 2.20 MB |
| CMU dictionary, slimmed (`default_cmudict_slim_data.h`) | `setDictionary` | 1.07 MB |
| Neural G2P fallback model (`default_dictionary_model_data.h`) | `setDictionaryModel` | 0.95 MB |
| **Total (recommended three)** | | **4.21 MB** |

The slimmed dictionary only contains the ~30% of words neither the neural fallback model
nor the crude character-level fallback already predicts correctly on its own, so it
*requires* `default_dictionary_model` also being wired in (via `setDictionaryModel()`) --
used alone, most lookups would silently fall through to the crude character-level fallback
instead.

**Simpler alternative: full dictionary, no G2P model** -- one fewer setter to call, but
larger and falls back to crude character-level phonemes for any word the dictionary
doesn't cover:

| Data | Setter | Size |
|---|---|---:|
| Weights (all four model stages) (`default_weights_data.h`) | `setWeights` | 2.20 MB |
| CMU dictionary (`default_cmudict_data.h`) | `setDictionary` | 3.31 MB |
| **Total (simpler two)** | | **5.51 MB** |

See `docs/architecture.md` for why the sizes above are what they are.

An 8MB-flash module can't fit either combination compiled into flash alongside app code —
pick which pieces to embed and which to load from storage (the dictionary is the one worth
moving to external storage first). A 16MB-flash module can embed everything, as
`examples/tts_i2s_output/` does.

## Model data

**`TinyTTS.h` does not embed or auto-include a model itself.** The library ships example
model data (`src/TinyTTS/data/*_data.h`, an umbrella-included `TinyTTS/Data.h` away) for
convenience, but it's only used if your sketch explicitly includes it and wires it in via
setters called before `begin()` — that way a sketch that doesn't need this particular model
doesn't pay for it, and swapping in a different checkpoint/quantization/dictionary doesn't
mean fighting a built-in default.

Each setter has two forms:

```cpp
#include <TinyTTS.h>
#include "TinyTTS/Data.h"   // default_weights, default_cmudict -- both in namespace tinytts

// Borrowed -- a flash const array. The pointer must outlive the TinyTTS
// object; nothing is copied.
tts.setWeights(default_weights, default_weights_len);
tts.setDictionary(default_cmudict, default_cmudict_len);

// Owned -- reads a File (LittleFS/FFat/SD/...) fully into a new PSRAM
// buffer TinyTTS allocates and frees automatically (no manual cleanup).
File f = LittleFS.open("/weights.bin", "r");
tts.setWeights(f);

// Optional: the neural G2P fallback, for words not in the dictionary.
#include "TinyTTS/data/default_dictionary_model_data.h"  // not pulled in by TinyTTS/Data.h
tts.setDictionaryModel(default_dictionary_model, default_dictionary_model_len);

// Optional ALTERNATIVE to setDictionary(default_cmudict, ...) above -- use
// one or the other, and only if you're also calling setDictionaryModel()
// (see "Model data sizes" above for why).
#include "TinyTTS/data/default_cmudict_slim_data.h"
tts.setDictionary(default_cmudict_slim, default_cmudict_slim_len);
```

`research/` has the Python tooling that produces this data — `export_weights_and_vectors.py`
(every hand-written-C++ stage's weights + validation vectors), `export_cmudict.py`
(dictionary, both the full and slimmed variants), `export_dictionary_model.py` (neural G2P
fallback model), `compute_cmudict_exceptions.py` (determines which dictionary words the
fallback model needs an exception for, i.e. the slimmed dictionary's contents), and
`export_headers.py` (which turns all of the above into the `.h` files under
`src/TinyTTS/data/`) — see `docs/research.md`. Useful as a starting point for regenerating
any of these with different settings.

## Status / known limitations

- **Out-of-dictionary words fall back to crude character-level phonemes unless
  `setDictionaryModel()` is called.** The neural GRU fallback (`DictionaryModel`/
  `default_dictionary_model`) gives much better results for proper nouns and unusual words,
  but costs ~0.95MB more flash, so it's opt-in rather than always-on. The CMU dictionary
  covers the large majority of real English words either way.
- **Splitting text across multiple `speak()` calls must happen on word boundaries, never
  mid-word** -- see `docs/text-input.md` for what input shapes are safe (individual words,
  full sentences, multiple sentences, and multi-call word-boundary splits all work; a
  multi-call split that cuts a word in half mispronounces it).
- **This project tests and tunes for one board** (the `partitions.csv` shipped alongside
  the example, the board settings above, etc.), even though none of the model code itself
  is chip-specific. `DataBuffer`'s `File`-loading path uses ESP32's own PSRAM allocator
  directly, `library.properties` scopes the library to the `esp32` architecture family, and
  the model's multi-megabyte weight/dictionary data needs a module with real PSRAM -- so
  this isn't a "runs on any microcontroller" library, just one with no inference-runtime
  dependency within that family.

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
fallback model, see "Model data sizes" above) is **4.21 MB total**, well within a 16MB-flash
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
