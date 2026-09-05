# TinyTTS

[![Arduino Library](https://img.shields.io/badge/Arduino-Library-00979D?logo=arduino&logoColor=white)](https://www.arduino.cc/reference/en/libraries/)
[![CMake](https://img.shields.io/badge/CMake-supported-064F8C?logo=cmake&logoColor=white)](CMakeLists.txt)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A header-only C++ port of [tronghieuit/tiny-tts](https://github.com/tronghieuit/tiny-tts) — a
~1.6M-parameter, VITS-style, end-to-end neural text-to-speech model — for the **ESP32-S3 with
PSRAM**, as an Arduino library.

```cpp
#include <TinyTTS.h>
#include "TinyTTS/Data.h"   // example model data shipped with the library
#include "AudioTools.h"

TinyTTS<> tts;
I2SStream i2s_out;

void setup() {
  // TinyTTS doesn't embed a model itself -- wire one in before begin().
  // See "Model data" below for loading from a File instead.
  tts.setWeights(default_weights, default_weights_len);
  tts.setDictionary(default_cmudict, default_cmudict_len);
  tts.setDurationPredictorModel(default_duration_predictor_model, default_duration_predictor_model_len, 32);
  tts.setDecoderModel(default_decoder_model, default_decoder_model_len, 96);

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
sketch (it needs specific board settings to compile — see "Requirements" below). That
example actually uses the slimmed dictionary + neural G2P model combo instead of the full
dictionary shown above (see "Model data sizes" below for why that combo is now the smaller,
better-covering option) — this snippet shows the simpler four-setter form since it doesn't
need the "why five headers instead of one `TinyTTS/Data.h` include" explanation to make sense.

## How it works

TinyTTS's model has four stages: `text_encoder` and `flow` run as hand-written C++
(the ONNX→TFLite converter can't handle their attention module); `duration_predictor` and
`decoder` run through **TFLite Micro** (via
[`tflm_esp32`](https://github.com/eloquentarduino/tflm_esp32)), each in a fixed-size window
since TFLM has no input-resize API -- `speak()` streams audio out one decoder chunk at a
time and isn't limited to short text either (see `docs/text-input.md` for what you can feed
it). Text becomes phonemes via `TextG2P` (dictionary lookup → neural G2P fallback →
character-level fallback), and an optional neural fallback model (`DictionaryModel`) covers
words the dictionary doesn't. See `docs/architecture.md` for the full breakdown of each
stage and the quantization choices behind them.

## Requirements

- **Board**: an ESP32-S3 *module* with enough external flash and PSRAM — the S3 die itself
  has neither; both come from the specific module, and "ESP32-S3" alone doesn't tell you
  how much of either you have. For example, an N8R8 module has 8MB flash/8MB PSRAM, an N4R2
  has 4MB flash/2MB PSRAM.
- **Board settings**: `PSRAM=opi` (or whichever PSRAM mode matches your module) is required
  regardless of how you load model data. `USBMode=hwcdc,CDCOnBoot=cdc` is needed for
  native-USB Serial to work on most ESP32-S3 boards at all. If you compile the example
  model data into flash as shown below, you additionally need `FlashSize=16M` +
  `PartitionScheme=custom` (using the `partitions.csv` shipped alongside
  `examples/tts_i2s_output/` — see that file's comments if adapting it for a different flash
  size).
- **Libraries**: [`tflm_esp32`](https://github.com/eloquentarduino/tflm_esp32) (TFLite Micro
  runtime) and [`arduino-audio-tools`](https://github.com/pschatzmann/arduino-audio-tools)
  (for `I2SStream`/audio output — `TinyTTS` itself only needs a plain Arduino `Print`, so any
  audio-tools output class works, or your own `Print` implementation).

### Model data sizes

TinyTTS needs at least four data buffers, provided via setters (see "Model data" below) --
each can be compiled into flash (`setWeights(data, len)`, ...) or loaded at runtime from
LittleFS/FFat/SD into PSRAM (`setWeights(file)`, ...). These are the sizes for the model
this project ships example headers for (`src/TinyTTS/data/*_data.h`); use them to decide
what fits in flash on your module and what should come from storage instead.

**Recommended: slimmed dictionary + neural G2P fallback model** (what
`examples/tts_i2s_output/` actually uses) -- smaller *and* better out-of-dictionary coverage
than the full dictionary alone:

| Data | Setter | Size |
|---|---|---:|
| Attention weights (`default_weights_data.h`) | `setWeights` | 1.30 MB |
| CMU dictionary, slimmed (`default_cmudict_slim_data.h`) | `setDictionary` | 1.07 MB |
| Neural G2P fallback model (`default_dictionary_model_data.h`) | `setDictionaryModel` | 0.95 MB |
| `duration_predictor` model (`default_dp_model_data.h`) | `setDurationPredictorModel` | 0.25 MB |
| `decoder` (vocoder) model (`default_decoder_model_data.h`) | `setDecoderModel` | 0.43 MB |
| **Total (recommended five)** | | **4.00 MB** |

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
| Attention weights (`default_weights_data.h`) | `setWeights` | 1.30 MB |
| CMU dictionary (`default_cmudict_data.h`) | `setDictionary` | 3.31 MB |
| `duration_predictor` model (`default_dp_model_data.h`) | `setDurationPredictorModel` | 0.25 MB |
| `decoder` (vocoder) model (`default_decoder_model_data.h`) | `setDecoderModel` | 0.43 MB |
| **Total (simpler four)** | | **5.29 MB** |

See `docs/architecture.md` for why the sizes above are what they are (quantization schemes,
what got dropped and why).

A module with 8MB flash (e.g. N8R8) can't fit either combination compiled into flash
alongside app code — pick which pieces to embed and which to load from storage (the two
`.tflite` models are small and cheap to embed even on an 8MB module; the weights and
especially the dictionary are the ones worth moving to external storage first). A
16MB-flash module (e.g. N16R8) can embed everything, as `examples/tts_i2s_output/` does.

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
#include "TinyTTS/Data.h"   // default_weights, default_cmudict, default_duration_predictor_model, default_decoder_model
using namespace tts_model_data;

// Borrowed -- a flash const array. The pointer must outlive the TinyTTS
// object; nothing is copied.
tts.setWeights(default_weights, default_weights_len);
tts.setDictionary(default_cmudict, default_cmudict_len);
tts.setDurationPredictorModel(default_duration_predictor_model, default_duration_predictor_model_len, /*max_phonemes=*/32);
tts.setDecoderModel(default_decoder_model, default_decoder_model_len, /*chunk_frames=*/96);

// Owned -- reads a File (LittleFS/FFat/SD/...) fully into a new PSRAM
// buffer TinyTTS allocates and frees automatically (no manual cleanup).
File f = LittleFS.open("/weights.bin", "r");
tts.setWeights(f);

// Optional: the neural G2P fallback, for words not in the dictionary.
#include "TinyTTS/data/default_dictionary_model_data.h"  // not pulled in by TinyTTS/Data.h
tts.setDictionaryModel(tts_model_data::default_dictionary_model, tts_model_data::default_dictionary_model_len);

// Optional ALTERNATIVE to setDictionary(default_cmudict, ...) above -- use
// one or the other, and only if you're also calling setDictionaryModel()
// (see "Model data sizes" above for why).
#include "TinyTTS/data/default_cmudict_slim_data.h"
tts.setDictionary(tts_model_data::default_cmudict_slim, tts_model_data::default_cmudict_slim_len);
```

`research/` has the Python tooling that produces this data — `export_weights_and_vectors.py`
(hand-written-C++ stage weights + validation vectors), `export_cmudict.py` (dictionary, both
the full and slimmed variants), `export_dictionary_model.py` (neural G2P fallback model),
`compute_cmudict_exceptions.py` (determines which dictionary words the fallback model needs
an exception for, i.e. the slimmed dictionary's contents), the `onnx2tf`/quantization
pipeline for the two TFLite stages, and `export_headers.py` (which turns all of the above
into the `.h` files under `src/TinyTTS/data/`) — see `docs/research.md`. Useful as a starting
point for regenerating any of these with different settings.

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

## Attribution

Ported from [tronghieuit/tiny-tts](https://github.com/tronghieuit/tiny-tts), licensed
Apache-2.0. The model architecture is VITS-style (windowed relative-position multi-head
attention text encoder, normalizing-flow decoder, HiFi-GAN-style vocoder); see the upstream
repository for training details and the original PyTorch implementation.

## License

Apache-2.0.
