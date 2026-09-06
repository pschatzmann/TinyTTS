# Model data: sizes, wiring, and where it comes from

TinyTTS needs at least two data buffers, provided via setters -- each can be compiled into
flash (`setWeights(data, len)`, ...) or loaded at runtime from LittleFS/FFat/SD into PSRAM
(`setWeights(file)`, ...). This covers how big those buffers are, how to wire them up, and
where they're generated from.

## Sizes

These are the sizes for the model this project ships example headers for
(`src/TinyTTS/data/*_data.h`); use them to decide what fits in flash on your module and what
should come from storage instead. `duration_predictor` and `decoder` have no data buffers of
their own -- both are hand-written C++, and their weights are folded into the attention
weights buffer below, same as `text_encoder`/`flow`.

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

## Wiring it up

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
// (see "Sizes" above for why).
#include "TinyTTS/data/default_cmudict_slim_data.h"
tts.setDictionary(default_cmudict_slim, default_cmudict_slim_len);
```

## Where it comes from

`research/` has the Python tooling that produces this data — `export_weights_and_vectors.py`
(every hand-written-C++ stage's weights + validation vectors), `export_cmudict.py`
(dictionary, both the full and slimmed variants), `export_dictionary_model.py` (neural G2P
fallback model), `compute_cmudict_exceptions.py` (determines which dictionary words the
fallback model needs an exception for, i.e. the slimmed dictionary's contents), and
`export_headers.py` (which turns all of the above into the `.h` files under
`src/TinyTTS/data/`) — see `docs/research.md`. Useful as a starting point for regenerating
any of these with different settings.
