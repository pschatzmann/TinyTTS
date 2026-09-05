# `research/` — offline model tooling

Everything in `research/` is **Python, PC-side, and not part of the shipped Arduino
library**. It's where the upstream PyTorch model was investigated, exported, converted, and
validated before any of its logic was ported into `src/`. Kept around because the export
scripts are how you'd regenerate `src/TinyTTS/data/*_data.h` (the example/reference model
data shipped with the library, included via `TinyTTS/Data.h` -- see the top-level
`README.md`'s "Model data" section) with a different checkpoint, different quantization
settings, or a different dictionary.

For the full investigative narrative (why things are structured the way they are, the bugs
hit along the way, the quantization quality study) see the project's planning notes; this
file is a reference for what's actually in the folder and how to reproduce it, not the story
of how it got that way.

## Setup

```bash
cd research
python3.12 -m venv venv          # 3.12 specifically -- TensorFlow/onnx2tf compatibility
source venv/bin/activate
pip install -r requirements-frozen.txt
```

`requirements-frozen.txt` is a full `pip freeze` of a working environment (torch 2.14 CPU,
TensorFlow 2.21, onnx2tf 2.6.8, and their transitive dependencies) — use it as-is for
reproducibility, or as a starting point if you need a lighter environment for just one
script (e.g. `export_cmudict.py` needs none of the ML libraries at all).

`research/tiny-tts/` is a clone of the upstream model repo (`git clone --depth 1 -b develop
https://github.com/tronghieuit/tiny-tts.git`) — regenerate it the same way; it's gitignored,
not vendored.

**Important**: the upstream repo's own committed `checkpoints/G.pth` and `onnx/*.onnx` are
stale/mismatched (a different, larger checkpoint than what `tiny_tts/utils/config.py`
describes). All the scripts below correctly pull the real checkpoint from the HuggingFace
Hub (`backtracking/tiny-tts`) instead — don't point them at the repo's own `checkpoints/`.

## Export scripts

| Script | Produces | Notes |
|---|---|---|
| `export_split_onnx.py` | `onnx_out/{text_encoder,duration_predictor,flow,decoder}.onnx` | Splits the model into the 4 independent ONNX graphs matching `tiny_tts/infer_onnx.py`'s stage boundaries. Loads the correct (HF Hub) checkpoint, folds weight-norm. Historical -- see "ONNX → TFLite conversion" below; not needed to regenerate anything the shipped library uses. |
| `export_weights_and_vectors.py` | `weights.bin`, `test_vectors.bin` | `weights.bin`: the `enc_p.*`/`flow.*`/`emb_g.*`/`dp.*`/`dec.*` tensors (everything the hand-written C++ `PhonemeEncoder`/`Flow`/`DurationPredictor`/`Decoder` need), in the named-tensor binary format `WeightStore.h` parses. Stored as float16 (`WeightStore.h` expands each to float32 at parse time -- see its own comment -- so none of the consumers ever see anything but plain float32), halving the size for a rounding-error-level precision cost (cos_sim stayed 1.000000 in `test/main.cpp`'s checks after switching). `enc_p.bert_proj.weight`/`enc_p.ja_bert_proj.weight` are dropped entirely -- multi-lingual BERT conditioning is disabled by default (`bert`/`ja_bert` are always a zero tensor), and a `Conv1d(kernel_size=1)`'s output with a zero input is just its bias term, so those two weight matrices (229,376 bytes) are mathematically dead weight; only their small bias vectors are still exported (see `PhonemeEncoder.h`'s `forward()` doc). `test_vectors.bin`: reference inputs/outputs (including intermediate values like `logw_ref`/`durations_ref`/`m_p_exp_ref`/`audio_ref`) for `test/main.cpp` to validate against -- kept float32 (full precision, since these are the numerical ground truth being validated against, not something to shrink). Re-run this any time any hand-written stage's PyTorch source changes, to regenerate both. |
| `export_cmudict.py` | `cmudict.bin`, `cmudict_slim.bin` | Compact binary CMU Pronouncing Dictionary for `CmuDict.h`, with symbol ids and stress tones pre-resolved against `tiny_tts.text.symbols` at export time. `cmudict.bin`: the full 123,463-entry dictionary, reads `tiny-tts/npm-package/cmudict.json` directly. `cmudict_slim.bin`: only the words `compute_cmudict_exceptions.py`'s `cmudict_exceptions.json` says neither the neural fallback model nor the character-level fallback gets right (skipped if that file doesn't exist yet) -- an alternative to the full dictionary, not a supplement; see the shipped headers' own comments and the top-level README's "Model data sizes" for the actual size tradeoff (it's a net increase in total data size once the model's own cost is counted, not a pure win). |
| `export_dictionary_model.py` | `dictionary_model.bin` | The neural G2P fallback model (`DictionaryModel.h`) -- a single-layer 256-dim GRU encoder-decoder, ~833K params -- converted from `tiny-tts/npm-package/g2p_model.json`'s base64-encoded tensors into a compact, purpose-built binary (not the named-tensor format `weights.bin` uses -- `DictionaryModel.h` parses it directly, no `WeightStore` involved). The four GRU weight matrices (~94% of the params) are INT8-quantized (symmetric, per output row); everything else stays float32 -- shrinks the model from ~3.19MB to ~0.95MB. Checked against a NumPy simulation of the same scheme before trusting it (see `g2p_reference_numpy.py`'s row below) -- a 5,000-word random sample of `cmudict.json` scored 70.8% correct vs. the float32 reference's 70.7%, with only 0.96% of individual predictions differing at all. Also computes and embeds a (symbol_id, tone) table mapping the model's 74-entry output phoneme vocabulary into the project's shared symbol table (same one `cmudict.bin` uses), via the exact same `parse_phone()`/`map_phoneme()` logic `export_cmudict.py` uses -- so `DictionaryModel::predict()`'s output is directly interchangeable with a `CmuDict` lookup's. |
| `compute_cmudict_exceptions.py` | `cmudict_exceptions.json` | Runs the neural fallback model (via `g2p_reference_numpy.py`) AND TextG2P.h's own character-level fallback (replicated in Python) over every word in `cmudict.json`, and keeps only the ones NEITHER gets right -- 37,045 of 123,463 (~30%); one of the two fallbacks already reproduces the other ~70% on its own. This is what makes `cmudict_slim.bin` possible; run this before `export_cmudict.py` if you want that output regenerated (takes a few minutes -- pure-Python/NumPy, ~250 words/sec). |
| `export_headers.py` | `src/TinyTTS/data/*_data.h` | Converts `weights.bin`, `cmudict.bin`, `cmudict_slim.bin` (if present), and `dictionary_model.bin` into the C++ headers the library actually ships. Uses octal-escaped string-literal encoding, not a comma-separated array -- see the script's docstring: a comma-list header for this much data made the Xtensa cross-compiler run out of memory/time; string literals fixed it. |

Typical regeneration order: `export_weights_and_vectors.py` + `export_dictionary_model.py` +
`compute_cmudict_exceptions.py` → `export_cmudict.py` → `export_headers.py`. No ONNX
export/conversion/quantization step is needed for anything the shipped library uses --
`export_split_onnx.py`/`onnx2tf`/`gen_calibration_data.py` and everything under "ONNX →
TFLite conversion" below are purely historical at this point (see that section).

## Validation scripts

| Script | Validates |
|---|---|
| `validate_onnx.py` | The 4 ONNX graphs against direct PyTorch inference, at 5 different sequence lengths (catches dynamic-shape export bugs) -- historical, see "ONNX → TFLite conversion" below. |
| `validate_tflite.py` | The float32 `.tflite` conversions (`tflite_out/`) against PyTorch, plus dumps the full TFLite builtin-op inventory each graph uses -- historical, see below. |
| `validate_int8.py` | The quantized `.tflite` variants (`tflite_int8/`) against PyTorch, using cosine similarity + RMS-based SNR -- historical, see below. |
| `g2p_reference_numpy.py` | A NumPy re-implementation of `g2p_model.json`'s GRU forward pass (same algorithm as `g2p_predict.js` and the C++ `DictionaryModel.h`), cross-checked against the actual Python `g2p_en` package's own output (`pip install g2p_en`) -- bit-exact for every word tried. Used both to validate `DictionaryModel.h`'s C++ port (`test/main.cpp`'s `testDictionaryModel()` checks against reference outputs computed this way) and, via `compute_cmudict_exceptions.py`, to determine `cmudict_slim.bin`'s contents. Still actively used -- not historical. |

`g2p_reference.json`: reference phoneme-id sequences for a few test sentences, generated by
running the *actual* upstream Python G2P pipeline (`tiny_tts.text.english.grapheme_to_phoneme`
+ `phonemes_to_ids` + `commons.insert_blanks`) — used to hand-verify `TextG2P.h`'s output
against ground truth (see `test/main.cpp`'s `CmuDict`/G2P checks).

## ONNX → TFLite conversion (historical -- nothing the shipped library uses goes through this anymore)

All four model stages are hand-written C++ now (see `docs/architecture.md`), so none of
this feeds anything `export_headers.py` produces. Kept as a record of *why* it ended up
that way, and because the artifacts (`onnx_out/`, `tflite_out/`, `tflite_int8/`, `calib/`)
are still on disk.

Conversion used [`onnx2tf`](https://github.com/PINTO0309/onnx2tf). Two backends came up:

- **`flatbuffer_direct`** (the default) — converted `duration_predictor`/`decoder` cleanly,
  dynamic shapes preserved correctly (validated across 5 lengths). Also where
  `text_encoder`/`flow` were *attempted* — those two graphs have the windowed
  relative-position attention module, and `onnx2tf` could not convert them correctly (3
  different bugs across 3 attempted PyTorch-side workarounds, none of which fully resolved
  it) — the original reason any of the four stages were hand-written C++ at all.
- **`-tb tf_converter`** (TensorFlow's own `TFLiteConverter`, more mature quantization
  support) — used for `duration_predictor`/`decoder`'s INT8 quantization
  (`tflite_int8/`), since `onnx2tf`'s own quantization IR builder rejected `SQRT` (used in
  layer norm) for strict full-integer quantization. `tf_converter`'s own dynamic-shape
  tracing had a separate bug (fails on the decoder's dilated resblocks with a too-small
  traced probe shape), worked around by exporting a fixed shape via `-ois`, in ONNX's own
  (channel-first) dimension order, matching the calibration data exactly.

Once `duration_predictor`/`decoder` were also moved to hand-written C++ (reusing the
Conv1d/ChannelNorm/ConvTranspose1d/LeakyReLU primitives `text_encoder`/`flow` already
needed -- see `docs/architecture.md`), this whole conversion/quantization pipeline became
unnecessary: every stage's weights now come directly from the PyTorch checkpoint via
`export_weights_and_vectors.py`, with no ONNX/TFLite step in between at all.

## Output directories

| Directory | Contents |
|---|---|
| `onnx_out/` | The 4 split ONNX graphs (`export_split_onnx.py`'s output), from the correct 1.618M-param checkpoint. Historical -- none of the 4 feed the shipped library anymore; every stage is hand-written C++, sourced from the PyTorch checkpoint directly via `export_weights_and_vectors.py`. |
| `tflite_out/` | `onnx2tf` (`flatbuffer_direct`) float32/float16 conversions of all 4 graphs. Historical -- conversion-bug/investigation artifacts kept for reference, not used by anything. |
| `tflite_int8/` | Quantized `duration_predictor`/`decoder` variants (`tf_converter` backend), e.g. `decoder_float32.tflite`, `decoder_float16.tflite`, `decoder_dynamic_range_quant.tflite`, `decoder_full_integer_quant.tflite` (int8 activations), `decoder_full_integer_quant_with_int16_act.tflite` (int16 activations). Historical -- this was the quantization-quality investigation that determined which variant loaded correctly on TFLite Micro before both stages were moved to hand-written C++ (see `docs/architecture.md`): float16 weights fail `AllocateTensors()` outright (TFLM's `DEQUANTIZE` kernel only accepts int8/int16/uint8), dynamic-range (weights-only int8) fails too (TFLite Micro has no support for "hybrid" models at all), and full INT8 with INT8 activations loads but measurably degrades audio quality (12.4dB SNR, audible noise) vs. INT16 activations' 22.6dB. On real ESP32-S3 hardware, feeding either of the first two unsupported variants to `tflm_esp32` didn't just fail cleanly -- `AllocateTensors()`'s failure path left the heap corrupted, surfacing later as a confusing TLSF assert on an unrelated allocation, which is what made this take a while to root-cause in the first place. None of this is used by the shipped library today. Also contains a partial TensorFlow `SavedModel` export for `duration_predictor` (`saved_model.pb`, `variables/`, `assets/`, `fingerprint.pb`) from an intermediate step. |
| `calib/` | `gen_calibration_data.py`'s output — `.npy` arrays of real (not synthetic) model activations, used as `-cind` calibration data for INT8 quantization. Historical, not used by anything the shipped library needs. |

## Regenerating the shipped library data end-to-end

No ONNX/TFLite/quantization step is needed -- every stage's weights come straight from the
PyTorch checkpoint:

```bash
cd research && source venv/bin/activate

python export_weights_and_vectors.py         # -> weights.bin (all 4 stages), test_vectors.bin
python export_dictionary_model.py            # -> dictionary_model.bin
python compute_cmudict_exceptions.py         # -> cmudict_exceptions.json (takes a few minutes)
python export_cmudict.py                     # -> cmudict.bin, cmudict_slim.bin
python export_headers.py                     # -> src/TinyTTS/data/*_data.h

cd .. && cmake -S . -B build -DTINYTTS_BUILD_TESTS=ON && cmake --build build && ./build/test/test_main
```
