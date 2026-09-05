# Architecture: how TinyTTS is actually built, and why

The design decisions behind TinyTTS's four model stages and their quantization choices --
the "why is it built this way" detail that isn't needed to just use the library (see the
top-level README for that), but matters if you're modifying it, debugging it, or curious.

## The four stages

| Stage | Implementation | Why |
|---|---|---|
| `text_encoder` (phoneme/tone/language embeddings + transformer) | hand-written C++ (`PhonemeEncoder`, `Attention`, `TransformerBlock`) | attention-bearing, `onnx2tf` can't convert it |
| `flow` (normalizing flow, prior → posterior latent) | hand-written C++ (`Flow`) | same reason — its coupling layers use the same attention module |
| `duration_predictor` | TFLite Micro (int8 weights, int16 activations) | plain conv + layernorm, converts cleanly |
| `decoder` (HiFi-GAN-style vocoder) | TFLite Micro (int8 weights, int16 activations) | plain conv/transposed-conv, converts cleanly |

`text_encoder` and `flow` both use a **windowed relative-position attention** module that
the ONNX→TFLite converter (`onnx2tf`) cannot handle (three different conversion bugs found
across three attempted PyTorch-side workarounds, none fully resolved -- see `docs/research.md`
for the specifics), so they're implemented directly as hand-written C++ instead of going
through TFLite Micro at all.

The hand-written C++ was verified bit-exact (cosine similarity 1.000000) against the
original PyTorch model at multiple sequence lengths before being trusted; see `research/`
for the validation scripts and `test/` for the ongoing regression harness.

## Why `duration_predictor` and `decoder` both ship as full INT8 with INT16 activations

TFLite Micro has no input-resize API, so both TFLite stages run in fixed-size windows:
`duration_predictor` processes 32 phonemes at a time (see `docs/text-input.md` for how
`TinyTTS` slides overlapping windows to handle longer text without that being a hard
limit), `decoder` generates audio in ~1.1s (96-frame) chunks, streamed to the output as
each chunk is ready.

`duration_predictor` is essentially lossless at this quantization (cosine similarity
0.9999 vs. the float32 reference). `decoder` took more investigation: full INT8 with INT16
activations is the *only* one of four quantization variants studied that TFLite Micro
actually loads, confirmed with a host-side (x86, ASan) build of the real TFLM
interpreter/allocator, not just assumed from PC-side quality numbers:

- **float16 weights**: `AllocateTensors()` fails outright -- TFLM's `DEQUANTIZE` kernel
  only accepts int8/int16/uint8 input, not float16.
- **dynamic-range int8** (weights-only, float32 activations): fails too -- TFLite Micro
  does not support "hybrid" models at all (`CONV_2D`'s own prepare step rejects it:
  "Hybrid models are not supported on TFLite Micro").
- **full INT8 with INT8 activations**: loads fine, but measurably degrades audio quality
  (12.4dB SNR, audible noise) vs. the INT16-activation variant's 22.6dB.

On real ESP32-S3 hardware, feeding either of the first two unsupported variants to
`tflm_esp32` didn't just fail cleanly -- `AllocateTensors()`'s failure path left the heap
corrupted, surfacing later as a confusing TLSF assert on an unrelated allocation, which is
what made this take a while to actually root-cause. The decoder's tensor arena also needs
to be at least ~1.5MB for this model -- `TinyTTS`'s default is 2MB.

## Text-to-phoneme pipeline

Text is turned into phonemes by `TextG2P`, a from-scratch port of the upstream project's
own Node.js reference pipeline (word-splitting → CMU Pronouncing Dictionary lookup →
neural G2P fallback → character-level fallback for anything else), backed by a
123,463-entry dictionary compiled into a compact binary lookup table (see `docs/research.md`
for how it's generated, and the top-level README's "Model data sizes" for the slimmed
alternative).

The neural fallback (`DictionaryModel`, `setDictionaryModel()`) is a from-scratch C++ port
of the reference project's own GRU encoder-decoder (`g2p_predict.js`/the Python `g2p_en`
package it's itself a port of) -- verified bit-exact against both before being trusted (see
`research/g2p_reference_numpy.py`, `docs/research.md`) -- for words not in the dictionary
(proper nouns, made-up words, etc.). Its four GRU weight matrices (~94% of its params) are
INT8-quantized (symmetric, per output row); the quantization was checked for accuracy loss
(a 5,000-word random sample of the dictionary scored 70.8% correct vs. the unquantized
model's 70.7%, with only 0.96% of individual predictions differing at all) before shipping
it, not assumed safe.

## Attention weights (`weights.bin`)

The hand-written `PhonemeEncoder`/`Flow` weights are stored float16 (halves their size for
a rounding-error-level precision cost -- confirmed via the same cosine-similarity checks
used elsewhere, `WeightStore.h` expands each value to float32 at parse time so no
consuming code needs to care), with two tensors dropped entirely: `bert_proj`/
`ja_bert_proj`'s weight matrices reduce to a constant, since multi-lingual BERT
conditioning is disabled by default (`bert`/`ja_bert` are always a zero tensor, and a
`Conv1d(kernel_size=1)`'s output with a zero input is just its bias term) -- only their
small bias vectors are still needed (see `PhonemeEncoder.h`'s `forward()` doc).
