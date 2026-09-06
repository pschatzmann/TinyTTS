# Potential improvements for TinyTTS on microcontrollers

What's genuinely still open for moving TinyTTS toward real-time (or near real-time)
speech on microcontrollers, based on the current architecture and measured performance
in this repository. For what's already been implemented and measured -- including two
approaches that were tried and turned out to disappoint (dual-core parallelism, INT8
activation quantization) -- see `docs/performance.md` instead; this document only covers
what's still worth doing.

## Current constraints (from existing measurements)

- Current end-to-end timing for `speak("Hello world!")` (~1.49s audio output):
  - ESP32-S3 (optimized): ~47.1s (about 32x slower than real time)
  - ESP32-P4 (optimized): ~28.4s (about 19x slower than real time)
- Memory is not the primary blocker (fits in available flash/PSRAM configurations used in this project).
- Compute is the primary blocker.
- `decoder` dominates runtime.
- Within `decoder`, `conv1d` compute dominates; weight fetch/decode is a small share.
- Dual-core parallelism (`TileSplitter`) and INT8 activation quantization have both
  already been tried -- real, ~1.2x and disappointing (~1.4x *slower*) respectively, see
  `docs/performance.md`. Neither is listed as an open lever below.

Implication: the remaining gains come mainly from a lower-sample-rate model, better
perceived latency, and/or a structurally lighter decoder path.

## Ranked approaches

### 1) Retrain at lower native sample rate (highest expected ROI)

**Status: not started -- a model-training effort, not a runtime change.**

Train/fine-tune the model for lower output sample rates (16kHz instead of 44.1kHz). The
single biggest remaining lever, and the only one confirmed compute-bound rather than
fetch-bound (see `docs/performance.md`'s fetch-vs-compute measurement).
`text_encoder`/`duration_predictor` operate on phoneme count (sample-rate-independent);
`flow`/`decoder` operate on frame count, which is proportional to sample rate for a
fixed spoken duration (assuming the hop-length/upsample architecture is unchanged).

Estimated, against the real, measured ESP32-S3 baseline (47.1s @ 44.1kHz):

| Target rate | Estimated total | vs. current |
|---|---|---|
| 44100 Hz (current) | 47.1 s | -- |
| 16000 Hz | ~18.4 s | ~2.6x faster |
| 8000 Hz | ~10.3 s | ~4.6x faster |

16kHz should stay close to perceptually transparent for a voice-assistant use case; 8kHz
would sound noticeably "telephone-quality" (loses fricative/sibilant detail) but stays
intelligible -- must validate both by listening tests + metrics, not just the timing
projection.

Stacks with the already-measured ESP32-P4 result (`docs/performance.md`): **P4 @ 16kHz
projected ~11s**, ~4.3x faster than the current S3 @ 44.1kHz baseline.

#### Trade-offs

- Pros: large speed gain without complex runtime refactors; stacks with ESP32-P4.
- Cons: requires training pipeline work; potential quality drop at lower sample rate.

### 2) Add streaming decode/output for low first-audio latency

**Status: not started.**

Currently one `speak()` call emits audio only after full synthesis. Add chunked vocoder
decoding and immediately write produced PCM to output buffers.

- Keep `text_encoder`/`duration_predictor`/`flow` as front-stage planning.
- Decode waveform in chunks with overlap/state carry where required by the decoder.
- Feed an output ring buffer (I2S task drains continuously).

Note this is a genuinely different, finer-grained thing than the desktop CLI's existing
sentence-level chunking (`docs/desktop.md`'s design notes): that splits *text* into
sentences and calls `speak()` once per sentence, which already gets audio started sooner
for multi-sentence input, but each individual `speak()` call still blocks until *that*
sentence's entire decode finishes. This approach would stream PCM out *during* a single
decode pass -- lower latency even for one long sentence, at real implementation cost.

#### Trade-offs

- Pros: dramatically better perceived responsiveness (audio starts earlier).
- Cons: may not significantly reduce total compute time; mainly improves
  latency-to-first-sound; real complexity (state carry across chunk boundaries, ring
  buffer management).

### 3) Replace vocoder with a lighter architecture

**Status: not started -- last resort.**

If throughput remains too far from real-time after the above, switch from the current
HiFi-GAN-style decoder to a lighter vocoder suitable for MCUs.

#### Trade-offs

- Pros: largest long-term speedup potential.
- Cons: major model/training/porting effort; quality tuning needed.

## Phased implementation roadmap

### Phase A: Improve responsiveness quickly

1. Implement chunked decoder output path (first-audio latency reduction, item 2 above).
2. Add profiling hooks for chunk timing and underrun detection.
3. Validate audio continuity (no clicks/pops at chunk boundaries).

### Phase B: Reduce required compute structurally

1. Train/fine-tune 16kHz model (item 1 above); evaluate quality/performance.
2. Re-export weights/data and run full validation.
3. Combine with ESP32-P4 for the compound ~4.3x projection above.

### Phase C: Last-resort throughput lever

1. Investigate lighter vocoder replacement if phases A-B are still not enough.
2. Prototype with identical front-end to isolate decoder impact.

## Suggested engineering guardrails

- Keep fast correctness sentinel checks in CI/bench runs (existing project checks should remain mandatory).
- Add a repeatable benchmark matrix:
  - board (S3/P4),
  - sample rate,
  - optimization level,
  - latency-to-first-audio and total synthesis time.
- Track both objective and perceptual quality:
  - cosine/SNR where relevant,
  - listening tests for artifacts, intelligibility, and prosody.

## Target framing

To reach real-time on the current S3 baseline, the aggregate improvement needed is
roughly 32x. A realistic path is stacking the two levers that are still open:

- lower sample rate model (~2.6x at 16kHz),
- ESP32-P4 instead of S3 (~1.66x, already measured),
- stacked: ~4.3x, still well short of 32x -- getting the rest of the way plausibly needs
  the lighter-decoder lever too.

The most practical near-term wins are:

1. stream audio earlier (better UX latency, item 2),
2. move to lower sample-rate training (item 1).
