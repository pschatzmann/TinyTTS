# Real-time audio generation approaches for TinyTTS on microcontrollers

This document summarizes practical approaches to move TinyTTS toward real-time (or near real-time) speech on microcontrollers, based on the current architecture and measured performance in this repository.

## Current constraints (from existing measurements)

- Current end-to-end timing for `speak("Hello world!")` (~1.49s audio output):
  - ESP32-S3 (optimized): ~47.1s (about 32x slower than real time)
  - ESP32-P4 (optimized): ~28.4s (about 19x slower than real time)
- Memory is not the primary blocker (fits in available flash/PSRAM configurations used in this project).
- Compute is the primary blocker.
- `decoder` dominates runtime.
- Within `decoder`, `conv1d` compute dominates; weight fetch/decode is a small share.

Implication: future gains must come mainly from reducing compute per generated second, improving compute parallelism, and/or using a lighter decoder path.

## Ranked approaches

## 1) Retrain at lower native sample rate (highest expected ROI)

Train/fine-tune the model for lower output sample rates (for example 22.05kHz or 16kHz instead of 44.1kHz).

- Why it helps: `flow` and especially `decoder` work scales with frame count, which scales with sample rate for fixed utterance duration.
- Existing project projection indicates roughly ~2.6x at 16kHz.

### Trade-offs

- Pros: large speed gain without complex runtime refactors.
- Cons: requires training pipeline work; potential quality drop at lower sample rate (must validate by listening tests + metrics).

## 2) Add streaming decode/output for low first-audio latency

Currently one `speak()` call emits audio only after full synthesis. Add chunked vocoder decoding and immediately write produced PCM to output buffers.

- Keep `text_encoder`/`duration_predictor`/`flow` as front-stage planning.
- Decode waveform in chunks with overlap/state carry where required by the decoder.
- Feed an output ring buffer (I2S task drains continuously).

### Trade-offs

- Pros: dramatically better perceived responsiveness (audio starts earlier).
- Cons: may not significantly reduce total compute time; mainly improves latency-to-first-sound.

## 3) Dual-core decoder parallelism

Split `decoder` hotspot work across both cores (ESP32-S3/P4).

- Suggested split: output-channel tile parallelization for `conv1d` blocks.
- Keep per-core scratch buffers and avoid dynamic allocation in the inner loops.
- Use low-overhead synchronization at layer/tile boundaries.

### Trade-offs

- Pros: likely meaningful throughput gain (~1.4x to ~1.8x range depending on overhead and balance).
- Cons: increased complexity (task pinning, synchronization correctness, buffer ownership).

## 4) Mixed-precision decoder quantization (beyond weights-only INT8)

Revisit activation quantization with better layer selection and calibration/training support.

- Keep sensitive layers in FP16/FP32.
- Quantize robust layers to INT8 (prefer per-channel scaling).
- Consider quantization-aware fine-tuning if post-training quantization quality is insufficient.

### Trade-offs

- Pros: potential strong compute reduction in decoder hotspot.
- Cons: quality regression risk; additional tooling/evaluation complexity.

## 5) Replace vocoder with a lighter architecture

If throughput remains too far from real-time, switch from the current HiFi-GAN-style decoder to a lighter vocoder suitable for MCUs.

### Trade-offs

- Pros: largest long-term speedup potential.
- Cons: major model/training/porting effort; quality tuning needed.

## 6) Hardware and toolchain co-optimization

Combine software improvements with stronger MCU targets and build settings.

- ESP32-P4 already demonstrates measurable gain over S3.
- Keep `-O2` optimization enabled (already shown as a practical win).
- Re-check toolchain options only with correctness guardrails in place.

### Trade-offs

- Pros: multiplicative benefit with model/runtime optimizations.
- Cons: deployment hardware constraints.

## Phased implementation roadmap

### Phase A: Improve responsiveness quickly

1. Implement chunked decoder output path (first-audio latency reduction).
2. Add profiling hooks for chunk timing and underrun detection.
3. Validate audio continuity (no clicks/pops at chunk boundaries).

### Phase B: Improve throughput on current architecture

1. Implement dual-core parallel decoder tiling.
2. Benchmark by stage and by kernel before/after.
3. Preserve current correctness checks and regression tests.

### Phase C: Reduce required compute structurally

1. Train/fine-tune 22.05kHz model; evaluate quality/performance.
2. Train/fine-tune 16kHz model if needed for additional speed.
3. Re-export weights/data and run full validation.

### Phase D: Last-resort throughput lever

1. Investigate lighter vocoder replacement if phases A-C are still not enough.
2. Prototype with identical front-end to isolate decoder impact.

## Suggested engineering guardrails

- Keep fast correctness sentinel checks in CI/bench runs (existing project checks should remain mandatory).
- Add a repeatable benchmark matrix:
  - board (S3/P4),
  - sample rate,
  - optimization level,
  - single-core vs dual-core,
  - latency-to-first-audio and total synthesis time.
- Track both objective and perceptual quality:
  - cosine/SNR where relevant,
  - listening tests for artifacts, intelligibility, and prosody.

## Target framing

To reach real-time on current S3 baseline, aggregate improvement needed is roughly 32x. A realistic path is stacking multiple independent gains:

- lower sample rate model,
- parallel execution,
- further quantization/runtime kernel optimization,
- and potentially a lighter decoder.

The most practical near-term wins are:

1. stream audio earlier (better UX latency),
2. parallelize decoder compute,
3. move to lower sample-rate training.
