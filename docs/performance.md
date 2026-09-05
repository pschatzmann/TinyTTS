# Performance: real-hardware optimization history

How synthesis went from ~7.3 minutes to ~47 seconds for "Hello world!" on a real ESP32-S3
(8MB PSRAM), what didn't work and why, and what's left on the table. Every number here is
from actual flashed hardware, not a simulation or estimate, unless explicitly marked as a
projection.

## Baseline

Before any of the work below, `TinyTTS.speak("Hello world!")` took **~439.7 seconds**
(~7.3 minutes) on real ESP32-S3 hardware, with `decoder` (the HiFi-GAN-style vocoder,
Conv1d/ConvTranspose1d only) accounting for ~90% of that (~393s). The other three stages
(`text_encoder`, `duration_predictor`, `flow`) were comparatively fast even unoptimized.

## What actually worked

### 1. Weights-only INT8 quantization for `decoder` (~2-3x on `decoder`)

`decoder`'s Conv1d/ConvTranspose1d weights are stored as symmetric per-row INT8
(`scale = max(abs(row))/127`, dtype 2 in `WeightStore.h`) instead of float16 -- activations
stay float32 throughout ("weights-only" quantization, not the full-INT8-with-INT16-activations
scheme the original TFLite-Micro-based decoder was forced into by TFLite Micro's "hybrid
model" rejection, a constraint that doesn't apply to hand-written code). Chosen over full
INT8 activations after measuring both: weights-only gave cos_sim 0.999072/SNR ~27dB on real
test utterances vs. full-INT8's ~12.4dB SNR (audible artifacts) -- see `research/validate_decoder_int8.py`
and `research/export_weights_and_vectors.py`'s module doc for the numbers and export format.

### 2. Tiled weight caching, not whole-tensor caching (fixes a real OOM crash)

`conv1d()`/`convTranspose1d()` (`Ops.h`) decode a small, fixed number of weight rows
(`kWeightTileRows = 8`) into internal RAM at a time, reused across every output timestep for
that tile, instead of re-fetching/re-decoding weights from flash on every single timestep (the
original, T-fold-redundant approach). A **whole-tensor** cache was tried first and crashed on
real hardware with `std::bad_alloc`: internal RAM left over after FreeRTOS/I2S/WiFi-BT
reservations turned out to be far less than assumed -- even a single ~49KB tensor failed to
allocate. The small fixed tile (worst case ~16KB) fits reliably while still turning most of
the redundant fetch traffic into a per-tile fetch instead of a per-timestep one.

### 3. esp-dsp SIMD acceleration for `conv1d`/`linear` (real win)

`Ops.h`'s `linear()` (Q/K/V/O projections, FFN -- used throughout `text_encoder`/`flow`) and
`conv1d()`'s gather-pattern dot product now use `esp-dsp`'s SIMD-accelerated
`dsps_dotprod_f32`/`dsps_dotprode_f32` (ESP32-S3 PIE instructions) when available, falling
back to a plain scalar loop otherwise (`TINYTTS_HAVE_ESP_DSP`, gated by `__has_include`, no
hard dependency). This is the change that actually moved the needle: combined with #1 and #2
above, `decoder` went from ~62.4s to ~41s. Real caveat found the hard way: upstream's doc
comment claims `dsps_dotprod(e)_f32` accumulates into `dest` (`*dest += ...`); every actual
implementation (checked directly in the vendored source, ansi/ae32/aes3 alike) does
`*dest = acc` -- a plain **overwrite**. Trusting the doc comment first silently dropped the
bias term and every kernel tap but the last, corrupting `duration_predictor`'s output
(`t_y=67` instead of the correct 128) before this was caught and fixed.

`esp-dsp` itself isn't usable as a normal Arduino library -- upstream
(github.com/espressif/esp-dsp) is an ESP-IDF component (headers nested under
`modules/*/include/`, no `library.properties`); `arduino-cli` rejects it outright
(`invalid library: no header files found`). A minimal, flattened vendored copy of just the
`dotprod`/`mulc`/`add` modules (not the `esp_dsp.h` umbrella, which pulls in ~15 unrelated
modules -- FFT/FIR/biquad/etc) lives directly at `src/esp-dsp-dotprod/`, inside TinyTTS's own
`src/` tree -- **not** a separate library a user has to install by hand. An earlier version
of this lived as a separate sibling Arduino library, which meant the SIMD path silently
never activated unless someone remembered a manual install step; moving it into TinyTTS's
own `src/` tree means Arduino's normal recursive header/source discovery finds it
automatically on any ESP32-family board, with zero extra setup -- see
`src/esp-dsp-dotprod/NOTICE.md` for what's vendored and why.

`convTranspose1d`'s scatter pattern (`y[t][co] += x*w`, not a gather) was also given a
SIMD path (`dsps_mulc_f32` scale + `dsps_add_f32` accumulate, since esp-dsp has no float32
multiply-*accumulate* primitive) -- kept, but it measured as **no real gain** (~41.06s vs.
~40.75s, noise-level). Two-call overhead evidently cancels out the SIMD win for this op.

### Net result so far

| Stage | ESP32-S3 baseline | ESP32-S3 current | Desktop (host build, current code) |
|---|---|---|---|
| `text_encoder` | ~1.1-1.3s | 117 ms | 2 ms |
| `duration_predictor` | -- | 247 ms | 8 ms |
| `flow` | -- | 4.4 s | 90 ms |
| `decoder` | ~393 s | 40.6 s | 657 ms |
| **Total** | **~439.7 s** | **~47.1 s** | **~0.76 s** |

**~9.3x faster overall** on the ESP32-S3 itself, verified correct via a real cosine-
similarity/SNR-checked pipeline plus a runtime sanity check (`duration_predictor`'s `t_y`
must equal 128 for "Hello world!" -- any deviation means something upstream broke).

The desktop column is the identical, unmodified code compiled for the host build (this
machine: an Intel Core i7-4650U laptop CPU @ 1.7GHz, nothing exotic) instead of an ESP32 --
~62x faster than the optimized ESP32-S3 number, and ~581x faster than the *unoptimized* S3
baseline. It's included as a reference point, not a target -- there's no PSRAM/flash-fetch
latency, no weight-tiling/internal-RAM pressure, and no comparatively weak embedded FPU to
work around on a desktop CPU. The gap between "desktop" and "optimized ESP32-S3" is roughly
the remaining headroom that's specifically an embedded-hardware problem (memory latency,
FPU/SIMD throughput) rather than the model architecture itself being slow -- consistent with
this document's finding that `conv1d` on the S3 is ~98.7% compute, not fetch, bound (see
below): the desktop CPU's much faster/wider FPU is doing that same compute far quicker, not
avoiding a fetch bottleneck that barely existed on-device either.

## Cross-platform: ESP32-P4 (measured)

Tested on a real Guition ESP32-P4 board (16MB flash, 32MB PSRAM) via a timing-only benchmark
sketch (`AudioChunkFn` callback, no I2S/codec output wired up) -- same weights, same INT8
quantization, same tiled weight caching, same `esp-dsp` SIMD dot product (P4 gets its own
genuine SIMD kernel, `dsps_dotprod_f32_arp4` -- RISC-V assembly, distinct from S3's Xtensa
`_aes3` kernel, both already vendored in `src/esp-dsp-dotprod/` and confirmed linked in via a
symbol check -- no separate library install needed, see above). Required
`USBMode=hwcdc,CDCOnBoot=cdc` for native-USB Serial to work at all, same as the S3 board.

| Stage | ESP32-S3 (240MHz) | ESP32-P4 (400MHz) | Speedup | Desktop (reference) |
|---|---|---|---|---|
| `text_encoder` | 117 ms | 57 ms | 2.05x | 2 ms |
| `duration_predictor` | 247 ms | 115 ms | 2.15x | 8 ms |
| `flow` | 4.4 s | 2.4 s | 1.79x | 90 ms |
| `decoder` | 40.6 s | 25.6 s | 1.59x | 657 ms |
| **Total** | **47.1 s** | **28.4 s** | **1.66x** | **~0.76 s** |

Correctness verified the same way as every other change in this document
(`duration_predictor`'s `t_y=128` for "Hello world!" on the P4 run too). The overall ratio
(1.66x) lands almost exactly on the raw clock ratio (400/240 = 1.67x) rather than the higher
CoreMark-based estimate that was floated before this was actually measured (P4 ~2491 vs. S3
~1329 CoreMark, ~1.87x) -- this workload is dominated by the SIMD dot-product kernel, and its
per-cycle throughput is apparently similar between S3's `_aes3` and P4's `_arp4`
implementations, so the win tracks clock speed almost 1:1 rather than exceeding it via better
IPC. `decoder` scales slightly less than the clock ratio (1.59x vs. 1.67x) while
`encoder`/`duration_predictor` scale slightly more (2.05x/2.15x) -- consistent with `decoder`
being dominated by the same SIMD kernel at both clock speeds, while the smaller stages have
more non-SIMD (scalar, memory-management) overhead that may benefit differently from P4's
newer core.

Combined with the 16kHz retraining projection below (not yet done, so still an estimate on
top of a real measurement): **P4 @ 16kHz projected ~11 s**, vs. the current S3 @ 44.1kHz
47.1s -- about 4.3x faster.

## Compiler optimization level: real, easy win (~15% faster, no code change)

The Arduino ESP32 core compiles sketches at **`-Os`** (optimize for size) by default --
confirmed directly from a verbose `arduino-cli compile` invocation, not assumed. For a
compute-bound workload like this one (see the fetch-vs-compute measurement below: `conv1d`
is ~98.7% compute), that's leaving real speed on the table. Tested on the same P4 board,
same "Hello world!" benchmark, overriding only `compiler.optimization_flags` via
`arduino-cli compile --build-property`, no source change:

| Optimization | Flash used | Total time | vs. default (`-Os`) |
|---|---:|---:|---:|
| `-Os` (Arduino default) | 4,543,508 B | 28.4 s | -- |
| `-O2` | 4,567,800 B | 24.7 s | ~15.2% faster |
| `-O3` | 4,570,992 B | 24.65 s | ~15.4% faster |

Correctness verified the same way as everything else in this document (`t_y=128` held for
all three builds). `-O2` captures essentially the entire available win -- `-O3` over `-O2` is
noise-level (24.65s vs. 24.7s, ~0.2%), not worth its marginally larger flash footprint. The
flash-size cost of either is negligible either way (`-O2`/`-O3` add ~24-27KB over `-Os`'s
4.54MB, against a 16MB budget already at only ~27% used).

**Practical takeaway**: pass `--build-property "compiler.optimization_flags=-O2"` to
`arduino-cli compile` (or the equivalent in the Arduino IDE / your build system) when
building a sketch that embeds TinyTTS, for a real ~15% speedup at essentially no cost. This
isn't board-specific -- the same `-Os` default and the same reasoning (compute-bound
workload, small flash-size cost) apply to the ESP32-S3 numbers throughout this document too,
though the exact percentage wasn't independently re-measured there.

## What was tried and didn't work (or wasn't worth it)

### DMA-prefetching weight tiles from flash: reverted, silently corrupted data

The idea: overlap the DMA fetch of the *next* weight tile with CPU compute on the *current*
one (GDMA async memcpy, `esp_async_memcpy`), hiding external-memory fetch latency behind
useful work. Implemented as a double-buffered pipeline (`DmaTilePrefetcher`/`TilePipeline`,
since removed) and flashed to real hardware -- it **silently corrupted weight data**
(`duration_predictor` produced `t_y=6637` instead of 128, then crashed downstream with
`std::bad_alloc` from a runaway allocation caused by the garbage duration value). Root cause:
TinyTTS's real, default weight storage is a `const uint8_t[]` compiled into flash rodata
(memory-mapped via the CPU's flash cache/MMU), and ESP-IDF's own `async_memcpy` docs say DMA
requires "DMA-accessible" buffers, explicitly calling out flash-mapped/MMU-cached read-only
data as not necessarily qualifying -- with no error returned on violation, just wrong bytes.
Reverted rather than chase a fix, given the real (not edge-case) weight storage format hits
this exactly.

### Measured before trying again: fetch is not the bottleneck (~1% of total time)

Before considering a flash→PSRAM copy (which would make weights DMA-accessible and might
make the above approach viable), added temporary instrumentation to measure the actual
fetch-vs-compute split on real hardware. Result, for `decoder`:

| Op | decode (fetch) | compute | decode share |
|---|---|---|---|
| `conv1d` | 468 ms | 34,720 ms | 1.3% |
| `convTranspose1d` | 3 ms | 2,245 ms | 0.1% |

`conv1d` alone is 73.7% of *total* synthesis time (35.2s of 47.1s), and only ~1.3% of that
is fetch. Even a perfect, zero-cost prefetch would save under 1% of total time -- not worth
the engineering risk of rebuilding a PSRAM-copy + DMA pipeline. The bottleneck is squarely
**compute** (the dot products in `conv1d`'s ConvResBlock stack, which runs after every
upsample stage on an increasingly long sequence), not memory latency. This instrumentation
was removed after answering the question -- see git history if it's needed again.

## What's left, roughly by effort-vs-payoff

1. **Retrain at a lower native sample rate (16kHz instead of 44.1kHz).** The single biggest
   remaining lever, and the only one confirmed compute-bound rather than fetch-bound above.
   `text_encoder`/`duration_predictor` operate on phoneme count (sample-rate-independent);
   `flow`/`decoder` operate on frame count, which is proportional to sample rate for a fixed
   spoken duration (assuming the hop-length/upsample architecture is unchanged). Estimated:

   | Target rate | Estimated total | vs. current |
   |---|---|---|
   | 44100 Hz (current) | 47.1 s | -- |
   | 16000 Hz | ~18.4 s | ~2.6x faster |
   | 8000 Hz | ~10.3 s | ~4.6x faster |

   16kHz should stay close to perceptually transparent for a voice-assistant use case; 8kHz
   would sound noticeably "telephone-quality" (loses fricative/sibilant detail) but stays
   intelligible. This is a model-training effort (fine-tuning tiny-tts's own training
   pipeline), not a runtime change -- not started.

2. **ESP32-P4 instead of S3 -- already measured, see above (~1.66x).** Combining it with
   #1 (16kHz retraining, not yet done) is projected at ~11s total, ~4.3x faster than the
   current S3 @ 44.1kHz baseline.

3. **Dual-core.** ESP32-S3 (and P4) have two cores; synthesis currently runs single-threaded
   on one. Splitting `conv1d`'s output-channel tiles across a second FreeRTOS task pinned to
   the other core could give up to ~1.5-1.8x. Not started -- real engineering cost (thread
   safety, work-splitting, sync overhead), no quality tradeoff.

4. **Full-INT8 activations for `decoder` via real int8 hardware kernels (e.g. `esp-nn`).**
   Rejected earlier in this project for quality (measured ~12.4dB SNR, audible, vs. ~22-27dB
   for the weights-only scheme actually shipped) -- worth revisiting only if more speed is
   wanted badly enough to trade real audio quality for it. `esp-nn` itself doesn't fit
   directly either way: INT8-activations-only, Conv2D-shaped (would need faking Conv1d as
   Conv2D), and an ESP-IDF component like upstream `esp-dsp` (not directly Arduino-loadable).

## Lessons that shaped how this was done

- **Real hardware is the only place these bugs show up.** The `std::bad_alloc` crashes (whole-
  tensor cache, then the DMA corruption's downstream effect), the flash-vs-PSRAM DMA
  restriction, and the `dsps_dotprod` overwrite-vs-accumulate bug were all invisible on the
  desktop/host build (uniform malloc'd RAM, no flash-cache-miss latency, no GDMA to violate).
  Every optimization in this document was flashed to real hardware and its output checked
  before being trusted, not just compiled.
- **Never trust a library's doc comment over its actual (vendored, readable) source** when a
  result looks even slightly off -- the `dsps_dotprod` accumulate claim and the DMA
  accessibility assumption were both exactly this class of mistake, one caught by a
  correctness regression (`t_y` deviating from the known-correct value), the other by
  checking ESP-IDF's own docs before committing to the DMA approach a second time.
  `duration_predictor`'s `t_y` for "Hello world!" (must equal 128) has been the standing
  sanity check for every change in this document -- cheap to check, sensitive to almost any
  correctness regression anywhere upstream of `decoder`.
