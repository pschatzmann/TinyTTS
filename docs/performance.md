# Performance: real-hardware optimization history

How synthesis went from ~7.3 minutes to ~47 seconds for "Hello world!" on a real ESP32-S3
(8MB PSRAM), and what didn't work and why. Every number here is from actual flashed
hardware, not a simulation or estimate, unless explicitly marked as a projection. For
work that's genuinely not started yet, see `docs/potential-improvements.md` instead --
this document only covers what's already been implemented and measured.

## At a glance

| Change | Result |
|---|---|
| Baseline (unoptimized) | 439.7 s |
| + weights-only INT8, tiled weight caching, float32 SIMD | **47.1 s** (~9.3x) |
| + ESP32-P4 instead of S3 | 28.4 s (~1.66x more) |
| + `-O2` instead of the Arduino default `-Os` | ~15% faster, no code change |
| + dual-core (`setNumWorkers(2)`) | ~1.21x more, after fixing a core-pinning bug |
| INT8 activations (prototype, opt-in) | still ~1.4x *slower* than float32 -- not a win yet |
| DMA-prefetching weight tiles | reverted -- silently corrupted data |

Details, numbers, and the bugs found along the way are below.

## Memory footprint (measured, not the bottleneck)

Flash/RAM was never the constraint this project ran into -- worth stating plainly since
everything else in this document is about compute time. The recommended data set (weights +
slimmed dictionary + neural G2P fallback model, see `docs/model-data.md`) is **4.21 MB
total**, comfortably within a 16MB-flash module's budget with room left for application
code. On real hardware, PSRAM actually used after `begin()` was only **~743 KB**, whether on
an 8MB-PSRAM ESP32-S3 module or a 32MB-PSRAM ESP32-P4 module -- the rest of PSRAM stays free
for the rest of the application.

## Baseline

Before any of the work below, `TinyTTS.speak("Hello world!")` took **~439.7 seconds**
(~7.3 minutes) on real ESP32-S3 hardware, with `decoder` (the HiFi-GAN-style vocoder,
Conv1d/ConvTranspose1d only) accounting for ~90% of that (~393s). The other three stages
(`text_encoder`, `duration_predictor`, `flow`) were comparatively fast even unoptimized.

## What actually worked

### 1. Weights-only INT8 quantization for `decoder`

**Result: ~2-3x on `decoder`.**

`decoder`'s Conv1d/ConvTranspose1d weights are stored as symmetric per-row INT8
(`scale = max(abs(row))/127`, dtype 2 in `WeightStore.h`) instead of float16. Activations
stay float32 throughout -- "weights-only" quantization, not the full-INT8-with-INT16-
activations scheme the original TFLite-Micro-based decoder was forced into by TFLite
Micro's "hybrid model" rejection (a constraint that doesn't apply to hand-written code).

Chosen over full INT8 activations after measuring both: weights-only gave cos_sim
0.999072 / SNR ~27dB on real test utterances vs. full-INT8's ~12.4dB SNR (audible
artifacts). See `research/validate_decoder_int8.py` and
`research/export_weights_and_vectors.py`'s module doc for the numbers and export format.

### 2. Tiled weight caching, not whole-tensor caching

**Result: fixes a real OOM crash.**

`conv1d()`/`convTranspose1d()` (`Ops.h`) decode a small, fixed number of weight rows
(`kWeightTileRows = 8`) into internal RAM at a time, reused across every output timestep
for that tile, instead of re-fetching/re-decoding weights from flash on every single
timestep (the original, T-fold-redundant approach).

A **whole-tensor** cache was tried first and crashed on real hardware with
`std::bad_alloc`: internal RAM left over after FreeRTOS/I2S/WiFi-BT reservations turned
out to be far less than assumed -- even a single ~49KB tensor failed to allocate. The
small fixed tile (worst case ~16KB) fits reliably while still turning most of the
redundant fetch traffic into a per-tile fetch instead of a per-timestep one.

### 3. esp-dsp SIMD acceleration for `conv1d`/`linear`

**Result: the change that actually moved the needle** -- combined with #1 and #2 above,
`decoder` went from ~62.4s to ~41s.

`Ops.h`'s `linear()` (Q/K/V/O projections, FFN -- used throughout `text_encoder`/`flow`)
and `conv1d()`'s gather-pattern dot product use `esp-dsp`'s SIMD-accelerated
`dsps_dotprod_f32`/`dsps_dotprode_f32` (ESP32-S3 PIE instructions) when available,
falling back to a plain scalar loop otherwise (`TINYTTS_HAVE_ESP_DSP`, gated by
`__has_include`, no hard dependency).

**Nothing to vendor for this part.** Arduino-ESP32 cores from ~3.3.x onward bundle their
own precompiled `espressif__esp-dsp` component, found automatically via
`<dsps_dotprod.h>`/`<dsps_mulc.h>`/`<dsps_add.h>` -- no separate install, no vendored
copy needed. (This wasn't always true -- an earlier core version required manually
vendoring esp-dsp, which is why `src/int8-dotprod/` still exists at all; see the
"INT8-activation `conv1d()`" section below for what it now actually contains, which is
unrelated to float32.)

`convTranspose1d`'s scatter pattern (`y[t][co] += x*w`, not a gather) was also given a
SIMD path (`dsps_mulc_f32` scale + `dsps_add_f32` accumulate, since esp-dsp has no
float32 multiply-*accumulate* primitive) -- kept, but it measured as **no real gain**
(~41.06s vs. ~40.75s, noise-level). Two-call overhead evidently cancels out the SIMD win
for this op.

> **Bug found here**: upstream's doc comment claims `dsps_dotprod(e)_f32` accumulates
> into `dest` (`*dest += ...`); every actual implementation (checked directly in the
> vendored source, ansi/ae32/aes3 alike) does `*dest = acc` -- a plain **overwrite**.
> Trusting the doc comment first silently dropped the bias term and every kernel tap but
> the last, corrupting `duration_predictor`'s output (`t_y=67` instead of the correct
> 128) before this was caught and fixed.

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
machine: an Intel Core i7-4650U laptop CPU @ 1.7GHz, nothing exotic) instead of an
ESP32 -- ~62x faster than the optimized ESP32-S3 number, and ~581x faster than the
*unoptimized* S3 baseline. It's a reference point, not a target: no PSRAM/flash-fetch
latency, no weight-tiling/internal-RAM pressure, no comparatively weak embedded FPU to
work around. The gap between "desktop" and "optimized ESP32-S3" is roughly the remaining
headroom that's specifically an embedded-hardware problem (memory latency, FPU/SIMD
throughput), not the model architecture itself being slow -- consistent with `conv1d` on
the S3 being ~98.7% compute-, not fetch-, bound (see below).

## Cross-platform: ESP32-P4 (measured)

Tested on a real Guition ESP32-P4 board (16MB flash, 32MB PSRAM) via a timing-only
benchmark sketch (`AudioChunkFn` callback, no I2S/codec output wired up) -- same
weights, same INT8 quantization, same tiled weight caching, same `esp-dsp` SIMD dot
product (P4 gets its own genuine SIMD kernel, `dsps_dotprod_f32_arp4` -- RISC-V
assembly, distinct from S3's Xtensa one, both provided by the Arduino-ESP32 core's own
bundled esp-dsp component, see item 3 above). Required `USBMode=hwcdc,CDCOnBoot=cdc` for
native-USB Serial to work at all, same as the S3 board.

| Stage | ESP32-S3 (240MHz) | ESP32-P4 (400MHz) | Speedup | Desktop (reference) |
|---|---|---|---|---|
| `text_encoder` | 117 ms | 57 ms | 2.05x | 2 ms |
| `duration_predictor` | 247 ms | 115 ms | 2.15x | 8 ms |
| `flow` | 4.4 s | 2.4 s | 1.79x | 90 ms |
| `decoder` | 40.6 s | 25.6 s | 1.59x | 657 ms |
| **Total** | **47.1 s** | **28.4 s** | **1.66x** | **~0.76 s** |

Correctness verified the same way as every other change in this document
(`duration_predictor`'s `t_y=128` for "Hello world!" on the P4 run too).

The overall ratio (1.66x) lands almost exactly on the raw clock ratio (400/240 = 1.67x)
rather than the higher CoreMark-based estimate floated before this was measured (P4
~2491 vs. S3 ~1329 CoreMark, ~1.87x). This workload is dominated by the SIMD
dot-product kernel, and its per-cycle throughput is apparently similar between S3's
`_aes3` and P4's `_arp4` implementations, so the win tracks clock speed almost 1:1
rather than exceeding it via better IPC. `decoder` scales slightly less than the clock
ratio (1.59x vs. 1.67x) while `encoder`/`duration_predictor` scale slightly more
(2.05x/2.15x) -- consistent with `decoder` being dominated by the same SIMD kernel at
both clock speeds, while the smaller stages have more non-SIMD (scalar,
memory-management) overhead that may benefit differently from P4's newer core.

Combined with the 16kHz retraining projection below (not yet done, so still an estimate
on top of a real measurement): **P4 @ 16kHz projected ~11 s**, vs. the current S3 @
44.1kHz 47.1s -- about 4.3x faster.

## Cross-platform: Raspberry Pi (desktop CLI, measured)

Not the Arduino/ESP-IDF microcontroller library -- both boards below run the `desktop/`
CLI's host code path (a normal Linux build of the same, unmodified `Ops.h`/model code),
included as reference points spanning from the weakest to a genuinely capable real chip
this code has been measured on.

**Pi Zero W** (single-core ARM1176JZF-S, ARMv6, no NEON/SIMD, ~1GHz): `text_encoder`
47ms, `duration_predictor` 126ms (`t_y=117`), `flow` 1869ms, `decoder` 13642ms (59904
samples, ~1.36s of audio at 44.1kHz) -- **~15.68s total**. `decoder` alone is ~87% of
that, consistent with it dominating on every other platform in this document too.

**Pi 4 Model B** (quad-core Cortex-A72, ARMv8-A, NEON, ~1.5GHz): `text_encoder` 13ms,
`duration_predictor` 13ms (`t_y=117`), `flow` 215ms, `decoder` 1515ms -- **~1.76s
total**, already faster than the ESP32-P4's optimized 28.4s by well over an order of
magnitude, and within striking distance of real time despite running the same
unmodified, single-threaded scalar code as the Pi Zero.

Two further experiments on the same Pi 4, neither of which moved the needle:

- **`--threads 2`** (`TileSplitter`, same mechanism as the ESP32 dual-core path):
  `flow` 222ms, `decoder` 1489ms, ~1.74s total -- essentially no change (noise-level,
  same pattern as the ESP32's own modest ~1.21x from this feature), consistent with
  this being a genuinely tiny model where per-call tiling/synchronization overhead
  eats most of the available parallelism on 2 threads.
- **NEON-accelerated `Ops.h`** (real ARM SIMD, prototyped in response to the above --
  `linear()`'s contiguous dot product, plus manually-gathered strided variants for
  `conv1d()`/`convTranspose1d()`'s tap-wise weight access, mirroring the existing
  ESP32 `esp-dsp` path; confirmed active, not silently falling back to scalar -- this
  board reports `aarch64`, which mandates NEON in the ISA with no extra compiler flags
  needed): `encoder` 12ms, `duration_predictor` 12ms (`t_y=117`), `flow` 220ms,
  `decoder` 1498ms, ~1.74s total -- again essentially no change from the plain-scalar
  run.

  Likely cause, not yet confirmed: this model's decoder channel counts are small (as
  low as 16-32, see the dual-core section above) and kernel sizes short, so most
  `conv1d()`/`convTranspose1d()` dot products are only 1-2 SIMD vector-widths long --
  too short to amortize the fixed per-call overhead the manual strided-gather adds
  (four individual scalar loads before each 4-wide vector multiply-accumulate). The
  same class of explanation was already reached above for esp-dsp's own
  `convTranspose1d` SIMD path (~41.06s vs ~40.75s, "two-call overhead evidently
  cancels out the SIMD win") -- short dot products and small channel counts appear to
  be a structural mismatch for gather-based SIMD on this model, not a platform-specific
  fluke. A real microbenchmark isolating just the dot-product loop at realistic
  channel counts (16-64) would confirm or rule this out; not yet done.

## Compiler optimization level: real, easy win

**Result: ~15% faster, no code change.**

The Arduino ESP32 core compiles sketches at **`-Os`** (optimize for size) by default --
confirmed directly from a verbose `arduino-cli compile` invocation, not assumed. For a
compute-bound workload like this one (`conv1d` is ~98.7% compute, see below), that's
leaving real speed on the table. Tested on the same P4 board, same "Hello world!"
benchmark, overriding only `compiler.optimization_flags` via
`arduino-cli compile --build-property`, no source change:

| Optimization | Flash used | Total time | vs. default (`-Os`) |
|---|---:|---:|---:|
| `-Os` (Arduino default) | 4,543,508 B | 28.4 s | -- |
| `-O2` | 4,567,800 B | 24.7 s | ~15.2% faster |
| `-O3` | 4,570,992 B | 24.65 s | ~15.4% faster |

Correctness verified the same way as everything else in this document (`t_y=128` held
for all three builds). `-O2` captures essentially the entire available win -- `-O3` over
`-O2` is noise-level (24.65s vs. 24.7s, ~0.2%), not worth its marginally larger flash
footprint. The flash-size cost of either is negligible either way (`-O2`/`-O3` add
~24-27KB over `-Os`'s 4.54MB, against a 16MB budget already at only ~27% used).

**Practical takeaway**: pass `--build-property "compiler.optimization_flags=-O2"` to
`arduino-cli compile` (or the equivalent in the Arduino IDE / your build system) when
building a sketch that embeds TinyTTS, for a real ~15% speedup at essentially no cost.
This isn't board-specific -- the same `-Os` default and the same reasoning apply to the
ESP32-S3 numbers throughout this document too, though the exact percentage wasn't
independently re-measured there.

## Dual-core (`TileSplitter`)

**Status: implemented, measured on real hardware. Result: ~1.21x**, after fixing a
genuine core-pinning bug the first measurement pass caught.

`conv1d()`/`convTranspose1d()` split their output/input-channel tiles across a second
FreeRTOS task pinned to the other core, opt-in via `TinyTTS::setNumWorkers(2)` (see
`src/TinyTTS/Concurrency/TileSplitter.h`, `docs/desktop.md`'s `--threads` flag). This
replaces an earlier ~1.5-1.8x *projection* that was never actually measured on hardware
-- this project's own house rule is to record only real, flashed numbers.

**First measurement showed no speedup at all** (~44.7s vs. ~44.4s vocoder time) --
suspicious enough to warrant digging rather than accepting a flat "no gain" result.

> **Bug found here**: `TileSplitter` pinned its worker task to core 1, on the hardcoded
> assumption "the calling thread runs on core 0 as Arduino sketches normally do" -- but
> Arduino-ESP32's own `core/main.cpp` pins the sketch's main task (`loopTask`, where
> `setup()`/`loop()`/every `speak()` call actually runs) to `ARDUINO_RUNNING_CORE`,
> which **defaults to 1, not 0**. Worker and caller were fighting over the *same*
> physical core the entire time -- confirmed with a direct probe (each `TileSplitter`
> participant reporting its own `xPortGetCoreID()`) showing both at core 1 before the
> fix, core 0 and core 1 after. Fixed by determining the calling thread's actual core at
> construction time (`1 - xPortGetCoreID()`) instead of hardcoding core 1.

| | 1 core | 2 cores (bug: same core) | 2 cores (fixed: different cores) |
|---|---:|---:|---:|
| `decoder` (vocoder stage) | ~44.4 s | ~44.7 s | **~36.9 s** |
| total `speak("Hello world!")` | ~49.6 s | ~49.9 s | **~41.0 s** (3 runs, <15 ms spread) |

A genuine **~1.21x** speedup -- smaller than the old ~1.5-1.8x projection (this is a
tiny model, per-call tile counts are modest, and there's real per-call synchronization
overhead: two atomics polled via `delay(1)`, at least one FreeRTOS tick each) -- but
real and reproducible, unlike the pre-fix number. Every stage improved, not just
`decoder` (encoder 117ms->82ms, duration_predictor 239ms->130ms, flow 4380ms->3435ms),
consistent with other stages also calling `conv1d()`/`linear()` with enough tiles to
benefit from the same fix.

## INT8-activation `conv1d()`, real SIMD on ESP32

**Status: prototyped, flashed, correct, opt-in. Result: still a net loss vs. float32**,
even with a working SIMD kernel -- see the final table below.

An earlier full-INT8-activations experiment (via TFLite's post-training quantizer,
`research/validate_int8.py`) measured only ~12.4dB SNR (audible noise) and was shelved.
That experiment used TFLite's asymmetric, *per-tensor* scale for both weights and
activations -- not a clean measurement of what a hand-written path using the *per-row*
weight scale the shipped decoder already has would sound like.

A new prototype (`ops::DecoderPrecision::kInt8Activations`, opt-in via
`TinyTTS::setDecoderPrecision()` / the desktop CLI's `--decoder-precision int8`)
quantizes `conv1d()`'s activations to INT8 *per-timestep* (not per-tensor) and does a
real INT8xINT8 dot product against the existing per-row-scaled INT8 weights, using
`dsps_dp_s8` (vendored into `src/int8-dotprod/`) on ESP32-S3, or an identical-math
scalar loop elsewhere. `convTranspose1d()` is intentionally left float32 -- its scatter
access pattern has no single dot-product primitive to swap in, unlike `conv1d()`'s
per-output-channel gather.

### Quality

Measured on desktop (float32 vs. INT8-activation output, same utterance):
**23.37dB SNR, 0.9977 cosine similarity** -- clearly better than the old per-tensor
TFLite experiment's 12.4dB, in the neighborhood of (if a bit below) the shipped
weights-only scheme's ~22-27dB.

### Speed

Measured on real ESP32-S3 hardware, "Hello world!" -- the full progression, including
two hardware-only-visible bugs found and fixed along the way:

| | `decoder` (vocoder stage) | total `speak()` |
|---|---:|---:|
| float32, 1 core | ~44.4 s | ~49.6 s |
| float32, 2 cores | ~36.9 s | ~41.0 s |
| INT8, scalar (broken SIMD kernel disabled), 1 core | ~88.8 s | ~94.0 s |
| INT8, corrected SIMD kernel, 2 cores | **~51.5 s** | **~55.6 s** |

The first pass (scalar-only, broken kernel disabled) measured INT8 as **~1.9x
*slower*** than float32.

> **Bug found here (1 of 2)**: the original SIMD kernel (`dsps_dp_s8_aes3`, from
> `espressif/esp-dsp`) was genuinely broken, not just suspicious. A self-test added
> specifically because of a leftover "always ANSI, remove before release" comment found
> in that vendored file caught it immediately on real hardware: for a 16-element vector,
> expected dot product 2360, the portable scalar reference correctly returned 2360,
> `dsps_dp_s8_aes3` returned 263 -- and different, still-wrong values on different
> inputs (359, -275), ruling out a one-off fluke. It's been removed entirely (not just
> disabled) and replaced -- see below.

> **Bug found here (2 of 2)**: a second, unrelated crash. The per-timestep
> activation-quantization buffers (`xq`/`x_scale` in `Ops.h`'s `conv1d()`) were
> originally declared as `InternalVector<T>` -- which, despite the name, means on-chip
> *internal* SRAM (~300KB total), not "internally managed". `conv1d()`'s existing
> `wtile` buffer safely uses the same allocator because its size is bounded by a fixed
> per-layer channel count, but `xq`/`x_scale` scale with `T`, the full frame count --
> tens of thousands of frames deep into the decoder's upsample stages. This crashed with
> `std::bad_alloc` resizing to a mere 128KB, which looked exactly like PSRAM
> fragmentation (`heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM)` reported 4.8MB
> free at the crash site) until a direct `heap_caps_malloc` probe of the identical size
> against `MALLOC_CAP_SPIRAM` succeeded fine -- proving the wrong heap was being
> exhausted, not that the right one was fragmented. Fixed by switching both to
> `PsramVector<T>`, matching how `Mat`'s own large activation buffers are already
> stored.

**Replacement SIMD kernel**: ported from `espressif/esp-nn` instead of hand-writing
Xtensa PIE assembly blind. Rather than risk repeating the exact class of mistake that
produced the `dsps_dp_s8_aes3` bug, `esp_nn_dot_s8_aligned_esp32s3` (vendored into
`src/int8-dotprod/esp_nn_dot_s8_esp32s3.S`) is used instead -- Espressif's own
production quantized-NN kernel library, used for real int8 conv/FC inference, far more
scrutinized than one orphaned esp-dsp function. It requires both operand pointers
16-byte aligned and `len` a multiple of 16 (undefined behavior otherwise -- it's a
128-bit vector load), so `conv1d()`'s INT8 path now allocates `xq`/`wtap` via
`AlignedStlAllocator.h` (16-byte-aligned `heap_caps_aligned_alloc`) and pads `cin` up to
the next multiple of 16 with zero bytes (zero-valued padding doesn't change a dot
product's result) -- guaranteeing the fast path's preconditions are always met by
construction. `dspsDotProdS8()` (`src/int8-dotprod/dsps_dp_s8.h`) still keeps the
portable scalar kernel as a correctness fallback for any caller that doesn't meet those
preconditions.

**Final result with the corrected kernel**: real, verified SIMD (self-test passes,
`t_y=128` holds, no crash) -- but still net slower than float32. The SIMD fix itself is
a genuine win in isolation (~88.8s -> ~51.5s vocoder time, ~1.72x faster than the scalar
path it replaced), but doesn't close the gap to float32's already-mature SIMD path
(~1.41x *slower* than float32 at 2 cores).

Not chased further in this pass, but the likely explanation: this is a genuinely tiny
model with decoder channel counts as small as 16-32 (see the dual-core section above),
so most `conv1d()` calls' INT8 dot products are only 1-2 SIMD vector loads long -- too
short for the kernel's real throughput advantage to outweigh the fixed per-call overhead
this path adds on top (the abs-max-scan + quantize pass over every activation element,
the weight-tile reshape into a padded per-tap layout, and any wasted work from padding
`cin` up to the next multiple of 16 for channel counts that aren't already one).

Weights-only INT8 (item 1 above, ~22-27dB SNR, no speed cost) remains the shipped
default; this prototype stays available via `setDecoderPrecision(kInt8Activations)` for
experimentation, not as something to switch to by default.

<details>
<summary>Note for anyone re-running this benchmark</summary>

The ~5-6 minute total runtime of 3x float32 + 3x int8 `speak()` calls back-to-back, with
no delay between calls, was originally causing a silent reset partway through (no
crash/panic printed, consistent with a task watchdog reset) -- adding a 2s `delay()`
between runs (matching the shipped `tts_i2s_output` example's own `loop()` pattern)
fixed it.

</details>

## What was tried and didn't work (or wasn't worth it)

### DMA-prefetching weight tiles from flash: reverted, silently corrupted data

The idea: overlap the DMA fetch of the *next* weight tile with CPU compute on the
*current* one (GDMA async memcpy, `esp_async_memcpy`), hiding external-memory fetch
latency behind useful work. Implemented as a double-buffered pipeline
(`DmaTilePrefetcher`/`TilePipeline`, since removed) and flashed to real hardware.

**It silently corrupted weight data**: `duration_predictor` produced `t_y=6637` instead
of 128, then crashed downstream with `std::bad_alloc` from a runaway allocation caused
by the garbage duration value.

Root cause: TinyTTS's real, default weight storage is a `const uint8_t[]` compiled into
flash rodata (memory-mapped via the CPU's flash cache/MMU), and ESP-IDF's own
`async_memcpy` docs say DMA requires "DMA-accessible" buffers, explicitly calling out
flash-mapped/MMU-cached read-only data as not necessarily qualifying -- with no error
returned on violation, just wrong bytes. Reverted rather than chase a fix, given the
real (not edge-case) weight storage format hits this exactly.

### Measured before trying again: fetch is not the bottleneck

**Result: fetch is ~1% of total time.**

Before considering a flash→PSRAM copy (which would make weights DMA-accessible and
might make the above approach viable), added temporary instrumentation to measure the
actual fetch-vs-compute split on real hardware. Result, for `decoder`:

| Op | decode (fetch) | compute | decode share |
|---|---|---|---|
| `conv1d` | 468 ms | 34,720 ms | 1.3% |
| `convTranspose1d` | 3 ms | 2,245 ms | 0.1% |

`conv1d` alone is 73.7% of *total* synthesis time (35.2s of 47.1s), and only ~1.3% of
that is fetch. Even a perfect, zero-cost prefetch would save under 1% of total time --
not worth the engineering risk of rebuilding a PSRAM-copy + DMA pipeline. The bottleneck
is squarely **compute** (the dot products in `conv1d`'s ConvResBlock stack, which runs
after every upsample stage on an increasingly long sequence), not memory latency. This
instrumentation was removed after answering the question -- see git history if it's
needed again.

## Lessons that shaped how this was done

- **Real hardware is the only place these bugs show up.** The `std::bad_alloc` crashes
  (whole-tensor cache, then the DMA corruption's downstream effect), the flash-vs-PSRAM
  DMA restriction, and the `dsps_dotprod` overwrite-vs-accumulate bug were all invisible
  on the desktop/host build (uniform malloc'd RAM, no flash-cache-miss latency, no GDMA
  to violate). Every optimization in this document was flashed to real hardware and its
  output checked before being trusted, not just compiled.
- **Never trust a library's doc comment over its actual (vendored, readable) source**
  when a result looks even slightly off -- the `dsps_dotprod` accumulate claim and the
  DMA accessibility assumption were both exactly this class of mistake, one caught by a
  correctness regression (`t_y` deviating from the known-correct value), the other by
  checking ESP-IDF's own docs before committing to the DMA approach a second time.
  `duration_predictor`'s `t_y` for "Hello world!" (must equal 128) has been the standing
  sanity check for every change in this document -- cheap to check, sensitive to almost
  any correctness regression anywhere upstream of `decoder`.
