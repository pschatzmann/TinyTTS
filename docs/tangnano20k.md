# Sipeed Tang Nano 20K

TinyTTS also builds for the [arduino-tangnano20k](https://github.com/pschatzmann/arduino-tangnano20k)
core: a picorv32 RISC-V soft CPU on the Tang Nano 20K's Gowin FPGA, with 8MB of embedded
SDRAM, the board's SPI flash, and an optional on-chip INT8 dot-product engine. This is an
experiment in running the model on very different hardware, not a practical target: the
CPU runs at 27MHz on one core and has no FPU, so synthesis takes far longer than on an
ESP32: "Hello world!" takes about 34 minutes on the board (see "Measured on the board").

## Board settings

Needs a core newer than the v0.2.2 release (the current `main` branch): it adds
`Serial.printf()`, `FS.h`, C++17, libm/libstdc++ linking and the `TANGNANO20K_AI_ACCEL`
define TinyTTS relies on. In the Arduino IDE's Tools menu:

| Setting | Value | Why |
|---|---|---|
| Boot Mode | **SRAM + SDRAM** (or Flash + SDRAM) | TinyTTS's code doesn't fit the 64KB internal SRAM |
| Hardware Multiply/Divide | **Enabled** | Integer and INT8 math, and faster software float |
| SD Card | **Enabled (GPLv3)** | `TinyTTS.h` includes `<FS.h>`, which the core's SD library provides |
| AI Accelerator | **Enabled** (optional) | Runs the dot products on the FPGA engine, see below |

The first build for each combination of options runs the full FPGA flow (15-25 minutes).

## How the pieces fit

- **Model data in flash.** The generated arrays in `src/TinyTTS/data/` are marked
  `TINYTTS_PROGMEM` (`src/TinyTTS/DataAttr.h`), which is `PROGMEM` on Arduino. On this core
  that places them in the memory-mapped SPI flash (about 4.2MB for weights + slim dictionary
  + G2P model) instead of SRAM, and they're read in place. On ESP32 `PROGMEM` is empty, so
  nothing changes there.
- **Heap.** `malloc()` on this core allocates from the SDRAM (about 7.9MB after the code
  image), so TinyTTS's PSRAM allocators map onto it unchanged.
- **Files.** `setWeights(File&)` etc. work with the SD library's `File` via the core's
  `FS.h`, which defines `fs::File` as SD's `File`.
- **Allocation failure.** The core builds without C++ exceptions, so the STL allocators
  call `abort()` (`src/TinyTTS/AllocFailure.h`) instead of throwing `std::bad_alloc`.
- **Timing logs** use `snprintf()` + `Serial.print()`, which works on every Arduino core.

## AI accelerator

With Tools > AI Accelerator enabled, the core defines `TANGNANO20K_AI_ACCEL=1`, and
`Ops.h` then includes `<AIAccelerator.h>` and defines `TINYTTS_HAVE_TANG_AI`. Define
`TINYTTS_NO_TANG_AI` before including `TinyTTS.h` to keep everything on the CPU.

The engine computes INT8 x INT8 -> INT32 dot products for 8 output channels x up to 16
taps per call, with a window of up to 1KB (`k * cin`). Every accelerated op goes through one
routine, `ops::detail::accelGather()`, which splits larger windows into input-channel chunks
and rescales the INT32 results to float. Layers with fewer than 16 input or 8 output
channels (`kAccelMinInChannels`/`kAccelMinOutChannels`) stay on the CPU: input channels are
padded to 16 bytes per tap and every call returns all 8 rows, so for the vocoder's last
stages (2-8 channels over up to 47,104 frames) the engine needs 2-8 bus accesses per useful
multiply-accumulate.

| Op | Used by | On the engine |
|---|---|---|
| `conv1d()`, INT8 weights | decoder (vocoder) | stored INT8 weights and per-row scales, as in the scalar INT8 path |
| `conv1d()`, float/float16 weights | text encoder FFN, duration predictor, phoneme projection | quantized on the fly |
| `linear()` | attention q/k/v/o, flow, duration predictor, vocoder | quantized on the fly, one tap |
| `convTranspose1d()` | vocoder upsampling | rewritten as a gather, one pass per stride residue class |

"Quantized on the fly" means symmetric abs-max INT8, weights per output channel and
activations per timestep, the same scheme as the decoder's INT8 path. Attention's per-head
score and value products stay on the CPU (see "CPU paths" below).

Two switches control it, both no-ops on other boards:

```cpp
// Default with the accelerator: kInt8Activations (the engine only helps the INT8 path).
tts.setDecoderPrecision(tinytts::ops::DecoderPrecision::kFloat32);

// Default with the accelerator: true. false keeps linear(), float conv1d() and
// convTranspose1d() in float, e.g. if the quantized output sounds worse.
tinytts::ops::setAccelerateFloatOps(false);
```

`examples/timing_benchmark` prints which modes are active.

## CPU paths

Whatever doesn't run on the engine runs on a 54MHz (at most) CPU without an FPU, where
every float operation is a libgcc software routine of several hundred cycles. Three things
in `Ops.h`/`Attention.h` address that:

- **`TINYTTS_HOT`** puts the inner loops (`conv1d`, the engine loop, `linear`,
  `convTranspose1d`, attention, the vocoder resblock, element-wise ops, quantization) in the
  core's `SRAM_CODE` section, so they run from internal SRAM instead of SDRAM (about 2.4x
  faster) with a "+ SDRAM" boot mode. About 34KB of the 64KB SRAM are used.
- **`TINYTTS_FIXED_POINT`** (on by default for RISC-V without an FPU, off wherever there is
  one) replaces the hot float arithmetic with integer arithmetic:
  - the INT8 conv paths sum their taps in 64-bit integers, with the per-timestep activation
    scales as Q24 fractions of the layer's largest scale, and convert to float once per
    output instead of three float operations per tap;
  - INT8 quantization finds the abs-max on the float bit patterns and computes
    `round(x * 127 / max)` from the mantissas with integer multiplies;
  - attention quantizes q/k/v and the relative embeddings to 16 bits per head and the
    softmax weights to Q15, and sums its dot products in 64-bit integers.
- **Attention's relative-position terms** are only computed within the trained window
  (+-4 frames), where the embeddings are non-zero. This one is exact and applies to every
  platform: about half of attention's work for long inputs was multiplying by zero.

## Profiling

Every `speak()` prints a line per stage (`g2p`, `encoder`, `duration_predictor`, `flow`,
`vocoder`). For a breakdown by operation, define `TINYTTS_PROFILE` before including
`TinyTTS.h`:

```cpp
#define TINYTTS_PROFILE
#include "TinyTTS.h"
```

After each `speak()` it then prints the total time and call count of `linear`, `conv1d`
(with the accelerated and quantization parts), `convTranspose1d`, the engine's `compute()`
calls, the attention score loops, element-wise ops, and G2P with its neural model
(`src/TinyTTS/Profile.h`). Totals are inclusive: nested entries (indented) are also counted
in their parent. Also defining `TINYTTS_PROFILE_TRACE` prints a line on entering and leaving each
of these operations (except the engine calls and element-wise ops), which shows where a
run is stuck.

## Verification

On the host, with a software stand-in for `AIAccelerator` that implements the same
semantics:

- The accelerated INT8 decoder path matches the scalar INT8 loop exactly (differences only
  from float summation order), and a full synthesis of "Hello world, this is a test." gives
  identical audio (15.2dB SNR against float, the existing INT8 decoder trade-off).
- The ops quantized on the fly are within about 0.5% relative RMS error of float, across
  chunked, dilated and partial-tile layers and transposed convolutions with stride > k.
- With every op accelerated, the INT8 duration predictor changes two phoneme durations
  (221 instead of 223 frames for that sentence), so the audio is about 1% shorter.
  Log-spectrogram correlation with the float output is 0.90, against 0.98 for the INT8
  decoder alone; part of that gap is the resulting misalignment.
- `TINYTTS_FIXED_POINT`: the INT8 conv results differ from the float rescaling by less than
  1e-7 relative; the integer quantization gives the same values as the float one on 2
  million random inputs; fixed-point attention is 53dB SNR against float attention. A full
  synthesis with all of it stays at the INT8 decoder's 15.2-15.5dB against float.
- The windowed attention matches the full loop to float rounding (100dB SNR).

The accelerator itself is verified on the board: `compute()` calls with random shapes,
weights and activations match a CPU dot product (11,600 calls at 27MHz, 12,000 at 54MHz).

## Measured on the board

`examples/timing_benchmark` with `TINYTTS_PROFILE`: one `speak("Hello world!")` (127-129
frames, 1.5s of audio) with Boot Mode: Flash + SDRAM, Clock Speed: Overclocked (54MHz),
Hardware Multiply/Divide and AI Accelerator enabled, after each optimization step:

| Stage | Working engine | + SRAM, faster rescale/quantize | + Q24 rescale | + integer quantize, attention |
|---|---|---|---|---|
| G2P | 189s | 189s | 189s | 189s |
| encoder | 14.9s | 13.0s | 12.2s | 6.5s |
| duration_predictor | 18.4s | 16.1s | 15.4s | 10.0s |
| flow | 818s | 712s | 682s | 226s |
| vocoder | 4,949s | 3,364s | 1,852s | 1,586s |
| **total** | **100 min** | **72 min** | **46 min** | **34 min** |

About 1,400 times slower than real time. Where the remaining time goes (last run):

| Operation | Time |
|---|---|
| INT8 conv1d on the CPU (the vocoder's 2-8-channel layers) | about 920s |
| element-wise ops (mostly the vocoder's leaky ReLU and residual adds) | 280s |
| INT8 conv1d on the engine | 247s (81s of it in `compute()`) |
| G2P (neural model) | 189s |
| attention | 163s |
| INT8 quantization | 155s |

Notes:

- **G2P**: `default_cmudict_slim` leaves out words the neural G2P model predicts correctly -
  including "hello" and "world" - and each model step reads 393KB of INT8 weights from flash
  one byte at a time (without Tools > Flash Cache every flash read is one SPI transaction of
  about 510 clocks). Use the full `default_cmudict` (weights + full dictionary fit the flash),
  or copy the model to the SDRAM heap before `setDictionaryModel()`.
- **Weights in flash**: every conv layer reads its weights from flash on every call.
  Copying `default_weights` into a `malloc()`ed buffer before `setWeights()` avoids that;
  weights, working memory and the code image fit the 8MB SDRAM, though not with much room
  to spare.
- **Memory**: about 34KB of code and data in SRAM, about 90KB of code in SDRAM (of 1MB), the
  model data in flash, and a peak heap of about 3.5MB for this sentence.
