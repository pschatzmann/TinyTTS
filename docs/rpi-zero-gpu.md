# Raspberry Pi Zero: the flow and the vocoder on the GPU

On the CPU, the Pi Zero W takes ~15.3 s for "Hello world!", ~10x slower than real time. The
ARM1176 has no NEON, so the CPU path has little left to give. The board's VideoCore IV GPU
has 12 QPUs: 16-wide float32 SIMD units at 300 MHz on this board, with a multiply and an
add in every instruction. That is a peak of about 14 G multiply-adds per second.

`QpuFlow` and `QpuVocoder` (`src/TinyTTS/`) run the flow and the vocoder on them. The
results match the CPU to rounding. Measured on a Pi Zero W:

| "Hello world!" (`t_y` = 128, 1.49 s of audio) | CPU | `--qpu` |
|---|---:|---:|
| flow | 1,333 ms | **33 ms** (40x) |
| vocoder | 13,809 ms | **308 ms** (45x) |
| whole synthesis (G2P, encoder, duration predictor, flow, vocoder) | ~15 s | **~0.58 s** |
| whole run of the CLI, including startup | 15.3 s | **1.0 s** -- faster than real time |

The CPU's flow time grows faster than linearly with the sentence, but the GPU's barely
does:

| Frames | CPU flow | GPU flow |
|---:|---:|---:|
| 128 | 1,239 ms | 30 ms |
| 349 | 7,298 ms | 80 ms |
| 512 | 14,622 ms | 135 ms |

The vocoder on the GPU takes 306 ms at 128 frames, 821 ms at 349 and 1.25–1.5 s at 512.
What's left on the CPU for "Hello world!" is ~240 ms: G2P, the text encoder and the
duration predictor.

## Using it

1. **Board setup.** Disable the kernel's KMS graphics driver, which otherwise owns the GPU's
   3D block (V3D). In `/boot/firmware/config.txt`:

   ```
   #dtoverlay=vc4-kms-v3d
   ```

   Then reboot, and check that `lsmod | grep vc4` prints nothing. Keep `gpu_mem` at 64 MB or
   more (`gpu=64M`, the default on Raspbian 13). This was only tried on a headless board.
   Measured on Raspbian 13 (trixie), kernel 6.18.39, firmware from 2026-05-21.
2. **Build the CLI** with `-DTINYTTS_RPI_QPU=ON` (`docs/desktop.md`). Building on the Pi
   Zero itself takes about 30 minutes, almost all of it compiling the model data headers.
3. **Run it as root,** because it needs `/dev/vcio` and `/dev/mem`:

   ```
   sudo ./build/desktop/tinytts --qpu "Hello world!"
   ```

   Weights and buffers are prepared for sentences of up to `--qpu-frames` frames (default
   512, ~6 s of audio at 44.1 kHz). That takes ~0.2 s at startup and ~30 MB of GPU memory.
   Longer sentences run on the CPU, with a message on stderr.

   With `gpu=64M`, the firmware hands out at most ~41 MB in one block, enough for
   `--qpu-frames 760` (~9 s of audio). More needs a larger `gpu_mem`, which leaves less
   memory for Linux.

In your own program:

```cpp
#include "TinyTTS/QpuFlow.h"
#include "TinyTTS/QpuVocoder.h"

// After tts.begin(); these must outlive tts's use of them.
QpuConv gpu;
QpuVocoder vocoder;
QpuFlow flow;
const TinyTTSCore& core = tts.core();
if (gpu.begin(QpuVocoder::arenaBytes(core.weights(), 512) + QpuFlow::arenaBytes(512)) &&
    vocoder.begin(core.weights(), core.vocoder(), gpu, 512) && flow.begin(core.weights(), gpu, 512)) {
  tts.setVocoder(vocoder.vocoderFn());
  tts.setFlow(flow.flowFn());
}
```

Each also works on its own, with its own GPU memory:
`vocoder.begin(tts.core())` and `flow.begin(tts.core())`.

## How it works

| Option | Verdict |
|---|---|
| QPU programs in assembly, run directly (like hello_fft's GPU_FFT, py-videocore) | **used** |
| VC4CL (OpenCL 1.2) | same hardware access underneath, plus a heavy, unmaintained toolchain |
| OpenGL ES 2 through Mesa | no float render targets on VC4: floats packed into RGBA8, one texture fetch per tap |
| Vulkan | not on VideoCore IV (v3dv starts with the Pi 4) |

### Files

| File | Role |
|---|---|
| `src/TinyTTS/QpuDevice.h` | GPU memory, QPU power, running programs |
| `src/TinyTTS/QpuConv.h` | the layers: conv1d, ConvTranspose1d, attention, LayerNorm |
| `src/TinyTTS/QpuVocoder.h` | the decoder as a sequence of those layers |
| `src/TinyTTS/QpuFlow.h` | the flow as a sequence of those layers |
| `src/TinyTTS/qpu/` | the assembled kernels: 37, 6,803 instructions |
| `research/rpi_qpu/gen_qpu_kernels.py` | generates the kernel source; assembled with [vc4asm](https://github.com/maazl/vc4asm), see `research/rpi_qpu/CMakeLists.txt` |
| `research/rpi_qpu/` | also the tests and benchmarks |

### The convolution kernel

One kernel family computes every convolution and linear layer. Activations stay in GPU
memory as channel-major rows with 32 zeros on each side, so a tap never needs a bounds
check.

- **Parallelism.** Each QPU processes blocks of 16 consecutive time steps, one per SIMD
  lane, for a tile of up to 16 output channels. The accumulators stay in registers.
- **Inner loop.** Per tap, the 16 inputs come from the TMU, prefetched two taps ahead. The
  tile's weights come from the uniform stream, broadcast to all lanes. Every instruction is
  one multiply-add: a multiply on one ALU, while the other adds the previous product.
- **ConvTranspose1d** runs polyphase: accumulator `col * stride + r` computes output time
  `m * stride + r`. A vertical VPM-to-memory DMA writes the `stride` phases out interleaved,
  in output order.
- **Epilogue.** After the taps, each variant can do any of the following, so no layer needs
  a separate pass:
  - add a residual;
  - scale and accumulate (the mean of the 3 resblocks);
  - zero the lanes past the end of the sequence, which keeps the next layer's padding at zero;
  - store the result and/or its LeakyReLU (ReLU with slope 0);
  - store the result time-major instead (the flow's keys and values).
- **Synchronisation.** Outputs go through the VPM, which all 12 QPUs share, under the
  hardware mutex.

The decoder is 97 GPU runs. Only z goes to the GPU, and only the audio comes back. The
speaker conditioning (one small linear layer) and the final tanh run on the CPU.

### The flow

The flow is 4 coupling layers, run in reverse. Each one is:
- a linear layer;
- a 3-layer transformer: windowed relative-position attention with 2 heads of 16 channels,
  an FFN of k=5 convolutions (32 → 128 → 32), and LayerNorm;
- a linear layer subtracted from the second half of the channels.

That's 104 GPU runs. Two new kernels serve it:

- **Attention.** One job is 16 queries (one per lane) of one head. q is pre-scaled by
  log2(e)/√16, so the scores are in log2 units.
  - **Pass 1** finds the maximum score.
  - **Pass 2** computes each weight as 2^(s − max), adds it to the sum, and accumulates the
    weight times v. The exponential uses a degree-7 polynomial and adds the integer part
    directly to the float's exponent; no SFU or lookup table is involved.
  - **Keys and values** stream through the uniform FIFO, broadcast to all lanes. That's why
    the projections write them time-major.
  - **Relative-position terms** only matter within ±4 of each query. The loop over the keys
    runs in three parts: before the band, the band, and after it. Only the band iterations
    add the per-lane relative-key term (selected with condition flags) and keep the weights
    for the relative-value term.
  - **Work per score** is 16 multiply-adds in pass 1 and about 60 instructions in pass 2,
    whatever the sentence length.
- **LayerNorm** normalizes the 32 channels per time step. The reciprocal square root comes
  from the SFU, refined by two Newton steps.

**Folded into the weights.** The channel flips between the coupling layers cost nothing:
they are folded into the order of the pre- and post-layer weights. The speaker conditioning
is folded into a LayerNorm's beta.

### Two firmware surprises

Both are handled in `QpuDevice.h`.

- **The firmware's `execute_qpu` mailbox call returns before the programs have finished.**
  With a 1.5 MB job it came back after 0.2 ms, and the QPUs kept writing in the background.
  The programs are therefore queued through the V3D registers (`V3D_SRQUA`/`V3D_SRQPC`), and
  the host waits for the completion count in `V3D_SRQCS`. That also skips the firmware
  round trip: launching 12 programs takes ~0.05 ms.
- **The firmware reserves every QPU against user programs** (`V3D_SQRSV0/1` =
  `0x11111111`/`0xffff1111`). `QpuDevice` clears those bits in `begin()` and restores them
  in `end()`. Programs queued while the QPUs are still reserved stay in the queue, and only
  a reboot clears it. `begin()` refuses to start if it finds any.

Memory comes from the firmware allocator (`mem_alloc`, flag `0xC` as in hello_fft), mapped
uncached. Before every run the host clears the V3D L2 cache and the per-slice caches.

One assembler trap: vc4asm reads `1e-05` as the expression `1e - 05`, i.e. -4.0. The
generator therefore writes every float constant as its bit pattern.

## Tests and measurements

All of these run on the Pi as root and are built with `research/rpi_qpu/CMakeLists.txt`.

| Program | Checks |
|---|---|
| `hello_qpu` | QPU access works at all |
| `test_qpu_conv` | every kernel against `tinytts::ops` or a reference, at odd lengths: 177 checks, all within 1e-5 relative error, tails exactly zero |
| `test_qpu_vocoder weights.bin` | the whole decoder against the CPU `Vocoder`: 81.4 dB SNR at 117 frames |
| `test_qpu_flow weights.bin [frames]` | the whole flow against the CPU `Flow`: 105–111 dB SNR at 128–512 frames |
| `bench_conv1d` | per-layer timing of the 45 resblock conv shapes |

End to end, the CLI's 16-bit output for "Hello world!" with `--qpu` differs from the CPU
version by at most 10 LSB (74.6 dB SNR).

Per-layer, from the first version of the conv kernel (the same inner loop), at 117 frames:

| Stage | T | C | CPU ms (k = 3 / 7 / 11) | QPU ms (k = 3 / 7 / 11) |
|---|---:|---:|---|---|
| 1 | 936 | 32 | 34 / 76 / 122 | 0.62 / 1.2 / 1.8 |
| 2 | 7,488 | 16 | 75 / 164 / 252 | 1.9 / 2.5 / 3.6 |
| 3 | 14,976 | 8 | 66 / 132 / 203 | 1.9 / 1.9 / 2.0 |
| 4 | 29,952 | 4 | 57 / 68 / 100 | 1.9 / 1.9 / 2.0 |
| 5 | 59,904 | 2 | 39 / 75 / 110 | 2.0 / 2.0 / 2.1 |

- **The 32-channel layers reach ~5.8 G multiply-adds per second,** about 40% of the GPU's
  peak.
- **The small-channel layers stay near 2 ms** whatever the kernel size. There, writing each
  16-sample row out (one DMA under the shared mutex) is the limit, not the arithmetic.

## Next steps

1. **The vocoder is now about half of the synthesis time.** Its small-channel layers are limited
   by writing one 16-sample row per DMA. Writing several rows per DMA, or more samples per
   block, would help.
2. **The CPU stages:** G2P, the text encoder and the duration predictor take ~240 ms for
   "Hello world!". The text encoder is the same kind of transformer as the flow, so it could
   reuse these kernels.
3. **Longer sentences on the GPU** are limited by GPU memory (see above). Splitting the
   vocoder into overlapping windows with `ChunkedVocoder` would need the kernels to mask the
   start of a window as well as its end.
