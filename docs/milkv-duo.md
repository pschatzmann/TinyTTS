# Milk-V Duo 256M / Duo S (TPU)

The Milk-V Duo 256M (SG2002, 256MB) and Duo S (SG2000, 512MB) run Linux on a 1GHz T-Head
C906 RISC-V core and have a TPU rated at 1 TOPS INT8 (it also runs BF16). TinyTTS runs
there as the desktop CLI (`docs/desktop.md`), with the decoder (vocoder) moved to the TPU.
The decoder is 87-90% of the synthesis time on every CPU measured so far
(`docs/performance.md`).

**Status: not yet run on the board.** Everything below is checked on a PC: the chunked
decoding against the one-pass decoder (in PyTorch and in `test/main.cpp`), the compiled
TPU models in TPU-MLIR's bit-accurate simulator, and the TPU wrapper against the real
`cviruntime.h`. The simulator gives no timing, so there are no speed numbers for the Duo
yet.

## What runs where

| Stage | Where | Why |
|---|---|---|
| G2P, text encoder, duration predictor | C906 CPU | small, variable length |
| flow | C906 CPU | attention over the whole utterance; see "Next steps" |
| decoder | TPU, in fixed-size windows | ~90% of the work, purely convolutional |

The TPU SDK's runtime (`libcviruntime.so` in
[tpu-sdk-sg200x](https://github.com/milkv-duo/tpu-sdk-sg200x)) is built for RISC-V/musl
only, so use the default RISC-V Linux image, not the Cortex-A53 (ARM) boot option.

## Why windows

TPU-MLIR compiles static shapes only, and the number of frames differs for every sentence.
`ChunkedVocoder` (`src/TinyTTS/ChunkedVocoder.h`) runs the decoder on fixed windows of `W`
frames (128 by default, 65,536 samples). Each window covers `W - 2*context` new frames plus
`context` frames on each side, and only the middle of its output is kept. The model has a
third input, `mask`, which is 0 for the zero frames past either end of the utterance and is
multiplied in before every convolution. That reproduces the zero padding the one-pass
decoder sees at the utterance's ends.

`research/milkv_duo/export_decoder_tpu.py --sweep` measures the error against the one-pass
decoder, float32, worst of 20 sentences plus 4 random-phoneme sequences:

| Context (frames per side) | W=64 | W=128 | Extra compute at W=128 |
|---|---:|---:|---:|
| 8 | 21.9 dB | 31.2 dB | x1.14 |
| 10 | 62.3 dB | 72.5 dB | x1.19 |
| **12** (default) | **111.6 dB** | **115.3 dB** | **x1.23** |
| 16 | 111.6 dB | 115.3 dB | x1.33 |

12 frames reproduce the one-pass result to float rounding; `test/main.cpp` checks the same
with the C++ decoder (>100dB for W = 32, 64, 128). Without the mask, the windows at the
utterance's ends are off by up to 13dB SNR when the utterance doesn't start/end in silence.

Audio is passed on window by window (`TinyTTS::setVocoder()`), so playback can start after
the first window (1.25s of audio) instead of after the whole sentence.

## Step 1: export the decoder (PC)

```bash
cd research && source venv/bin/activate     # see docs/research.md
python milkv_duo/export_decoder_tpu.py      # --window 128 --context 12
```

Writes `research/milkv_duo/out/`: `decoder_w128.onnx` (inputs `z` [1,32,W], `g` [1,128,1],
`mask` [1,1,W], output `audio` [1,1,W*512]), 71 calibration windows from 20 real
sentences, and a test input/reference. It also checks the ONNX model's chunked output in
onnxruntime (108dB SNR against PyTorch).

The mask is upsampled to each layer's rate with unsqueeze/expand/reshape, not
`F.interpolate()`: TPU-MLIR 1.30 crashes compiling `Resize` for cv181x (`std::length_error`
in `tpuc-opt`, even for a single [1,1,128] tensor). Every other op in the model compiles,
including the 2-channel convolutions over 65,536 samples.

## Step 2: compile for the TPU (PC)

Needs [TPU-MLIR](https://github.com/sophgo/tpu-mlir) 1.30 or newer. Tested with the
`sophgo/tpuc_dev:v3.4` Docker image plus `pip install tpu_mlir` (1.30.2). The image's own
Python 3.10 already has the dependencies, so `pip install --no-deps` on a downloaded wheel
is enough. Run the container as your own user, or the outputs in the repo end up owned by
root:

```bash
docker run -d --name tpumlir -v $HOME:/workspace sophgo/tpuc_dev:v3.4 sleep infinity
docker exec tpumlir pip install tpu_mlir
docker exec -u $(id -u):$(id -g) -e HOME=/tmp -w /workspace/<path to TinyTTS>/research \
  tpumlir milkv_duo/build_cvimodel.sh 128
docker exec -u $(id -u):$(id -g) -e HOME=/tmp -w /workspace/<path to TinyTTS>/research \
  tpumlir python3 -I milkv_duo/validate_cvimodel.py
```

If `pip` in the container can't resolve hostnames, the container inherited a DNS server it
can't reach: add `{"dns": ["8.8.8.8", "1.1.1.1"]}` to `/etc/docker/daemon.json` and restart
Docker.

`build_cvimodel.sh` builds variants for processor `cv181x` (SG2002/SG2000).
`validate_cvimodel.py` runs them in the simulator over the 20 sentences, chunked as on the
device, against the float32 one-pass decoder, and writes WAV files for listening:

| File | Precision | SNR median | SNR worst | Cosine worst |
|---|---|---:|---:|---:|
| `decoder_w128_bf16.cvimodel` | BF16 | **27.4 dB** | **23.9 dB** | 0.9981 |
| `decoder_w128_int8.cvimodel` | INT8, per-tensor activation scales | -2.5 dB | -8.1 dB | 0.119 |
| `decoder_w128_mix.cvimodel` | INT8, sensitive layers in BF16 | pending | | |

**Use BF16.** It's as close to float as the CPU decoder this library ships (weights-only
INT8, 22-27dB, `docs/performance.md`). INT8 output is mostly noise: TPU-MLIR's own INT8
reference is already down to 4.9dB at its worst layer on the test window, and the compiled
model differs from that reference again (cosine 0.25). That matches earlier experience in
this project: per-tensor INT8 activation scales measured only 12.4dB with TFLite
(`docs/research.md`). The mixed variant's layer search takes about 2 hours.

## Step 3: cross-compile the CLI (PC)

Needs the T-Head musl toolchain from [host-tools](https://github.com/milkv-duo/host-tools)
(`gcc/riscv64-linux-musl-x86_64`) and a checkout of
[tpu-sdk-sg200x](https://github.com/milkv-duo/tpu-sdk-sg200x):

```bash
cmake -S . -B build_duo -DCMAKE_TOOLCHAIN_FILE=cmake/milkv-duo.cmake \
  -DDUO_TOOLCHAIN_DIR=$HOME/host-tools/gcc/riscv64-linux-musl-x86_64 \
  -DTINYTTS_BUILD_DESKTOP_MAIN=ON \
  -DTINYTTS_CVI_TPU=ON -DTPU_SDK_DIR=$HOME/tpu-sdk-sg200x
cmake --build build_duo -j
```

`cmake/milkv-duo.cmake` uses the SDK's own CPU flags (`-mcpu=c906fdv
-march=rv64gcv0p7_zfh_xthead`). This hasn't been tried yet: the desktop CLI also builds
arduino-audio-tools and miniaudio, which haven't been built with this toolchain before.

## Step 4: run on the board

```bash
scp build_duo/desktop/tinytts research/milkv_duo/out/decoder_w128_*.cvimodel root@192.168.42.1:
scp tpu-sdk-sg200x/lib/*.so* root@192.168.42.1:/root/tpu-lib/

# on the Duo
export LD_LIBRARY_PATH=/root/tpu-lib
./tinytts --tpu-model decoder_w128_bf16.cvimodel -o hello.wav "Hello world!"
./tinytts --no-play "Hello world!"        # CPU-only decoder, for comparison
```

The CLI prints the time per stage on stderr. For a first check, compare `t_y` (128 for
"Hello world!", as everywhere else in this project) and listen to `hello.wav`. The Duo has
no audio output of its own, so `-o`/`--stdout` is the normal way to get the audio out.

`CviTpuVocoder` expects all model inputs and outputs in float32. That's TPU-MLIR's default
unless `--quant_input`/`--quant_output` are given, and it reports an error otherwise. In
the compiled model the tensors are 4-D (`z` [1,32,128,1], output [1,1,65536,1]), since
TPU-MLIR runs the 1-D convolutions as 2-D ones. The memory layout is the same.

## Things to check first on the hardware

- **TPU memory**: the runtime allocates from the ION heap the Duo image reserves. A 128-frame
  window needs a few MB of activations. If `CVI_NN_RegisterModel` fails, try `W=64`.
- **Whether the TPU pays off**: the decoder needs about 0.54 GMAC per second of audio
  (0.65 GMAC per 128-frame window, counted from the ONNX graph), so real time takes about
  1.1 GOPS against the TPU's rated 1,000 (INT8; BF16 is slower). But 72% of that work is in
  layers with only 4-16 input channels, a poor fit for a wide accelerator, so utilization
  will be low. The simulator doesn't give timing. Compare the `vocoder:` line of a
  `--tpu-model` run against a CPU run.
- **The new bottleneck**: with the decoder off the CPU, flow is likely to dominate. On a Pi
  Zero (1GHz ARMv6, no SIMD) flow alone took 1.9s for 1.36s of audio.

## Next steps

1. **Flow on the TPU.** Flow masks padded frames everywhere (its attention and
   convolutions take `y_mask`), so it can be compiled for a few fixed lengths (e.g.
   256/512/1024 frames) and padded to the next one, without the decoder's overlap. Checked
   in PyTorch: padding 79 frames to 256 or 300 to 512 gives the unpadded result to 133dB
   SNR. Still to check: whether TPU-MLIR converts its relative-position attention, and
   which precision keeps it accurate (BF16 most likely).
2. **Overlap CPU and TPU.** `CVI_NN_ForwardAsync()` lets the CPU run the next sentence's
   G2P/encoder/flow while the TPU decodes the current one.
3. **RVV on the C906.** The C906 has a vector unit (RVV 0.7.1, T-Head's `xtheadvector`
   variant). Whatever stays on the CPU could use it for `linear()`/`conv1d()`, as NEON does
   on ARM. Needs the T-Head toolchain's intrinsics, and a board to test on.
