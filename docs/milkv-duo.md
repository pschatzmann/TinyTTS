# Milk-V Duo 256M / Duo S (TPU)

The Milk-V Duo 256M (SG2002, 256MB) and Duo S (SG2000, 512MB) run Linux on a 1GHz T-Head
C906 RISC-V core and have a TPU rated at 1 TOPS INT8 (it also runs BF16). TinyTTS runs
there as the desktop CLI (`docs/desktop.md`), with the decoder (vocoder) and the flow moved
to the TPU. The decoder is 87-90% of the synthesis time on every CPU measured so far, the
flow most of the rest (`docs/performance.md`).

**Status: not yet run on the board.** Everything below is checked on a PC: the chunked
decoder and the padded flow against the one-pass PyTorch model, the compiled TPU models in
TPU-MLIR's bit-accurate simulator, and the C++ TPU wrappers against the real `cviruntime.h`
and, with a fake runtime, against the CPU pipeline (`test/test_cvi.cpp`). The simulator
gives no timing, so there are no speed numbers for the Duo yet.

## What runs where

| Stage | Where | Why |
|---|---|---|
| G2P, text encoder, duration predictor | C906 CPU | small, variable length |
| flow | TPU, padded to fixed lengths (CPU for longer sentences) | attention over the whole sentence |
| decoder | TPU, in fixed-size windows | ~90% of the work, purely convolutional |

The TPU SDK's runtime (`libcviruntime.so` in
[tpu-sdk-sg200x](https://github.com/milkv-duo/tpu-sdk-sg200x)) is built for RISC-V/musl
only, so use the default RISC-V Linux image, not the Cortex-A53 (ARM) boot option.

TPU-MLIR compiles static shapes only, and the number of frames differs for every sentence.
The decoder and the flow get around that differently.

## Decoder: overlapping windows

`ChunkedVocoder` (`src/TinyTTS/ChunkedVocoder.h`) runs the decoder on fixed windows of `W`
frames (128 by default, 65,536 samples at 44.1 kHz). Each window covers `W - 2*context` new
frames plus `context` frames on each side, and only the middle of its output is kept. The
model has a third input, `mask`, which is 0 for the zero frames past either end of the
utterance and is multiplied in before every convolution. That reproduces the zero padding
the one-pass decoder sees at the utterance's ends.

`research/milkv_duo/export_decoder_tpu.py --sweep` measures the error against the one-pass
decoder, float32, worst of 20 sentences plus 4 random-phoneme sequences (released 44.1 kHz
model):

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

The mask is upsampled to each layer's rate with unsqueeze/expand/reshape, not
`F.interpolate()`: TPU-MLIR 1.30 crashes compiling `Resize` for cv181x (`std::length_error`
in `tpuc-opt`, even for a single [1,1,128] tensor). Every other op in the decoder compiles,
including the 2-channel convolutions over 65,536 samples.

## Flow: fixed lengths

The flow's attention spans the whole sentence, so it can't be split into windows. But its
convolutions and attention all take `y_mask`, so a sentence padded to a fixed length (mask 0
on the padding) gives the unpadded result. `BucketedFlow` (`src/TinyTTS/BucketedFlow.h`)
runs a sentence on the smallest of several fixed-length models that fits: 128, 256, 512
and 1024 frames by default (1.5s to 11.9s of speech at 44.1 kHz). Longer sentences run on
the CPU flow; the desktop CLI already splits text into sentences. Padded vs. unpadded:
133-138dB SNR (`research/milkv_duo/export_flow_tpu.py`).

Three changes to the exported graph get it through TPU-MLIR for cv181x, all exact in float
(checked against the upstream flow):

- **Relative-position attention as a band.** Upstream builds the relative-position terms
  with `gather` over [L, 2L-1] indices; TPU-MLIR runs `GatherElements` on the CPU. The
  relative embeddings only cover +-4 frames, so the terms are 9 diagonals, moved between
  band and [L, L] layout with pad/reshape/slice.
- **The attention mask as a [1,1,1,L] key mask.** Upstream builds a [1,1,L,L] mask as the
  outer product of the frame mask; TPU-MLIR 1.30 compiles that two-sided broadcast multiply
  wrongly for cv181x (0dB SNR, even for an all-ones mask), which brought the whole flow down
  to 11-15dB. The rows of padded queries then differ, but they're masked out before they
  reach any valid frame. `masked_fill(mask == 0)` also becomes an additive -1e4, since the
  CV18xx backend can't compile the compare.
- **Channel flips as 1x1 convolutions** (an anti-diagonal 0/1 weight): the CV18xx backend
  only reverses along axis 0 or 3.

The keys are also centered before q.k (`k - mean(k)`). That changes all scores in a row by
the same amount, which softmax ignores, but removes a per-row offset of up to +-90 that
otherwise costs the scores most of BF16's 8-bit mantissa (attention layer alone: 28 -> 38dB).

## Step 1: export (PC)

```bash
cd research && source venv/bin/activate     # see docs/research.md
python milkv_duo/export_decoder_tpu.py      # --window 128 --context 12
python milkv_duo/export_flow_tpu.py         # --lengths 128 256 512 1024
```

Both take `--checkpoint G_*.pth` for a model from `research/training/` (default: the
released 44.1 kHz model); see "Models at other sample rates". They write to
`research/milkv_duo/out/`, with the sample rate in every name:

- `decoder_44100hz_w128.onnx`: inputs `z` [1,32,W], `g` [1,128,1], `mask` [1,1,W], output
  `audio` [1,1,W*hop]. Plus `_calib/` (71 windows from 20 real sentences, for INT8
  calibration), `_test_input.npz`/`_test_ref.npz` and `_utterances.npz` (for the
  validation, with the model's hop and sample rate). The script also checks the chunked
  ONNX output in onnxruntime (108dB SNR against PyTorch).
- `flow_44100hz_l<L>.onnx` per length: inputs `z_p` [1,32,L], `y_mask` [1,1,L], `g`
  [1,128,1], output `z` [1,32,L]. Plus test inputs and `flow_44100hz_utterances.npz`.

## Step 2: compile for the TPU (PC)

Needs [TPU-MLIR](https://github.com/sophgo/tpu-mlir) 1.30 or newer. Tested with the
`sophgo/tpuc_dev:v3.4` Docker image plus `pip install tpu_mlir` (1.30.2). The image's own
Python 3.10 already has the dependencies, so `pip install --no-deps` on a downloaded wheel
is enough. Run the container as your own user, or the outputs in the repo end up owned by
root:

```bash
docker run -d --name tpumlir -v $HOME:/workspace sophgo/tpuc_dev:v3.4 sleep infinity
docker exec tpumlir pip install tpu_mlir
run() { docker exec -u $(id -u):$(id -g) -e HOME=/tmp -w /workspace/<path to TinyTTS>/research tpumlir "$@"; }

run milkv_duo/build_cvimodel.sh decoder_44100hz_w128
run milkv_duo/build_flow_cvimodels.sh 44100                 # 128 256 512 1024
run python3 -I milkv_duo/validate_cvimodel.py out/decoder_44100hz_w128_bf16.cvimodel
run python3 -I milkv_duo/validate_flow_cvimodel.py --sample-rate 44100 out/decoder_44100hz_w128_bf16.cvimodel
```

If `pip` in the container can't resolve hostnames, the container inherited a DNS server it
can't reach: add `{"dns": ["8.8.8.8", "1.1.1.1"]}` to `/etc/docker/daemon.json` and restart
Docker.

The build scripts compile for processor `cv181x` (SG2002/SG2000), each model checked
against its ONNX output in the simulator. The validation scripts run the models in the
simulator over the 20 sentences (chunked/padded as on the device) against the float32
model, and write WAV files for listening. Released 44.1 kHz model:

| Model | Precision | SNR median | SNR worst | Cosine worst |
|---|---|---:|---:|---:|
| `decoder_44100hz_w128_bf16.cvimodel` | BF16 | **27.4 dB** | **23.9 dB** | 0.9981 |
| `decoder_44100hz_w128_int8.cvimodel` | INT8, per-tensor activation scales | -2.5 dB | -8.1 dB | 0.119 |
| `decoder_44100hz_w128_mix.cvimodel` | INT8, 47 sensitive layers in BF16 | -2.3 dB | (2 sentences) | |
| `flow_44100hz_l*_bf16.cvimodel` (z) | BF16 | **43.0 dB** | **42.1 dB** | |
| flow + decoder, both BF16 (audio) | BF16 | 15.2 dB | 11.0 dB | |

**Use BF16.** The decoder is as close to float as the CPU decoder this library ships
(weights-only INT8, 22-27dB, `docs/performance.md`). The flow is closer still, 42-45dB in
all four lengths. With both on the TPU, the waveform SNR drops to 11-21dB, but that's phase:
the decoder turns the flow's small errors into slightly shifted waveforms. The log-spectrogram
correlation with float is the same with or without the TPU flow (0.900 vs. 0.899 for
"Hello world!", 0.907 vs. 0.906 for the next sentence; at least 0.894 over all 20), so the
TPU flow adds no audible difference of its own. For scale: synthesizing the same sentence
with another noise seed gives -3dB waveform SNR. Listen to `float32_44100hz_<n>.wav`,
`decoder_44100hz_w128_bf16_<n>.wav` and `tpu_flow_decoder_44100hz_<n>.wav` in
`research/milkv_duo/out/`. INT8 output is mostly noise: TPU-MLIR's own INT8 reference is
already down to 4.9dB at its worst layer on the test window, and the compiled model differs
from that reference again (cosine 0.25). That matches earlier experience in this project:
per-tensor INT8 activation scales measured only 12.4dB with TFLite (`docs/research.md`).
Mixed precision doesn't rescue it: TPU-MLIR's layer search kept 47 layers in BF16, its own
reference for that mix is down to 12.3dB at the worst layer, and the compiled model is noise
again (cosine 0.34 on the test window, -2.3 and -1.8dB on the first two sentences). The
build scripts therefore build BF16 only; `build_cvimodel.sh <name> all` also builds the
INT8 and mixed variants for comparison (the mixed search alone takes about 4 hours).

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
cd research/milkv_duo/out
scp ../../../build_duo/desktop/tinytts decoder_44100hz_w128_bf16.cvimodel flow_44100hz_l*_bf16.cvimodel root@192.168.42.1:
scp $HOME/tpu-sdk-sg200x/lib/*.so* root@192.168.42.1:/root/tpu-lib/

# on the Duo
export LD_LIBRARY_PATH=/root/tpu-lib
./tinytts --tpu-model decoder_44100hz_w128_bf16.cvimodel \
  --tpu-flow flow_44100hz_l128_bf16.cvimodel --tpu-flow flow_44100hz_l256_bf16.cvimodel \
  --tpu-flow flow_44100hz_l512_bf16.cvimodel --tpu-flow flow_44100hz_l1024_bf16.cvimodel \
  -o hello.wav "Hello world!"
./tinytts --no-play "Hello world!"        # all on the CPU, for comparison
```

The CLI prints the time per stage on stderr; the flow line says `(external)` when the TPU
ran it. For a first check, compare `t_y` (128 for "Hello world!", as everywhere else in this
project) and listen to `hello.wav`. The Duo has no audio output of its own, so
`-o`/`--stdout` is the normal way to get the audio out.

`CviTpuVocoder` and `CviTpuFlow` expect all model inputs and outputs in float32. That's
TPU-MLIR's default unless `--quant_input`/`--quant_output` are given, and they report an
error otherwise. In the compiled models the tensors are 4-D (`z` [1,32,128,1], output
[1,1,65536,1]), since TPU-MLIR runs the 1-D convolutions as 2-D ones. The memory layout is
the same.

## Models at other sample rates

A model trained at a lower sample rate (`docs/training.md`) needs its own TPU models,
exported from the same checkpoint as its `weights.bin`:

```bash
python export_weights_and_vectors.py --checkpoint runs/x/G_5000.pth --out-dir runs/x/export
python milkv_duo/export_decoder_tpu.py --checkpoint runs/x/G_5000.pth --sweep   # check the context
python milkv_duo/export_decoder_tpu.py --checkpoint runs/x/G_5000.pth
python milkv_duo/export_flow_tpu.py --checkpoint runs/x/G_5000.pth
run milkv_duo/build_cvimodel.sh decoder_16000hz_w128
run milkv_duo/build_flow_cvimodels.sh 16000
```

The scripts take the sample rate, the output samples per frame (hop) and the vocoder's
upsample stack from the checkpoint (`research/training/audio_config.py`). Points to note:

- **Mismatches are caught at startup.** A decoder from another model than the weights would
  play at the wrong speed and pitch. After `begin()`, the desktop CLI calls
  `CviTpuVocoder::checkModel()` and `CviTpuFlow::checkModel()`: they compare the decoder's
  samples per frame with the weights, and the sample rate in each model's file name with
  the weights' rate. Models for 16, 22.05 and 24 kHz all have 256 samples per frame, so
  keep the names as the scripts write them.
- **The context** (12 frames) was measured on the released 5-stage decoder. Lower rates
  drop stages from the end, which shouldn't need more, but `--sweep` checks it per model.
- **The flow buckets are frames**, so they cover more time at lower rates: at 16 kHz (256
  samples per frame) the 1024-frame bucket holds 16.4s instead of 11.9s.
- **Less work per second of audio.** A 16 kHz model produces 2.8x fewer samples and drops
  the decoder's last stage, the long 2-channel layers that suit the TPU worst. Not measured
  yet: all numbers in this document are for the released 44.1 kHz model.

## Things to check first on the hardware

- **TPU memory**: the runtime allocates from the ION heap the Duo image reserves. A 128-frame
  decoder window needs a few MB of activations, the 1024-frame flow more (its attention
  scores alone are [2, 1024, 1024]). If `CVI_NN_RegisterModel` fails, try `W=64` or leave
  out the largest flow model.
- **Whether the TPU pays off**: the decoder needs about 0.54 GMAC per second of audio
  (0.65 GMAC per 128-frame window, counted from the ONNX graph), so real time takes about
  1.1 GOPS against the TPU's rated 1,000 (INT8; BF16 is slower). But 72% of that work is in
  layers with only 4-16 input channels, a poor fit for a wide accelerator, so utilization
  will be low. The simulator doesn't give timing. Compare the `vocoder:` and `flow:` lines
  of a TPU run against a CPU run.
- **Padding cost**: a sentence just over a bucket length pays for up to twice its frames
  (and four times the attention). If the flow on the TPU is slow, more bucket lengths
  (`--lengths`) help.

## Next steps

1. **Overlap CPU and TPU.** `CVI_NN_ForwardAsync()` lets the CPU run the next sentence's
   G2P/encoder/duration predictor while the TPU runs the current one's flow and decoder.
2. **RVV on the C906.** The C906 has a vector unit (RVV 0.7.1, T-Head's `xtheadvector`
   variant). Whatever stays on the CPU could use it for `linear()`/`conv1d()`, as NEON does
   on ARM. Needs the T-Head toolchain's intrinsics, and a board to test on.
