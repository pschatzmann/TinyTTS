# Milk-V Duo 256M / Duo S

The Milk-V Duo 256M (SG2002, 256MB) and Duo S (SG2000, 512MB) run Linux on a 1GHz T-Head
C906 RISC-V core and have a TPU rated at 1 TOPS INT8 (it also runs BF16). TinyTTS runs there
as the desktop CLI (`docs/desktop.md`). The decoder (vocoder) and the flow can run on the TPU:
the decoder is 87-90% of the synthesis time on every CPU measured so far, the flow most of
the rest (`docs/performance.md`).

**Status (Duo 256M):**

| | "Hello world!", 44.1 kHz model (1.49s of audio) | 16 kHz model (1.04s of audio) |
|---|---:|---:|
| CPU only | 8.85s | 4.04s |
| Flow and decoder on the TPU | **0.61s**, startup included | **0.67s** |

- The TPU makes the board faster than real time. Sentences longer than the largest flow model
  (1,024 frames, 11.9s of speech at 44.1 kHz) still run their flow on the CPU, which is slow.
- Audio output through an I2S amplifier (MAX98357A) needs a rebuilt kernel, which runs on the
  board; the sound on a connected amplifier hasn't been listened to yet.
- The Duo S uses the same SoC family and build, but hasn't been run.

## Contents

1. [Setting up the board](#setting-up-the-board)
   - [Flash the system image](#flash-the-system-image)
   - [Boot and log in](#boot-and-log-in)
   - [What's on the board](#whats-on-the-board)
2. [TinyTTS on the CPU](#tinytts-on-the-cpu)
   - [Toolchain](#toolchain)
   - [Build](#build)
   - [Run](#run)
3. [TinyTTS on the TPU](#tinytts-on-the-tpu)
   - [Overview](#overview)
   - [Install TPU-MLIR](#install-tpu-mlir)
   - [Export and compile the models](#export-and-compile-the-models)
   - [Build the CLI with the TPU runtime](#build-the-cli-with-the-tpu-runtime)
   - [Install and run](#install-and-run)
   - [Results](#results)
   - [The 16 kHz voice and other sample rates](#the-16-khz-voice-and-other-sample-rates)
4. [Audio output](#audio-output)
   - [Options](#options)
   - [Linux remote execution](#linux-remote-execution)
   - [I2S amplifier (MAX98357A)](#i2s-amplifier-max98357a)
   - [USB speaker](#usb-speaker)
5. [Technical background](#technical-background)
   - [Decoder: overlapping windows](#decoder-overlapping-windows)
   - [Flow: fixed lengths](#flow-fixed-lengths)
   - [Precision: BF16, not INT8](#precision-bf16-not-int8)
   - [Notes from the hardware](#notes-from-the-hardware)
6. [Next steps](#next-steps)

---

## Setting up the board

### Flash the system image

The board boots from a microSD card (4GB or more). Download the image from the
[duo-buildroot-sdk-v2 releases](https://github.com/milkv-duo/duo-buildroot-sdk-v2/releases).
The file name must match the board exactly:

| Board | Image |
|---|---|
| Duo 256M (SG2002) | `milkv-duo256m-musl-riscv64-sd_*.img.zip` |
| Duo S (SG2000) | `milkv-duos-musl-riscv64-sd_*.img.zip` |

Use the default RISC-V image, not the Cortex-A53 (ARM) boot option: the TPU runtime is built
for RISC-V/musl only.

```bash
unzip milkv-duo256m-musl-riscv64-sd_*.img.zip
lsblk                                   # find the card, e.g. /dev/sda
sudo dd if=milkv-duo256m-musl-riscv64-sd_*.img of=/dev/sdX bs=4M status=progress conv=fsync
sync
udisksctl power-off -b /dev/sdX         # eject before pulling the card
```

Two things that stop the board before Linux (no blue LED, no USB network):

- **An image for another board.** The first-stage loader (`fip.bin`) holds board-specific DDR
  and pin setup. To check a card, mount it on the PC: `strings boot/boot.sd | grep "board sg"`
  names the board (`sg2002_milkv_duo256m_...`), and so does `board=` in
  `rootfs/mnt/system/sdk-release`.
- **A card pulled while mounted.** Most desktops mount the card's partitions as soon as `dd`
  is done; pulling it without ejecting can corrupt the FAT boot partition.

### Boot and log in

Insert the card and connect the board to the PC with a USB-C data cable (not through a hub).
The red LED shows power; after 10-20 seconds the blue LED starts blinking, which means Linux
is up. The board then appears as a USB network adapter and gives the PC an address in
`192.168.42.x`; the board itself is `192.168.42.1`.

```bash
ssh root@192.168.42.1                   # password: milkv
ssh-copy-id root@192.168.42.1           # so ssh and scp don't ask for the password
ssh root@192.168.42.1 passwd            # change the password
```

On the first boot the SSH server generates its host keys, so the first connection can time
out "during banner exchange" even though `ping 192.168.42.1` already works: wait a moment
and try again.

### What's on the board

The default image is a minimal Buildroot system (Duo 256M, image from 2025-05-30):

| | |
|---|---|
| Kernel / libc | Linux 5.10, musl |
| CPU | T-Head C906, 1GHz, `rv64imafdvcsu` (with RVV 0.7.1), one core |
| RAM | 165MB for Linux; the other ~75MB is the ION heap reserved for the TPU and multimedia |
| Root file system | 739MB partition (about 470MB free), whatever the card's size |
| Boot partition | mounted at `/boot`: `fip.bin` (boot loader) and `boot.sd` (kernel + device tree) |
| TPU | runtime in `/mnt/system/lib` (`libcviruntime.so`), device `/dev/cvi-tpu0` |
| Tools | BusyBox, `python3`, `aplay`; no compiler and no package manager |
| Audio | no usable output out of the box, see [Audio output](#audio-output) |

Programs are cross-compiled on the PC and copied over with `scp`.

---

## TinyTTS on the CPU

### Toolchain

The CLI is cross-compiled with the T-Head musl toolchain from
[host-tools](https://github.com/milkv-duo/host-tools), the same GCC 10.2 the board's kernel
and libraries are built with. The repository holds several toolchains (several GB); a sparse
clone fetches only the one needed (about 900MB):

```bash
cd ~
git clone --depth 1 --filter=blob:none --sparse https://github.com/milkv-duo/host-tools.git
cd host-tools && git sparse-checkout set gcc/riscv64-linux-musl-x86_64
gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-g++ --version
```

### Build

From the TinyTTS checkout:

```bash
cmake -S . -B build_duo -DCMAKE_TOOLCHAIN_FILE=cmake/milkv-duo.cmake \
  -DDUO_TOOLCHAIN_DIR=$HOME/host-tools/gcc/riscv64-linux-musl-x86_64 \
  -DTINYTTS_BUILD_DESKTOP_MAIN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_duo -j
```

The result is `build_duo/desktop/tinytts`, a RISC-V binary that uses the board's
`/lib/ld-musl-riscv64v0p7_xthead.so.1`. The model, dictionary and G2P data are compiled in,
so it needs no other files on the board. `cmake/milkv-duo.cmake` uses the TPU SDK's own CPU
flags (`-mcpu=c906fdv -march=rv64gcv0p7_zfh_xthead -mabi=lp64d`).

The configure step clones arduino-audio-tools with its full history, which can take a long
time. If a desktop build in `build/` already has the sources, reuse them:

```bash
cmake -S . -B build_duo ... \
  -DFETCHCONTENT_SOURCE_DIR_ARDUINO-AUDIO-TOOLS=$PWD/build/_deps/arduino-audio-tools-src \
  -DFETCHCONTENT_SOURCE_DIR_MINIAUDIO=$PWD/build/_deps/miniaudio-src
```

### Run

```bash
riscv64-unknown-linux-musl-strip -o tinytts build_duo/desktop/tinytts   # optional, from the toolchain's bin/
scp tinytts data/ljspeech16k_weights.bin root@192.168.42.1:/root/

ssh root@192.168.42.1
./tinytts --no-play -o hello.wav "Hello world!"                                    # 44.1 kHz, compiled-in model
./tinytts --no-play --weights ljspeech16k_weights.bin -o hello16k.wav "Hello world!"  # 16 kHz
```

Copy the WAV back with `scp root@192.168.42.1:/root/hello.wav .` to listen to it, or see
[Linux remote execution](#linux-remote-execution). Without a speaker (see [Audio output](#audio-output)),
`-o` or `--stdout` is the way to get the audio out.

The CLI prints the time per stage on stderr. "Hello world!" on a Duo 256M, CPU only:

| Model | g2p | encoder | duration | flow | vocoder | total | audio |
|---|---:|---:|---:|---:|---:|---:|---:|
| released, 44.1 kHz | 48 ms | 32 ms | 52 ms | 808 ms | 7,781 ms | 8.85 s | 1.49 s |
| `ljspeech16k`, 16 kHz | 42 ms | 31 ms | 56 ms | 382 ms | 3,393 ms | 4.04 s | 1.04 s |

That is about 6x (44.1 kHz) and 4x (16 kHz) slower than real time, and as on every other
CPU the vocoder is close to 90% of it. The board has one core, so `--threads` makes no
difference (`--threads 1`: 3,310 ms vocoder at 16 kHz). `t_y=128` for the 44.1 kHz model
matches the value on every other platform.

---

## TinyTTS on the TPU

### Overview

| Stage | Where | Why |
|---|---|---|
| G2P, text encoder, duration predictor | C906 CPU | small, variable length |
| flow | TPU, padded to fixed lengths (CPU for longer sentences) | attention over the whole sentence |
| decoder | TPU, in fixed-size windows | ~90% of the work, purely convolutional |

TPU-MLIR compiles static shapes only, and the number of frames differs for every sentence:
the decoder runs in overlapping windows, the flow on a few fixed-length models (see
[Technical background](#technical-background)). The steps:

1. On the PC, export the decoder and the flow as ONNX (`research/milkv_duo/`).
2. On the PC, compile them for the TPU with TPU-MLIR (`.cvimodel`, BF16).
3. Build `tinytts` with the TPU runtime.
4. Copy the program and the models to the board and run it there.

### Install TPU-MLIR

[TPU-MLIR](https://github.com/sophgo/tpu-mlir) 1.30 or newer compiles the models (tested:
1.30.2). There are two ways to install it.

**With pip** (about 600MB; tested on Ubuntu 24.04, glibc 2.39). It needs Python 3.10, e.g.
from conda, and three fixes:

```bash
conda create -y -p ~/.venvs/tpumlir python=3.10
~/.venvs/tpumlir/bin/pip install "tpu_mlir[onnx]" psutil "setuptools<70"
# The wheel bundles an older glibc and puts it on LD_LIBRARY_PATH, which breaks the
# system's libdl/libutil ("GLIBC_2.36 not found"): move it aside, the system glibc is newer.
D=~/.venvs/tpumlir/lib/python3.10/site-packages/tpu_mlir/lib
mkdir -p $D/third_party_glibc_disabled
mv $D/third_party/{libc.so.6,libm.so.6,libpthread.so.0} $D/third_party_glibc_disabled/
export PATH=~/.venvs/tpumlir/bin:$PATH       # for the build scripts below
```

`psutil` is used but not declared by the package, and `setuptools<70` still has
`pkg_resources`, which TPU-MLIR imports.

**With Docker** (the `sophgo/tpuc_dev:v3.4` image, several GB). The image's Python 3.10
already has the dependencies. Run the container as your own user, or the outputs in the repo
end up owned by root:

```bash
docker run -d --name tpumlir -v $HOME:/workspace sophgo/tpuc_dev:v3.4 sleep infinity
docker exec tpumlir pip install tpu_mlir
run() { docker exec -u $(id -u):$(id -g) -e HOME=/tmp -w /workspace/<path to TinyTTS>/research tpumlir "$@"; }
```

Then prefix the build commands below with `run`. If `pip` in the container can't resolve
hostnames, the container inherited a DNS server it can't reach: add
`{"dns": ["8.8.8.8", "1.1.1.1"]}` to `/etc/docker/daemon.json` and restart Docker.

### Export and compile the models

Export (Python environment from `docs/research.md`):

```bash
cd research && source venv/bin/activate
python milkv_duo/export_decoder_tpu.py      # --window 128 --context 12
python milkv_duo/export_flow_tpu.py         # --lengths 128 256 512 1024
```

Compile (TPU-MLIR on the `PATH`, or with `run` for Docker):

```bash
milkv_duo/build_cvimodel.sh decoder_44100hz_w128              # about a minute
milkv_duo/build_flow_cvimodels.sh 44100                       # 128 256 512 1024, about five minutes
```

Optionally, measure the accuracy over 20 sentences in TPU-MLIR's simulator (see
[Precision](#precision-bf16-not-int8)):

```bash
python3 -I milkv_duo/validate_cvimodel.py out/decoder_44100hz_w128_bf16.cvimodel
python3 -I milkv_duo/validate_flow_cvimodel.py --sample-rate 44100 out/decoder_44100hz_w128_bf16.cvimodel
```

Everything goes to `research/milkv_duo/out/`, with the sample rate in every name:

- `decoder_44100hz_w128.onnx`: inputs `z` [1,32,W], `g` [1,128,1], `mask` [1,1,W], output
  `audio` [1,1,W*hop]. Plus `_calib/` (71 windows from 20 real sentences, for INT8
  calibration), `_test_input.npz`/`_test_ref.npz` and `_utterances.npz` (for the
  validation). The export also checks the chunked ONNX output in onnxruntime (108dB SNR
  against PyTorch).
- `flow_44100hz_l<L>.onnx` per length: inputs `z_p` [1,32,L], `y_mask` [1,1,L], `g`
  [1,128,1], output `z` [1,32,L]. Plus test inputs and `flow_44100hz_utterances.npz`.
- `decoder_44100hz_w128_bf16.cvimodel` and `flow_44100hz_l{128,256,512,1024}_bf16.cvimodel`:
  the models for the board. The build scripts compile for processor `cv181x`
  (SG2002/SG2000) and check each model against its ONNX output in the simulator.

Both export scripts take `--checkpoint` for another model, see
[The 16 kHz voice and other sample rates](#the-16-khz-voice-and-other-sample-rates).

### Build the CLI with the TPU runtime

The same build as in [TinyTTS on the CPU](#build), plus the TPU runtime's headers from
[tpu-sdk-sg200x](https://github.com/milkv-duo/tpu-sdk-sg200x) (44MB):

```bash
git clone --depth 1 https://github.com/milkv-duo/tpu-sdk-sg200x.git ~/tpu-sdk-sg200x
cmake -S . -B build_duo -DCMAKE_TOOLCHAIN_FILE=cmake/milkv-duo.cmake \
  -DDUO_TOOLCHAIN_DIR=$HOME/host-tools/gcc/riscv64-linux-musl-x86_64 \
  -DTINYTTS_BUILD_DESKTOP_MAIN=ON -DCMAKE_BUILD_TYPE=Release \
  -DTINYTTS_CVI_TPU=ON -DTPU_SDK_DIR=$HOME/tpu-sdk-sg200x
cmake --build build_duo -j
```

### Install and run

The board already has the TPU runtime (`/mnt/system/lib/libcviruntime.so`), so only the
program and the models need to be copied:

```bash
cd research/milkv_duo/out
ssh root@192.168.42.1 mkdir -p /root/tpu
scp ../../../build_duo/desktop/tinytts decoder_44100hz_w128_bf16.cvimodel flow_44100hz_l*_bf16.cvimodel root@192.168.42.1:/root/tpu/
```

On the board:

```bash
cd /root/tpu
export LD_LIBRARY_PATH=/mnt/system/lib
./tinytts --no-play --tpu-model decoder_44100hz_w128_bf16.cvimodel \
  --tpu-flow flow_44100hz_l128_bf16.cvimodel --tpu-flow flow_44100hz_l256_bf16.cvimodel \
  --tpu-flow flow_44100hz_l512_bf16.cvimodel --tpu-flow flow_44100hz_l1024_bf16.cvimodel \
  -o hello.wav "Hello world!"
./tinytts --no-play -o cpu.wav "Hello world!"        # all on the CPU, for comparison
```

The flow line on stderr says `(external)` when the TPU ran it. For a first check, compare
`t_y` (128 for "Hello world!", as everywhere else in this project) and listen to `hello.wav`.
Without `--no-play` and `-o`, `tinytts` plays on the board's sound card, see
[Audio output](#audio-output); to hear it on the PC instead, see
[Linux remote execution](#linux-remote-execution).

### Results

Duo 256M, default image (2025-05-30), released 44.1 kHz model, BF16 models from TPU-MLIR
1.30.2, "Hello world!" (`t_y=128`, 1.49s of audio):

| Stage | CPU only | Decoder on the TPU | Flow and decoder on the TPU |
|---|---:|---:|---:|
| g2p | 48 ms | 44 ms | 42 ms |
| encoder | 32 ms | 30 ms | 31 ms |
| duration predictor | 52 ms | 51 ms | 61 ms |
| flow | 808 ms | 810 ms | **12 ms** |
| vocoder | 7,781 ms | **139 ms** | **138 ms** |
| **total, incl. startup** | **8.85 s** | **1.16 s** | **0.61 s** |

With both on the TPU, synthesis is about 2.4x faster than real time, startup and loading the
five models included. Compared with the CPU output (same text and seed), the audio is as close
as the simulator predicted: 22.0 dB waveform SNR with the decoder on the TPU, 18.8 dB with the
flow as well; log-spectrogram correlation 0.880 and 0.877. The waveform SNR of the TPU flow
understates how close it sounds, see [Precision](#precision-bf16-not-int8).

The 16 kHz voice (`data/ljspeech16k_weights.bin`, "Hello world!": `t_y=65`, 1.04s of audio):

| Stage | CPU only | Flow and decoder on the TPU |
|---|---:|---:|
| flow | 383 ms | **12 ms** |
| vocoder | 3,390 ms | **49 ms** |
| **total, incl. startup** | **4.04 s** | **0.67 s** |

Its decoder on the TPU is closer to the CPU output than the 44.1 kHz one: 31.8 dB waveform
SNR (log-spectrogram correlation 0.917); with the flow on the TPU as well, 10.7 dB (0.910).
Per second of audio it is the cheaper model (49 ms of vocoder per 1.04s vs. 139 ms per
1.49s), but for a short sentence the fixed costs (startup, G2P, encoder) dominate, so both
totals are about the same.

**Long sentences.** A sentence longer than the largest flow model falls back to the CPU flow
(`flow: ... (external flow failed, used CPU)`). Attention makes that grow with the square of
the length: a 1,072-frame sentence (12.5s of speech) took 30.5s for the flow alone, while its
decoder on the TPU took 773 ms. Shorter sentences, or a larger flow bucket, avoid this.

### The 16 kHz voice and other sample rates

A model trained at another sample rate (`docs/training.md`) needs its own TPU models, made
from the same weights as its `weights.bin`. The export scripts' `--checkpoint` takes a
`G_*.pth`, or an exported `weights.bin` (`load_weights_bin()` in
`export_weights_and_vectors.py`). The model rebuilt from a `weights.bin` has exactly the
float16 and dequantized INT8 values the C++ library runs on (re-exporting it gives a
byte-identical file), so this also works when the training checkpoint is gone, as for
`data/ljspeech16k_weights.bin` (trained on Kaggle).

```bash
cd research && source venv/bin/activate
python milkv_duo/export_decoder_tpu.py --checkpoint ../data/ljspeech16k_weights.bin --sweep   # check the context
python milkv_duo/export_decoder_tpu.py --checkpoint ../data/ljspeech16k_weights.bin
python milkv_duo/export_flow_tpu.py --checkpoint ../data/ljspeech16k_weights.bin
milkv_duo/build_cvimodel.sh decoder_16000hz_w128          # with TPU-MLIR
milkv_duo/build_flow_cvimodels.sh 16000

scp milkv_duo/out/*16000hz*_bf16.cvimodel ../data/ljspeech16k_weights.bin root@192.168.42.1:/root/tpu/
```

On the board, in `/root/tpu` (with `LD_LIBRARY_PATH=/mnt/system/lib`):

```bash
./tinytts --no-play --weights ljspeech16k_weights.bin \
  --tpu-model decoder_16000hz_w128_bf16.cvimodel \
  --tpu-flow flow_16000hz_l128_bf16.cvimodel --tpu-flow flow_16000hz_l256_bf16.cvimodel \
  --tpu-flow flow_16000hz_l512_bf16.cvimodel --tpu-flow flow_16000hz_l1024_bf16.cvimodel \
  -o hello16k.wav "Hello world!"
```

The scripts take the sample rate, the output samples per frame (hop) and the vocoder's
upsample stack from the checkpoint or `weights.bin` (`research/training/audio_config.py`).

- **Mismatches are caught at startup.** A decoder from another model than the weights would
  play at the wrong speed and pitch. After `begin()`, the desktop CLI calls
  `CviTpuVocoder::checkModel()` and `CviTpuFlow::checkModel()`: they compare the decoder's
  samples per frame with the weights, and the sample rate in each model's file name with the
  weights' rate. Models for 16, 22.05 and 24 kHz all have 256 samples per frame, so keep the
  names as the scripts write them.
- **The context** (12 frames) was measured on the released 5-stage decoder. Lower rates drop
  stages from the end, which shouldn't need more, but `--sweep` checks it per model. For
  `ljspeech16k` 12 frames are exact as well (107 dB at W=128).
- **The flow buckets are frames**, so they cover more time at lower rates: at 16 kHz (256
  samples per frame) the 1024-frame bucket holds 16.4s instead of 11.9s.
- **Less work per second of audio.** A 16 kHz model produces 2.8x fewer samples and drops the
  decoder's last stage, the long 2-channel layers that suit the TPU worst: its decoder takes
  47 ms per second of audio on the TPU, the 44.1 kHz one 93 ms.

---

## Audio output

### Options

The board has no usable audio output of its own with the default image. The ways to hear
the speech:

| Option | Quality | What it takes |
|---|---|---|
| [Linux remote execution](#linux-remote-execution): synthesize on the board, play on a Linux PC | good | nothing on the board; SSH and `aplay` on the PC |
| [I2S amplifier (MAX98357A)](#i2s-amplifier-max98357a) on I2S2 | good | a rebuilt kernel (`boot.sd`); uses the Ethernet pads |
| [USB speaker](#usb-speaker) / sound card | good | the USB-C port in host mode, which ends the USB network |
| Built-in DAC | poor | Duo 256M users report mostly pops and noise |
| PWM from Linux | poor | 59 µs per duty-cycle update through sysfs (measured): too slow and too jittery for 16 kHz |

### Linux remote execution

The board doesn't need a speaker of its own: a Linux PC can run `tinytts` on the board over
SSH and play the result itself. `tinytts --stdout` writes the WAV to standard output, SSH
carries it to the PC, and `aplay` plays it there. The board's sound card isn't involved, so
this works with the default image.

**With the script.** `research/milkv_duo/duo-say.sh` does this with the flow and the decoder
on the TPU. It expects the program and the models in `/root/tpu`, as installed in
[Install and run](#install-and-run) (for `--16k` also the 16 kHz files from
[The 16 kHz voice and other sample rates](#the-16-khz-voice-and-other-sample-rates)):

```bash
research/milkv_duo/duo-say.sh "Hello world!"
research/milkv_duo/duo-say.sh --16k "The 16 kHz voice"
echo "Text from a pipe" | research/milkv_duo/duo-say.sh
research/milkv_duo/duo-say.sh -o out.wav "Save instead of playing"
DUO=root@192.168.1.50 research/milkv_duo/duo-say.sh "Another board"
```

The board's timings go to stderr; add `2>/dev/null` to hide them. A 2.6s sentence takes about
1s from the command to the WAV, SSH included.

**By hand.** The same with a plain `ssh` command. On the CPU only, with the `tinytts` from
[TinyTTS on the CPU](#tinytts-on-the-cpu) in `/root`:

```bash
ssh root@192.168.42.1 './tinytts --stdout "Hello world!"' | aplay
```

With the TPU:

```bash
ssh root@192.168.42.1 'cd /root/tpu && LD_LIBRARY_PATH=/mnt/system/lib ./tinytts --stdout \
  --tpu-model decoder_44100hz_w128_bf16.cvimodel \
  --tpu-flow flow_44100hz_l128_bf16.cvimodel --tpu-flow flow_44100hz_l256_bf16.cvimodel \
  --tpu-flow flow_44100hz_l512_bf16.cvimodel --tpu-flow flow_44100hz_l1024_bf16.cvimodel \
  "Hello world!"' | aplay
```

Any player that reads a WAV from standard input works in place of `aplay`, e.g.
`paplay`, `pw-play -` or `ffplay -nodisp -autoexit -`. Since the whole WAV is written at the
end (its header needs the length), playback on the PC starts once the board has synthesized
all of the text; for long texts, call it once per sentence or paragraph.

### I2S amplifier (MAX98357A)

**Status:** runs on a Duo 256M. The rebuilt kernel boots, the TPU and SSH work as before, the
sound card is ALSA card 0, and `aplay` and `tinytts` play on it at 16 and 44.1 kHz in real
time. The sound on a connected MAX98357A hasn't been listened to yet, so the wiring below is
the forum's, not yet confirmed on this board. The approach follows the working setup in the
Milk-V forum ([Duo256M: add max98357a sound card](https://community.milkv.io/t/duo256m-add-max98357a-sound-card-done/1494),
[I2S2 thread](https://community.milkv.io/t/i2s-milk-v-duo-speaker-max98357a/668)).

None of the 26 header pins has an I2S function (`cvi-pinmux -l`). I2S2 is only available on
the four Ethernet pads, so the Ethernet MAC is disabled. SSH is not affected: it runs over the
USB network (`usb0`), not `eth0`.

#### Wiring

As measured in the forum's working setup (it doesn't follow the pinmux names):

| Duo Ethernet pad | MAX98357A |
|---|---|
| TX- | DIN |
| RX+ | BCLK |
| RX- | LRC |
| GND | GND |
| 5V (VBUS/VSYS pin) | Vin |

TX+ stays unconnected. SD_MODE can stay open on the usual breakout boards (pulled up: left
and right averaged). The MAX98357A needs no MCLK.

#### Installation and setup

All files are in `research/milkv_duo/i2s/`:

| File | What it is |
|---|---|
| `fetch_sdk.sh` | downloads the parts of the Milk-V SDK the kernel build needs |
| `build_boot_sd.sh` | builds the kernel and packs `boot.sd` |
| `build_dtb.sh` | compiles the device tree (used by `build_boot_sd.sh`) |
| `kernel/sg2002_milkv_duo256m_max98357a.dts` | the board's device tree plus I2S2 and the amplifier |
| `kernel/max98357a-sound-card.patch` | the sound card driver, as a patch for the SDK's kernel (`sound/soc/cvitek/`) |
| `eth2i2s.c` | disconnects the Ethernet pads from the Ethernet PHY |
| `i2s-max98357a.sh` | the setup at every boot: pad functions, then `eth2i2s` |
| `asound.conf` | makes the amplifier the default ALSA device |

The commands on the PC run in `research/milkv_duo/i2s/`.

**1. Install the build tools (PC).** Besides the toolchain from [Toolchain](#toolchain):

```bash
sudo apt install u-boot-tools lz4
```

**2. Fetch the SDK sources (PC).** `fetch_sdk.sh` makes a sparse checkout of just the kernel
and board files of [duo-buildroot-sdk-v2](https://github.com/milkv-duo/duo-buildroot-sdk-v2)
(1.2GB on disk), one directory at a time, since git can't resume a broken download:

```bash
./fetch_sdk.sh ~/duo-buildroot-sdk-v2
```

**3. Build the kernel (PC).** The kernel configuration comes from the running board:

```bash
ssh root@192.168.42.1 'zcat /proc/config.gz' > running.config
./build_boot_sd.sh ~/duo-buildroot-sdk-v2 ~/host-tools/gcc/riscv64-linux-musl-x86_64 running.config out
```

`build_boot_sd.sh` first applies `kernel/max98357a-sound-card.patch` to the SDK (with
`git apply`; it skips this if the patch is already in), then builds. To apply the patch by
hand, or to undo it:

```bash
git -C ~/duo-buildroot-sdk-v2 apply $PWD/kernel/max98357a-sound-card.patch
git -C ~/duo-buildroot-sdk-v2 apply --reverse $PWD/kernel/max98357a-sound-card.patch
```

The build takes a few minutes and writes `out/boot.sd`. If the kernel configuration reports
"can't open file" for a file the sparse checkout lacks,
`git -C ~/duo-buildroot-sdk-v2 sparse-checkout add /linux_5.10/<file>` fetches it.

**4. Build `eth2i2s` (PC):**

```bash
~/host-tools/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-gcc -O2 -static -o eth2i2s eth2i2s.c
```

**5. Copy the files to the board (PC):**

```bash
ssh root@192.168.42.1 mkdir -p /mnt/system/i2s
scp eth2i2s i2s-max98357a.sh asound.conf root@192.168.42.1:/mnt/system/i2s/
scp out/boot.sd root@192.168.42.1:/root/boot.sd.i2s
```

**6. Set up the boot script and ALSA (board):**

```bash
chmod +x /mnt/system/i2s/eth2i2s /mnt/system/i2s/i2s-max98357a.sh
cp /mnt/system/i2s/asound.conf /etc/asound.conf
echo /mnt/system/i2s/i2s-max98357a.sh >> /mnt/system/auto.sh
```

**7. Install the kernel and reboot (board).** Keep the original next to it:

```bash
cp /boot/boot.sd /boot/boot.sd.orig
cp /root/boot.sd.i2s /boot/boot.sd && sync
reboot
```

**8. Check (board):**

```bash
cat /proc/asound/cards              # card 0: cv1835max98357a
cvi-pinmux -r PAD_ETH_TXP           # [v] IIS2_LRCK
lsmod | grep cv181x_tpu             # the TPU driver still loads
aplay hello.wav                     # plays on the amplifier
```

`tinytts` without `--no-play` and `-o` now plays on the amplifier, with or without the TPU
(see [Install and run](#install-and-run)).

**If the board doesn't come back** (no blue LED, no USB network): put the card into the PC
and copy `boot.sd.orig` over `boot.sd` on its boot partition. To go back to the original
kernel on a running board: `cp /boot/boot.sd.orig /boot/boot.sd && sync && reboot`.

#### How it works

**The kernel** (only `boot.sd` changes; the boot loader `fip.bin` stays as it is):

- **Device tree:** the board's own `.dts` from the SDK plus the `i2s2` node,
  `&ethernet0 { status = "disabled"; }`, a `maxim,max98357a` codec node and a
  `cvitek,cv1835-max98357a` sound card node. The `i2s2` node is declared again, with its
  interrupt, because `cv181x_asic_qfn.dtsi` deletes it for this package. Compiled from the
  SDK, the unmodified board `.dts` gives exactly the device tree in the image's `boot.sd`, so
  these are the only differences.
- **Sound card driver** (`kernel/max98357a-sound-card.patch`): adds
  `sound/soc/cvitek/cv1835_max98357a.c` and its Kconfig option and Makefile entry to the
  SDK's kernel; nothing else in the SDK changes. The driver links I2S2 (`4120000.i2s`) to the
  codec. The SoC is the I2S master; the SDK's I2S driver needs no changes.
- **Configuration:** the running kernel's (`/proc/config.gz`) plus `CONFIG_SND_SOC_MAX98357A`,
  `CONFIG_SND_SOC_CV1835_MAX98357A` and `CONFIG_SND_PROC_FS`. The release string must stay
  `5.10.4-tag-`: module versioning is off, and the image's own drivers (`/mnt/system/ko`,
  including the TPU) only load into a kernel with the same string. `build_boot_sd.sh` checks
  both.
- **`boot.sd`:** the same FIT layout as the SDK's: lzma-compressed kernel at `0x80200000`,
  one device tree, configuration `config-sg2002_milkv_duo256m_musl_riscv64_sd`.

**At every boot** (`i2s-max98357a.sh`): `cvi-pinmux -w` selects the `IIS2_*` functions on
the four pads, then `eth2i2s` switches the Ethernet PHY to "GPIO from top" mode through
`/dev/mem`. The pinmux alone isn't enough: the pads stay connected to the PHY until then.
`eth2i2s` is Eric Brombaugh's register sequence from the forum, with one register offset
corrected (`0x808`; the original has `0x806`).

**ALSA** (`/etc/asound.conf`): the default device always sends two channels. The I2S driver
puts mono into the left slot only, and a MAX98357A with SD_MODE open plays
(left + right) / 2, which would halve the volume.

**DMA buffer sizes.** The SoC's DMA driver (`drivers/dma/cvitek/cvitek-dma.c`) splits the
ring buffer into descriptors of at most 64 bytes, but also of at most half the largest power
of two dividing the buffer's length. `aplay` picks 88,200 bytes for 44.1 kHz (0.5s,
2^3 x 11,025), which makes 22,050 descriptors of 4 bytes; the allocation fails, and the error
pointer it returns crashes the kernel in `snd_dmaengine_pcm_trigger()`, which only checks for
NULL. That is the 44.1 kHz crash the forum reports for the built-in DAC. The sound card
driver therefore restricts buffer and period sizes to multiples of 128 bytes.

### USB speaker

The kernel has `snd-usb-audio`, and `aplay` and `libasound` are on the board. But the Duo
256M has a single USB-C port: in host mode the USB network, and with it SSH, is gone, and the
board must be powered through VSYS.

```bash
ln -sf usb-host.sh /mnt/system/usb.sh && reboot     # host mode
ln -sf usb-ncm.sh /mnt/system/usb.sh && reboot      # back to the USB network
```

The console is then UART0 (GP12 TX, GP13 RX, 115200 baud, 3.3V). A USB Ethernet adapter on
a hub doesn't help with the default image: its kernel has no USB network drivers
(`CONFIG_USB_NET_DRIVERS` is not set).

---

## Technical background

### Decoder: overlapping windows

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

### Flow: fixed lengths

The flow's attention spans the whole sentence, so it can't be split into windows. But its
convolutions and attention all take `y_mask`, so a sentence padded to a fixed length (mask 0
on the padding) gives the unpadded result. `BucketedFlow` (`src/TinyTTS/BucketedFlow.h`)
runs a sentence on the smallest of several fixed-length models that fits: 128, 256, 512 and
1024 frames by default (1.5s to 11.9s of speech at 44.1 kHz). Longer sentences run on the
CPU flow; the desktop CLI already splits text into sentences. Padded vs. unpadded:
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

### Precision: BF16, not INT8

The validation scripts run the models in TPU-MLIR's simulator over 20 sentences (chunked and
padded as on the device) against the float32 model, and write WAV files for listening.
Released 44.1 kHz model:

| Model | Precision | SNR median | SNR worst | Cosine worst |
|---|---|---:|---:|---:|
| `decoder_44100hz_w128_bf16.cvimodel` | BF16 | **27.4 dB** | **23.9 dB** | 0.9981 |
| `decoder_44100hz_w128_int8.cvimodel` | INT8, per-tensor activation scales | -2.5 dB | -8.1 dB | 0.119 |
| `decoder_44100hz_w128_mix.cvimodel` | INT8, 47 sensitive layers in BF16 | -2.3 dB | (2 sentences) | |
| `flow_44100hz_l*_bf16.cvimodel` (z) | BF16 | **43.0 dB** | **42.1 dB** | |
| flow + decoder, both BF16 (audio) | BF16 | 15.2 dB | 11.0 dB | |

**BF16:** the decoder is as close to float as the CPU decoder this library ships (weights-only
INT8, 22-27dB, `docs/performance.md`). The flow is closer still, 42-45dB in all four lengths.

**Flow and decoder together:** the waveform SNR drops to 11-21dB, but that's phase: the
decoder turns the flow's small errors into slightly shifted waveforms. The log-spectrogram
correlation with float is the same with or without the TPU flow (0.900 vs. 0.899 for "Hello
world!", 0.907 vs. 0.906 for the next sentence; at least 0.894 over all 20), so the TPU flow
adds no audible difference of its own. For scale: synthesizing the same sentence with another
noise seed gives -3dB waveform SNR. Listen to `float32_44100hz_<n>.wav`,
`decoder_44100hz_w128_bf16_<n>.wav` and `tpu_flow_decoder_44100hz_<n>.wav` in
`research/milkv_duo/out/`.

**INT8** output is mostly noise: TPU-MLIR's own INT8 reference is already down to 4.9dB at
its worst layer on the test window, and the compiled model differs from that reference again
(cosine 0.25). That matches earlier experience in this project: per-tensor INT8 activation
scales measured only 12.4dB with TFLite (`docs/research.md`). Mixed precision doesn't rescue
it: TPU-MLIR's layer search kept 47 layers in BF16, its own reference for that mix is down to
12.3dB at the worst layer, and the compiled model is noise again (cosine 0.34 on the test
window, -2.3 and -1.8dB on the first two sentences). The build scripts therefore build BF16
only; `build_cvimodel.sh <name> all` also builds the INT8 and mixed variants for comparison
(the mixed search alone takes about 4 hours).

### Notes from the hardware

- **TPU memory:** the runtime allocates from the ION heap the Duo image reserves (75MB on the
  Duo 256M). The 128-frame decoder and all four flow models (128-1024 frames) load and run
  together without problems.
- **The TPU pays off:** the decoder needs about 0.54 GMAC per second of audio, and 72% of it
  is in layers with only 4-16 input channels, a poor fit for a wide accelerator. It still
  runs 56x faster than on the C906 (139 ms vs. 7.8s for "Hello world!"), the flow 67x.
- **Padding cost:** a sentence just over a bucket length pays for up to twice its frames (and
  four times the attention). With the flow at 12 ms for a 128-frame bucket this hasn't
  mattered so far; more bucket lengths (`--lengths`) are cheap to add. A bucket above 1,024
  frames would avoid the slow CPU fallback for long sentences, if it fits in the ION heap.
- **Model inputs and outputs:** `CviTpuVocoder` and `CviTpuFlow` expect them in float32.
  That's TPU-MLIR's default unless `--quant_input`/`--quant_output` are given, and they
  report an error otherwise. In the compiled models the tensors are 4-D (`z` [1,32,128,1],
  output [1,1,65536,1]), since TPU-MLIR runs the 1-D convolutions as 2-D ones. The memory
  layout is the same.

---

## Next steps

1. **Listen to the I2S amplifier,** and confirm the wiring.
2. **Overlap CPU and TPU.** `CVI_NN_ForwardAsync()` lets the CPU run the next sentence's
   G2P/encoder/duration predictor while the TPU runs the current one's flow and decoder.
3. **RVV on the C906.** The C906 has a vector unit (RVV 0.7.1, T-Head's `xtheadvector`
   variant). Whatever stays on the CPU could use it for `linear()`/`conv1d()`, as NEON does
   on ARM. Needs the T-Head toolchain's intrinsics. With the flow and the decoder on the
   TPU, G2P, encoder and duration predictor are now about half of the synthesis time.
