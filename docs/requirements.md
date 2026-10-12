# Requirements

TinyTTS has two genuinely different consumers with different requirements: the Arduino/ESP-IDF
library (what runs *on* a microcontroller) and the `desktop/` CLI (a normal CMake-built
program that runs on your own machine). Nothing below the "Microcontroller" section applies
to the desktop build, and vice versa.

## Microcontroller (Arduino / ESP-IDF)

- **Board**: a module with enough external flash and PSRAM for the model data (see
  `docs/model-data.md`) — the chip die itself typically has neither; both come from the
  specific module, so check your module's actual flash/PSRAM size rather than assuming from
  the chip name alone.
- **Board settings**: whichever PSRAM mode matches your module (e.g. `PSRAM=opi`) is
  required regardless of how you load model data. If native-USB Serial doesn't work on your
  board out of the box, check whether it needs a USB-mode board setting. If you compile the
  example model data into flash, size the partition scheme to fit it (see the
  `partitions.csv` shipped alongside `examples/tts_i2s_output/` for a worked example, and
  adapt it for your own flash size).
- **Libraries**: just [`arduino-audio-tools`](https://github.com/pschatzmann/arduino-audio-tools)
  (for `I2SStream`/audio output — `TinyTTS` itself only needs a plain Arduino `Print`, so any
  audio-tools output class works, or your own `Print` implementation). No inference-runtime
  library (TFLite Micro or otherwise) is needed -- every model stage is hand-written C++.
- **No extra step needed for the ESP32-S3/P4 SIMD speedup.** Float32 SIMD comes from the
  Arduino-ESP32 core's own bundled `esp-dsp` component (~3.3.x+ cores ship one) —
  `Ops.h` detects it via `__has_include` and falls back to a plain scalar loop on
  non-ESP32/host builds or older cores; see `docs/performance.md` for the measured
  difference (a real ~1.5-2x on the affected ops). INT8 SIMD (the decoder-precision
  prototype, see `docs/performance.md`) is genuinely vendored — the core doesn't provide
  it — directly inside this library's own `src/int8-dotprod/` tree, so Arduino picks it
  up automatically like any other TinyTTS header, nothing to install separately; see
  `src/int8-dotprod/NOTICE.md` for what it is.
- **Chip family**: ESP32 is the tested and tuned target. The library also builds for the
  Sipeed Tang Nano 20K (`tangnano20k` architecture) as an experiment -- different board
  settings, and an optional on-chip INT8 accelerator; see `docs/tangnano20k.md`. None of the
  model code is chip-specific, but the multi-megabyte weight/dictionary data needs a board
  with that much flash and RAM either way -- see `docs/limitations.md`.

### How much flash and PSRAM

**Flash.** The model data is 4.21 MB with the recommended slimmed dictionary + neural G2P
model, or 5.51 MB with the full dictionary alone (`docs/model-data.md`). With the data
compiled in, the `timing_benchmark` image measured **4.54 MB** on ESP32-P4 (slimmed + G2P,
`-Os`), so TinyTTS's code plus the Arduino core and arduino-audio-tools add only ~0.3 MB. The
stock ESP32 partition schemes have no app partition that big, so use a custom one (see the
`partitions.csv` in the examples, sized for 16 MB flash; an 8 MB module needs a smaller
`factory` partition of at least ~4.7 MB, or ~6 MB for the full dictionary). Loading the
data from LittleFS/FFat instead doesn't save flash, it only moves the data to the file
system. Only an SD card takes it off the internal flash.

**PSRAM** has three parts:

| Part | Size |
|---|---:|
| After `begin()`, data compiled into flash | ~0.75-0.9 MB (743 KB measured on ESP32-S3 and P4) |
| Extra after `begin()` with the full dictionary (no G2P model) | +0.7 MB |
| Data loaded from a file (`setWeights(File&)` etc.) instead of flash | + the file sizes (4.21 MB or 5.51 MB) |
| Working memory during one `speak()` call | ~1.75 MB per second of audio |

The working memory is freed again after every `speak()` call, but it grows with the length
of the audio that call produces (44.1 kHz model, data in flash, slimmed dictionary + G2P,
measured with a heap counter on the host build; the 1.5 s case matches the ~3.5 MB peak
measured on the Tang Nano 20K):

| Text of one `speak()` call | Audio | Peak heap |
|---|---:|---:|
| "Hello world!" | 1.5 s | 3.4 MB |
| one short sentence (9 words) | 4.1 s | 7.9 MB |
| one long sentence (27 words) | 9.8 s | 17.9 MB |
| a paragraph (3 sentences) | 21.8 s | 38.7 MB |

So the PSRAM size limits how long a single `speak()` call may be, not what you can say in
total. **Split longer text into sentences and call `speak()` once per sentence**
(`docs/text-input.md`), as the desktop CLI does. Rough limits per `speak()` call, with the
data in flash:

| PSRAM | Longest audio per `speak()` call |
|---|---|
| 2 MB | not enough, even for "Hello world!" |
| 4 MB | ~1.5 s (a few words) |
| 8 MB | ~3.5-4 s (one short sentence) |
| 16 MB | ~8 s |
| 32 MB (e.g. ESP32-P4 boards) | ~17 s |

With the data loaded from files into PSRAM, subtract another 4.2-5.5 MB first: an 8 MB
module then has room for only about 1-1.5 s of audio per call ("Hello world!" is borderline). `setLengthScale()` above 1.0
makes the same text longer, so it needs more memory too. A model trained at a lower sample
rate (`docs/training.md`) needs less per second of audio; that hasn't been measured.

**Internal RAM** use is small: weight tiles of at most 16 KB, a few small scratch buffers,
and a 16 KB stack per worker task with `setNumWorkers(2)`. The large buffers all go to
PSRAM.

Building as a plain ESP-IDF component (no Arduino dependency) is also supported -- see
`docs/esp-idf.md`. For model data sizes and how to wire weights/dictionary buffers up, see
`docs/model-data.md`.

## Desktop (the `tinytts` CLI)

None of the board/flash/PSRAM requirements above apply here -- this is a normal native
executable, not something flashed to a chip. See `docs/desktop.md` for the full build/install/
usage guide; the short version:

- **CMake 3.16+** and a **C++17 compiler** (tested with GCC on Linux; no platform-specific
  code beyond what `arduino-audio-tools`' desktop emulation already handles, so macOS/Windows
  should work too, just not independently verified here).
- **Internet access at first configure**: `-DTINYTTS_BUILD_DESKTOP_MAIN=ON` fetches
  `arduino-audio-tools` and `miniaudio.h` via `FetchContent` (git) -- a one-time cost, not
  needed again once populated.
- **A real-time audio device, only if you want to hear it play.** `-o`/`--stdout` (render to
  a WAV file or pipe) don't need a working audio device at all -- useful in headless/CI/
  sandboxed environments where live playback isn't available.
- **Cross-compiling for the Milk-V Duo** (decoder and flow on its TPU) additionally needs
  the T-Head musl toolchain, `tpu-sdk-sg200x` and, to build the TPU models, TPU-MLIR -- see
  `docs/milkv-duo.md`.
- No flash/PSRAM sizing to think about at all: the embedded model data (~4.2MB, see
  `docs/model-data.md`) is trivial for a desktop machine's storage/RAM, and there's no
  partition scheme or board setting of any kind to configure.
