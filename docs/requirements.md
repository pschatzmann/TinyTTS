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
- **No extra step needed for the ESP32-S3/P4 SIMD speedup.** `src/esp-dsp-dotprod/` ships a
  minimal, flattened vendored copy of three modules from
  [espressif/esp-dsp](https://github.com/espressif/esp-dsp) (upstream is an ESP-IDF
  component `arduino-cli` can't load directly) directly inside this library's own `src/`
  tree, so Arduino picks it up automatically like any other TinyTTS header — nothing to
  install separately. `Ops.h` detects it via `__has_include` and falls back to a plain
  scalar loop on non-ESP32/host builds; see `src/esp-dsp-dotprod/NOTICE.md` for what it is
  and `docs/performance.md` for the measured difference (a real ~1.5-2x on the affected ops).
- **Chip family**: ESP32 only (`library.properties` scopes it to the `esp32` architecture).
  None of the model code is chip-specific, but `DataBuffer`'s `File`-loading path uses
  ESP32's own PSRAM allocator directly, and the multi-megabyte weight/dictionary data needs a
  module with real PSRAM either way -- see `docs/limitations.md`.

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
- No flash/PSRAM sizing to think about at all: the embedded model data (~4.2MB, see
  `docs/model-data.md`) is trivial for a desktop machine's storage/RAM, and there's no
  partition scheme or board setting of any kind to configure.
