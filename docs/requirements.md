# Requirements

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

Building as a plain ESP-IDF component (no Arduino dependency) is also supported -- see
`docs/esp-idf.md`. For model data sizes and how to wire weights/dictionary buffers up, see
`docs/model-data.md`.
