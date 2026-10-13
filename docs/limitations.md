# Status / known limitations

## Status per platform

| Platform | Status | "Hello world!" (1.49s of audio) |
|---|---|---:|
| ESP32-S3 (Arduino / ESP-IDF) | Runs on the board, tested and tuned | ~34.1 s (~23x slower than real time) |
| ESP32-P4 (Arduino / ESP-IDF) | Runs on the board | ~28.4 s (~19x slower than real time) |
| Sipeed Tang Nano 20K (`docs/tangnano20k.md`) | Runs on the board, with the FPGA INT8 accelerator | ~34 min (~1,400x slower than real time) |
| Raspberry Pi Zero W / Pi 4 (desktop CLI) | Runs | ~15.7 s / ~1.55 s |
| Raspberry Pi Zero W, flow and decoder on the GPU (`--qpu`, `docs/rpi-zero-gpu.md`) | Runs on the board | ~0.58 s synthesis (1.0 s with startup) -- faster than real time |
| Desktop PC (desktop CLI) | Runs | ~0.76 s (faster than real time) |
| Milk-V Duo 256M / Duo S, TPU (`docs/milkv-duo.md`) | Checked on a PC only (simulator, fake runtime); **not yet run on the board** | not measured |

See `docs/performance.md` for how these numbers were measured.

## Known limitations

- **Microcontrollers are too slow for real-time speech.** Even the fastest board measured
  (ESP32-P4) is ~19x slower than real time. What's still open is listed in
  `docs/potential-improvements.md`.
- **Only the released 44.1 kHz voice is shipped and measured.** The library, the desktop CLI
  and the Milk-V tooling all accept models at a lower sample rate (`weights.bin` carries
  the rate, `getAudioSampleRate()` reports it after `begin()`), and `research/training/`
  can train one (`docs/training.md`). No lower-rate voice is shipped yet, and none has been
  timed on a microcontroller.
- **Out-of-dictionary words fall back to crude character-level phonemes unless
  `setDictionaryModel()` is called.** The neural GRU fallback (`DictionaryModel`/
  `default_dictionary_model`) gives much better results for proper nouns and unusual words,
  but costs ~0.95MB more flash, so it's opt-in rather than always-on. The CMU dictionary
  covers the large majority of real English words either way.
- **English only.** `docs/training.md` describes what adding another language takes; the C++
  G2P (`TextG2P`) is English-specific.
- **On the CPU, `speak()` hands over the whole utterance's audio at once**, after synthesis
  finishes. Only an external vocoder set with `setVocoder()` (the Milk-V TPU's
  `ChunkedVocoder`) delivers audio window by window. See `docs/text-input.md`.
- **Splitting text across multiple `speak()` calls must happen on word boundaries, never
  mid-word** -- see `docs/text-input.md` for what input shapes are safe (individual words,
  full sentences, multiple sentences, and multi-call word-boundary splits all work; a
  multi-call split that cuts a word in half mispronounces it).
- **Not a "runs on any microcontroller" library.** None of the model code is chip-specific,
  but the multi-megabyte weight/dictionary data needs a board with that much flash and RAM.
  The `partitions.csv` shipped alongside the examples and the board settings in
  `docs/requirements.md` are for ESP32 modules, and `DataBuffer`'s `File`-loading path uses
  ESP32's PSRAM allocator where available. `library.properties` lists `esp32` and
  `tangnano20k`.
- **INT8 activations (`setDecoderPrecision(kInt8Activations)`) are a prototype.** On ESP32
  they're still slower than the float32 path and audibly lossier (`docs/performance.md`); on
  the Tang Nano 20K they're the default, since its accelerator only does INT8.
