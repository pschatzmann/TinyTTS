# Desktop: building and running TinyTTS as a command-line tool

TinyTTS's `desktop/` directory builds a real, standalone CLI (`tinytts`) on top of the same
portable core (`TinyTTS::begin(const AudioChunkFn&)`) the ESP32 examples use -- useful for
trying the model out, generating WAV files for testing, or piping synthesized speech into
other Unix tools, all without any embedded hardware. It is **not** part of the Arduino/ESP-IDF
library surface: nothing under `src/` includes anything from `desktop/`, and it only builds
when you explicitly ask for it (see below).

## What it is

- `desktop/DesktopMain.h` -- owns a `TinyTTS` instance and an
  [`arduino-audio-tools`](https://github.com/pschatzmann/arduino-audio-tools) `MiniAudioStream`
  (real-time audio output via [miniaudio](https://miniaud.io/), works on Linux/macOS/Windows),
  wired together with the same embedded weights/dictionary data
  (`src/TinyTTS/data/default_weights_data.h` + the slimmed dictionary + neural G2P fallback)
  `examples/tts_i2s_output/` uses.
- `desktop/main.cpp` -- a one-line `main()` that just calls `DesktopMain::run(argc, argv)`.
- `desktop/CMakeLists.txt` -- fetches `arduino-audio-tools` and `miniaudio.h`, builds the
  `tinytts` binary.

This is a genuinely separate build path from the host **test** harness (`test/`,
`-DTINYTTS_BUILD_TESTS=ON`) -- that one is a correctness/regression suite with no audio output
at all; this one is a usable tool.

## Building

```sh
cmake -S . -B build -DTINYTTS_BUILD_DESKTOP_MAIN=ON
cmake --build build --target tinytts_desktop
./build/desktop/tinytts "Hello world!"
```

The first configure fetches `arduino-audio-tools` (git) and `miniaudio.h` (git, header only --
see below for why its own CMake project is deliberately not built) via `FetchContent`; expect
that step to take a couple of minutes the first time, not on every rebuild.

Note the CMake **target** name is `tinytts_desktop` (`--target tinytts_desktop`), not `tinytts`
-- the top-level `CMakeLists.txt` already has a target literally named `tinytts` (the
header-only library every consumer links against), and CMake target names must be unique
within a project. `set_target_properties(... PROPERTIES OUTPUT_NAME tinytts)` in
`desktop/CMakeLists.txt` is what makes the **produced binary** `./build/desktop/tinytts`
despite the target itself being named differently.

### Installing

A system-wide prefix like `/usr/local` is normally owned by root, so installing there needs
`sudo` -- without it, `cmake --install` fails partway through with a permissions error (e.g.
`file INSTALL cannot set permissions on "/usr/local/include/...": Operation not permitted`):

```sh
sudo cmake --install build --prefix /usr/local
tinytts "Hello world!"                       # if that prefix's bin/ is on PATH
```

If you'd rather not use `sudo`, install to a location your own account already owns instead
-- no elevated privileges needed, just make sure its `bin/` is on `PATH`:

```sh
cmake --install build --prefix "$HOME/.local"
~/.local/bin/tinytts "Hello world!"
```

### Combining with dual-core (`TileSplitter`)

`-DTINYTTS_BUILD_DESKTOP_MAIN=ON` alone gets you a working `tinytts` binary, single-core.
To also exercise the parallel decoder path (see `docs/performance.md`'s dual-core section)
add `-DTINYTTS_BUILD_PARALLEL_OPS=ON` at configure time -- this is what makes `--threads`
(below) actually split work across more than one thread; without it, `--threads` is silently
a no-op (the underlying `conv1d()`/`convTranspose1d()` fall back to a single-threaded loop
when `arduino-audio-tools`' `Task` support isn't compiled in -- see `Ops.h`'s
`TINYTTS_HAVE_TASK` gate).

```sh
cmake -S . -B build -DTINYTTS_BUILD_DESKTOP_MAIN=ON -DTINYTTS_BUILD_PARALLEL_OPS=ON
cmake --build build --target tinytts_desktop
./build/desktop/tinytts --threads 4 "Hello world!"
```

## Usage

```
tinytts [options] [TEXT]

Input:
  TEXT                  Text to speak. If omitted (and --file isn't given), read from
                        stdin -- e.g. echo "hi" | tinytts
  -f, --file FILE       Read text from FILE instead.

Output:
  -o, --output FILE     Write synthesized audio to FILE as a WAV file, instead of playing it.
  --stdout              Write WAV bytes to stdout instead of playing -- for piping,
                        e.g. tinytts --stdout | aplay, or > out.wav
  --no-play             Skip playback (implied by -o/--stdout).

Voice tuning:
  --volume VALUE        0.0-1.0, default 1.0
  --noise-scale VALUE   default 0.667
  --length-scale VALUE  default 1.0 (speaking rate; >1 slower, <1 faster)
  --speaker ID          default 0
  --threads N           worker count for the decoder's parallel conv loops, default 2
                        (needs -DTINYTTS_BUILD_PARALLEL_OPS=ON -- see above)
  --full-dict           Use the full CMU dictionary instead of the slimmed one (still
                        paired with the neural G2P fallback) -- see docs/model-data.md
                        for why slimmed+G2P is the recommended default even so.

  -h, --help            Show this help text.
```

### Examples

```sh
# Play through the default audio device
tinytts "Hello world!"

# Render to a WAV file instead of playing
tinytts -o hello.wav "Hello world!"

# Read text from a file
tinytts -f script.txt -o script.wav

# Unix pipelines: text in via stdin, audio out via stdout
echo "This came from a pipe" | tinytts --stdout > piped.wav
cat script.txt | tinytts --stdout | aplay

# Quieter, slower, a different speaker
tinytts --volume 0.4 --length-scale 1.3 --speaker 2 "Take it slow."
```

## Design notes (why it's built this way)

- **Portable callback, not the `Print&`-based API.** `DesktopMain` uses
  `TinyTTS::begin(const AudioChunkFn&)` rather than the `Print&`-based constructor/`begin()`
  the ESP32 examples use. The latter needs `ARDUINO` defined, which pulls in more of
  `TinyTTSCore.h`'s `#ifdef ARDUINO` code than `arduino-audio-tools`' `IS_MIN_DESKTOP`
  emulation actually provides (e.g. `Serial.printf()`, an ESP32-specific `HardwareSerial`
  extension the emulation's base `Print`/`Stream` doesn't have) -- not worth chasing down for
  what the portable callback path already does equally well.
- **Diagnostics on stderr, audio on stdout.** `TinyTTSCore.h`'s per-stage timing log
  (`[TinyTTS] encoder: ... ms`) goes to `stderr`, specifically so `--stdout`'s WAV bytes on
  `stdout` are never corrupted by interleaved log lines -- standard Unix convention (data on
  stdout, diagnostics on stderr), not a `desktop/`-only workaround.
- **Input text is split into sentences, each spoken with its own `speak()` call.**
  `TinyTTS::speak()` synthesizes its whole argument as one batch -- all four model stages run
  once over the entire phoneme sequence, and the `AudioChunkFn` callback fires exactly once,
  at the end, with the complete waveform. For a short phrase that's irrelevant, but for
  `-f somefile.txt` on a real document it would mean nothing reaches the speaker until the
  *entire* file has finished synthesizing. `DesktopMain` instead splits the input on sentence
  boundaries (`.`/`!`/`?` followed by whitespace, or a blank line -- with a hard length cap so
  one long unpunctuated line can't become a single slow chunk either) and calls `speak()` once
  per sentence, so the first sentence starts playing as soon as *it* is ready. Per
  `docs/text-input.md`, splitting text across multiple `speak()` calls on sentence/word
  boundaries never mispronounces anything (only a mid-word split does) -- this applies that
  same guarantee for latency, not just correctness. `-o`/`--stdout` output still gets one
  complete WAV covering every chunk (`onAudio()` appends each chunk's samples rather than
  overwriting).
- **`BufferedStream` chunks the audio write.** `TinyTTS`'s `AudioChunkFn` fires once per
  utterance with the *entire* waveform. Handing `MiniAudioStream::write()` all of that in one
  call overwhelms its internal ring-buffer backpressure (confirmed via a minimal,
  TinyTTS-free repro: one huge `write()` times out after filling exactly one ring-buffer's
  worth and never draining further; the same bytes written in ~2KB pieces succeed
  completely) -- `audio_tools::BufferedStream` provides that chunking for free.
- **miniaudio is fetched source-only (`FetchContent_Populate`, not `_MakeAvailable`).**
  `arduino-audio-tools` doesn't vendor `miniaudio.h` itself, and `MiniAudioStream.h` compiles
  the whole implementation directly (`#define MINIAUDIO_IMPLEMENTATION`) -- we never build or
  link any of miniaudio's *own* CMake targets. `FetchContent_MakeAvailable` would also
  `add_subdirectory()` miniaudio's project, defining an installable `libminiaudio.a` target
  that's never actually built; `cmake --install` would then fail outright looking for it.
  Populate-only avoids that entirely -- the same pattern `arduino-audio-tools`' own top-level
  `CMakeLists.txt` already uses for its optional `portaudio`/`arduino_emulator` fetches, for
  the identical reason.
- **`setNumWorkers()`/`--threads` must be set before the first `speak()`.** The underlying
  worker pool (`TileSplitter`, see `docs/performance.md`) is created lazily on first use and
  persists for the process's lifetime -- changing it afterward has no effect. `DesktopMain`
  calls `setNumWorkers()` right after parsing arguments, before `begin()`.
