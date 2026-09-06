# int8-dotprod (vendored)

INT8 dot product for `Ops.h`'s `conv1d()` INT8-activation path
(`ops::DecoderPrecision::kInt8Activations`, see `docs/performance.md`).

- **`dsps_dp_s8_ansi.c`** -- portable scalar fallback, from
  [espressif/esp-dsp](https://github.com/espressif/esp-dsp).
- **`esp_nn_dot_s8_esp32s3.S`** -- ESP32-S3 SIMD kernel
  (`esp_nn_dot_s8_aligned_esp32s3`), from
  [espressif/esp-nn](https://github.com/espressif/esp-nn), Espressif's own
  production quantized-NN kernel library. Requires both operand pointers
  16-byte aligned and `len` a multiple of 16 (>=16) -- `Ops.h`'s `conv1d()`
  guarantees this via `AlignedStlAllocator.h` and zero-padding, so the
  scalar fallback is a correctness safety net, not the expected hot path.
- `dsp_err.h`/`dsp_err_codes.h`/`dsps_dotprod.h`/`dsps_dotprod_platform.h`
  -- small shim headers `dsps_dp_s8_ansi.c` needs, also from esp-dsp.
- `dsps_dp_s8.h` -- this project's own dispatcher (`tinytts::dspsDotProdS8`),
  declares both kernels directly rather than via `<dsps_dotprod.h>`, since
  Arduino-ESP32 cores bundle their own esp-dsp component whose include
  paths take priority over this directory's on the compiler command line
  (confirmed empirically) -- a quoted, project-relative include can't be
  shadowed that way.

**`Ops.h`'s float32 dot product does NOT come from this directory** --
Arduino-ESP32 cores (~3.3.x+) bundle a precompiled `espressif__esp-dsp`
component that already provides `dsps_dotprod_f32`/`dsps_mulc_f32`/
`dsps_add_f32`, found automatically via `<dsps_dotprod.h>`/etc. It just
doesn't provide `dsps_dp_s8` at all, which is the entire reason this
directory still exists.

**Lives directly under TinyTTS's own `src/` tree** (not a separate
library): Arduino's normal recursive `src/` discovery picks these files up
automatically on any ESP32-family board, no separate install step.

Still fully optional at compile time -- `Ops.h` guards every include/call
site with `__has_include(<dsps_dotprod.h>)` and a plain scalar fallback.

**License**: Apache-2.0, same as TinyTTS itself -- every file here carries
its original Espressif Systems copyright/license header, unmodified.
