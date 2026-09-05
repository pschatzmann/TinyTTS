# esp-dsp-dotprod (vendored)

A minimal, flattened copy of three modules from
[espressif/esp-dsp](https://github.com/espressif/esp-dsp) -- `dotprod`
(`dsps_dotprod_f32`/`dsps_dotprode_f32`), `mulc` (`dsps_mulc_f32`), and `add`
(`dsps_add_f32`) -- SIMD-accelerated float32 kernels for ESP32-S3 (Xtensa PIE) and ESP32-P4
(RISC-V), used by `../TinyTTS/Ops.h`'s hot dot-product loops when available.

**Lives directly under TinyTTS's own `src/` tree on purpose**, not as a separate library:
upstream `esp-dsp` is an ESP-IDF component (headers nested under `modules/*/include/`, no
`library.properties`) that `arduino-cli`/the Arduino IDE reject outright
(`invalid library: no header files found`), and an earlier version of this vendoring lived as
a separate sibling library a user had to install by hand -- that meant a fresh TinyTTS
install silently lost the SIMD speedup unless someone remembered the extra step. Placing the
files here instead means Arduino's normal recursive `src/` header/source discovery picks them
up automatically as part of TinyTTS itself: no separate install, no extra step, works out of
the box on any ESP32-family board.

Still fully optional at compile time -- `Ops.h` guards every include/call site with
`__has_include(<dsps_dotprod.h>)` and a plain scalar fallback, so nothing here is a hard
requirement; it also means a host/desktop build (or a non-ESP32 target, if TinyTTS is ever
ported to one) simply never sees these files matter, since `__has_include` still gates them --
they're just always *found* now, on any ESP32 board, rather than needing a separate install.

**License**: Apache-2.0, same as upstream and same as TinyTTS itself -- every file here still
carries its original Espressif Systems copyright/license header, unmodified. File contents
are unmodified from upstream apart from being flattened into one directory (no subfolders,
matching this project's existing `src/TinyTTS/*.h` convention) instead of `esp-dsp`'s own
`modules/*/include|float|misc/` layout.

Last vendored from `espressif/esp-dsp` @ `master` (no pinned tag/commit at vendoring time --
if you re-vendor later, consider pinning and recording that here). Re-vendoring, if ever
needed: copy `modules/{common,dotprod,math/mulc,math/add}/{include,float,misc}/*` from a
fresh checkout, flattened into this directory -- small and stable enough to be a five-minute
manual copy, not worth scripting.
