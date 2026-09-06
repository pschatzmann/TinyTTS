# esp-dsp-dotprod (vendored)

A minimal, flattened copy of three modules from
[espressif/esp-dsp](https://github.com/espressif/esp-dsp) -- `dotprod`
(`dsps_dotprod_f32`/`dsps_dotprode_f32`, plus `dsps_dp_s8` -- see below), `mulc`
(`dsps_mulc_f32`), and `add` (`dsps_add_f32`) -- SIMD-accelerated float32 kernels for
ESP32-S3 (Xtensa PIE) and ESP32-P4 (RISC-V), used by `../TinyTTS/Ops.h`'s hot dot-product
loops when available.

**`dsps_dp_s8_ansi.c`/`dsps_dp_s8_aes3.S`** (INT8 dot product, ESP32-S3 SIMD +
portable-C fallback) were added later than the rest of this directory, for `Ops.h`'s
INT8-activation `conv1d()` prototype (see `docs/performance.md`) -- `dsps_dotprod.h`
already declared these two functions and the `dsps_dp_s8` platform-dispatch macro from
the original vendoring pass; only the implementation files themselves were missing.
**`dsps_dp_s8_aes3` is confirmed broken on real hardware, not just suspicious.**
`dsps_dp_s8_aes3.S` (fetched verbatim from `espressif/esp-dsp@master`, byte-for-byte, not
modified) contains a leftover developer comment in Russian right before its vectorized
path -- roughly "DEBUG: always ANSI; remove before release". That alone was reason enough
to distrust it; a self-test written specifically because of it (see
`docs/performance.md`) then caught it red-handed on a real ESP32-S3: for a 16-element
vector, expected dot product 2360, `dsps_dp_s8_ansi` (portable scalar) correctly returns
2360, `dsps_dp_s8_aes3` returns 263 -- and different, still-wrong values across different
runs/inputs (359, -275, ruling out a one-off fluke). `tinytts::dspsDotProdS8()`
(`dsps_dp_s8.h`, this directory) currently always routes to the scalar `_ansi` kernel,
unconditionally, on every platform, until this is root-caused or upstream fixes it -- see
that function's own doc for how to re-enable the SIMD dispatch once it's trustworthy.

**`<dsps_dotprod.h>` does NOT reliably resolve to this directory's copy.** Arduino-ESP32
cores from ~3.3.x onward bundle their own precompiled `espressif__esp-dsp` component, and
that core's include directories are placed *before* any library's own `src/` on the
compiler command line -- so `#include <dsps_dotprod.h>` from `Ops.h` picks up the core's
bundled header, not this one, confirmed empirically against a real ESP32-S3 core build.
That bundled header has the float32 dotprod functions this project already relied on
(so that part kept working invisibly), but not `dsps_dp_s8` at all -- which is exactly why
`dsps_dp_s8.h` exists as a separate, quoted, project-relative include: a quoted include
can't be shadowed the way `<dsps_dotprod.h>` was. Its precompiled `.a` doesn't define
`dsps_dp_s8_ansi`/`_aes3` either (checked via `nm`), so linking this directory's own
definitions in is safe -- they're the only ones that exist anywhere in the final binary.

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
