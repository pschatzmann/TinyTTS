# Desktop CLI: ./tinytts "some text" -- MiniAudioStream output
# (arduino-audio-tools, fetched by the top-level CMakeLists.txt), text
# passed as argv[1]. Not part of the Arduino/ESP-IDF library surface at all
# (see DesktopMain.h's own doc) -- only built when -DTINYTTS_BUILD_DESKTOP_MAIN=ON.
#
#   cmake -S . -B build -DTINYTTS_BUILD_DESKTOP_MAIN=ON
#   cmake --build build
#   ./build/desktop/tinytts "Hello world!"
#
# The CMake *target* below stays named tinytts_desktop, not tinytts -- the
# top-level CMakeLists.txt already has a CMake target literally named
# `tinytts` (the header-only INTERFACE library every consumer links
# against), and target names must be unique within a project. OUTPUT_NAME
# below only renames the produced binary file, not the target itself.

# audio-tools doesn't vendor miniaudio.h itself (single-header
# github.com/mackron/miniaudio -- only the header matters here, we never
# use any of its own CMake-built targets: MiniAudioStream.h compiles the
# whole implementation directly via #define MINIAUDIO_IMPLEMENTATION) --
# every arduino-audio-tools example that uses MiniAudioStream keeps a
# local copy for the same reason.
#
# Deliberately FetchContent_Populate(), not FetchContent_MakeAvailable():
# the latter also runs add_subdirectory() on miniaudio's own CMakeLists.txt,
# which defines an installable libminiaudio.a target we never build or
# link -- `cmake --install` then fails outright looking for that file
# ("file INSTALL cannot find .../libminiaudio.a"). Populate-only avoids
# pulling in that project at all; this is the same pattern
# arduino-audio-tools' own top-level CMakeLists.txt already uses for its
# optional portaudio/arduino_emulator fetches, for the identical reason.
FetchContent_Declare(miniaudio
  GIT_REPOSITORY "https://github.com/mackron/miniaudio.git"
  GIT_TAG master)
FetchContent_GetProperties(miniaudio)
if(NOT miniaudio_POPULATED)
  FetchContent_Populate(miniaudio)
endif()

add_executable(tinytts_desktop main.cpp)
set_target_properties(tinytts_desktop PROPERTIES OUTPUT_NAME tinytts)
target_link_libraries(tinytts_desktop PRIVATE tinytts arduino-audio-tools)
target_include_directories(tinytts_desktop PRIVATE ${miniaudio_SOURCE_DIR})
# NO_MAIN: AudioTools/AudioLibs/Emulation/Main.h (pulled in transitively by
# IS_MIN_DESKTOP) otherwise defines its own int main(void) -- conflicts with
# this target's own main.cpp.
target_compile_definitions(tinytts_desktop PRIVATE IS_MIN_DESKTOP TINYTTS_DESKTOP_TASK NO_MAIN)

# Needed on Linux: std::thread (audio-tools' Desktop Task) links against
# pthread, and miniaudio's implementation needs libm/libdl for its backend
# probing.
find_package(Threads REQUIRED)
target_link_libraries(tinytts_desktop PRIVATE Threads::Threads ${CMAKE_DL_LIBS} m)

# On 32-bit ARM without native 8-byte atomic instructions (e.g. Raspberry
# Pi Zero's ARMv6), GCC/glibc lower std::atomic<uint64_t>/ma_uint64 ops
# (used throughout miniaudio) to __atomic_*_8 libcalls instead of inline
# instructions -- those live in libatomic. Linked unconditionally: it ships
# with every GCC toolchain (including x86_64/aarch64, where it's simply
# unused), so there's no downside, and a check_library_exists() probe here
# isn't reliable -- it links its test program with the C compiler, which
# doesn't always see GCC's bundled libatomic.so the same way the real
# C++ link step does.
target_link_libraries(tinytts_desktop PRIVATE atomic)

# Raspberry Pi Zero (W)/Pi 1 (ARM1176JZF-S, ARMv6, armv6l) default to a
# generic soft-float-ish ARM target unless told otherwise, missing the
# chip's actual VFPv2 hardware FPU -- this matters a lot here since the
# vocoder/decoder stage is dominated by scalar float conv1d loops (no
# NEON exists on ARMv6 to vectorize with instead). Scoped to armv6l
# specifically so it doesn't mis-tune armv7/aarch64 Pi boards (2/3/4/5),
# which have their own (better) NEON-capable cores.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv6l")
  target_compile_options(tinytts_desktop PRIVATE
    -mcpu=arm1176jzf-s -mfpu=vfp -mfloat-abi=hard -funsafe-math-optimizations)
endif()

# Milk-V Duo 256M / Duo S: runs the decoder on the TPU (--tpu-model), see
# docs/milkv-duo.md. TPU_SDK_DIR points at a checkout of
# https://github.com/milkv-duo/tpu-sdk-sg200x (include/cviruntime.h,
# lib/libcviruntime.so) built for the same CPU as this binary.
option(TINYTTS_CVI_TPU "Build the desktop CLI with the CVITEK TPU vocoder (Milk-V Duo)" OFF)
set(TPU_SDK_DIR "" CACHE PATH "tpu-sdk-sg200x directory (include/, lib/)")
if(TINYTTS_CVI_TPU)
  find_path(CVIRUNTIME_INCLUDE_DIR cviruntime.h HINTS ${TPU_SDK_DIR}/include NO_CMAKE_FIND_ROOT_PATH REQUIRED)
  find_library(CVIRUNTIME_LIBRARY cviruntime HINTS ${TPU_SDK_DIR}/lib NO_CMAKE_FIND_ROOT_PATH REQUIRED)
  get_filename_component(CVIRUNTIME_LIB_DIR ${CVIRUNTIME_LIBRARY} DIRECTORY)
  target_include_directories(tinytts_desktop PRIVATE ${CVIRUNTIME_INCLUDE_DIR})
  target_compile_definitions(tinytts_desktop PRIVATE TINYTTS_HAVE_CVI_TPU)
  # libcviruntime.so pulls in libcvikernel.so etc. from the same directory.
  target_link_libraries(tinytts_desktop PRIVATE ${CVIRUNTIME_LIBRARY})
  target_link_options(tinytts_desktop PRIVATE "-Wl,-rpath-link,${CVIRUNTIME_LIB_DIR}")
  # Where the board's image keeps libcviruntime.so: the program then also
  # starts without LD_LIBRARY_PATH, which only login shells get from
  # /etc/profile -- not e.g. `ssh root@192.168.42.1 'tinytts ...'`.
  target_link_options(tinytts_desktop PRIVATE "-Wl,-rpath,/mnt/system/lib")
endif()

# Raspberry Pi 1 / Zero: the decoder on the VideoCore IV GPU (QpuVocoder,
# --qpu), see docs/rpi-zero-gpu.md. No extra libraries: the kernels are
# compiled in.
option(TINYTTS_RPI_QPU "Build the desktop CLI with the Raspberry Pi 1/Zero GPU vocoder (--qpu)" OFF)
if(TINYTTS_RPI_QPU)
  target_compile_definitions(tinytts_desktop PRIVATE TINYTTS_HAVE_RPI_QPU)
endif()

# MP3 replies for `tinytts --serve` (MP3EncoderShine from arduino-audio-tools,
# see DesktopMain.h's serve()): fetches https://github.com/pschatzmann/codec-shine,
# a small fixed-point MP3 encoder in plain C. With OFF, --serve replies with WAV
# only. To use a local checkout instead of fetching it, configure with
# -DFETCHCONTENT_SOURCE_DIR_CODEC_SHINE=/path/to/codec-shine.
option(TINYTTS_DESKTOP_MP3 "Build the desktop CLI with MP3 output for --serve (fetches codec-shine)" ON)
if(TINYTTS_DESKTOP_MP3)
  FetchContent_Declare(codec_shine
    GIT_REPOSITORY "https://github.com/pschatzmann/codec-shine.git"
    GIT_TAG main)
  FetchContent_MakeAvailable(codec_shine)
  target_link_libraries(tinytts_desktop PRIVATE codec-shine)
  target_compile_definitions(tinytts_desktop PRIVATE TINYTTS_HAVE_MP3)
endif()

# Without this, `cmake --install`/`make install` silently installs nothing
# of ours at all (only arduino-audio-tools' own headers, as an incidental
# side effect of its unrelated install() rules) -- there's no default
# "install every executable" behavior in CMake.
install(TARGETS tinytts_desktop RUNTIME DESTINATION bin)
