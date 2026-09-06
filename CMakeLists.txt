# TinyTTS is header-only, so this needs no sources of its own to compile --
# just registering its include path. Supports three consumers from this one
# file:
#   - ESP-IDF: point EXTRA_COMPONENT_DIRS (or idf_component.yml) at this repo
#     root -- IDF sets ESP_PLATFORM, detected below, and this registers as a
#     normal IDF component instead of running the plain-CMake path.
#   - Plain CMake (PlatformIO's CMake mode, a host build, ...): the INTERFACE
#     library further down, consumed as TinyTTS::TinyTTS.
#   - Arduino: doesn't use this file at all -- see library.properties.
#
# Every model stage is hand-written C++ -- no TFLite Micro or other
# inference-runtime dependency on any of the three, so there's nothing else
# to add here or in a consuming project.
if(ESP_PLATFORM)
  idf_component_register(INCLUDE_DIRS "src")
  return()
endif()

cmake_minimum_required(VERSION 3.16)
project(TinyTTS CXX)

# Applies to every target below (test/, desktop/) -- without this, an
# unspecified CMAKE_BUILD_TYPE means no optimization flags at all (~15-20x
# slower measured on this project's compute-bound hot loops), easy to miss
# since the build still succeeds, just silently unoptimized.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#   cmake -S . -B build -DTINYTTS_BUILD_TESTS=ON
#   cmake --build build
#   ./build/test/test_main
add_library(tinytts INTERFACE)
add_library(TinyTTS::TinyTTS ALIAS tinytts)
target_include_directories(tinytts INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(tinytts INTERFACE cxx_std_17)

option(TINYTTS_BUILD_TESTS "Build the host-side test harness (test/)" OFF)

# Optional: fetches arduino-audio-tools and enables TileSplitter.h's real
# dual-core (std::thread-backed) fork/join path in conv1d()/convTranspose1d()
# for the host test build too, not just ESP32 -- off by default so the
# ordinary host test build stays exactly as dependency-free as it is today
# (see Ops.h's TileSplitter include -- __has_include-gated, single-core
# fallback with no dependency at all when this is OFF).
option(TINYTTS_BUILD_PARALLEL_OPS "Build the host test target with dual-core TileSplitter enabled (fetches arduino-audio-tools)" OFF)

# Optional: builds desktop/'s tinytts CLI (MiniAudioStream output, text
# passed as argv[1]) -- also needs arduino-audio-tools, plus the
# single-header mackron/miniaudio library it doesn't vendor itself.
option(TINYTTS_BUILD_DESKTOP_MAIN "Build the desktop CLI (fetches arduino-audio-tools + miniaudio)" OFF)

if(TINYTTS_BUILD_PARALLEL_OPS OR TINYTTS_BUILD_DESKTOP_MAIN)
  include(FetchContent)
  set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
  # ADD_ARDUINO_EMULATOR OFF: skips the separate, heavier
  # pschatzmann/Arduino-Emulator fetch (full WiFi/Client emulation) --
  # IS_MIN_DESKTOP's own lightweight Emulation/{Arduino,Time,Main}.h
  # (audio-tools' own src/, no extra fetch) cover everything TinyTTS needs.
  # ADD_PORTAUDIO OFF: MiniAudioStream doesn't use PortAudio.
  set(ADD_ARDUINO_EMULATOR OFF CACHE BOOL "" FORCE)
  set(ADD_PORTAUDIO OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(arduino-audio-tools
    GIT_REPOSITORY "https://github.com/pschatzmann/arduino-audio-tools.git"
    GIT_TAG main)
  FetchContent_MakeAvailable(arduino-audio-tools)
endif()

if(TINYTTS_BUILD_PARALLEL_OPS)
  # NO_MAIN: AudioTools/AudioLibs/Emulation/Main.h (pulled in transitively
  # by IS_MIN_DESKTOP) otherwise defines its own int main(void) -- conflicts
  # with test/main.cpp's own main().
  target_compile_definitions(tinytts INTERFACE IS_MIN_DESKTOP TINYTTS_DESKTOP_TASK NO_MAIN)
  target_link_libraries(tinytts INTERFACE arduino-audio-tools)
endif()

if(TINYTTS_BUILD_TESTS)
  add_subdirectory(test)
endif()

if(TINYTTS_BUILD_DESKTOP_MAIN)
  add_subdirectory(desktop)
endif()
