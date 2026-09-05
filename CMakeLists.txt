# TinyTTS is header-only, so this is an INTERFACE library (no sources of its
# own to compile) -- for consuming src/ from a CMake-based build (e.g. an
# ESP-IDF component, PlatformIO's CMake mode, or a plain host build) instead
# of Arduino's own build system, which doesn't use this file at all.
#
# On ARDUINO/ESP32 builds, TinyTTS.h needs tflm_esp32 (TFLite Micro,
# precompiled for ESP32-S3) as a compiled dependency -- add that yourself in
# the consuming project, this file doesn't fetch it. A non-ARDUINO build
# (this library's own host tests, see test/) instead needs the vendored
# TFLite Micro source in test/vendor/, which is test-only and not part of
# this target -- see test/CMakeLists.txt.
#
#   cmake -S . -B build -DTINYTTS_BUILD_TESTS=ON
#   cmake --build build
#   ./build/test/test_main
cmake_minimum_required(VERSION 3.16)
project(TinyTTS CXX)

add_library(tinytts INTERFACE)
add_library(TinyTTS::TinyTTS ALIAS tinytts)
target_include_directories(tinytts INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(tinytts INTERFACE cxx_std_17)

option(TINYTTS_BUILD_TESTS "Build the host-side test harness (test/)" OFF)
if(TINYTTS_BUILD_TESTS)
  add_subdirectory(test)
endif()
