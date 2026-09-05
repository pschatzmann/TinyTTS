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

#   cmake -S . -B build -DTINYTTS_BUILD_TESTS=ON
#   cmake --build build
#   ./build/test/test_main
add_library(tinytts INTERFACE)
add_library(TinyTTS::TinyTTS ALIAS tinytts)
target_include_directories(tinytts INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(tinytts INTERFACE cxx_std_17)

option(TINYTTS_BUILD_TESTS "Build the host-side test harness (test/)" OFF)
if(TINYTTS_BUILD_TESTS)
  add_subdirectory(test)
endif()
