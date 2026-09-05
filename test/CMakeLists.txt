# Host-side test build for TinyTTS. Not used by Arduino/PlatformIO builds --
# those compile src/ directly via their own toolchains. Every model stage is
# hand-written C++ (no TFLite Micro or other inference-runtime dependency),
# so this is a plain, dependency-free host build. Normally reached via the
# top-level CMakeLists.txt:
#
#   cmake -S . -B build -DTINYTTS_BUILD_TESTS=ON && cmake --build build && ./build/test/test_main
#
# but also buildable standalone (e.g. `cmake -S test -B test/build`) for
# quick iteration without the rest of the project.
cmake_minimum_required(VERSION 3.16)

if(NOT TARGET tinytts)
  project(tinytts_test CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  add_library(tinytts INTERFACE)
  target_include_directories(tinytts INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
  target_compile_features(tinytts INTERFACE cxx_std_17)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TINYTTS_ASAN "Build with AddressSanitizer (catches the class of heap-corruption bugs this project has hit on real hardware)" OFF)

add_executable(test_main main.cpp)
target_link_libraries(test_main PRIVATE tinytts)

if(TINYTTS_ASAN)
  target_compile_options(test_main PRIVATE -fsanitize=address -g -O1)
  target_link_options(test_main PRIVATE -fsanitize=address)
endif()
