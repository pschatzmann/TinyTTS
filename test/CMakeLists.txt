# Host-side test build for TinyTTS. Not used by Arduino/PlatformIO builds --
# those compile src/ directly via their own toolchains (real hardware uses
# tflm_esp32; this target fetches the real TFLite Micro source instead, see
# cmake/FetchTFLiteMicro.cmake and main.cpp's top comment). Normally reached
# via the top-level CMakeLists.txt:
#
#   cmake -S . -B build -DTINYTTS_BUILD_TESTS=ON && cmake --build build && ./build/test/test_main
#
# but also buildable standalone (e.g. `cmake -S test -B test/build`) for
# quick iteration without the rest of the project. First configure needs
# network access (fetches ~15MB of source, cached under the build dir after
# that -- re-configuring doesn't re-download).
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

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(FetchTFLiteMicro)

# TFLM's own tree includes kernel variants for other hardware targets
# (xtensa/cmsis_nn/ceva/arc_mli/ethos_u/hexagon accelerators), its own unit
# tests/test-only helpers, and the Signal library (audio-frontend ops --
# RFFT, framer, filter bank, etc. -- kissfft-dependent). TinyTTS's models
# never register any Signal op, so signal/micro/kernels/*.cc is deliberately
# NOT compiled here at all -- only its *headers* are needed, because
# micro_mutable_op_resolver.h unconditionally #includes a couple of them
# even though our resolvers never call the corresponding Add*() methods;
# TFLM_SOURCE_DIR is already on the include path below, so those headers
# resolve fine without adding kissfft or compiling any Signal .cc.
# Also exclude every other-target micro_time.cc/system_setup.cc override
# (arc_custom/arc_emsdp/bluepill/chre/cortex_m_*/riscv32_generic -- each
# provides its own; only tensorflow/lite/micro/{micro_time,system_setup}.cc,
# the portable reference versions, should get compiled) and non-source
# directories (benchmarks/examples/integration_tests/models/python/docs)
# that don't need to be globbed at all.
# A couple of small pieces (ErrorReporter::Report's base impl, GetBuiltinCode)
# live in tensorflow/compiler/mlir/lite/{core,schema}/ -- a "mirror" TFLM
# depends on for a few core API bits -- not just tensorflow/lite/ itself.
file(GLOB_RECURSE TFLM_SOURCES
  ${TFLM_SOURCE_DIR}/tensorflow/lite/*.cc
  ${TFLM_SOURCE_DIR}/tensorflow/compiler/mlir/lite/*.cc
)
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/(xtensa|cmsis_nn|ceva|arc_mli|ethos_u|hexagon|testdata|experimental)/")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/(arc_custom|arc_emsdp|bluepill|chre|cortex_m_generic|cortex_m_corstone_300|riscv32_generic)/")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/(benchmarks|examples|integration_tests|models|python|docs|tools|testing|test_data_generation)/")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "_test\\.cc$")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/(test_helper_custom_ops|mock_micro_graph|test_helpers|conv_test_common|kernel_runner)\\.cc$")

# TFLM itself needs -fno-rtti -fno-exceptions (matches how it's actually
# built/shipped, e.g. tflm_esp32); main.cpp does not (it uses exceptions for
# its own file-reading error handling) -- an OBJECT library keeps the flag
# scoped to just the vendored sources instead of the whole target.
add_library(tflm_objs OBJECT ${TFLM_SOURCES})
target_include_directories(tflm_objs PUBLIC
  ${TFLM_SOURCE_DIR}
  ${GEMMLOWP_INCLUDE_DIR}
  ${FLATBUFFERS_INCLUDE_DIR}
  ${RUY_INCLUDE_DIR}
)
target_compile_definitions(tflm_objs PUBLIC TF_LITE_STATIC_MEMORY)
target_compile_options(tflm_objs PRIVATE -fno-rtti -fno-exceptions)

add_executable(test_main main.cpp)
target_link_libraries(test_main PRIVATE tinytts tflm_objs)

if(TINYTTS_ASAN)
  target_compile_options(tflm_objs PRIVATE -fsanitize=address -g -O1)
  target_compile_options(test_main PRIVATE -fsanitize=address -g -O1)
  target_link_options(test_main PRIVATE -fsanitize=address)
endif()
