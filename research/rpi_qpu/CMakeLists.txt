# Tests and benchmarks for the Raspberry Pi 1 / Zero GPU flow and vocoder
# (src/TinyTTS/QpuFlow.h, QpuVocoder.h), see docs/rpi-zero-gpu.md. Build on the Pi
# (armv6l), run as root:
#
#   cmake -S research/rpi_qpu -B build_qpu && cmake --build build_qpu
#   sudo ./build_qpu/hello_qpu                  # QPU access works at all
#   sudo ./build_qpu/test_qpu_conv              # every kernel vs. the CPU ops
#   sudo ./build_qpu/test_qpu_vocoder research/weights.bin   # whole decoder vs. the CPU
#   sudo ./build_qpu/test_qpu_flow research/weights.bin      # whole flow vs. the CPU
#   sudo ./build_qpu/bench_conv1d               # per-layer timing
#
# The kernels are generated and assembled ahead of time, so the Pi needs no
# assembler: src/TinyTTS/qpu/qpu_kernels.hex and qpu_kernels_index.h. To
# regenerate them after changing gen_qpu_kernels.py (on any machine, with
# https://github.com/maazl/vc4asm built):
#
#   python3 gen_qpu_kernels.py /tmp/qpu
#   vc4asm -V -C ../../src/TinyTTS/qpu/qpu_kernels.hex <vc4asm>/share/vc4inc/vc4.qinc /tmp/qpu/qpu_kernels.qasm
#   cp /tmp/qpu/qpu_kernels_index.h ../../src/TinyTTS/qpu/
#
# The .hex must have as many lines as the index's instruction total.
# hello.hex is assembled the same way from hello.qasm.
cmake_minimum_required(VERSION 3.16)
project(tinytts_rpi_qpu CXX)

set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(Threads REQUIRED)

foreach(target hello_qpu test_qpu_conv test_qpu_vocoder test_qpu_flow bench_conv1d)
  add_executable(${target} ${target}.cpp)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
  target_link_libraries(${target} PRIVATE Threads::Threads)
  # Same CPU flags as desktop/CMakeLists.txt, so CPU baselines match the CLI.
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv6l")
    target_compile_options(${target} PRIVATE
      -mcpu=arm1176jzf-s -mfpu=vfp -mfloat-abi=hard -funsafe-math-optimizations)
  endif()
endforeach()
