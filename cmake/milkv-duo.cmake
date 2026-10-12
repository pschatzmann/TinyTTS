# Cross-compiles for the Milk-V Duo 256M / Duo S running the default RISC-V
# (C906, musl) Linux image -- see docs/milkv-duo.md.
#
#   cmake -S . -B build_duo -DCMAKE_TOOLCHAIN_FILE=cmake/milkv-duo.cmake \
#     -DDUO_TOOLCHAIN_DIR=/path/to/host-tools/gcc/riscv64-linux-musl-x86_64 \
#     -DTINYTTS_BUILD_DESKTOP_MAIN=ON -DTINYTTS_CVI_TPU=ON -DTPU_SDK_DIR=/path/to/tpu-sdk-sg200x
#
# DUO_TOOLCHAIN_DIR: the T-Head musl toolchain from
# https://github.com/milkv-duo/host-tools (gcc/riscv64-linux-musl-x86_64).
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(DUO_TOOLCHAIN_DIR "$ENV{DUO_TOOLCHAIN_DIR}" CACHE PATH "host-tools/gcc/riscv64-linux-musl-x86_64")
set(CMAKE_C_COMPILER ${DUO_TOOLCHAIN_DIR}/bin/riscv64-unknown-linux-musl-gcc)
set(CMAKE_CXX_COMPILER ${DUO_TOOLCHAIN_DIR}/bin/riscv64-unknown-linux-musl-g++)

# Same CPU flags as the TPU SDK's own toolchain file
# (tpu-sdk-sg200x/cmake/toolchain-riscv64-linux-musl-x86_64.cmake).
set(DUO_CPU_FLAGS "-mcpu=c906fdv -march=rv64gcv0p7_zfh_xthead -mabi=lp64d")
set(CMAKE_C_FLAGS_INIT "${DUO_CPU_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${DUO_CPU_FLAGS}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
