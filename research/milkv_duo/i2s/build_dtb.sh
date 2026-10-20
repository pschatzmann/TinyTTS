#!/bin/bash
# Compiles the Duo 256M device tree from duo-buildroot-sdk-v2 the way the SDK's
# kernel build does (C preprocessor, then dtc).
#
#   build_dtb.sh SDK_DIR BOARD_DTS OUT.dtb
set -euo pipefail
SDK=$1
DTS=$2
OUT=$3
D=$SDK/build/boards/default/dts
BOARD_DIR=$SDK/build/boards/cv181x/sg2002_milkv_duo256m_musl_riscv64_sd
GEN=$(mktemp -d)
trap 'rm -rf "$GEN"' EXIT
# The memory map header is generated from the board's memmap.py (build/scripts/mmap.mk).
python3 "$SDK/build/scripts/mmap_conv.py" --type h "$BOARD_DIR/memmap.py" "$GEN/cvi_board_memmap.h"
cpp -nostdinc -undef -D__DTS__ -x assembler-with-cpp \
  -I "$(dirname "$DTS")" -I "$BOARD_DIR/dts_riscv" -I "$D/cv181x" -I "$D/cv181x_riscv" -I "$GEN" \
  -I "$SDK/linux_5.10/include" -I "$SDK/linux_5.10/arch/riscv/boot/dts" \
  "$DTS" -o "$OUT.pre"
dtc -q -I dts -O dtb -o "$OUT" "$OUT.pre"
rm -f "$OUT.pre"
