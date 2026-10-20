#!/bin/sh
# Routes I2S2 to the Ethernet pads of a Milk-V Duo 256M for a MAX98357A, after
# booting the kernel from build_boot_sd.sh. Run once per boot, as root, e.g.
# from /mnt/system/auto.sh. See docs/milkv-duo.md, "I2S amplifier (MAX98357A)".
DIR=$(dirname "$0")

# The pads' IIS2 functions (the default is GPIO).
cvi-pinmux -w PAD_ETH_TXP/IIS2_LRCK >/dev/null
cvi-pinmux -w PAD_ETH_TXM/IIS2_BCLK >/dev/null
cvi-pinmux -w PAD_ETH_RXP/IIS2_DO >/dev/null
cvi-pinmux -w PAD_ETH_RXM/IIS2_DI >/dev/null

# Disconnect the pads from the Ethernet PHY.
"$DIR/eth2i2s" >/dev/null

grep -q max98357a /proc/asound/cards 2>/dev/null || echo "i2s-max98357a: no max98357a sound card (wrong kernel?)" >&2
