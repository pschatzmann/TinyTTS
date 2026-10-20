#!/bin/bash
# Builds boot.sd (kernel + device tree) for a Milk-V Duo 256M with a MAX98357A
# I2S amplifier on I2S2. See docs/milkv-duo.md, "I2S amplifier (MAX98357A)".
#
#   build_boot_sd.sh SDK_DIR TOOLCHAIN_DIR BASE_CONFIG OUT_DIR
#
# SDK_DIR        duo-buildroot-sdk-v2 checkout (build/, linux_5.10/)
# TOOLCHAIN_DIR  host-tools/gcc/riscv64-linux-musl-x86_64
# BASE_CONFIG    the board's running kernel configuration:
#                ssh root@192.168.42.1 'zcat /proc/config.gz' > running.config
# OUT_DIR        gets boot.sd, Image, the .dtb and the build tree (kbuild/)
#
# The kernel release must stay what the image's own modules in
# /mnt/system/ko (TPU, video, ...) were built for: module versioning is off,
# and insmod refuses a module whose vermagic doesn't match.
set -euo pipefail

SDK=$(realpath "$1")
TC=$(realpath "$2")
BASE_CONFIG=$(realpath "$3")
OUT=$(realpath -m "$4")
HERE=$(cd "$(dirname "$0")" && pwd)
KSRC=$SDK/linux_5.10
KOUT=$OUT/kbuild
RELEASE=5.10.4-tag-
mkdir -p "$KOUT"

# 1. The sound card driver into the kernel tree, unless the patch is already in.
PATCH=$HERE/kernel/max98357a-sound-card.patch
if git -C "$SDK" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "kernel patch already applied"
else
  git -C "$SDK" apply "$PATCH"
fi

# 2. Configuration: the running one plus the amplifier.
cp "$BASE_CONFIG" "$KOUT/.config"
cat >> "$KOUT/.config" <<EOF
CONFIG_SND_PROC_FS=y
CONFIG_SND_SOC_MAX98357A=y
CONFIG_SND_SOC_CV1835_MAX98357A=y
# CONFIG_LOCALVERSION_AUTO is not set
CONFIG_LOCALVERSION="-tag-"
EOF
MAKE=(make -C "$KSRC" O="$KOUT" ARCH=riscv CROSS_COMPILE="$TC/bin/riscv64-unknown-linux-musl-" LOCALVERSION=)
"${MAKE[@]}" olddefconfig >/dev/null
for opt in SND_PROC_FS SND_SOC_MAX98357A SND_SOC_CV1835_MAX98357A; do
  grep -q "^CONFIG_$opt=y" "$KOUT/.config" || { echo "CONFIG_$opt didn't stick" >&2; exit 1; }
done
got=$("${MAKE[@]}" -s kernelrelease)
[[ $got == "$RELEASE" ]] || { echo "kernel release is '$got', expected '$RELEASE'" >&2; exit 1; }

# 3. Kernel and device tree.
"${MAKE[@]}" -j"$(nproc)" Image
"$HERE/build_dtb.sh" "$SDK" "$HERE/kernel/sg2002_milkv_duo256m_max98357a.dts" "$OUT/sg2002_milkv_duo256m_max98357a.dtb"
cp "$KOUT/arch/riscv/boot/Image" "$OUT/Image"

# 4. boot.sd: the same FIT layout as the SDK's (lzma kernel at 0x80200000, one FDT).
lzma -9 -k -f -c "$OUT/Image" > "$OUT/Image.lzma"
cat > "$OUT/boot.its" <<EOF
/dts-v1/;
/ {
	description = "Various kernels, ramdisks and FDT blobs";
	#address-cells = <2>;
	images {
		kernel-1 {
			description = "cvitek kernel";
			data = /incbin/("Image.lzma");
			type = "kernel";
			arch = "riscv";
			os = "linux";
			compression = "lzma";
			load = <0x0 0x80200000>;
			entry = <0x0 0x80200000>;
			hash-2 { algo = "crc32"; };
		};
		fdt-sg2002_milkv_duo256m_musl_riscv64_sd {
			description = "cvitek device tree - sg2002_milkv_duo256m_musl_riscv64_sd";
			data = /incbin/("sg2002_milkv_duo256m_max98357a.dtb");
			type = "flat_dt";
			arch = "riscv";
			compression = "none";
			hash-1 { algo = "sha256"; };
		};
	};
	configurations {
		default = "config-sg2002_milkv_duo256m_musl_riscv64_sd";
		config-sg2002_milkv_duo256m_musl_riscv64_sd {
			description = "boot cvitek system with board sg2002_milkv_duo256m_musl_riscv64_sd";
			kernel = "kernel-1";
			fdt = "fdt-sg2002_milkv_duo256m_musl_riscv64_sd";
		};
	};
};
EOF
(cd "$OUT" && mkimage -f boot.its boot.sd >/dev/null)
dumpimage -l "$OUT/boot.sd" | grep -E "Image|Configuration|Data Size|Load"
echo "-> $OUT/boot.sd (kernel $RELEASE)"
