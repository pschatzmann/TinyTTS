#!/bin/bash
# Fetches the parts of duo-buildroot-sdk-v2 that build_boot_sd.sh needs (about
# 450MB instead of the whole SDK), one directory at a time: git can't resume a
# broken fetch, so on a slow or unreliable connection smaller pieces lose less.
# A piece that stalls for 10 minutes is retried.
#
#   fetch_sdk.sh [SDK_DIR]      # default ~/duo-buildroot-sdk-v2
set -uo pipefail
SDK=${1:-$HOME/duo-buildroot-sdk-v2}
if [[ ! -d $SDK/.git ]]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/milkv-duo/duo-buildroot-sdk-v2.git "$SDK" || exit 1
fi
cd "$SDK" || exit 1
git sparse-checkout init --no-cone

add() {  # add PATH...: fetch these paths, retrying a stalled fetch
  for try in 1 2 3 4 5 6; do
    if timeout 600 git sparse-checkout add "$@" >/dev/null 2>&1; then
      echo "ok $*"
      return 0
    fi
    echo "retry $try $*"
    rm -f .git/objects/pack/tmp_pack_*
    sleep 10
  done
  echo "failed: $*" >&2
  return 1
}

K=linux_5.10
for d in build device scripts .version \
  $K/scripts $K/arch/riscv $K/include $K/init $K/kernel $K/mm $K/lib $K/fs $K/block $K/certs \
  $K/crypto $K/ipc $K/security $K/usr $K/virt $K/sound $K/net \
  $K/tools/include $K/tools/arch/riscv $K/tools/lib $K/tools/scripts; do
  add "/$d" || exit 1
done
for d in $(git ls-tree --name-only HEAD $K/drivers/); do
  add "/$d" || exit 1
done
# Loose files (Makefile, Kconfig, ...) next to the directories above, and the
# few Kconfig files of other directories that the kernel's Kconfig includes.
loose=$(for d in $K $K/arch $K/drivers $K/tools; do
  git ls-tree HEAD $d/ | awk '$2 == "blob" { print "/" $4 }'
done)
add $loose /$K/arch/x86/ras/Kconfig /$K/samples/Kconfig /$K/Documentation/Kconfig || exit 1
echo "done: $(du -sh . | cut -f1) in $SDK"
