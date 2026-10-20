#!/bin/bash
# Synthesizes text on a Milk-V Duo (flow and decoder on its TPU) and plays it on
# this machine. Expects tinytts, the BF16 models and (for --16k) the 16 kHz
# weights in /root/tpu on the board (docs/milkv-duo.md, "Install and run"):
# tinytts loads the TPU models for the voice's sample rate from its own
# directory by itself. Timings from the board go to stderr.
#
#   research/milkv_duo/duo-say.sh "Hello world!"
#   research/milkv_duo/duo-say.sh --16k "The 16 kHz LJSpeech voice"
#   echo "Text from a pipe" | research/milkv_duo/duo-say.sh
#   DUO=root@192.168.42.1 research/milkv_duo/duo-say.sh -o out.wav "Save instead"
set -euo pipefail

DUO=${DUO:-root@192.168.42.1}
WEIGHTS=
OUT=
while [[ $# -gt 0 ]]; do
  case $1 in
    --16k) WEIGHTS=" --weights /root/tpu/ljspeech16k_weights.bin"; shift ;;
    -o) OUT=$2; shift 2 ;;
    *) break ;;
  esac
done

# Quote each argument for the remote shell; with no arguments, stdin is passed on.
ARGS=
if [[ $# -gt 0 ]]; then ARGS=$(printf ' %q' "$@"); fi
REMOTE="/root/tpu/tinytts --stdout$WEIGHTS$ARGS"

if [[ -n $OUT ]]; then
  ssh "$DUO" "$REMOTE" > "$OUT"
else
  ssh "$DUO" "$REMOTE" | aplay -q
fi
