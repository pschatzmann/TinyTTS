#!/bin/bash
# Synthesizes text on a Milk-V Duo (flow and decoder on its TPU) and plays it on
# this machine. Expects tinytts, the BF16 models and (for --16k) the 16 kHz
# weights in /root/tpu on the board (docs/milkv-duo.md, "Install and run"). Timings from
# the board go to stderr.
#
#   research/milkv_duo/duo-say.sh "Hello world!"
#   research/milkv_duo/duo-say.sh --16k "The 16 kHz LJSpeech voice"
#   echo "Text from a pipe" | research/milkv_duo/duo-say.sh
#   DUO=root@192.168.42.1 research/milkv_duo/duo-say.sh -o out.wav "Save instead"
set -euo pipefail

DUO=${DUO:-root@192.168.42.1}
RATE=44100
WEIGHTS=
OUT=
while [[ $# -gt 0 ]]; do
  case $1 in
    --16k) RATE=16000; WEIGHTS="--weights ljspeech16k_weights.bin"; shift ;;
    -o) OUT=$2; shift 2 ;;
    *) break ;;
  esac
done

# Quote each argument for the remote shell; with no arguments, stdin is passed on.
ARGS=
if [[ $# -gt 0 ]]; then ARGS=$(printf ' %q' "$@"); fi
FLOWS=
for L in 128 256 512 1024; do FLOWS="$FLOWS --tpu-flow flow_${RATE}hz_l${L}_bf16.cvimodel"; done
REMOTE="cd /root/tpu && LD_LIBRARY_PATH=/mnt/system/lib ./tinytts --stdout $WEIGHTS \
  --tpu-model decoder_${RATE}hz_w128_bf16.cvimodel$FLOWS$ARGS"

if [[ -n $OUT ]]; then
  ssh "$DUO" "$REMOTE" > "$OUT"
else
  ssh "$DUO" "$REMOTE" | aplay -q
fi
