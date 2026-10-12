#!/usr/bin/env bash
# Compiles export_flow_tpu.py's fixed-length flow models into BF16 cvimodels
# for the Milk-V Duo 256M / Duo S TPU (processor cv181x), checking each
# against the ONNX output in TPU-MLIR's simulator. See docs/milkv-duo.md.
#
# Same requirements as build_cvimodel.sh. Run from research/ after
# export_flow_tpu.py:
#
#   milkv_duo/build_flow_cvimodels.sh [RATE [LENGTH ...]]   # default 44100 128 256 512 1024
#
# RATE selects the model, as in export_flow_tpu.py's names flow_<RATE>hz_l<L>
# (e.g. 16000 for a model exported with --checkpoint from a 16 kHz training
# run). Produces out/flow_<RATE>hz_l<L>_bf16.cvimodel. BF16 only: the flow's attention and
# exp() need more range than per-tensor INT8 gives (the decoder's INT8 build
# is already unusable, see docs/milkv-duo.md).
set -euo pipefail

RATE=${1:-44100}
shift || true
LENGTHS=${*:-128 256 512 1024}
OUT=$(cd "$(dirname "$0")" && pwd)/out
PROCESSOR=cv181x

run() { echo "+ $*"; "$@"; }

for L in $LENGTHS; do
  NAME=flow_${RATE}hz_l$L
  mkdir -p "$OUT/work_$NAME"
  cd "$OUT/work_$NAME"
  run model_transform \
    --model_name "$NAME" \
    --model_def "../$NAME.onnx" \
    --input_shapes "[[1,32,$L],[1,1,$L],[1,128,1]]" \
    --test_input "../${NAME}_test_input.npz" \
    --test_result "${NAME}_top_outputs.npz" \
    --mlir "$NAME.mlir"
  run model_deploy \
    --mlir "$NAME.mlir" \
    --processor "$PROCESSOR" \
    --quantize BF16 \
    --test_input "../${NAME}_test_input.npz" \
    --test_reference "${NAME}_top_outputs.npz" \
    --tolerance 0.99,0.9 \
    --model "../${NAME}_bf16.cvimodel"
done

ls -la "$OUT/flow_${RATE}hz"_l*_bf16.cvimodel
