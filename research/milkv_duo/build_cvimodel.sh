#!/usr/bin/env bash
# Compiles export_decoder_tpu.py's ONNX window into cvimodels for the Milk-V
# Duo 256M / Duo S TPU (SG2002/SG2000 = TPU-MLIR processor cv181x), checking
# each against the ONNX output in TPU-MLIR's simulator. See docs/milkv-duo.md.
#
# Needs TPU-MLIR >= 1.30 (https://github.com/sophgo/tpu-mlir): `pip install
# tpu_mlir[onnx]` in a Python 3.10 environment, or the sophgo/tpuc_dev Docker
# image. Run from research/ after export_decoder_tpu.py:
#
#   milkv_duo/build_cvimodel.sh [NAME [all]]    # default decoder_44100hz_w128, BF16 only
#
# NAME is the ONNX model's name as export_decoder_tpu.py wrote it,
# decoder_<sample rate>hz_w<W> -- e.g. decoder_16000hz_w128 for a model
# exported with --checkpoint from a 16 kHz training run.
#
# Produces out/<NAME>_bf16.cvimodel: 27dB median SNR vs. float32 for the
# released 44.1 kHz model. With "all" also, for comparison (both unusable,
# see docs/milkv-duo.md, and both fail model_deploy's accuracy check):
#   int8  all-INT8: -2.5dB median SNR
#   mix   INT8 with the 47 layers most sensitive to quantization in BF16:
#         -2dB; its layer search alone takes about 4 hours
# validate_cvimodel.py measures them over whole utterances.
set -euo pipefail

NAME=${1:-decoder_44100hz_w128}
ALL=${2:-}
W=${NAME##*_w}
OUT=$(cd "$(dirname "$0")" && pwd)/out
PROCESSOR=cv181x
[[ "$W" =~ ^[0-9]+$ && -f "$OUT/$NAME.onnx" ]] || {
  echo "expected out/decoder_<rate>hz_w<W>.onnx from export_decoder_tpu.py, got $NAME" >&2
  exit 1
}
mkdir -p "$OUT/work_$NAME"
cd "$OUT/work_$NAME"

run() { echo "+ $*"; "$@"; }

run model_transform \
  --model_name "$NAME" \
  --model_def "../$NAME.onnx" \
  --input_shapes "[[1,32,$W],[1,128,1],[1,1,$W]]" \
  --test_input "../${NAME}_test_input.npz" \
  --test_result "${NAME}_top_outputs.npz" \
  --mlir "$NAME.mlir"

CALIB="../${NAME}_calib"
N_CAL=$(ls "$CALIB"/*.npz | wc -l)
run run_calibration "$NAME.mlir" \
  --dataset "$CALIB" \
  --input_num "$N_CAL" \
  -o "${NAME}_cali_table"

deploy() {  # deploy <suffix> <model_deploy args...>
  local suffix=$1
  shift
  run model_deploy \
    --mlir "$NAME.mlir" \
    --processor "$PROCESSOR" \
    --test_input "../${NAME}_test_input.npz" \
    --test_reference "${NAME}_top_outputs.npz" \
    --model "../${NAME}_${suffix}.cvimodel" \
    "$@"
}

deploy bf16 --quantize BF16 --tolerance 0.99,0.9

if [[ "$ALL" != all ]]; then
  ls -la "$OUT/${NAME}_bf16.cvimodel"
  exit 0
fi

# Mixed precision: TPU-MLIR's qtable search keeps the layers that lose the
# most accuracy in INT8 in BF16 instead.
run run_calibration "$NAME.mlir" \
  --dataset "$CALIB" \
  --input_num "$N_CAL" \
  --inference_num 10 \
  --search search_qtable \
  --expected_cos 0.99 \
  --processor "$PROCESSOR" \
  --fp_type BF16 \
  --calibration_table "${NAME}_cali_table_mix" \
  --quantize_table "${NAME}_qtable"
deploy mix --quantize INT8 --calibration_table "${NAME}_cali_table_mix" \
  --quantize_table "${NAME}_qtable" --tolerance 0.8,0.4 ||
  echo "mix: accuracy check failed (expected, see docs/milkv-duo.md)"

deploy int8 --quantize INT8 --calibration_table "${NAME}_cali_table" --tolerance 0.8,0.4 ||
  echo "int8: accuracy check failed (expected, see docs/milkv-duo.md)"

ls -la "$OUT/$NAME"_*.cvimodel
