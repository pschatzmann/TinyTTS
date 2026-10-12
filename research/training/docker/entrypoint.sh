#!/bin/sh
# Entrypoint of the TinyTTS training image: `<command> [args...]`, where the
# args go straight to the matching script. Runs in /research/training, so
# relative paths like data/... and runs/... refer to the mounted volumes.
set -e

usage() {
  cat <<'EOF'
TinyTTS training container -- see docs/training.md

  prepare    [args]  prepare_dataset.py   (dataset -> 44.1 kHz wavs + filelists)
  train      [args]  train.py             (fine-tune / train; --resume to continue)
  synthesize [args]  synthesize.py        (checkpoint + text -> wav)
  export     [args]  export_weights_and_vectors.py (checkpoint -> weights.bin for the C++ library)
  shell              interactive shell
  <anything else>    run as a command

Mount your data at /research/training/data and outputs at /research/training/runs, e.g.:
  docker run --rm --shm-size=2g -v "$PWD/data:/research/training/data" \
      -v "$PWD/runs:/research/training/runs" tinytts-train:cpu train --config configs/en_ljspeech.json
Pass `<command> --help` for a script's own options.
EOF
}

cmd="${1:-help}"
[ $# -gt 0 ] && shift

case "$cmd" in
  prepare)    exec python prepare_dataset.py "$@" ;;
  train)      exec python train.py "$@" ;;
  synthesize) exec python synthesize.py "$@" ;;
  export)     exec python /research/export_weights_and_vectors.py "$@" ;;
  shell)      exec bash "$@" ;;
  help|-h|--help) usage ;;
  *)          exec "$cmd" "$@" ;;
esac
