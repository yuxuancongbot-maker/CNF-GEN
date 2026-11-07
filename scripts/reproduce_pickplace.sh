#!/usr/bin/env bash
# End-to-end reproduction of the 6-DOF pick-and-place experiments.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
DATA="${DATA:-runs/pickplace256}"
EVAL="${EVAL:-runs/pickplace256_eval}"
OUT="${OUT:-runs}"

export PYTHONPATH="${PYTHONPATH:-}:src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" scripts/generate_dataset.py --output "$DATA" --count 128 --seed 0
"$PY" scripts/generate_dataset.py --output "$EVAL" --count 16 --seed 1
"$PY" scripts/train_motion_prior.py --data "$DATA" --output "$OUT/motion_model.pt"
"$PY" scripts/train_soft_prior.py \
  --data "$DATA" --output "$OUT/soft_motion_model_w20.pt" --soft-weight 20
"$PY" scripts/train_video_wm.py --data "$DATA" --output "$OUT/video_wm.pt"
"$PY" scripts/generate_controls.py \
  --eval "$EVAL" --motion-prior "$OUT/motion_model.pt" \
  --output "$OUT/controls_sparse" --mode sparse --samples 16
"$PY" scripts/generate_controls.py \
  --eval "$EVAL" --motion-prior "$OUT/motion_model.pt" \
  --output "$OUT/controls_dense" --mode dense --samples 16
"$PY" scripts/eval_action_wm.py \
  --eval "$EVAL" --motion-prior "$OUT/motion_model.pt" \
  --soft-prior "$OUT/soft_motion_model_w20.pt" \
  --video-wm "$OUT/video_wm.pt" \
  --output "$OUT/eval_action_wm" --samples 16 --device cuda

echo "done: $OUT"
