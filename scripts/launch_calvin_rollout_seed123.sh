#!/usr/bin/env bash
# Full CALVIN official rollout eval — Full VERA seed 123 best checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
CALVIN_ROOT="${CALVIN_ROOT:-$HOME/work/calvin}"
DATA="${CALVIN_DATA:-$HOME/calvin_task_D/task_D_D}"
CKPT="${CKPT:-$ROOT/checkpoints/calvin_core6_ltdev/full_vera/seed123/best_sft_vera.pt}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

export PYTHONPATH="$CALVIN_ROOT/calvin_env:$CALVIN_ROOT/calvin_models:$ROOT:${PYTHONPATH:-}"
export PYOPENGL_PLATFORM=egl
export EGL_VISIBLE_DEVICES="$GPU"
export CUDA_VISIBLE_DEVICES="$GPU"

cd "$ROOT"
exec "$VENV" scripts/run_calvin_rollout_eval.py \
  --checkpoint "$CKPT" \
  --dataset_path "$DATA" \
  --calvin_root "$CALVIN_ROOT" \
  --num_sequences 1000 \
  --device 0 \
  --action_mode hybrid \
  --action_magnitude 1.0 \
  --action_hold_steps 30 \
  --lang_goal task_key \
  --eval_log_dir "$(dirname "$CKPT")/calvin_rollout_eval_taskkey"
