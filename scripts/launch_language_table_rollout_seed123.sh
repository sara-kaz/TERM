#!/usr/bin/env bash
# Language-Table official sim task-success eval — seed 123 best checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
LT_ROOT="${LANGUAGE_TABLE_ROOT:-$ROOT/vendor/language-table}"
CKPT="${CKPT:-$ROOT/checkpoints/Language_table/seed123/best_sft_vera.pt}"
# GPU 1 by default — leave GPU 0 free for CALVIN rollout (launch_calvin_rollout_seed123.sh).
GPU="${CUDA_VISIBLE_DEVICES:-1}"

export PYTHONPATH="$LT_ROOT:$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

cd "$ROOT"
exec "$VENV" scripts/run_language_table_rollout_eval.py \
  --checkpoint "$CKPT" \
  --language_table_root "$LT_ROOT" \
  --eval_seed 123 \
  --num_episodes 50 \
  --device 0 \
  --tune \
  --tune_episodes 8 \
  --action_mode hybrid \
  --action_magnitude 0.05 \
  --action_hold_steps 4 \
  --crop_factor 0.95 \
  --eval_log_dir "$(dirname "$CKPT")/lt_rollout_eval"
