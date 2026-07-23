#!/usr/bin/env bash
# Train full MCIL (static + gripper + rel actions) — multi-day, targets ~70%+ 1-task success.
# This is the official CALVIN baseline architecture, NOT VERA.
#
# Usage:
#   CALVIN_DATA=$HOME/calvin_task_D/task_D_D TRAIN_GPU=1 \
#   nohup bash scripts/train_calvin_mcil_gripper.sh >> checkpoints/calvin_mcil_train/train.log 2>&1 &
#
set -euo pipefail

CALVIN_DATA="${CALVIN_DATA:-$HOME/calvin_task_D/task_D_D}"
CALVIN_ROOT="${CALVIN_ROOT:-$HOME/work/calvin}"
VENV="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
GPU="${TRAIN_GPU:-1}"
OUT="${MCIL_TRAIN_OUT:-$HOME/work/RLConditionedVLA/checkpoints/calvin_mcil_train/gripper_lang}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$CALVIN_ROOT/calvin_models:$CALVIN_ROOT/calvin_env:$PYTHONPATH"

mkdir -p "$OUT"
cd "$CALVIN_ROOT/calvin_models/calvin_agent"

# Shared-memory loader + gripper RGB + relative actions (CALVIN paper setup)
"$VENV" training.py \
  hydra.run.dir="$OUT" \
  datamodule.root_data_dir="$CALVIN_DATA" \
  datamodule/datasets=vision_lang_shm \
  datamodule/observation_space=lang_rgb_static_gripper_rel_act \
  model/perceptual_encoder=gripper_cam \
  trainer.gpus=1 \
  trainer.max_epochs=50 \
  ~callbacks/rollout \
  ~callbacks/rollout_lh

echo "Training finished. Eval with:"
echo "  MCIL_TRAIN_FOLDER=$OUT bash scripts/run_calvin_mcil_eval_1000.sh"
