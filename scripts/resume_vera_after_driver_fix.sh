#!/usr/bin/env bash
# Resume VERA max pipeline after driver fix / reboot.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export VENV_PYTHON="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
export TRAIN_GPU="${TRAIN_GPU:-1}"
export EVAL_GPU="${EVAL_GPU:-2}"
export CALVIN_DATA="${CALVIN_DATA:-$HOME/calvin_task_D/task_D_D}"
export CALVIN_ROOT="${CALVIN_ROOT:-$HOME/work/calvin}"

MAX="$ROOT/checkpoints/calvin_max/seed123"
TARGET_EPOCHS=$("$VENV_PYTHON" -c "
import yaml
from pathlib import Path
print(yaml.safe_load(Path('$ROOT/configs/calvin_vera_max.yaml').read_text())['training']['epochs'])
")

LAST_EPOCH=0
if [[ -f "$MAX/sft_vera_log.json" ]]; then
  LAST_EPOCH=$("$VENV_PYTHON" -c "
import json
d=json.load(open('$MAX/sft_vera_log.json'))
print(d[-1]['epoch'] if d else 0)
")
fi

REG="$ROOT/checkpoints/calvin_vera_reg_only/seed123"

if [[ "$LAST_EPOCH" -ge "$TARGET_EPOCHS" && -f "$MAX/best_sft_vera.pt" ]]; then
  export SKIP_P1=1
  echo "Phase 1 complete ($LAST_EPOCH/$TARGET_EPOCHS epochs) — SKIP_P1=1"
else
  unset SKIP_P1
  echo "Phase 1 incomplete ($LAST_EPOCH/$TARGET_EPOCHS) — will resume training"
fi

if [[ -f "$REG/best_sft_vera.pt" ]]; then
  export SKIP_P2=1
  echo "Phase 2 reg-only checkpoint found — SKIP_P2=1"
else
  unset SKIP_P2
fi

echo "CALVIN_DATA=$CALVIN_DATA"
echo "Quick sim checks..."
nvidia-smi 2>&1 | head -5 || true
if nvidia-smi 2>&1 | grep -qi "Driver/library version mismatch"; then
  echo "ERROR: NVIDIA driver/library mismatch — run: bash $ROOT/scripts/fix_nvidia_driver.sh"
  exit 1
fi

export EVAL_GPU="$EVAL_GPU"
bash "$ROOT/scripts/check_calvin_egl.sh"
"$VENV_PYTHON" "$ROOT/scripts/calvin_expert_replay_sanity.py" || {
  echo "WARN: expert replay sanity failed — EGL may still be broken"
}

PIPE="$ROOT/checkpoints/vera_max_calvin"
rm -f "$PIPE/pipeline.lock"
rm -rf "$PIPE"/score_* "$PIPE/best_embodied.pt" "$PIPE/embodied_pick_p3.json"
echo "Cleared stale 0% embodied picks — will re-run phase 3+ with fixed gripper actions"
cd "$ROOT"
bash scripts/start_vera_max_calvin.sh
