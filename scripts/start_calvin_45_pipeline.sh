#!/usr/bin/env bash
# Fastest path to >=45% CALVIN 1-task success:
#   1) Official pretrained MCIL eval (usually 45–70%+)
#   2) If below 45%, launch full MCIL gripper training (multi-day)
#
#   cd ~/work/RLConditionedVLA && bash scripts/start_calvin_45_pipeline.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIPE="$ROOT/checkpoints/calvin_45_pipeline"
mkdir -p "$PIPE"
chmod +x "$ROOT/scripts/"*.sh 2>/dev/null || true

LOG="$PIPE/pipeline.log"

{
  echo "======== CALVIN 45%+ pipeline ========"
  echo "[1/2] Setup pretrained MCIL"
  bash "$ROOT/scripts/setup_calvin_mcil_pretrained.sh"

  echo "[2/2] 1000-chain MCIL eval"
  bash "$ROOT/scripts/run_calvin_mcil_eval_1000.sh"

  SR="$(python3 -c "
import json
from pathlib import Path
p = Path('$ROOT/checkpoints/calvin_mcil_pretrained/eval_1000/mcil_summary.json')
if p.exists():
    print(json.loads(p.read_text()).get('one_task_success', 0))
else:
    print(0)
" 2>/dev/null || echo 0)"

  if python3 -c "exit(0 if float('$SR') >= 0.45 else 1)"; then
    echo "SUCCESS: 1-task rate = $(python3 -c "print(f'{100*float(\"$SR\"):.1f}%')")"
  else
    echo "Pretrained MCIL below 45%. Starting full MCIL gripper training in background..."
    echo "  Monitor: tail -f $ROOT/checkpoints/calvin_mcil_train/train.log"
    TRAIN_GPU="${TRAIN_GPU:-1}" nohup bash "$ROOT/scripts/train_calvin_mcil_gripper.sh" \
      >> "$ROOT/checkpoints/calvin_mcil_train/train.log" 2>&1 &
    echo "  After training, eval:"
    echo "    MCIL_TRAIN_FOLDER=$ROOT/checkpoints/calvin_mcil_train/gripper_lang \\"
    echo "    bash scripts/run_calvin_mcil_eval_1000.sh"
  fi
} 2>&1 | tee -a "$LOG"

echo "Log: $LOG"
