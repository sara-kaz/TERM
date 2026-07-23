#!/usr/bin/env bash
# Maximum CALVIN embodied training: static + gripper + proprio, warm-start from Full VERA BC.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${CALVIN_SIM_GPU:-2}"
CALVIN="${CALVIN_DATA:-$HOME/calvin_task_D/task_D_D}"
LOG="$ROOT/checkpoints/calvin_sim_vera/seed123/train.log"

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"
mkdir -p "$(dirname "$LOG")"

python3 scripts/compute_calvin_proprio_stats.py --calvin_path "$CALVIN"

nohup python3 -u -c "
import yaml
from pathlib import Path
import sys
sys.path.insert(0, '$ROOT')
from training.sft_trainer_vera import train
cfg = yaml.safe_load(Path('configs/calvin_vera_sim.yaml').read_text())
cfg['data']['episodes_path'] = '$CALVIN'
ws = cfg.pop('warmstart', None)
train(cfg, resume_from=ws)
" >> "$LOG" 2>&1 &

echo "CALVIN sim VERA training PID=$! GPU=$GPU log=$LOG"
