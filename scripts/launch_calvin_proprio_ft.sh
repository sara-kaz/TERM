#!/usr/bin/env bash
# Finetune Full VERA + CALVIN proprio on GPU 1; warm-start from seed-123 peak BC checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
CALVIN="${CALVIN_DATA:-$HOME/calvin_task_D/task_D_D}"
PRETRAIN="${PRETRAIN:-$ROOT/checkpoints/calvin_core6_ltdev/full_vera/seed123/best_sft_vera.pt}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
PYTHON="${PYTHON:-python3}"
LOG="${ROOT}/checkpoints/calvin_proprio_ft/seed123/train.log"

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"
mkdir -p "$(dirname "$LOG")"

# Use system python3 (ablations env); venv may break torch/typing_extensions.
nohup env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PYTHON" -u -c "
import yaml
from pathlib import Path
from training.sft_trainer_vera import train

cfg = yaml.safe_load(Path('configs/calvin_proprio_ft.yaml').read_text())
cfg['data']['episodes_path'] = '${CALVIN}'
cfg['training']['seed'] = 123
train(cfg, resume_from='${PRETRAIN}')
" >> "$LOG" 2>&1 &

echo "Proprio finetune started PID=$!  GPU=$GPU  log=$LOG"
