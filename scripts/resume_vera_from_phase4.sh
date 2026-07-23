#!/usr/bin/env bash
# Resume VERA max pipeline from DAgger R1 (phase 4) after prior phases completed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export VENV_PYTHON="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
export TRAIN_GPU="${TRAIN_GPU:-1}"
export EVAL_GPU="${EVAL_GPU:-2}"
export CALVIN_DATA="${CALVIN_DATA:-$HOME/calvin_task_D/task_D_D}"
export CALVIN_ROOT="${CALVIN_ROOT:-$HOME/work/calvin}"

export SKIP_P0=1 SKIP_P1=1 SKIP_P2=1 SKIP_P3=1
# Re-collect DAgger (validation split was partial; no pkl was saved)
export FORCE_DAGGER=1

PIPE="$ROOT/checkpoints/vera_max_calvin"
WORK="$PIPE/best_embodied.pt"
DAG1="$ROOT/checkpoints/calvin_max/seed123/dagger_r1.pkl"

if [[ ! -f "$WORK" ]]; then
  echo "ERROR: missing $WORK — run phase 3 pick first or unset SKIP_P3"
  exit 1
fi

echo "Resuming from phase 4 (DAgger R1 collect)"
echo "  checkpoint: $WORK"
echo "  train GPU:  $TRAIN_GPU   eval GPU: $EVAL_GPU"
rm -f "$PIPE/pipeline.lock"
rm -f "$DAG1"

cd "$ROOT"
bash scripts/start_vera_max_calvin.sh
