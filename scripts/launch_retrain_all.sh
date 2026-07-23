#!/usr/bin/env bash
# Launch VERA retrain after preflight passes.
# Usage:
#   bash scripts/launch_retrain_all.sh                    # preflight + CALVIN core6 on GPU 0
#   bash scripts/launch_retrain_all.sh --lt /path/to/lt   # also LT on GPU 1
#   bash scripts/launch_retrain_all.sh --preflight-only
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${VENV_PYTHON:-$ROOT/../ConvIR with Lewin/venv/bin/python}"
CALVIN_PATH="${CALVIN_PATH:-$HOME/calvin_task_D/task_D_D}"
GPU_CALVIN="${GPU_CALVIN:-0}"
GPU_LT="${GPU_LT:-1}"
LT_DATA=""
PREFLIGHT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lt) LT_DATA="$2"; shift 2 ;;
    --calvin-path) CALVIN_PATH="$2"; shift 2 ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=== VERA preflight ==="
"$PY" "$ROOT/scripts/preflight_vera.py" --calvin_path "$CALVIN_PATH"

if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
  echo "Preflight-only mode — done."
  exit 0
fi

OUT="$ROOT/checkpoints/calvin_core6_fixed"
LOG="$OUT/launch.log"
mkdir -p "$OUT"

echo "=== CALVIN core-6 ablations (GPU $GPU_CALVIN) ==="
CUDA_VISIBLE_DEVICES="$GPU_CALVIN" nohup "$PY" -u "$ROOT/scripts/run_calvin_ablations.py" \
  --calvin_path "$CALVIN_PATH" \
  --config configs/calvin_config_ltdev.yaml \
  --out "$OUT" \
  --core6 \
  --no_skip_complete \
  >> "$LOG" 2>&1 &
echo "CALVIN ablations PID $!  log: $LOG"

if [[ -n "$LT_DATA" ]]; then
  LT_OUT="$ROOT/checkpoints/Language_table_fixed"
  mkdir -p "$LT_OUT"
  echo "=== Language-Table training (GPU $GPU_LT) ==="
  CUDA_VISIBLE_DEVICES="$GPU_LT" nohup "$PY" -u -m training.sft_trainer_vera \
    --config configs/config.yaml \
    --episodes_path "$LT_DATA" \
    --output_dir "$LT_OUT" \
    >> "$LT_OUT/train.log" 2>&1 &
  echo "LT training PID $!  log: $LT_OUT/train.log"
fi

echo "Done. Monitor: tail -f $LOG"
