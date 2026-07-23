#!/usr/bin/env bash
# Continue CALVIN core-6 ablations on a dedicated GPU (default: GPU 0).
# Skips finished full_vera / bc_baseline; resumes no_exp seed42; runs no_act, no_lang, no_history_tf.
#
#   cd ~/work/RLConditionedVLA
#   bash scripts/launch_continue_calvin_core6_ablations.sh        # background
#   bash scripts/launch_continue_calvin_core6_ablations.sh --fg   # foreground
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
CALVIN="${CALVIN_DATA:-$HOME/calvin_task_D/task_D_D}"
OUT="${OUT:-$ROOT/checkpoints/calvin_core6_ltdev}"
LOG="${LOG:-$OUT/ablations_resume.log}"
GPU="${ABLATION_GPU:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$OUT"
cd "$ROOT"

run_ablations() {
  "$PY" -u "$ROOT/scripts/run_calvin_ablations.py" \
    --calvin_path "$CALVIN" \
    --config configs/calvin_config_ltdev.yaml \
    --out "$OUT" \
    --core6 \
    --start_from 2
}

echo "CALVIN ablations — GPU $GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  data:   $CALVIN"
echo "  out:    $OUT"
echo "  log:    $LOG"
echo "  range:  ablations 2–5 (no_exp, no_act, no_lang, no_history_tf)"
echo ""

if [[ "${1:-}" == "--fg" || "${1:-}" == "--foreground" ]]; then
  run_ablations 2>&1 | tee -a "$LOG"
  exit 0
fi

nohup "$PY" -u "$ROOT/scripts/run_calvin_ablations.py" \
  --calvin_path "$CALVIN" \
  --config configs/calvin_config_ltdev.yaml \
  --out "$OUT" \
  --core6 \
  --start_from 2 \
  >>"$LOG" 2>&1 &
PID=$!
echo "Started PID=$PID"
echo "  tail -f $LOG"
