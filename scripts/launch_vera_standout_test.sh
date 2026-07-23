#!/usr/bin/env bash
# VERA stand-out test — history ON vs OFF vs BC vs no-history-TF (CALVIN rollouts).
#
#   cd ~/work/RLConditionedVLA
#   bash scripts/launch_vera_standout_test.sh           # 200 seq, GPU 2
#   bash scripts/launch_vera_standout_test.sh smoke     # 10 seq quick check
#   bash scripts/launch_vera_standout_test.sh --fg      # foreground + live log
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
CALVIN="${CALVIN_DATA:-$HOME/calvin_task_D/task_D_D}"
CALVIN_ROOT="${CALVIN_ROOT:-$HOME/work/calvin}"
GPU="${EVAL_GPU:-2}"
OUT="$ROOT/checkpoints/vera_standout_test"
LOG="$OUT/standout.log"

mkdir -p "$OUT"
EXTRA=()
[[ "${1:-}" == "smoke" ]] && EXTRA+=(--smoke) && shift
[[ "${1:-}" == "--smoke" ]] && EXTRA+=(--smoke) && shift

run() {
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/scripts/run_vera_standout_test.py" \
    --dataset_path "$CALVIN" \
    --calvin_root "$CALVIN_ROOT" \
    --device 0 \
    --out "$OUT" \
    "${EXTRA[@]}" \
    "$@"
}

echo "VERA stand-out test — GPU $GPU"
echo "  out: $OUT"
echo "  log: $LOG"
echo ""
echo "Conditions:"
echo "  1. Full VERA + history ON   (same checkpoint as below)"
echo "  2. Full VERA + history OFF  (eval-time ablation — proves Stream 4)"
echo "  3. BC/SFT baseline"
echo "  4. No history TF (trained)"
echo ""

if [[ "${1:-}" == "--fg" || "${1:-}" == "--foreground" ]]; then
  shift || true
  run 2>&1 | tee -a "$LOG"
else
  nohup env CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/scripts/run_vera_standout_test.py" \
    --dataset_path "$CALVIN" \
    --calvin_root "$CALVIN_ROOT" \
    --device 0 \
    --out "$OUT" \
    "${EXTRA[@]}" \
    "$@" >>"$LOG" 2>&1 &
  echo "Started PID=$!"
  echo "  tail -f $LOG"
fi
