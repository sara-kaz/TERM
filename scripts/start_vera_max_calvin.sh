#!/usr/bin/env bash
# Start VERA max CALVIN pipeline — all phases A→Z run automatically.
#
#   cd ~/work/RLConditionedVLA
#   bash scripts/start_vera_max_calvin.sh          # background
#   bash scripts/start_vera_max_calvin.sh --fg     # live output in this terminal
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIPE="$ROOT/checkpoints/vera_max_calvin"
LOG="$PIPE/pipeline.log"
STATUS="$PIPE/pipeline_status.json"

mkdir -p "$PIPE"
chmod +x "$ROOT/scripts/run_vera_max_calvin_pipeline.sh" 2>/dev/null || true
chmod +x "$ROOT/scripts/watch_vera_max_progress.sh" 2>/dev/null || true

# Stop old run if restarting
pkill -f "run_vera_max_calvin_pipeline.sh" 2>/dev/null || true
sleep 1
rm -f "$PIPE/pipeline.lock"

if [[ "${1:-}" == "--fg" || "${1:-}" == "--foreground" ]]; then
  echo "Running in FOREGROUND (Ctrl+C stops pipeline)."
  echo "Log also saved to: $LOG"
  echo ""
  exec bash "$ROOT/scripts/run_vera_max_calvin_pipeline.sh"
fi

nohup bash "$ROOT/scripts/run_vera_max_calvin_pipeline.sh" &
PID=$!

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " VERA max pipeline started  PID=$PID"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Python: ${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
echo "Everything runs automatically in order (phases 0–10)."
echo ""
echo "Watch progress (pick one):"
echo ""
echo "  # Live log (epochs, eval %, phase banners):"
echo "  tail -f $LOG"
echo ""
echo "  # Short status summary every 30s:"
echo "  bash $ROOT/scripts/watch_vera_max_progress.sh"
echo ""
echo "  # Current phase only:"
echo "  watch -n 10 cat $STATUS"
echo ""
echo "When finished:"
echo "  cat $PIPE/final_report.json"
echo ""
