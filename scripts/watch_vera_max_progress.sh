#!/usr/bin/env bash
# Print pipeline phase + latest training/eval progress every 30s.
INTERVAL="${1:-30}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
PIPE="$ROOT/checkpoints/vera_max_calvin"
LOG="$PIPE/pipeline.log"
STATUS="$PIPE/pipeline_status.json"

echo "Watching VERA pipeline (every ${INTERVAL}s). Ctrl+C to stop."
echo "Full log: tail -f $LOG"
echo ""

while true; do
  clear
  echo "═══ VERA MAX CALVIN — $(date) ═══"
  echo ""
  if [[ -f "$STATUS" ]]; then
    "$PY" -c "
import json
from pathlib import Path
d = json.loads(Path('$STATUS').read_text())
print('Phase:   ', d.get('phase', '?'))
print('Detail:  ', d.get('detail', ''))
print('Updated: ', d.get('updated', ''))
"
  else
    echo "Status file not created yet..."
  fi
  echo ""
  echo "--- Last 15 pipeline log lines ---"
  tail -15 "$LOG" 2>/dev/null || echo "(no log yet)"
  echo ""
  for f in \
    "$ROOT/checkpoints/calvin_max/seed123/train.log" \
    "$ROOT/checkpoints/calvin_vera_reg_only/seed123/train.log" \
    "$ROOT/checkpoints/calvin_max_dagger_r1/seed123/train.log" \
    "$ROOT/checkpoints/calvin_max_dagger_r2/seed123/train.log"
  do
    if [[ -f "$f" ]] && { [[ ! -f "$LOG" ]] || [[ "$f" -nt "$LOG" ]]; }; then
      echo "--- Active train: $f (last epoch line) ---"
      grep -E "Epoch |best checkpoint|Early stop" "$f" 2>/dev/null | tail -3 || tail -2 "$f"
      echo ""
    fi
  done
  sleep "$INTERVAL"
done
