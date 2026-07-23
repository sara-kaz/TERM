#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/checkpoints/calvin_fix_pipeline"
chmod +x "$ROOT/scripts/run_calvin_fix_pipeline.sh"

nohup bash "$ROOT/scripts/run_calvin_fix_pipeline.sh" \
  > "$ROOT/checkpoints/calvin_fix_pipeline/pipeline.log" 2>&1 &

echo "Fix pipeline started PID=$!"
echo "Monitor: tail -f $ROOT/checkpoints/calvin_fix_pipeline/pipeline.log"
echo "Report:  cat $ROOT/checkpoints/calvin_fix_pipeline/final_report.json"
