#!/usr/bin/env bash
# Start the maximum-quality CALVIN pipeline (multi-day, unattended).
#
#   cd ~/work/RLConditionedVLA
#   bash scripts/start_calvin_max_pipeline.sh
#
# GPUs (default): train on GPU 1, sim eval on GPU 2
#   TRAIN_GPU=0 EVAL_GPU=1 bash scripts/start_calvin_max_pipeline.sh
#
# Resume after interruption (skip completed stages):
#   SKIP_MAX_TRAIN=1 bash scripts/start_calvin_max_pipeline.sh
#   SKIP_MAX_TRAIN=1 SKIP_DAGGER1_TRAIN=1 bash scripts/start_calvin_max_pipeline.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/checkpoints/calvin_max_pipeline"
chmod +x "$ROOT/scripts/run_calvin_max_success_pipeline.sh"

nohup bash "$ROOT/scripts/run_calvin_max_success_pipeline.sh" \
  > "$ROOT/checkpoints/calvin_max_pipeline/pipeline.log" 2>&1 &

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " CALVIN MAX SUCCESS pipeline started  PID=$!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Phases (automatic):"
echo "  0. Expert replay sanity + diagnostics"
echo "  1. Max continuous train (~80 epochs, regression-focused)"
echo "  2. Eval sweep (hold 1/5/10/30) → pick best"
echo "  3. DAgger round 1: collect 1600 eps + train 30 epochs"
echo "  4. DAgger round 2: collect 1600 eps + train 25 epochs"
echo "  5. Official 1000-chain CALVIN eval"
echo ""
echo "Est. time: 4–7 days total"
echo ""
echo "Monitor:"
echo "  tail -f $ROOT/checkpoints/calvin_max_pipeline/pipeline.log"
echo "  cat $ROOT/checkpoints/calvin_max_pipeline/pipeline_status.json"
echo ""
echo "Final report:"
echo "  cat $ROOT/checkpoints/calvin_max_pipeline/final_report.json"
echo ""
