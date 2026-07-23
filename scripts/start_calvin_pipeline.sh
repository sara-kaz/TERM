#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# ONE COMMAND — starts the full unattended CALVIN pipeline:
#   stats → train → sweep → pick best → 1000-chain eval
#
# Usage (from repo root):
#   cd ~/work/RLConditionedVLA
#   bash scripts/start_calvin_pipeline.sh
#
# Optional env vars:
#   TRAIN_GPU=1          GPU for training (default 1)
#   EVAL_GPU=2           GPU for CALVIN sim eval (default 2)
#   SKIP_TRAIN_IF_CKPT=1 Skip training if best_sft_vera.pt already exists
#   SKIP_SWEEP=1         Skip sweep if you already ran it
#   SWEEP_SEQS=200       Sequences per sweep config (default 200)
#   FINAL_SEQS=1000      Final official eval sequences (default 1000)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/checkpoints/calvin_pipeline"
chmod +x "$ROOT/scripts/run_calvin_success_pipeline.sh"

# If a stale pipeline is stuck, remove lock and restart:
#   pkill -f run_calvin_success_pipeline
#   rm -f checkpoints/calvin_pipeline/pipeline.lock

nohup bash "$ROOT/scripts/run_calvin_success_pipeline.sh" \
  > "$ROOT/checkpoints/calvin_pipeline/pipeline.log" 2>&1 &

echo ""
echo "════════════════════════════════════════════════════════════"
echo " CALVIN pipeline started  PID=$!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Steps (automatic, in order):"
echo "  1. Proprio stats"
echo "  2. Sim-aligned VERA train (40 ep, gripper+proprio) — or wait if already running"
echo "  3. Rollout sweep (hybrid/regression × magnitudes, 200 seq each)"
echo "  4. Pick best inference config"
echo "  5. Official 1000-chain CALVIN eval"
echo ""
echo "Monitor:"
echo "  tail -f $ROOT/checkpoints/calvin_pipeline/pipeline.log"
echo ""
echo "Status:"
echo "  cat $ROOT/checkpoints/calvin_pipeline/pipeline_status.json"
echo ""
echo "Final results (when done):"
echo "  cat $ROOT/checkpoints/calvin_pipeline/final_report.json"
echo ""
