#!/usr/bin/env bash
# Run 1000-chain CALVIN eval when calvin_sim_vera best checkpoint exists.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="$ROOT/checkpoints/calvin_sim_vera/seed123/best_sft_vera.pt"
LOGDIR="$ROOT/checkpoints/calvin_sim_vera/seed123/calvin_rollout_eval_1000"
OUT="$LOGDIR/nohup.log"

mkdir -p "$LOGDIR"
echo "[watch] waiting for $CKPT ..." | tee -a "$OUT"
while [ ! -f "$CKPT" ]; do sleep 120; done
echo "[watch] starting eval" | tee -a "$OUT"
CKPT="$CKPT" LOGDIR="$LOGDIR" CUDA_VISIBLE_DEVICES=2 \
  nohup bash "$ROOT/scripts/launch_calvin_rollout_fixed.sh" >> "$OUT" 2>&1 &
echo "[watch] eval PID=$!"
