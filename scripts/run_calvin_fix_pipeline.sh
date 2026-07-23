#!/usr/bin/env bash
# Fix pipeline: continuous actions + DAgger (addresses 0% embodied success).
#
#   0. Expert replay sanity + offline diagnostics
#   1. Continuous-action finetune (regression-heavy)
#   2. Embodied eval (continuous, replan every step)
#   3. DAgger data collection + finetune
#   4. Final 1000-chain eval
#
# Start:
#   cd ~/work/RLConditionedVLA && bash scripts/start_calvin_fix_pipeline.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
CALVIN="${CALVIN_DATA:-$HOME/calvin_task_D/task_D_D}"
CALVIN_ROOT="${CALVIN_ROOT:-$HOME/work/calvin}"
TRAIN_GPU="${TRAIN_GPU:-1}"
EVAL_GPU="${EVAL_GPU:-2}"
PIPE_DIR="$ROOT/checkpoints/calvin_fix_pipeline"
CONT_DIR="$ROOT/checkpoints/calvin_continuous/seed123"
DAGGER_PKL="$ROOT/checkpoints/calvin_dagger/seed123/dagger_episodes.pkl"
DAGGER_DIR="$ROOT/checkpoints/calvin_dagger_ft/seed123"
WARMSTART="${WARMSTART:-$ROOT/checkpoints/calvin_sim_vera/seed123/best_sft_vera.pt}"
LOCK="$PIPE_DIR/pipeline.lock"
LOG="$PIPE_DIR/pipeline.log"

mkdir -p "$PIPE_DIR" "$CONT_DIR" "$(dirname "$DAGGER_PKL")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

exec 200>"$LOCK"
flock -n 200 || { log "Fix pipeline already running."; exit 0; }

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

log "======== CALVIN fix pipeline (continuous + DAgger) ========"

# ── Step 0: sanity + diagnostics ──────────────────────────────────────────────
log "Step 0: expert replay sanity (env must be >>0%)"
setup_eval() {
  export PYTHONPATH="$CALVIN_ROOT/calvin_env:$CALVIN_ROOT/calvin_models:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export PYOPENGL_PLATFORM=egl
  export EGL_VISIBLE_DEVICES="$EVAL_GPU"
  export CUDA_VISIBLE_DEVICES="$EVAL_GPU"
}
setup_eval
"$VENV" "$ROOT/scripts/calvin_expert_replay_sanity.py" 2>&1 | tee -a "$LOG" || true

log "Step 0b: offline diagnostics on sim checkpoint"
python3 "$ROOT/scripts/diagnose_calvin_policy.py" \
  --checkpoint "$WARMSTART" \
  --calvin_path "$CALVIN" \
  --device "cuda:0" 2>&1 | tee -a "$LOG" || true

# ── Step 1: continuous finetune ───────────────────────────────────────────────
CKPT_CONT="$CONT_DIR/best_sft_vera.pt"
if [[ -f "$CKPT_CONT" && "${SKIP_CONT_TRAIN:-0}" == "1" ]]; then
  log "Step 1: skip continuous train (checkpoint exists)"
else
  log "Step 1: continuous-action finetune on GPU $TRAIN_GPU"
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
  cd "$ROOT"
  python3 -u -c "
import yaml, sys
from pathlib import Path
sys.path.insert(0, '$ROOT')
from training.sft_trainer_vera import train
cfg = yaml.safe_load(Path('configs/calvin_vera_continuous.yaml').read_text())
cfg['data']['episodes_path'] = '$CALVIN'
train(cfg, resume_from='$WARMSTART')
" 2>&1 | tee -a "$CONT_DIR/train.log"
fi
[[ -f "$CKPT_CONT" ]] || { log "ERROR: missing $CKPT_CONT"; exit 1; }

# ── Step 2: embodied eval (continuous) ────────────────────────────────────────
setup_eval
for HOLD in 1 5; do
  OUT="$CONT_DIR/eval_continuous_hold${HOLD}"
  if [[ -f "$OUT/vera_calvin_summary.json" && "${SKIP_EVAL:-0}" == "1" ]]; then
    log "Step 2: skip eval hold=$HOLD (exists)"
    continue
  fi
  log "Step 2: eval continuous hold=$HOLD (200 seq smoke)"
  "$VENV" "$ROOT/scripts/run_calvin_rollout_eval.py" \
    --checkpoint "$CKPT_CONT" \
    --dataset_path "$CALVIN" \
    --calvin_root "$CALVIN_ROOT" \
    --num_sequences 200 \
    --device 0 \
    --action_mode continuous \
    --action_hold_steps "$HOLD" \
    --lang_goal task_key \
    --eval_log_dir "$OUT" 2>&1 | tee -a "$LOG"
done

# ── Step 3: DAgger collection ─────────────────────────────────────────────────
if [[ ! -f "$DAGGER_PKL" || "${FORCE_DAGGER_COLLECT:-0}" == "1" ]]; then
  log "Step 3: collect DAgger episodes"
  "$VENV" "$ROOT/scripts/collect_calvin_dagger_data.py" \
    --checkpoint "$CKPT_CONT" \
    --dataset_path "$CALVIN" \
    --calvin_root "$CALVIN_ROOT" \
    --action_mode continuous \
    --action_hold_steps 1 \
    --max_trials 400 \
    --device 0 2>&1 | tee -a "$LOG"
fi

# ── Step 4: DAgger finetune ───────────────────────────────────────────────────
CKPT_DAG="$DAGGER_DIR/best_sft_vera.pt"
if [[ -f "$DAGGER_PKL" ]]; then
  if [[ -f "$CKPT_DAG" && "${SKIP_DAGGER_TRAIN:-0}" == "1" ]]; then
    log "Step 4: skip DAgger finetune"
  else
    log "Step 4: DAgger finetune on GPU $TRAIN_GPU"
    export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
    cd "$ROOT"
    python3 -u -c "
import yaml, sys
from pathlib import Path
sys.path.insert(0, '$ROOT')
from training.sft_trainer_vera import train
cfg = yaml.safe_load(Path('configs/calvin_dagger_ft.yaml').read_text())
cfg['data']['episodes_path'] = '$CALVIN'
cfg['data']['dagger_episodes_pkl'] = '$DAGGER_PKL'
train(cfg, resume_from='$CKPT_CONT')
" 2>&1 | tee -a "$DAGGER_DIR/train.log"
  fi
  FINAL_CKPT="${CKPT_DAG:-$CKPT_CONT}"
else
  log "Step 4: no DAgger pkl — using continuous checkpoint"
  FINAL_CKPT="$CKPT_CONT"
fi
[[ -f "$FINAL_CKPT" ]] || FINAL_CKPT="$CKPT_CONT"

# ── Step 5: official 1000 eval ────────────────────────────────────────────────
setup_eval
FINAL_OUT="$(dirname "$FINAL_CKPT")/calvin_rollout_eval_1000_continuous"
log "Step 5: 1000-chain eval continuous hold=1 -> $FINAL_OUT"
"$VENV" "$ROOT/scripts/run_calvin_rollout_eval.py" \
  --checkpoint "$FINAL_CKPT" \
  --dataset_path "$CALVIN" \
  --calvin_root "$CALVIN_ROOT" \
  --num_sequences 1000 \
  --device 0 \
  --action_mode continuous \
  --action_hold_steps 1 \
  --lang_goal task_key \
  --eval_log_dir "$FINAL_OUT" 2>&1 | tee -a "$LOG"

python3 -c "
import json
from pathlib import Path
report = {
  'continuous_ckpt': '$CKPT_CONT',
  'dagger_ckpt': '$FINAL_CKPT',
  'dagger_episodes': '$DAGGER_PKL',
}
s = Path('$FINAL_OUT/vera_calvin_summary.json')
if s.exists():
    report['final_results'] = json.loads(s.read_text())
Path('$PIPE_DIR/final_report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
"

log "======== CALVIN fix pipeline finished ========"
