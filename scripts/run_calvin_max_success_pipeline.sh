#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# MAXIMUM CALVIN task-success pipeline (unattended, multi-day).
#
# What makes this "top tier" within VERA:
#   • Full 7-DoF continuous actions (not discrete single-axis)
#   • 80-epoch regression-focused training + closed-loop history dropout
#   • Best checkpoint by val regression MSE (not classification acc)
#   • 2 rounds of DAgger (val + train splits, 800 eps each)
#   • Embodied eval sweep + official 1000-chain final eval
#
# Estimated wall time: ~4–7 days on 1× A800 (train) + 1× GPU (eval)
#
# Start:
#   cd ~/work/RLConditionedVLA && bash scripts/start_calvin_max_pipeline.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
CALVIN="${CALVIN_DATA:-$HOME/calvin_task_D/task_D_D}"
CALVIN_ROOT="${CALVIN_ROOT:-$HOME/work/calvin}"
TRAIN_GPU="${TRAIN_GPU:-1}"
EVAL_GPU="${EVAL_GPU:-2}"
PIPE="$ROOT/checkpoints/calvin_max_pipeline"
LOCK="$PIPE/pipeline.lock"
LOG="$PIPE/pipeline.log"

MAX_DIR="$ROOT/checkpoints/calvin_max/seed123"
DAG1_DIR="$ROOT/checkpoints/calvin_max_dagger_r1/seed123"
DAG2_DIR="$ROOT/checkpoints/calvin_max_dagger_r2/seed123"
DAG1_PKL="$MAX_DIR/dagger_r1.pkl"
DAG2_PKL="$MAX_DIR/dagger_r2.pkl"
WARM="${WARMSTART:-$ROOT/checkpoints/calvin_sim_vera/seed123/best_sft_vera.pt}"

mkdir -p "$PIPE" "$MAX_DIR" "$DAG1_DIR" "$DAG2_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

write_status() {
  PIPELINE_STATUS_JSON="$1" python3 -c "
import json, datetime, os
from pathlib import Path
p = Path('$PIPE/pipeline_status.json')
d = json.loads(p.read_text()) if p.exists() else {}
d.update(json.loads(os.environ['PIPELINE_STATUS_JSON']))
d['updated'] = datetime.datetime.now().isoformat()
p.write_text(json.dumps(d, indent=2))
"
}

exec 200>"$LOCK"
flock -n 200 || { log "Max pipeline already running."; exit 0; }

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

setup_eval() {
  export PYTHONPATH="$CALVIN_ROOT/calvin_env:$CALVIN_ROOT/calvin_models:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export PYOPENGL_PLATFORM=egl
  export EGL_VISIBLE_DEVICES="$EVAL_GPU"
  export CUDA_VISIBLE_DEVICES="$EVAL_GPU"
}

run_train() {
  local cfg="$1" resume="$2" logfile="$3"
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
  cd "$ROOT"
  python3 -u -c "
import yaml, sys
from pathlib import Path
sys.path.insert(0, '$ROOT')
from training.sft_trainer_vera import train
cfg = yaml.safe_load(Path('$cfg').read_text())
cfg['data']['episodes_path'] = '$CALVIN'
train(cfg, resume_from='$resume')
" 2>&1 | tee -a "$logfile"
}

run_eval() {
  local ckpt="$1" out="$2" hold="$3" nseq="$4"
  setup_eval
  mkdir -p "$out"
  "$VENV" "$ROOT/scripts/run_calvin_rollout_eval.py" \
    --checkpoint "$ckpt" \
    --dataset_path "$CALVIN" \
    --calvin_root "$CALVIN_ROOT" \
    --num_sequences "$nseq" \
    --device 0 \
    --action_mode continuous \
    --action_hold_steps "$hold" \
    --lang_goal task_key \
    --eval_log_dir "$out" 2>&1 | tee -a "$LOG"
}

pick_best_hold() {
  python3 -c "
import json
from pathlib import Path
best_sr, best_hold, best_path = -1.0, 1, None
base = Path('$MAX_DIR')
for d in base.glob('eval_hold*'):
    s = d / 'vera_calvin_summary.json'
    if not s.exists():
        continue
    j = json.loads(s.read_text())
    sr = float(j.get('chain_success_rate', {}).get('1', 0.0))
    if sr > best_sr:
        best_sr, best_hold, best_path = sr, int(d.name.replace('eval_hold','')), s
print(best_hold)
" 2>/dev/null || echo "1"
}

log "======== CALVIN MAX SUCCESS pipeline ========"
write_status '{"phase":"started"}'

# ── 0. Sanity ─────────────────────────────────────────────────────────────────
log "Phase 0: expert replay + offline diagnostics"
setup_eval
"$VENV" "$ROOT/scripts/calvin_expert_replay_sanity.py" 2>&1 | tee -a "$LOG" || true
python3 "$ROOT/scripts/diagnose_calvin_policy.py" \
  --checkpoint "$WARM" --calvin_path "$CALVIN" --device cuda:0 2>&1 | tee -a "$LOG" || true
python3 "$ROOT/scripts/compute_calvin_proprio_stats.py" --calvin_path "$CALVIN" 2>&1 | tee -a "$LOG"

# ── 1. Max continuous pretrain (80 ep) ────────────────────────────────────────
CKPT_MAX="$MAX_DIR/best_sft_vera.pt"
if [[ -f "$CKPT_MAX" && "${SKIP_MAX_TRAIN:-0}" == "1" ]]; then
  log "Phase 1: skip (checkpoint exists)"
else
  log "Phase 1: max continuous pretrain (~80 epochs, GPU $TRAIN_GPU)"
  write_status '{"phase":"max_train"}'
  run_train "configs/calvin_vera_max.yaml" "$WARM" "$MAX_DIR/train.log"
fi
[[ -f "$CKPT_MAX" ]] || { log "FATAL: $CKPT_MAX missing"; exit 1; }

# ── 2. Embodied smoke sweep (pick hold) ───────────────────────────────────────
log "Phase 2: embodied eval sweep (200 seq, continuous)"
write_status '{"phase":"eval_sweep"}'
for HOLD in 1 5 10 30; do
  OUT="$MAX_DIR/eval_hold${HOLD}"
  [[ -f "$OUT/vera_calvin_summary.json" && "${SKIP_EVAL_SWEEP:-0}" == "1" ]] && continue
  log "  eval hold=$HOLD"
  run_eval "$CKPT_MAX" "$OUT" "$HOLD" 200
done
BEST_HOLD="$(pick_best_hold)"
log "Best hold from sweep: $BEST_HOLD"

# ── 3. DAgger round 1 ─────────────────────────────────────────────────────────
if [[ ! -f "$DAG1_PKL" || "${FORCE_DAGGER:-0}" == "1" ]]; then
  log "Phase 3a: DAgger R1 data collection"
  write_status '{"phase":"dagger_r1_collect"}'
  setup_eval
  "$VENV" "$ROOT/scripts/collect_calvin_dagger_data.py" \
    --checkpoint "$CKPT_MAX" \
    --dataset_path "$CALVIN" \
    --calvin_root "$CALVIN_ROOT" \
    --out_pkl "$DAG1_PKL" \
    --splits validation,training \
    --max_trials 800 \
    --action_mode continuous \
    --action_hold_steps "$BEST_HOLD" \
    --device 0 2>&1 | tee -a "$LOG"
fi

CKPT_D1="$DAG1_DIR/best_sft_vera.pt"
if [[ -f "$DAG1_PKL" ]]; then
  if [[ -f "$CKPT_D1" && "${SKIP_DAGGER1_TRAIN:-0}" == "1" ]]; then
    log "Phase 3b: skip DAgger R1 train"
  else
    log "Phase 3b: DAgger R1 finetune (~30 epochs)"
    write_status '{"phase":"dagger_r1_train"}'
    export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
    cd "$ROOT"
    python3 -u -c "
import yaml, sys
from pathlib import Path
sys.path.insert(0, '$ROOT')
from training.sft_trainer_vera import train
cfg = yaml.safe_load(Path('configs/calvin_dagger_max_r1.yaml').read_text())
cfg['data']['episodes_path'] = '$CALVIN'
cfg['data']['dagger_episodes_pkl'] = '$DAG1_PKL'
train(cfg, resume_from='$CKPT_MAX')
" 2>&1 | tee -a "$DAG1_DIR/train.log"
  fi
fi

# ── 4. DAgger round 2 ─────────────────────────────────────────────────────────
CKPT_FOR_R2="${CKPT_D1:-$CKPT_MAX}"
if [[ -f "$CKPT_FOR_R2" && (! -f "$DAG2_PKL" || "${FORCE_DAGGER:-0}" == "1") ]]; then
  log "Phase 4a: DAgger R2 data collection"
  write_status '{"phase":"dagger_r2_collect"}'
  setup_eval
  "$VENV" "$ROOT/scripts/collect_calvin_dagger_data.py" \
    --checkpoint "$CKPT_FOR_R2" \
    --dataset_path "$CALVIN" \
    --calvin_root "$CALVIN_ROOT" \
    --out_pkl "$DAG2_PKL" \
    --splits validation,training \
    --max_trials 800 \
    --action_mode continuous \
    --action_hold_steps "$BEST_HOLD" \
    --device 0 2>&1 | tee -a "$LOG"
fi

CKPT_D2="$DAG2_DIR/best_sft_vera.pt"
if [[ -f "$DAG2_PKL" ]]; then
  if [[ -f "$CKPT_D2" && "${SKIP_DAGGER2_TRAIN:-0}" == "1" ]]; then
    log "Phase 4b: skip DAgger R2 train"
  else
    log "Phase 4b: DAgger R2 finetune (~25 epochs)"
    write_status '{"phase":"dagger_r2_train"}'
    export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
    cd "$ROOT"
    python3 -u -c "
import yaml, sys
from pathlib import Path
sys.path.insert(0, '$ROOT')
from training.sft_trainer_vera import train
cfg = yaml.safe_load(Path('configs/calvin_dagger_max_r2.yaml').read_text())
cfg['data']['episodes_path'] = '$CALVIN'
cfg['data']['dagger_episodes_pkl'] = '$DAG2_PKL'
train(cfg, resume_from='$CKPT_FOR_R2')
" 2>&1 | tee -a "$DAG2_DIR/train.log"
  fi
fi

FINAL_CKPT="$CKPT_D2"
[[ -f "$FINAL_CKPT" ]] || FINAL_CKPT="$CKPT_FOR_R2"
[[ -f "$FINAL_CKPT" ]] || FINAL_CKPT="$CKPT_MAX"

# ── 5. Official 1000-chain eval ───────────────────────────────────────────────
log "Phase 5: final 1000-chain eval (continuous, hold=$BEST_HOLD)"
write_status "{\"phase\":\"final_eval\",\"checkpoint\":\"$FINAL_CKPT\"}"
FINAL_OUT="$(dirname "$FINAL_CKPT")/calvin_rollout_1000_max"
run_eval "$FINAL_CKPT" "$FINAL_OUT" "$BEST_HOLD" 1000

python3 -c "
import json
from pathlib import Path
r = {
  'final_checkpoint': '$FINAL_CKPT',
  'best_hold': int('$BEST_HOLD'),
  'max_ckpt': '$CKPT_MAX',
  'dagger_r1_ckpt': '$CKPT_D1',
  'dagger_r2_ckpt': '$CKPT_D2',
}
s = Path('$FINAL_OUT/vera_calvin_summary.json')
if s.exists():
    r['final_results'] = json.loads(s.read_text())
Path('$PIPE/final_report.json').write_text(json.dumps(r, indent=2))
print(json.dumps(r, indent=2))
"

log "======== MAX SUCCESS pipeline finished ========"
write_status '{"phase":"done"}'
