#!/usr/bin/env bash
# Boost VERA embodied CALVIN: factored gripper + 10× DAgger data + R3 finetune + eval.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
CALVIN="${CALVIN_DATA:-$HOME/calvin_task_D/task_D_D}"
CALVIN_ROOT="${CALVIN_ROOT:-$HOME/work/calvin}"
TRAIN_GPU="${TRAIN_GPU:-1}"
EVAL_GPU="${EVAL_GPU:-2}"
PIPE="$ROOT/checkpoints/vera_boost_calvin"
LOCK="$PIPE/boost.lock"
CKPT="${CKPT:-$ROOT/checkpoints/calvin_max_dagger_r2/seed123/best_sft_vera.pt}"
DAG_PKL="$ROOT/checkpoints/calvin_max/seed123/dagger_r3.pkl"
D3="$ROOT/checkpoints/calvin_max_dagger_r3/seed123"
LOG="$PIPE/boost.log"

mkdir -p "$PIPE" "$D3"
touch "$LOG"

exec 200>"$LOCK"
flock -n 200 || { echo "Boost pipeline already running (lock $LOCK). Exit."; exit 0; }

exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

setup_eval() {
  export PYTHONPATH="$CALVIN_ROOT/calvin_env:$CALVIN_ROOT/calvin_models:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export PYOPENGL_PLATFORM=egl
  unset EGL_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES="$EVAL_GPU"
}

run_eval() {
  local tag="$1" ckpt="$2" nseq="$3" mode="$4" hold="$5" mag="${6:-0.45}"
  local edir="$PIPE/eval_${tag}"
  setup_eval
  log "Eval $tag: $nseq seq, mode=$mode hold=$hold"
  "$PY" "$ROOT/scripts/run_calvin_rollout_eval.py" \
    --checkpoint "$ckpt" \
    --dataset_path "$CALVIN" \
    --calvin_root "$CALVIN_ROOT" \
    --num_sequences "$nseq" \
    --action_mode "$mode" \
    --action_magnitude "$mag" \
    --action_hold_steps "$hold" \
    --device 0 \
    --eval_log_dir "$edir"
  "$PY" -c "
import json
s=json.load(open('$edir/vera_calvin_summary.json'))
sr=s.get('chain_success_rate',{})
print('  1-task:', sr.get('1',0), ' avg_chain:', s.get('avg_chain_length',0))
"
}

log "=== VERA BOOST === checkpoint=$CKPT"

# 0) Optional baseline eval (skip with SKIP_PREEVAL=1)
if [[ "${SKIP_PREEVAL:-0}" != "1" ]]; then
  run_eval "r2_hybrid_pre" "$CKPT" 100 hybrid 1 0.45 || true
  run_eval "r2_continuous_pre" "$CKPT" 100 continuous 1 "1.0" || true
fi

# 1) DAgger R3 — ~500 eps/split, all lang segments (not 1-per-task)
setup_eval
log "DAgger R3 collect → $DAG_PKL"
"$PY" "$ROOT/scripts/collect_calvin_dagger_data.py" \
  --checkpoint "$CKPT" \
  --dataset_path "$CALVIN" \
  --calvin_root "$CALVIN_ROOT" \
  --out_pkl "$DAG_PKL" \
  --splits training,validation \
  --max_trials 500 \
  --action_mode hybrid \
  --action_magnitude 0.45 \
  --action_hold_steps 1 \
  --device 0

# 2) Train R3
export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
cd "$ROOT"
log "DAgger R3 train → $D3/train.log"
"$PY" -u -c "
import yaml, sys
from pathlib import Path
sys.path.insert(0, '$ROOT')
from training.sft_trainer_vera import train
cfg = yaml.safe_load(Path('configs/calvin_dagger_boost_r3.yaml').read_text())
cfg['data']['episodes_path'] = '$CALVIN'
cfg['data']['dagger_episodes_pkl'] = '$DAG_PKL'
train(cfg, resume_from='$CKPT')
" 2>&1 | tee -a "$D3/train.log"

BEST="$D3/best_sft_vera.pt"
[[ -f "$BEST" ]] || BEST="$CKPT"

# 3) Eval boosted checkpoint
run_eval "r3_hybrid" "$BEST" 200 hybrid 1 0.45
run_eval "r3_continuous" "$BEST" 200 continuous 1 1.0

# 4) Official 1000-chain on best mode (continuous + gripper fix)
setup_eval
FINAL="$PIPE/final_boost_report.json"
log "Official 1000-chain eval (continuous hold=1 mag=1.0)"
"$PY" "$ROOT/scripts/run_calvin_rollout_eval.py" \
  --checkpoint "$BEST" \
  --dataset_path "$CALVIN" \
  --calvin_root "$CALVIN_ROOT" \
  --num_sequences 1000 \
  --action_mode continuous \
  --action_magnitude 1.0 \
  --action_hold_steps 1 \
  --device 0 \
  --eval_log_dir "$PIPE/final_1000"
"$PY" -c "
import json
from pathlib import Path
s=json.load(open('$PIPE/final_1000/vera_calvin_summary.json'))
report={
  'method':'VERA_boost_r3',
  'checkpoint': '$BEST',
  'action_mode':'continuous',
  'action_magnitude':1.0,
  'action_hold_steps':1,
  'final_1000':s,
}
Path('$FINAL').write_text(json.dumps(report, indent=2))
sr=s.get('chain_success_rate',{}).get('1',0)
print(f'BOOST 1-task success: {100*sr:.2f}%')
"
log "Done → $FINAL"
