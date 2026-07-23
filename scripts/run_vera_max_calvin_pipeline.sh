#!/usr/bin/env bash
# VERA-only max CALVIN pipeline — runs A→Z automatically with live progress in pipeline.log
set -euo pipefail

export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
PY="$VENV"   # never use system python3 — broken torch/NCCL on this machine
CALVIN="${CALVIN_DATA:-$HOME/calvin_task_D/task_D_D}"
CALVIN_ROOT="${CALVIN_ROOT:-$HOME/work/calvin}"
TRAIN_GPU="${TRAIN_GPU:-1}"
EVAL_GPU="${EVAL_GPU:-2}"
WARM="${WARMSTART:-$ROOT/checkpoints/calvin_sim_vera/seed123/best_sft_vera.pt}"

PIPE="$ROOT/checkpoints/vera_max_calvin"
MAX="$ROOT/checkpoints/calvin_max/seed123"
REG="$ROOT/checkpoints/calvin_vera_reg_only/seed123"
D1="$ROOT/checkpoints/calvin_max_dagger_r1/seed123"
D2="$ROOT/checkpoints/calvin_max_dagger_r2/seed123"
DAG1_PKL="$MAX/dagger_r1.pkl"
DAG2_PKL="$MAX/dagger_r2.pkl"
LOG="${PIPE}/pipeline.log"
STATUS="${PIPE}/pipeline_status.json"
LOCK="${PIPE}/pipeline.lock"
TOTAL_PHASES=10

mkdir -p "$PIPE" "$MAX" "$REG" "$D1" "$D2"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

pct() { awk -v s="${1:-0}" 'BEGIN{printf "%.2f%%", 100*s+0}'; }

banner() {
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  $*"
  echo "════════════════════════════════════════════════════════════"
  echo ""
}

write_status() {
  local phase="$1" detail="${2:-}"
  PIPELINE_PHASE="$phase" PIPELINE_DETAIL="$detail" PIPELINE_STATUS="$STATUS" \
  PIPELINE_UPDATED="$(date -Iseconds)" \
  "$PY" -c "
import json, os
from pathlib import Path
p = Path(os.environ['PIPELINE_STATUS'])
d = json.loads(p.read_text()) if p.exists() else {}
d['phase'] = os.environ['PIPELINE_PHASE']
d['detail'] = os.environ.get('PIPELINE_DETAIL', '')
d['updated'] = os.environ['PIPELINE_UPDATED']
p.write_text(json.dumps(d, indent=2))
"
}

phase_start() {
  local n="$1"
  shift
  local name="$*"
  banner "PHASE $n/$TOTAL_PHASES: $name"
  write_status "phase_${n}" "$name"
  log ">>> START phase $n/$TOTAL_PHASES: $name"
}

phase_done() {
  local n="$1"
  shift
  local name="$*"
  log ">>> DONE  phase $n/$TOTAL_PHASES: $name"
  write_status "phase_${n}_done" "$name"
}

tail_train_hint() {
  local f="$1"
  log "    (training log: $f)"
  if [[ -f "$f" ]]; then
    log "    --- last training lines ---"
    tail -3 "$f" | while read -r line; do log "    | $line"; done
    log "    ---------------------------"
  fi
}

exec 200>"$LOCK"
flock -n 200 || { log "Pipeline already running (lock $LOCK). Exit."; exit 0; }

# All stdout/stderr → pipeline.log (after helpers are defined)
exec >>"$LOG" 2>&1

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

log "Python for train+eval: $PY"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || {
  log "FATAL: $PY cannot import torch/CUDA. Set VENV_PYTHON to a working env."
  exit 1
}

setup_eval() {
  export PYTHONPATH="$CALVIN_ROOT/calvin_env:$CALVIN_ROOT/calvin_models:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export PYOPENGL_PLATFORM=egl
  unset EGL_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES="$EVAL_GPU"
}

run_train() {
  local cfg="$1" resume="$2" logf="$3" extra_py="${4:-}"
  mkdir -p "$(dirname "$logf")"
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
  cd "$ROOT"
  log "Training started → also saved to $logf"
  log "Resume from: $resume"
  # -u + pipe: every epoch line appears in pipeline.log in real time
  "$PY" -u -c "
import yaml, sys
from pathlib import Path
sys.path.insert(0, '$ROOT')
from training.sft_trainer_vera import train
cfg = yaml.safe_load(Path('$cfg').read_text())
cfg['data']['episodes_path'] = '$CALVIN'
${extra_py}
train(cfg, resume_from='$resume')
" 2>&1 | tee -a "$logf"
  log "Training finished."
  tail_train_hint "$logf"
}

run_cmd() {
  log "Running: $*"
  "$@" 2>&1
  log "Command finished (exit=$?)"
}

pick_best_embodied() {
  local ckpt_a="$1" ckpt_b="$2" ckpt_c="$3" hold="$4" out_json="$5"
  setup_eval
  local best_ckpt="" best_sr="-1"
  for C in "$ckpt_a" "$ckpt_b" "$ckpt_c"; do
    [[ -f "$C" ]] || { log "  skip missing $C"; continue; }
    EDIR="$PIPE/score_$(basename "$(dirname "$C")")_hold${hold}"
    log "  embodied eval 100 seq: $(basename "$C")"
    "$PY" "$ROOT/scripts/run_vera_embodied_score.py" \
      --checkpoint "$C" \
      --dataset_path "$CALVIN" \
      --calvin_root "$CALVIN_ROOT" \
      --num_sequences 100 \
      --action_hold_steps "$hold" \
      --device 0 \
      --eval_log_dir "$EDIR"
    SR=$("$PY" -c "import json; print(json.load(open('$EDIR/embodied_score.json'))['one_task_success'])")
    log "  → 1-task success = $(pct "$SR")"
    if "$PY" -c "exit(0 if float('$SR') > float('$best_sr') else 1)"; then
      best_sr="$SR"
      best_ckpt="$C"
    fi
  done
  [[ -n "$best_ckpt" ]] || { log "ERROR: no checkpoint to pick"; exit 1; }
  cp -f "$best_ckpt" "$PIPE/best_embodied.pt"
  "$PY" -c "
import json
json.dump({
  'checkpoint': '$PIPE/best_embodied.pt',
  'source': '$best_ckpt',
  'one_task_success': float('$best_sr'),
  'hold': int('$hold'),
}, open('$out_json','w'), indent=2)
"
  log "★ best_embodied.pt ← $best_ckpt  (1-task SR=$(pct "$best_sr"))"
}

# ─── A→Z pipeline ─────────────────────────────────────────────────────────────

banner "VERA MAX CALVIN — automatic pipeline (VERA-only)"
log "Log file: $LOG"
log "Train GPU=$TRAIN_GPU  Eval GPU=$EVAL_GPU"
write_status "starting" "pipeline boot"

phase_start 0 "Proprio stats and sanity checks"
if [[ "${SKIP_P0:-0}" == "1" ]]; then
  log "SKIP_P0=1 — skipping sanity checks"
else
  "$PY" "$ROOT/scripts/compute_calvin_proprio_stats.py" --calvin_path "$CALVIN" || true
  setup_eval
  "$PY" "$ROOT/scripts/calvin_expert_replay_sanity.py" || log "WARN: expert replay sanity failed (check env)"
  "$PY" "$ROOT/scripts/diagnose_calvin_policy.py" --checkpoint "$WARM" --calvin_path "$CALVIN" --device cuda:0 || true
fi
phase_done 0 "Proprio stats and sanity checks"

phase_start 1 "Max continuous train (~80 epochs)"
if [[ -f "$MAX/best_sft_vera.pt" && "${SKIP_P1:-0}" == "1" ]]; then
  log "SKIP_P1=1 — using existing $MAX/best_sft_vera.pt"
else
  P1_RESUME="$WARM"
  P1_EXTRA=""
  if [[ -f "$MAX/sft_vera_log.json" ]]; then
    LAST_EPOCH=$("$PY" -c "
import json
d=json.load(open('$MAX/sft_vera_log.json'))
print(d[-1]['epoch'] if d else 0)
")
    TARGET=$("$PY" -c "
import yaml
from pathlib import Path
print(yaml.safe_load(Path('$ROOT/configs/calvin_vera_max.yaml').read_text())['training']['epochs'])
")
    if [[ "$LAST_EPOCH" -gt 0 && "$LAST_EPOCH" -lt "$TARGET" ]]; then
      EPOCH_PT="$MAX/sft_vera_epoch$(printf '%04d' "$LAST_EPOCH").pt"
      if [[ -f "$EPOCH_PT" ]]; then
        P1_RESUME="$EPOCH_PT"
        P1_EXTRA="cfg['training']['finetune'] = False"
        log "Resuming phase 1 from epoch $LAST_EPOCH/$TARGET → $EPOCH_PT"
      elif [[ -f "$MAX/best_sft_vera.pt" ]]; then
        P1_RESUME="$MAX/best_sft_vera.pt"
        P1_EXTRA="cfg['training']['finetune'] = False"
        log "Resuming phase 1 from best checkpoint (epoch $LAST_EPOCH/$TARGET)"
      fi
    fi
  fi
  run_train "configs/calvin_vera_max.yaml" "$P1_RESUME" "$MAX/train.log" "$P1_EXTRA"
fi
[[ -f "$MAX/best_sft_vera.pt" ]] || { log "FATAL: missing $MAX/best_sft_vera.pt"; exit 1; }
phase_done 1 "Max continuous train"

phase_start 2 "Regression-only finetune (~20 epochs)"
if [[ -f "$REG/best_sft_vera.pt" && "${SKIP_P2:-0}" == "1" ]]; then
  log "SKIP_P2=1 — using existing $REG/best_sft_vera.pt"
else
  run_train "configs/calvin_vera_reg_only.yaml" "$MAX/best_sft_vera.pt" "$REG/train.log"
fi
phase_done 2 "Regression-only finetune"

phase_start 3 "Pick best VERA checkpoint (embodied 100-seq)"
if [[ "${SKIP_P3:-0}" == "1" && -f "$PIPE/best_embodied.pt" ]]; then
  log "SKIP_P3=1 — using existing $PIPE/best_embodied.pt"
else
  pick_best_embodied "$MAX/best_sft_vera.pt" "$REG/best_sft_vera.pt" "$WARM" 1 "$PIPE/embodied_pick_p3.json"
fi
WORK="$PIPE/best_embodied.pt"
[[ -f "$WORK" ]] || { log "FATAL: missing $WORK"; exit 1; }
phase_done 3 "Embodied checkpoint pick"

phase_start 4 "DAgger round 1 — collect data"
if [[ -f "$DAG1_PKL" && "${FORCE_DAGGER:-0}" != "1" ]]; then
  log "Using existing $DAG1_PKL"
else
  setup_eval
  "$PY" "$ROOT/scripts/collect_calvin_dagger_data.py" \
    --checkpoint "$WORK" \
    --dataset_path "$CALVIN" \
    --calvin_root "$CALVIN_ROOT" \
    --out_pkl "$DAG1_PKL" \
    --splits validation,training \
    --max_trials 1000 \
    --action_mode continuous \
    --action_hold_steps 1 \
    --device 0
fi
phase_done 4 "DAgger R1 collect"

phase_start 5 "DAgger round 1 — train (~30 epochs)"
if [[ -f "$DAG1_PKL" ]]; then
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
  cd "$ROOT"
  log "DAgger R1 training → $D1/train.log"
  "$PY" -u -c "
import yaml, sys
from pathlib import Path
sys.path.insert(0, '$ROOT')
from training.sft_trainer_vera import train
cfg = yaml.safe_load(Path('configs/calvin_dagger_max_r1.yaml').read_text())
cfg['data']['episodes_path'] = '$CALVIN'
cfg['data']['dagger_episodes_pkl'] = '$DAG1_PKL'
train(cfg, resume_from='$WORK')
" 2>&1 | tee -a "$D1/train.log"
  tail_train_hint "$D1/train.log"
else
  log "WARN: no DAgger R1 pkl — skipping train"
fi
phase_done 5 "DAgger R1 train"

WORK2="${D1}/best_sft_vera.pt"
[[ -f "$WORK2" ]] || WORK2="$WORK"

phase_start 6 "DAgger round 2 — collect data"
if [[ -f "$DAG2_PKL" && "${FORCE_DAGGER:-0}" != "1" ]]; then
  log "Using existing $DAG2_PKL"
else
  setup_eval
  "$PY" "$ROOT/scripts/collect_calvin_dagger_data.py" \
    --checkpoint "$WORK2" \
    --dataset_path "$CALVIN" \
    --calvin_root "$CALVIN_ROOT" \
    --out_pkl "$DAG2_PKL" \
    --splits validation,training \
    --max_trials 1000 \
    --action_mode continuous \
    --action_hold_steps 1 \
    --device 0
fi
phase_done 6 "DAgger R2 collect"

phase_start 7 "DAgger round 2 — train (~25 epochs)"
if [[ -f "$DAG2_PKL" ]]; then
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
  cd "$ROOT"
  log "DAgger R2 training → $D2/train.log"
  "$PY" -u -c "
import yaml, sys
from pathlib import Path
sys.path.insert(0, '$ROOT')
from training.sft_trainer_vera import train
cfg = yaml.safe_load(Path('configs/calvin_dagger_max_r2.yaml').read_text())
cfg['data']['episodes_path'] = '$CALVIN'
cfg['data']['dagger_episodes_pkl'] = '$DAG2_PKL'
train(cfg, resume_from='$WORK2')
" 2>&1 | tee -a "$D2/train.log"
  tail_train_hint "$D2/train.log"
fi
phase_done 7 "DAgger R2 train"

phase_start 8 "Final embodied checkpoint selection"
pick_best_embodied "${D2}/best_sft_vera.pt" "${D1}/best_sft_vera.pt" "${REG}/best_sft_vera.pt" 1 "$PIPE/embodied_pick_final.json"
FINAL="$PIPE/best_embodied.pt"
phase_done 8 "Final embodied pick"

phase_start 9 "Hold sweep (200 seq × holds 1,5,10)"
for HOLD in 1 5 10; do
  OUT="$PIPE/final_hold${HOLD}"
  if [[ -f "$OUT/vera_calvin_summary.json" ]]; then
    log "  skip existing hold=$HOLD"
    continue
  fi
  log "  eval hold=$HOLD (200 sequences)..."
  setup_eval
  "$PY" "$ROOT/scripts/run_calvin_rollout_eval.py" \
    --checkpoint "$FINAL" \
    --dataset_path "$CALVIN" \
    --calvin_root "$CALVIN_ROOT" \
    --num_sequences 200 \
    --device 0 \
    --action_mode continuous \
    --action_hold_steps "$HOLD" \
    --lang_goal task_key \
    --eval_log_dir "$OUT"
  SR=$("$PY" -c "import json; print(json.load(open('$OUT/vera_calvin_summary.json'))['chain_success_rate']['1'])")
  log "  hold=$HOLD → 1-task success $(pct "$SR")"
done
BEST_HOLD=$("$PY" -c "
import json
from pathlib import Path
best, bh = -1.0, 1
for d in Path('$PIPE').glob('final_hold*'):
    s = d/'vera_calvin_summary.json'
    if not s.exists(): continue
    sr = float(json.loads(s.read_text())['chain_success_rate'].get('1',0))
    if sr > best: best, bh = sr, int(d.name.replace('final_hold',''))
print(bh)
")
log "Best hold for final eval: $BEST_HOLD"
phase_done 9 "Hold sweep"

phase_start 10 "Official 1000-chain VERA eval"
FINAL_OUT="$PIPE/calvin_rollout_1000_vera"
setup_eval
log "Final eval: 1000 sequences, continuous, hold=$BEST_HOLD"
"$PY" "$ROOT/scripts/run_calvin_rollout_eval.py" \
  --checkpoint "$FINAL" \
  --dataset_path "$CALVIN" \
  --calvin_root "$CALVIN_ROOT" \
  --num_sequences 1000 \
  --device 0 \
  --action_mode continuous \
  --action_hold_steps "$BEST_HOLD" \
  --lang_goal task_key \
  --eval_log_dir "$FINAL_OUT"

"$PY" -c "
import json
from pathlib import Path
r = {
  'method': 'VERA',
  'best_embodied_checkpoint': '$FINAL',
  'best_hold': int('$BEST_HOLD'),
}
s = Path('$FINAL_OUT/vera_calvin_summary.json')
if s.exists():
    j = json.loads(s.read_text())
    r['final_1000'] = j
    sr1 = j.get('chain_success_rate', {}).get('1', 0)
    print()
    print('=' * 60)
    print('  VERA CALVIN FINAL RESULTS')
    print('=' * 60)
    print(f\"  1-task success: {100*float(sr1):.2f}%\")
    print(f\"  Avg chain length: {j.get('avg_chain_length', 0):.3f}\")
    for k in sorted(j.get('chain_success_rate', {}), key=lambda x: int(x)):
        print(f\"  {k}-chain: {100*j['chain_success_rate'][k]:.2f}%\")
    print('=' * 60)
Path('$PIPE/final_report.json').write_text(json.dumps(r, indent=2))
"

write_status "done" "pipeline complete"
banner "PIPELINE COMPLETE"
log "Report: $PIPE/final_report.json"
log "Best VERA ckpt: $PIPE/best_embodied.pt"
