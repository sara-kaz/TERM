#!/usr/bin/env bash
# Fully automated CALVIN embodied pipeline (runs steps in order, unattended):
#   1. Proprio stats
#   2. Sim-aligned VERA train (or wait if already running)
#   3. Rollout hyperparameter sweep (200 sequences × configs)
#   4. Pick best inference config
#   5. Official 1000-chain eval with best config
#   6. Write pipeline_status.json + final_report.json
#
# Start everything with ONE command:
#   bash scripts/start_calvin_pipeline.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
CALVIN="${CALVIN_DATA:-$HOME/calvin_task_D/task_D_D}"
CALVIN_ROOT="${CALVIN_ROOT:-$HOME/work/calvin}"
WARMSTART="${WARMSTART:-$ROOT/checkpoints/calvin_core6_ltdev/full_vera/seed123/best_sft_vera.pt}"
TRAIN_GPU="${TRAIN_GPU:-1}"
EVAL_GPU="${EVAL_GPU:-2}"
SWEEP_SEQS="${SWEEP_SEQS:-200}"
FINAL_SEQS="${FINAL_SEQS:-1000}"
SKIP_SWEEP="${SKIP_SWEEP:-0}"
SKIP_TRAIN_IF_CKPT="${SKIP_TRAIN_IF_CKPT:-0}"
PIPE_DIR="$ROOT/checkpoints/calvin_pipeline"
SIM_DIR="$ROOT/checkpoints/calvin_sim_vera/seed123"
STATUS="$PIPE_DIR/pipeline_status.json"
REPORT="$PIPE_DIR/final_report.json"
LOCK="$PIPE_DIR/pipeline.lock"
CKPT="$SIM_DIR/best_sft_vera.pt"
SWEEP_DIR="$SIM_DIR/rollout_sweep"
BEST_CFG="$PIPE_DIR/best_rollout_config.json"
FINAL_OUT="$SIM_DIR/calvin_rollout_eval_1000_best"

mkdir -p "$PIPE_DIR" "$SIM_DIR" "$SWEEP_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

write_status() {
  PIPELINE_STATUS_JSON="$1" python3 -c "
import json, datetime, os
from pathlib import Path
p = Path('$STATUS')
data = json.loads(p.read_text()) if p.exists() else {}
data.update(json.loads(os.environ['PIPELINE_STATUS_JSON']))
data['updated'] = datetime.datetime.now().isoformat()
p.write_text(json.dumps(data, indent=2))
"
}

setup_eval_env() {
  export PYTHONPATH="$CALVIN_ROOT/calvin_env:$CALVIN_ROOT/calvin_models:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export PYOPENGL_PLATFORM=egl
  export EGL_VISIBLE_DEVICES="$EVAL_GPU"
  export CUDA_VISIBLE_DEVICES="$EVAL_GPU"
}

run_eval() {
  local out_dir="$1"
  local mode="$2"
  local mag="$3"
  local hold="${4:-30}"
  local num_seq="$5"
  local reset_hist="${6:-0}"
  local extra=()
  if [[ "$reset_hist" == "1" ]]; then
    extra+=(--reset_history_each_step)
  fi
  mkdir -p "$out_dir"
  cd "$ROOT"
  "$VENV" scripts/run_calvin_rollout_eval.py \
    --checkpoint "$CKPT" \
    --dataset_path "$CALVIN" \
    --calvin_root "$CALVIN_ROOT" \
    --num_sequences "$num_seq" \
    --device 0 \
    --action_mode "$mode" \
    --action_magnitude "$mag" \
    --action_hold_steps "$hold" \
    --lang_goal task_key \
    --eval_log_dir "$out_dir" \
    "${extra[@]}"
}

# Prevent duplicate pipelines
exec 200>"$LOCK"
flock -n 200 || { log "Another pipeline is running (lock $LOCK). Exit."; exit 0; }

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

log "======== CALVIN automated pipeline start ========"
write_status "{\"phase\":\"starting\",\"train_gpu\":$TRAIN_GPU,\"eval_gpu\":$EVAL_GPU,\"steps\":[\"stats\",\"train\",\"sweep\",\"pick_best\",\"final_eval\"]}"

# Stop redundant lighter runs
log "Stopping old watchers / proprio-only jobs…"
pkill -f "configs/calvin_proprio_ft.yaml" 2>/dev/null || true
pkill -f "watch_proprio_and_eval.sh" 2>/dev/null || true
pkill -f "watch_sim_vera_and_eval.sh" 2>/dev/null || true
sleep 2

# ── Step 1: proprio stats ─────────────────────────────────────────────────────
log "Step 1/5: compute proprio stats"
python3 "$ROOT/scripts/compute_calvin_proprio_stats.py" --calvin_path "$CALVIN"
write_status "{\"phase\":\"stats_done\"}"

# ── Step 2: sim-aligned training ──────────────────────────────────────────────
SIM_TRAIN_RE='calvin_vera_sim\.yaml|checkpoints/calvin_sim_vera'

sim_train_running() {
  pgrep -af "$SIM_TRAIN_RE" 2>/dev/null | grep -v "run_calvin_success_pipeline" | grep -q python || return 1
}

if sim_train_running; then
  log "Step 2/5: existing sim train detected — waiting (no duplicate launch)"
  write_status "{\"phase\":\"waiting_train\",\"out_dir\":\"$SIM_DIR\"}"
  while sim_train_running; do
    sleep 60
  done
elif [[ -f "$CKPT" && "$SKIP_TRAIN_IF_CKPT" == "1" ]]; then
  log "Step 2/5: checkpoint exists — skipping training ($CKPT)"
else
  log "Step 2/5: sim-aligned VERA training on GPU $TRAIN_GPU (40 epochs)"
  write_status "{\"phase\":\"training\",\"out_dir\":\"$SIM_DIR\"}"
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
  cd "$ROOT"
  python3 -u -c "
import yaml, sys
from pathlib import Path
sys.path.insert(0, '$ROOT')
from training.sft_trainer_vera import train
cfg = yaml.safe_load(Path('configs/calvin_vera_sim.yaml').read_text())
cfg['data']['episodes_path'] = '$CALVIN'
train(cfg, resume_from='$WARMSTART')
" 2>&1 | tee -a "$SIM_DIR/train.log"
fi

if [[ ! -f "$CKPT" ]]; then
  log "ERROR: no checkpoint at $CKPT"
  write_status "{\"phase\":\"failed\",\"reason\":\"no checkpoint\"}"
  exit 1
fi
log "Training done: $CKPT"
write_status "{\"phase\":\"train_done\",\"checkpoint\":\"$CKPT\"}"

setup_eval_env

# ── Step 3: rollout sweep ─────────────────────────────────────────────────────
if [[ "$SKIP_SWEEP" == "1" && -f "$BEST_CFG" ]]; then
  log "Step 3/5: SKIP_SWEEP=1 and best config exists — skipping sweep"
else
  log "Step 3/5: rollout sweep ($SWEEP_SEQS seq each) on GPU $EVAL_GPU"
  write_status "{\"phase\":\"sweep\",\"sweep_dir\":\"$SWEEP_DIR\",\"sequences_per_config\":$SWEEP_SEQS}"

  for MAG in 0.20 0.28 0.32 0.40; do
    for MODE in hybrid regression; do
      OUT="$SWEEP_DIR/${MODE}_mag${MAG}"
      if [[ -f "$OUT/vera_calvin_summary.json" ]]; then
        log "  skip existing $OUT"
        continue
      fi
      log "  sweep: mode=$MODE mag=$MAG"
      run_eval "$OUT" "$MODE" "$MAG" 30 "$SWEEP_SEQS" 0
    done
  done

  # Extra: reset-history variant on default hybrid config
  OUT="$SWEEP_DIR/hybrid_mag0.32_reset_hist"
  if [[ ! -f "$OUT/vera_calvin_summary.json" ]]; then
    log "  sweep: hybrid mag=0.32 reset_history"
    run_eval "$OUT" hybrid 0.32 30 "$SWEEP_SEQS" 1
  fi
fi

# ── Step 4: pick best config ──────────────────────────────────────────────────
log "Step 4/5: pick best rollout config"
python3 "$ROOT/scripts/pick_best_calvin_sweep.py" \
  --sweep_dir "$SWEEP_DIR" \
  --out "$BEST_CFG"
write_status "{\"phase\":\"best_config_chosen\",\"best_config_file\":\"$BEST_CFG\"}"

read -r BEST_MODE BEST_MAG BEST_HOLD BEST_RESET <<< "$(
python3 -c "
import json
b = json.load(open('$BEST_CFG'))
print(b['action_mode'], b['action_magnitude'], b['action_hold_steps'], int(b.get('reset_history_each_step', False)))
"
)"
log "Best config: mode=$BEST_MODE mag=$BEST_MAG hold=$BEST_HOLD reset_hist=$BEST_RESET"

# ── Step 5: official full eval ────────────────────────────────────────────────
log "Step 5/5: full $FINAL_SEQS-chain eval with best config on GPU $EVAL_GPU"
write_status "{\"phase\":\"final_eval\",\"eval_out\":\"$FINAL_OUT\",\"num_sequences\":$FINAL_SEQS}"

run_eval "$FINAL_OUT" "$BEST_MODE" "$BEST_MAG" "$BEST_HOLD" "$FINAL_SEQS" "$BEST_RESET"

SUMMARY="$FINAL_OUT/vera_calvin_summary.json"
if [[ ! -f "$SUMMARY" ]]; then
  log "ERROR: final eval missing $SUMMARY"
  write_status "{\"phase\":\"failed\",\"reason\":\"no final summary\"}"
  exit 1
fi

python3 -c "
import json
from pathlib import Path
report = {
    'checkpoint': '$CKPT',
    'best_rollout_config': json.load(open('$BEST_CFG')),
    'final_results': json.load(open('$SUMMARY')),
    'sweep_dir': '$SWEEP_DIR',
}
Path('$REPORT').write_text(json.dumps(report, indent=2))
"

log "Final results:"
cat "$SUMMARY"
log "Full report: $REPORT"
write_status "$(python3 -c "import json; print(json.dumps({'phase':'done','report':'$REPORT','results':json.load(open('$SUMMARY'))}))")"
log "======== CALVIN automated pipeline finished ========"
