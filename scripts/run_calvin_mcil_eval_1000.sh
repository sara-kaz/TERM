#!/usr/bin/env bash
# Official CALVIN MCIL 1000-chain eval — expect ~45–76% 1-task success (not VERA).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
CALVIN_DATA="${CALVIN_DATA:-$HOME/calvin_task_D/task_D_D}"
CALVIN_ROOT="${CALVIN_ROOT:-$HOME/work/calvin}"
TRAIN_FOLDER="${MCIL_TRAIN_FOLDER:-$ROOT/checkpoints/calvin_mcil_pretrained/D_D_static_rgb_baseline}"
EVAL_GPU="${EVAL_GPU:-2}"
OUT="${MCIL_EVAL_OUT:-$ROOT/checkpoints/calvin_mcil_pretrained/eval_1000}"
NUM_SEQ="${NUM_SEQUENCES:-1000}"

export PYTHONPATH="$CALVIN_ROOT/calvin_models:$CALVIN_ROOT/calvin_env:$PYTHONPATH"
export PYOPENGL_PLATFORM=egl
export EGL_VISIBLE_DEVICES="$EVAL_GPU"
export CUDA_VISIBLE_DEVICES="$EVAL_GPU"

mkdir -p "$OUT"

CKPT="${MCIL_CHECKPOINT:-}"
if [[ -z "$CKPT" ]]; then
  CKPT="$(find "$TRAIN_FOLDER" -name '*.ckpt' 2>/dev/null | head -1)"
fi
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "ERROR: no .ckpt in $TRAIN_FOLDER — run: bash scripts/setup_calvin_mcil_pretrained.sh"
  exit 1
fi

echo "MCIL eval: checkpoint=$CKPT"
echo "           dataset=$CALVIN_DATA"
echo "           sequences=$NUM_SEQ  out=$OUT"

cd "$CALVIN_ROOT/calvin_models/calvin_agent"
"$VENV" evaluation/evaluate_policy.py \
  --dataset_path "$CALVIN_DATA" \
  --train_folder "$TRAIN_FOLDER" \
  --checkpoint "$CKPT" \
  --eval_log_dir "$OUT" \
  2>&1 | tee "$OUT/eval.log"

# Parse 1-task success from results.json
python3 -c "
import json
from pathlib import Path
rpath = Path('$OUT/results.json')
if not rpath.exists():
    print('WARN: no results.json')
    raise SystemExit(0)
data = json.loads(rpath.read_text())
epoch = list(data.keys())[-1]
d = data[epoch]
sr1 = d['chain_sr'].get('1', d['chain_sr'].get(1, 0))
print()
print('=== MCIL CALVIN results ===')
print(f\"  1-task success: {100*float(sr1):.1f}%\")
print(f\"  Avg chain length: {d['avg_seq_len']:.3f}\")
for k, v in sorted(d['chain_sr'].items(), key=lambda x: int(x[0])):
    print(f\"  {k}-chain: {100*v:.1f}%\")
summary = {
    'method': 'MCIL_pretrained_D_D_static_rgb',
    'train_folder': '$TRAIN_FOLDER',
    'checkpoint': '$CKPT',
    'num_sequences': int('$NUM_SEQ'),
    'one_task_success': float(sr1),
    'avg_chain_length': float(d['avg_seq_len']),
    'chain_success_rate': d['chain_sr'],
}
Path('$OUT/mcil_summary.json').write_text(json.dumps(summary, indent=2))
print(f\"Saved: $OUT/mcil_summary.json\")
if float(sr1) >= 0.45:
    print('TARGET MET: >=45% 1-task success')
else:
    print('Below 45% — train full MCIL with gripper (see scripts/train_calvin_mcil_gripper.sh)')
"
