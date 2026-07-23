#!/usr/bin/env bash
# Quick CALVIN EGL probe — run before embodied eval phases.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CALVIN_ROOT="${CALVIN_ROOT:-$HOME/work/calvin}"
PY="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
GPU="${EVAL_GPU:-0}"

export PYOPENGL_PLATFORM=egl
unset EGL_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$GPU"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Probing EGL (CALVIN egl_check)…"
if [[ -x "$CALVIN_ROOT/calvin_env/egl_check/EGL_options.o" ]]; then
  (cd "$CALVIN_ROOT/calvin_env/egl_check" && ./EGL_options.o 2>&1 | head -5) || true
fi

"$PY" -c "
import os, sys
sys.path.insert(0, '$ROOT')
sys.path.insert(0, '$CALVIN_ROOT/calvin_env')
from data.calvin_utils import setup_calvin_egl
egl = setup_calvin_egl(0, '$CALVIN_ROOT')
print('Mapped cuda:0 -> EGL_VISIBLE_DEVICES=' + str(egl))
import pybullet as p
cid = p.connect(p.DIRECT)
print('PyBullet DIRECT ok, cid=', cid)
p.disconnect()
print('EGL OK for CALVIN sim')
" && echo "PASS" || {
  echo ""
  echo "FAIL: EGL/PyBullet headless init failed."
  echo "Common fix: NVIDIA driver was updated — reboot the machine, then re-run."
  echo "  nvidia-smi   # should not show 'Driver/library version mismatch'"
  exit 1
}
