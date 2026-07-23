#!/usr/bin/env bash
# Fix NVIDIA 550/570 driver mismatch without full reboot (or reboot if rmmod fails).
# Run from NoMachine terminal:  bash scripts/fix_nvidia_driver.sh
set -euo pipefail

echo "=== Stop GPU jobs ==="
pkill -f run_vera_max_calvin_pipeline 2>/dev/null || true
pkill -f sft_trainer_vera 2>/dev/null || true
pkill -f "training.sft_trainer_vera import train" 2>/dev/null || true
pkill -f train_dehaze 2>/dev/null || true
sleep 3

if pgrep -f "sft_trainer_vera|train_dehaze" >/dev/null 2>&1; then
  echo "Some GPU jobs still running — force kill:"
  pgrep -af "sft_trainer_vera|train_dehaze" || true
  pkill -9 -f sft_trainer_vera 2>/dev/null || true
  pkill -9 -f train_dehaze 2>/dev/null || true
  sleep 2
fi

echo ""
echo "=== Processes still on /dev/nvidia* (need sudo) ==="
sudo fuser -v /dev/nvidia* 2>/dev/null || true

echo ""
echo "=== Reload kernel module (550 -> 570) ==="
if sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null; then
  sudo modprobe nvidia
  sudo modprobe nvidia_uvm nvidia_drm nvidia_modeset
  echo "Module reload OK."
else
  echo ""
  echo "rmmod failed (GPU/display still in use)."
  echo "Options:"
  echo "  1) Disconnect NoMachine, run this script again from SSH/console"
  echo "  2) Reboot (recommended):  sudo reboot"
  exit 1
fi

echo ""
echo "=== Verify ==="
nvidia-smi
echo ""
export VENV_PYTHON="${VENV_PYTHON:-$HOME/work/ConvIR with Lewin/venv/bin/python}"
bash "$(dirname "$0")/check_calvin_egl.sh"
echo ""
echo "Driver fix done. Resume VERA:"
echo "  bash $(dirname "$0")/resume_vera_after_driver_fix.sh"
