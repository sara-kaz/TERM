#!/usr/bin/env python3
"""Quick CALVIN embodied score for VERA checkpoint selection (returns JSON)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--calvin_root", default=os.environ.get("CALVIN_ROOT", str(Path.home() / "work/calvin")))
    p.add_argument("--num_sequences", type=int, default=100)
    p.add_argument("--action_mode", default="continuous")
    p.add_argument("--action_hold_steps", type=int, default=1)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--eval_log_dir", required=True)
    args = p.parse_args()

    venv = os.environ.get(
        "VENV_PYTHON",
        str(Path.home() / "work/ConvIR with Lewin/venv/bin/python"),
    )
    script = ROOT / "scripts/run_calvin_rollout_eval.py"
    env = os.environ.copy()
    env["PYOPENGL_PLATFORM"] = "egl"
    env.pop("EGL_VISIBLE_DEVICES", None)
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)

    cmd = [
        venv,
        str(script),
        "--checkpoint", args.checkpoint,
        "--dataset_path", args.dataset_path,
        "--calvin_root", args.calvin_root,
        "--num_sequences", str(args.num_sequences),
        "--device", "0",
        "--action_mode", args.action_mode,
        "--action_hold_steps", str(args.action_hold_steps),
        "--lang_goal", "task_key",
        "--eval_log_dir", args.eval_log_dir,
    ]
    subprocess.run(cmd, check=True, env=env)

    summary_path = Path(args.eval_log_dir) / "vera_calvin_summary.json"
    summary = json.loads(summary_path.read_text())
    sr1 = float(summary.get("chain_success_rate", {}).get("1", 0.0))
    out = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "one_task_success": sr1,
        "avg_chain_length": float(summary.get("avg_chain_length", 0.0)),
        "chain_success_rate": summary.get("chain_success_rate", {}),
        "eval_log_dir": str(summary_path.parent),
    }
    score_path = Path(args.eval_log_dir) / "embodied_score.json"
    score_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out))
    return 0 if sr1 >= 0 else 0


if __name__ == "__main__":
    sys.exit(main())
