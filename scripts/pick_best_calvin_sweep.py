#!/usr/bin/env python3
"""Pick best CALVIN rollout config from a sweep directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def score(summary: dict) -> tuple:
    chain = summary.get("chain_success_rate") or {}
    one = float(chain.get("1", 0.0))
    avg = float(summary.get("avg_chain_length", 0.0))
    # Prefer any 1-task success, then avg chain length
    return (one, avg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", required=True, help="Parent dir containing sweep_* subdirs")
    p.add_argument("--out", required=True, help="Write best config JSON here")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    best_path = None
    best_summary = None
    best_score = (-1.0, -1.0)

    for sub in sorted(sweep_dir.iterdir()):
        summary_path = sub / "vera_calvin_summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        s = score(summary)
        if s > best_score:
            best_score = s
            best_path = summary_path
            best_summary = summary

    if best_summary is None:
        raise SystemExit(f"No vera_calvin_summary.json found under {sweep_dir}")

    best = {
        "action_mode": best_summary["action_mode"],
        "action_magnitude": best_summary["action_magnitude"],
        "action_hold_steps": best_summary["action_hold_steps"],
        "lang_goal": best_summary.get("lang_goal", "task_key"),
        "reset_history_each_step": best_summary.get("reset_history_each_step", False),
        "avg_chain_length": best_summary["avg_chain_length"],
        "chain_success_rate": best_summary["chain_success_rate"],
        "source_summary": str(best_path),
    }
    Path(args.out).write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
