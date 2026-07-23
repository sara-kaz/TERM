#!/usr/bin/env python3
"""
VERA stand-out test — closed-loop CALVIN comparison on the same eval protocol.

Runs rollout eval for conditions designed to show *where VERA differs from BC*:

  1. full_vera_history_on   — Full VERA checkpoint, temporal history active (default)
  2. full_vera_history_off  — Same checkpoint, --reset_history_each_step (ablates Stream 4 at eval)
  3. bc_baseline            — BC/SFT checkpoint (no lang streams, no temporal TF)
  4. no_history_tf          — Model trained without temporal history transformer

If VERA is effective, expect:
  - (1) > (2)  — history helps closed-loop control on the *same* weights
  - (1) >= (3) — full VERA beats or matches BC in sim (not just offline val acc)

Usage
-----
  cd ~/work/RLConditionedVLA
  CUDA_VISIBLE_DEVICES=2 python scripts/run_vera_standout_test.py \\
      --dataset_path ~/calvin_task_D/task_D_D \\
      --num_sequences 200

Smoke (10 seq, ~5 min):
  python scripts/run_vera_standout_test.py --num_sequences 10 --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "checkpoints" / "vera_standout_test"
PY = os.environ.get(
    "VENV_PYTHON",
    str(Path.home() / "work" / "ConvIR with Lewin" / "venv" / "bin" / "python"),
)


def _ckpt(*parts: str) -> str:
    return str(ROOT / "checkpoints" / "calvin_core6_ltdev" / Path(*parts))


CONDITIONS = [
    {
        "id": "full_vera_history_on",
        "label": "Full VERA (history ON)",
        "checkpoint": _ckpt("full_vera", "seed123", "best_sft_vera.pt"),
        "reset_history": False,
    },
    {
        "id": "full_vera_history_off",
        "label": "Full VERA (history OFF, same ckpt)",
        "checkpoint": _ckpt("full_vera", "seed123", "best_sft_vera.pt"),
        "reset_history": True,
    },
    {
        "id": "bc_baseline",
        "label": "BC/SFT baseline",
        "checkpoint": _ckpt("bc_baseline", "seed123", "best_sft_vera.pt"),
        "reset_history": False,
    },
    {
        "id": "no_history_tf",
        "label": "No history TF (trained ablation)",
        "checkpoint": _ckpt("no_history_tf", "seed123", "best_sft_vera.pt"),
        "reset_history": False,
    },
]


def _default_eval_device() -> int:
    """When CUDA_VISIBLE_DEVICES is set, only cuda:0 exists — use 0."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        return 0
    return int(os.environ.get("EVAL_GPU", "2"))


def run_one(
    cond: dict,
    *,
    dataset_path: str,
    calvin_root: str,
    device: int,
    num_sequences: int,
    action_mode: str,
    action_magnitude: float,
    action_hold_steps: int,
    out_root: Path,
) -> dict:
    ckpt = Path(cond["checkpoint"])
    if not ckpt.is_file():
        return {
            "id": cond["id"],
            "label": cond["label"],
            "checkpoint": str(ckpt),
            "error": "checkpoint missing",
        }

    edir = out_root / cond["id"]
    edir.mkdir(parents=True, exist_ok=True)
    summary_path = edir / "vera_calvin_summary.json"

    if summary_path.is_file() and not os.environ.get("FORCE_STANDOUT_RERUN"):
        print(f"[skip] {cond['id']} — exists {summary_path}")
        return json.loads(summary_path.read_text())

    cmd = [
        PY,
        str(ROOT / "scripts" / "run_calvin_rollout_eval.py"),
        "--checkpoint",
        str(ckpt),
        "--dataset_path",
        dataset_path,
        "--calvin_root",
        calvin_root,
        "--device",
        str(device),
        "--num_sequences",
        str(num_sequences),
        "--action_mode",
        action_mode,
        "--action_magnitude",
        str(action_magnitude),
        "--action_hold_steps",
        str(action_hold_steps),
        "--lang_goal",
        "task_key",
        "--eval_log_dir",
        str(edir),
    ]
    if cond["reset_history"]:
        cmd.append("--reset_history_each_step")

    print(f"\n{'=' * 60}")
    print(f"  {cond['label']}")
    print(f"  ckpt: {ckpt.name}  history_off={cond['reset_history']}")
    print(f"{'=' * 60}\n", flush=True)

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=os.environ.copy())
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {
            "id": cond["id"],
            "label": cond["label"],
            "checkpoint": str(ckpt),
            "error": f"eval exited {proc.returncode}",
            "elapsed_sec": elapsed,
        }
    if not summary_path.is_file():
        return {
            "id": cond["id"],
            "label": cond["label"],
            "checkpoint": str(ckpt),
            "error": "summary json missing after eval",
            "elapsed_sec": elapsed,
        }
    s = json.loads(summary_path.read_text())
    s["id"] = cond["id"]
    s["label"] = cond["label"]
    return s


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("  VERA STAND-OUT TEST — CALVIN chain success")
    print("=" * 72)
    print(f"  {'Condition':<42} {'1-task':>8} {'avg chain':>10} {'2-task':>8}")
    print(f"  {'-' * 42} {'-' * 8} {'-' * 10} {'-' * 8}")
    for r in results:
        if r.get("error"):
            print(f"  {r.get('label', r.get('id', '?')):<42}  ERROR: {r['error']}")
            continue
        sr = r.get("chain_success_rate", {})
        s1 = 100 * float(sr.get("1", 0))
        s2 = 100 * float(sr.get("2", 0))
        avg = float(r.get("avg_chain_length", 0))
        print(f"  {r['label']:<42} {s1:7.2f}% {avg:10.3f} {s2:7.2f}%")

    # Deltas vs full_vera_history_on
    base = next((r for r in results if r.get("id") == "full_vera_history_on" and not r.get("error")), None)
    if base:
        b1 = float(base.get("chain_success_rate", {}).get("1", 0))
        print(f"\n  Δ 1-task success vs Full VERA (history ON):")
        for r in results:
            if r.get("error") or r.get("id") == "full_vera_history_on":
                continue
            s1 = float(r.get("chain_success_rate", {}).get("1", 0))
            print(f"    {r['label']:<40} {100 * (s1 - b1):+.2f} pp")

    print("\n  Interpretation:")
    print("    • History ON > History OFF  → temporal memory helps closed-loop control")
    print("    • Full VERA > BC            → language+history design beats plain SFT")
    print("    • Full VERA > No-history TF → trained history stream matters in sim")
    print("=" * 72 + "\n")


def main():
    p = argparse.ArgumentParser(description="VERA stand-out CALVIN comparison test")
    p.add_argument("--dataset_path", default=os.path.expanduser("~/calvin_task_D/task_D_D"))
    p.add_argument("--calvin_root", default=os.path.expanduser("~/work/calvin"))
    p.add_argument("--device", type=int, default=_default_eval_device())
    p.add_argument("--num_sequences", type=int, default=200)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument(
        "--action_mode",
        default="continuous",
        choices=("continuous", "hybrid", "discrete", "regression"),
    )
    p.add_argument("--action_magnitude", type=float, default=0.45)
    p.add_argument("--action_hold_steps", type=int, default=1)
    p.add_argument(
        "--only",
        nargs="*",
        choices=[c["id"] for c in CONDITIONS],
        help="Run subset of conditions (default: all)",
    )
    p.add_argument("--smoke", action="store_true", help="10 sequences, fast sanity check")
    args = p.parse_args()

    if args.smoke:
        args.num_sequences = 10

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    conds = CONDITIONS
    if args.only:
        conds = [c for c in CONDITIONS if c["id"] in args.only]

    results = []
    for c in conds:
        results.append(
            run_one(
                c,
                dataset_path=args.dataset_path,
                calvin_root=args.calvin_root,
                device=args.device,
                num_sequences=args.num_sequences,
                action_mode=args.action_mode,
                action_magnitude=args.action_magnitude,
                action_hold_steps=args.action_hold_steps,
                out_root=out_root,
            )
        )

    report = {
        "num_sequences": args.num_sequences,
        "action_mode": args.action_mode,
        "action_magnitude": args.action_magnitude,
        "action_hold_steps": args.action_hold_steps,
        "results": results,
    }
    report_path = out_root / "standout_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print_table(results)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
