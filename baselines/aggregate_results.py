"""
Aggregate 3-seed baseline results into Mean ± Std for the paper table.

Usage:
    python -m baselines.aggregate_results \\
        --dirs /checkpoints/octo_lt_s42 /checkpoints/octo_lt_s123 /checkpoints/octo_lt_s456 \\
        --model Octo

    python -m baselines.aggregate_results \\
        --dirs /checkpoints/openvla_lt_s42 /checkpoints/openvla_lt_s123 /checkpoints/openvla_lt_s456 \\
        --model OpenVLA
"""

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs",  nargs="+", required=True)
    p.add_argument("--model", default="Baseline")
    args = p.parse_args()

    accs = []
    for d in args.dirs:
        rfile = Path(d) / "results.json"
        if not rfile.exists():
            print(f"[warn] {rfile} not found — skipping")
            continue
        r = json.loads(rfile.read_text())
        best = r.get("best_val_acc", max(r.get("val_acc", [0])))
        accs.append(best * 100)
        print(f"  {d}  seed={r.get('seed','?')}  best_val_acc={best*100:.2f}%")

    if not accs:
        print("No results found.")
        return

    mean = np.mean(accs)
    std  = np.std(accs, ddof=0)

    print(f"\n{'─'*50}")
    print(f"  {args.model}")
    print(f"  Seeds:    {accs}")
    print(f"  Mean ± Std: {mean:.2f} ± {std:.2f}%")
    print(f"{'─'*50}")
    print(f"\nLaTeX table row:")
    seed_strs = " & ".join(f"{a:.1f}" for a in accs)
    print(f"  {args.model} & {seed_strs} & ${mean:.1f} \\pm {std:.1f}$ \\\\")


if __name__ == "__main__":
    main()
