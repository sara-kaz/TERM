"""
Plot training curves for VERA paper (Figure 2).

Usage
-----
  python scripts/plot_results.py \
      --sft  checkpoints/vera_full/train_log.json \
      --rl   checkpoints/vera_full/sample_efficiency.csv \
              checkpoints/ablation_A/sample_efficiency.csv \
              checkpoints/ablation_F/sample_efficiency.csv \
              checkpoints/ablation_G/sample_efficiency.csv \
      --labels "VERA (full)" "Abl.A — Memoryless" "Abl.F — Act only" "Abl.G — Exp only" \
      --out  figures/training_curves.pdf

Output
------
  figures/training_curves.pdf  — two-panel figure for NEURIPS_PAPER.tex

Panel 1 (SFT phase):
  - train_loss, val_loss  vs epoch
  - cos_act, cos_exp      vs epoch  (right y-axis, [0,1])

Panel 2 (RL phase):
  - mean_return ± std     vs cumulative_steps  (4 model variants)

train_log.json format (one dict per epoch, list of dicts):
  [{"epoch": 1, "train_loss": 2.3, "val_loss": 2.1, "val_acc": 0.18,
    "align_loss": 0.05, "reg_loss": 0.12, "cos_exp": 0.12, "cos_act": 0.08}, ...]

sample_efficiency.csv format:
  cumulative_steps,mean_return,std_return,success_rate
  1000,0.12,0.04,0.05
  ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_sft_log(path: str) -> dict:
    with open(path) as f:
        records = json.load(f)
    keys = ["epoch", "train_loss", "val_loss", "val_acc",
            "align_loss", "reg_loss", "cos_exp", "cos_act"]
    out = {k: [] for k in keys}
    for rec in records:
        for k in keys:
            out[k].append(rec.get(k, float("nan")))
    return out


def load_rl_csv(path: str) -> dict:
    import csv
    steps, ret, std = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(float(row["cumulative_steps"]))
            ret.append(float(row["mean_return"]))
            std.append(float(row.get("std_return", 0)))
    return {"steps": np.array(steps), "ret": np.array(ret), "std": np.array(std)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft",    required=True,  help="Path to SFT train_log.json")
    parser.add_argument("--rl",     nargs="+",       required=True,
                        help="Paths to sample_efficiency.csv (one per model variant)")
    parser.add_argument("--labels", nargs="+",       default=None,
                        help="Legend labels for RL curves (same order as --rl)")
    parser.add_argument("--out",    default="figures/training_curves.pdf")
    args = parser.parse_args()

    labels = args.labels or [Path(p).parent.name for p in args.rl]
    assert len(labels) == len(args.rl), "Number of labels must match number of RL csv paths"

    # ── colours matching the paper diagram ──────────────────────────────────────
    COLORS = ["#006064", "#BF360C", "#1A237E", "#4A148C", "#1B5E20"]

    sft = load_sft_log(args.sft)
    rl_data = [load_rl_csv(p) for p in args.rl]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VERA — Training Dynamics", fontsize=13, fontweight="bold")

    # ── Panel 1: SFT ──────────────────────────────────────────────────────────
    epochs = sft["epoch"]
    ax1.plot(epochs, sft["train_loss"], label="Train loss", color="#1565C0", lw=2)
    ax1.plot(epochs, sft["val_loss"],   label="Val loss",   color="#1565C0", lw=2, ls="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss", color="#1565C0")
    ax1.tick_params(axis="y", labelcolor="#1565C0")

    ax1r = ax1.twinx()
    ax1r.plot(epochs, sft["cos_act"], label=r"$\cos_{\mathrm{act}}$",
              color="#BF360C", lw=2)
    ax1r.plot(epochs, sft["cos_exp"], label=r"$\cos_{\mathrm{exp}}$",
              color="#006064", lw=2)
    ax1r.set_ylabel("Cosine similarity (alignment)", color="#555555")
    ax1r.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax1.set_title("Phase 1 — SFT (Meta-World)")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: RL ───────────────────────────────────────────────────────────
    for i, (d, lbl) in enumerate(zip(rl_data, labels)):
        col = COLORS[i % len(COLORS)]
        ax2.plot(d["steps"], d["ret"], label=lbl, color=col, lw=2)
        ax2.fill_between(d["steps"],
                         d["ret"] - d["std"],
                         d["ret"] + d["std"],
                         color=col, alpha=0.15)
    ax2.set_xlabel("Cumulative environment steps")
    ax2.set_ylabel("Mean return ± std")
    ax2.set_title("Phase 2 — RL Sample Efficiency (Meta-World)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {args.out}")


if __name__ == "__main__":
    main()
