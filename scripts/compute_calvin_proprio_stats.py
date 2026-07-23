#!/usr/bin/env python3
"""Compute and save CALVIN robot_obs mean/std for z-score normalization."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.calvin_utils import compute_robot_obs_stats, save_proprio_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calvin_path", default=str(Path.home() / "calvin_task_D/task_D_D"))
    p.add_argument("--out", default="data/calvin_proprio_stats.json")
    p.add_argument("--proprio_dim", type=int, default=15)
    args = p.parse_args()
    mean, std = compute_robot_obs_stats(args.calvin_path, "training", proprio_dim=args.proprio_dim)
    out = Path(__file__).resolve().parents[1] / args.out
    save_proprio_stats(str(out), mean, std)
    print(f"Saved {out}  mean[:3]={mean[:3]}  std[:3]={std[:3]}")


if __name__ == "__main__":
    main()
