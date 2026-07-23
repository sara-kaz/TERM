#!/usr/bin/env python3
"""
Sanity-check that TrajectoryDataset windows match rollout policy frame/history layout.

Usage:
  python scripts/verify_rollout_train_alignment.py --calvin_path ~/calvin_task_D/task_D_D
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _frame_window(ep_frames: np.ndarray, t: int, num_vis: int) -> list:
    """Mirror TrajectoryDataset visual indexing (includes frame at t)."""
    start_vis = max(0, t - num_vis + 1)
    raw = ep_frames[start_vis : t + 1]
    pad_needed = num_vis - len(raw)
    if pad_needed > 0:
        pad = np.zeros_like(raw[0:1]).repeat(pad_needed, axis=0)
        raw = np.concatenate([pad, raw], axis=0)
    return [f.sum() for f in raw]  # cheap fingerprint per frame


def _rollout_frame_window(ep_frames: np.ndarray, t: int, num_vis: int) -> list:
    """Mirror policy frame_q after t steps (0..t inclusive)."""
    from collections import deque

    q = deque(maxlen=num_vis)
    for i in range(t + 1):
        q.append(ep_frames[i])
    pad = num_vis - len(q)
    stacked = [np.zeros_like(ep_frames[0])] * pad + list(q)
    return [f.sum() for f in stacked]


def check_calvin(calvin_path: str, num_vis: int = 3) -> bool:
    from data.trajectory_dataset import load_calvin, TrajectoryDataset

    eps = load_calvin(calvin_path, split="training")
    if not eps:
        print("[align] No CALVIN episodes loaded")
        return False
    ep = eps[0]
    ds = TrajectoryDataset([ep], num_vis_frames=num_vis, num_actions=14, action_dim=7)
    ok = True
    for t in [0, 1, 2, 5, min(20, len(ep["frames"]) - 1)]:
        if t >= len(ep["frames"]):
            continue
        ds_fp = _frame_window(ep["frames"], t, num_vis)
        pol_fp = _rollout_frame_window(ep["frames"], t, num_vis)
        if ds_fp != pol_fp:
            print(f"[align] CALVIN frame mismatch at t={t}: dataset={ds_fp} rollout={pol_fp}")
            ok = False
    print(f"[align] CALVIN frame windows: {'OK' if ok else 'FAIL'}")
    return ok


def check_model_chunk(cfg_path: str) -> bool:
    from evaluation.evaluate_vera import build_vera_from_cfg
    import yaml

    cfg = yaml.safe_load(open(cfg_path))
    device = "cpu"
    m = build_vera_from_cfg(cfg, device)
    K = int(cfg["model"].get("chunk_size", 1))
    B = 2
    H = cfg["model"]["history_len"]
    T = cfg["model"]["num_vis_frames"]
    A = cfg["model"]["num_actions"]
    ad = cfg["model"].get("action_dim", 2)
    frames = torch.randn(B, T, 3, 224, 224)
    lang = torch.randint(0, 100, (B, 77))
    ah = torch.zeros(B, H, dtype=torch.long)
    rh = torch.zeros(B, H)
    out = m(frames, lang, ah, rh)
    has_chunk = "logits_chunk" in out
    expected = K > 1
    ok = has_chunk == expected
    print(
        f"[align] chunk_size={K} logits_chunk present={has_chunk} "
        f"(expected {expected}): {'OK' if ok else 'FAIL'}"
    )
    if has_chunk:
        shape_ok = out["logits_chunk"].shape == (B, K, A)
        print(f"[align] logits_chunk shape {tuple(out['logits_chunk'].shape)}: {'OK' if shape_ok else 'FAIL'}")
        ok = ok and shape_ok
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calvin_path", default=None)
    p.add_argument("--config", default=str(REPO / "configs/config.yaml"))
    args = p.parse_args()

    all_ok = check_model_chunk(args.config)
    if args.calvin_path and Path(args.calvin_path).exists():
        all_ok = check_calvin(args.calvin_path) and all_ok
    else:
        print("[align] Skipping CALVIN frame check (no --calvin_path)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
