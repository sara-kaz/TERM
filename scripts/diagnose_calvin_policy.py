#!/usr/bin/env python3
"""Offline diagnostics: discrete acc vs continuous MSE on CALVIN val windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.trajectory_dataset import TrajectoryDataset, load_calvin
from evaluation.evaluate_vera import build_vera_from_cfg, load_checkpoint
from evaluation.vera_calvin_policy import discretise_rel_action, postprocess_rel_action


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--calvin_path", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_batches", type=int, default=500)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    device = args.device

    episodes = load_calvin(args.calvin_path, split="validation")
    if not episodes:
        episodes = load_calvin(args.calvin_path, split="training")
    ds_kwargs = dict(
        history_len=cfg["model"]["history_len"],
        num_vis_frames=cfg["model"].get("num_vis_frames", 3),
        img_size=cfg["data"].get("img_size", 224),
        num_actions=cfg["model"]["num_actions"],
        action_dim=cfg["model"].get("action_dim", 7),
        proprio_dim=cfg["model"].get("proprio_dim", 0),
        use_gripper_cam=cfg["data"].get("use_gripper_cam", False),
    )
    stats_path = cfg["data"].get("proprio_stats_path")
    if stats_path:
        sp = Path(stats_path)
        if not sp.is_absolute():
            sp = ROOT / sp
        if sp.exists():
            from data.calvin_utils import load_proprio_stats
            mean, std = load_proprio_stats(str(sp))
            ds_kwargs["proprio_mean"] = mean
            ds_kwargs["proprio_std"] = std
    ds = TrajectoryDataset(episodes, **ds_kwargs)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    model = build_vera_from_cfg(cfg, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    ce_ok = reg_mse_sum = reg_mse_n = 0
    zero_hist_ce = 0
    n = 0

    for bi, batch in enumerate(loader):
        if bi >= args.max_batches:
            break
        frames = batch["frames"].to(device)
        lang = batch["lang_tokens"].to(device)
        ah = batch["action_hist"].to(device)
        rh = batch["reward_hist"].to(device)
        target = batch["target"].to(device)
        tvec = batch.get("target_vec")
        r_max = rh.max().clamp(min=1e-6)
        rh_norm = (rh / r_max).clamp(0.0, 1.0)
        sd = batch.get("state_delta")
        sd = sd.to(device) if isinstance(sd, torch.Tensor) else None
        avh = batch.get("action_vec_hist")
        if isinstance(avh, torch.Tensor):
            avh = avh.to(device)
        ro = batch.get("robot_obs")
        if isinstance(ro, torch.Tensor):
            ro = ro.to(device)

        with torch.no_grad():
            out = model(
                frames, lang, ah, rh_norm,
                state_delta=sd, action_vec_hist=avh, robot_obs=ro,
            )
        pred = out["logits"].argmax(-1)
        ce_ok += (pred == target).sum().item()
        n += target.numel()

        if isinstance(tvec, torch.Tensor) and out.get("action_vec") is not None:
            reg_mse_sum += F.mse_loss(out["action_vec"], tvec.to(device), reduction="sum").item()
            reg_mse_n += tvec.numel()

        # Zero history ablation
        pad_action = cfg["model"]["num_actions"]
        z_ah = torch.full_like(ah, pad_action)
        z_rh = torch.zeros_like(rh_norm)
        z_avh = torch.zeros_like(avh) if isinstance(avh, torch.Tensor) else None
        with torch.no_grad():
            out_z = model(
                frames, lang, z_ah, z_rh,
                state_delta=sd, action_vec_hist=z_avh, robot_obs=ro,
            )
        zero_hist_ce += (out_z["logits"].argmax(-1) == target).sum().item()

    print(f"Teacher-forcing discrete acc: {ce_ok / max(n, 1):.3f}  ({n} samples)")
    print(f"Zero-history discrete acc:      {zero_hist_ce / max(n, 1):.3f}")
    if reg_mse_n:
        print(f"Regression MSE (7-DoF):       {reg_mse_sum / reg_mse_n:.5f}")
        print("  (lower is better; <0.05 often needed for smooth control)")


if __name__ == "__main__":
    main()
