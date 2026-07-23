"""
Supervised Fine-Tuning (Behavioral Cloning) Trainer
====================================================
Phase 1 of training: learn to imitate expert demonstrations.
Loss: cross-entropy over discrete action logits.

Usage
-----
  python -m training.sft_trainer --config configs/config.yaml
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.vla_model import RLConditionedVLA
from data.trajectory_dataset import TrajectoryDataset, load_episodes, make_random_episodes


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict, device: str):
    # Load or generate episodes
    if cfg["data"].get("episodes_path") and Path(cfg["data"]["episodes_path"]).exists():
        episodes = load_episodes(cfg["data"]["episodes_path"])
        print(f"[data] Loaded {len(episodes)} episodes from {cfg['data']['episodes_path']}")
    else:
        print("[data] No dataset found — generating random synthetic episodes for testing.")
        episodes = make_random_episodes(
            num_episodes=cfg["data"].get("synthetic_episodes", 100),
            ep_len=cfg["data"].get("ep_len", 30),
            num_actions=cfg["model"]["num_actions"],
        )

    full_ds = TrajectoryDataset(
        episodes,
        history_len=cfg["model"]["history_len"],
        num_vis_frames=cfg["model"]["num_vis_frames"],
        num_actions=cfg["model"]["num_actions"],
        img_size=cfg["data"].get("img_size", 224),
        device=device,
    )

    val_frac = cfg["training"].get("val_fraction", 0.1)
    n_val    = max(1, int(len(full_ds) * val_frac))
    n_train  = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 2),
        pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 2),
        pin_memory=(device != "cpu"),
    )
    return train_loader, val_loader


def build_model(cfg: dict) -> RLConditionedVLA:
    model = RLConditionedVLA(
        num_actions=cfg["model"]["num_actions"],
        history_len=cfg["model"]["history_len"],
        fusion_layers=cfg["model"].get("fusion_layers", 6),
        fusion_heads=cfg["model"].get("fusion_heads", 8),
        dropout=cfg["model"].get("dropout", 0.1),
        freeze_clip=cfg["model"].get("freeze_clip", True),
    )
    return model


# ── Training / validation loops ───────────────────────────────────────────────

def run_epoch(
    model: RLConditionedVLA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    is_train: bool,
    grad_clip: float = 1.0,
) -> dict:
    model.train() if is_train else model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            frames      = batch["frames"].to(device)       # (B, T, 3, H, W)
            lang_tokens = batch["lang_tokens"].to(device)  # (B, 77)
            action_hist = batch["action_hist"].to(device)  # (B, H)
            reward_hist = batch["reward_hist"].to(device)  # (B, H)
            target      = batch["target"].to(device)       # (B,)

            logits = model(frames, lang_tokens, action_hist, reward_hist)  # (B, A)
            loss   = criterion(logits, target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            B = target.size(0)
            total_loss    += loss.item() * B
            total_correct += (logits.argmax(dim=-1) == target).sum().item()
            total_samples += B

    return {
        "loss":     total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def train(cfg: dict):
    device = cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[trainer] device = {device}")

    train_loader, val_loader = build_dataloaders(cfg, device)
    model = build_model(cfg).to(device)
    print(f"[model] Trainable params: {model.num_trainable_params():,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"].get("lr_min", 1e-6),
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg["training"].get("label_smoothing", 0.05)
    )

    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "sft_log.json"
    log = []

    best_val_acc = 0.0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        train_metrics = run_epoch(
            model, train_loader, optimizer, criterion, device, is_train=True,
            grad_clip=cfg["training"].get("grad_clip", 1.0),
        )
        val_metrics = run_epoch(
            model, val_loader, optimizer, criterion, device, is_train=False,
        )
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            "epoch":    epoch,
            "train_loss": round(train_metrics["loss"],     4),
            "train_acc":  round(train_metrics["accuracy"], 4),
            "val_loss":   round(val_metrics["loss"],       4),
            "val_acc":    round(val_metrics["accuracy"],   4),
            "lr":         round(scheduler.get_last_lr()[0], 8),
            "time_s":     round(elapsed, 1),
        }
        log.append(row)
        print(f"Epoch {epoch:3d} | "
              f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} | "
              f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.3f} | "
              f"lr {row['lr']:.2e} | {elapsed:.1f}s")

        # Save best checkpoint
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            ckpt_path = out_dir / "best_sft.pt"
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_acc":    best_val_acc,
                "cfg":        cfg,
            }, ckpt_path)
            print(f"  -> saved best checkpoint (val_acc={best_val_acc:.3f})")

        # Periodic checkpoint
        if epoch % cfg["training"].get("save_every", 10) == 0:
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "cfg":         cfg,
            }, out_dir / f"sft_epoch{epoch:04d}.pt")

    # Save log
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n[trainer] Training complete. Best val acc: {best_val_acc:.3f}")
    print(f"[trainer] Logs saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
