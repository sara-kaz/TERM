"""
OpenVLA Fine-Tuning for CALVIN D→D (14-bin action classification)
=================================================================
Same protocol as finetune_openvla_lt.py but with CALVIN's 14-bin
arm-priority discretisation and task_D_D dataset.

14-bin vocabulary (semantically meaningful, all single LLaMA tokens):
  0 (+x arm)    → "forward"    6 (+roll)  → "tilt"    12 (grip open) → "open"
  1 (-x arm)    → "backward"   7 (-roll)  → "lean"    13 (grip close) → "close"
  2 (+y arm)    → "left"       8 (+pitch) → "pitch"
  3 (-y arm)    → "right"      9 (-pitch) → "dip"
  4 (+z arm)    → "up"         10 (+yaw)  → "turn"
  5 (-z arm)    → "down"       11 (-yaw)  → "twist"

Random baseline: 7.1% (1/14).

Usage:
    python -m baselines.finetune_openvla_calvin \\
        --data_path /data/calvin_episodes.pkl \\
        --output_dir /results \\
        --seed 42 --epochs 5
"""

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


def load_episodes(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 14-bin action vocabulary — all single LLaMA tokens
# ─────────────────────────────────────────────────────────────────────────────

ACTION_TOKENS = {
    0:  "forward",
    1:  "backward",
    2:  "left",
    3:  "right",
    4:  "up",
    5:  "down",
    6:  "tilt",
    7:  "lean",
    8:  "pitch",
    9:  "dip",
    10: "turn",
    11: "twist",
    12: "open",
    13: "close",
}
TOKEN_TO_BIN = {v: k for k, v in ACTION_TOKENS.items()}
NUM_ACTIONS  = 14
IMG_SIZE     = 224
NUM_FRAMES   = 3


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _resize(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


class CalvinWindowDataset(Dataset):
    """CALVIN windows for OpenVLA: (image, prompt) → action token."""

    def __init__(self, episodes, processor, seed: int = 0):
        self.processor = processor
        self.windows   = []
        rng = random.Random(seed)
        for ep in episodes:
            frames = ep["frames"]
            instr  = ep["instruction"]
            acts   = ep["actions"]
            T      = len(acts)
            for t in range(NUM_FRAMES - 1, T):
                self.windows.append({
                    "frame":        _resize(frames[t]),
                    "instruction":  instr,
                    "action_bin":   int(acts[t]),
                    "action_token": ACTION_TOKENS[int(acts[t])],
                })
        rng.shuffle(self.windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        prompt = (
            f"In: What action should the robot take to {w['instruction']}?\n"
            f"Out: {w['action_token']}"
        )
        inputs = self.processor(
            text=prompt,
            images=w["frame"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=64,
        )
        input_ids      = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values   = inputs["pixel_values"].squeeze(0)

        labels = torch.full_like(input_ids, fill_value=-100)
        labels[-1] = input_ids[-1]

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "pixel_values":   pixel_values,
            "labels":         labels,
            "action_bin":     torch.tensor(w["action_bin"], dtype=torch.long),
        }


def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "pixel_values":   torch.stack([b["pixel_values"]   for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "action_bin":     torch.stack([b["action_bin"]     for b in batch]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, processor, dataloader, device):
    model.eval()
    # Token ids for each action word
    action_token_ids = torch.tensor(
        [processor.tokenizer.encode(t, add_special_tokens=False)[0]
         for t in ACTION_TOKENS.values()],
        device=device,
    )
    correct, total = 0, 0
    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values   = batch["pixel_values"].to(device).to(torch.float16)
        gt_bins        = batch["action_bin"].to(device)

        inp_ids_trimmed = input_ids[:, :-1]
        attn_trimmed    = attention_mask[:, :-1]

        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(
                input_ids=inp_ids_trimmed,
                attention_mask=attn_trimmed,
                pixel_values=pixel_values,
            )

        last_logits   = out.logits[:, -1, :]
        action_logits = last_logits[:, action_token_ids]
        pred_idx      = action_logits.argmax(dim=-1)
        correct += (pred_idx == gt_bins).sum().item()
        total   += gt_bins.size(0)

    model.train()
    return correct / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(ckpt_dir: Path, model, optimizer, scheduler, state: dict):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir / "lora_adapter"))
    torch.save(optimizer.state_dict(),  ckpt_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(),  ckpt_dir / "scheduler.pt")
    (ckpt_dir / "training_state.json").write_text(json.dumps(state, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",   required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--resume_from", default=None)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--val_frac",    type=float, default=0.1)
    p.add_argument("--lora_r",      type=int,   default=16)
    p.add_argument("--lora_alpha",  type=int,   default=32)
    p.add_argument("--openvla_ckpt", default="openvla/openvla-7b")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # DDP init
    ddp        = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    is_main    = rank == 0

    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out      = Path(args.output_dir)
    ckpt_dir = out / "checkpoint"
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    if is_main:
        print(f"[data] Loading from {args.data_path}")
    all_episodes = load_episodes(args.data_path)
    random.Random(42).shuffle(all_episodes)
    n_val  = max(1, int(len(all_episodes) * args.val_frac))
    val_ep = all_episodes[:n_val]
    trn_ep = all_episodes[n_val:]
    if is_main:
        print(f"[data] {len(trn_ep)} train / {len(val_ep)} val episodes")

    # ── Model ─────────────────────────────────────────────────────────────────
    resuming    = False
    resume_path = Path(args.resume_from) if args.resume_from else None
    if resume_path and (resume_path / "training_state.json").exists():
        resuming = True

    if is_main:
        if resuming:
            print(f"[train] Resuming from {resume_path}")
        else:
            print("[train] No checkpoint found — starting fresh.")

    if is_main:
        print(f"[model] Loading {args.openvla_ckpt} …")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.openvla_ckpt,
        quantization_config=bnb_cfg,
        device_map={"": device},
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.openvla_ckpt, trust_remote_code=True)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if resuming:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, str(resume_path / "lora_adapter"), is_trainable=True)
        if is_main:
            print(f"[resume] LoRA adapter loaded from {resume_path / 'lora_adapter'}")
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)

    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"[lora] Trainable: {trainable/1e6:.1f}M / {total/1e6:.0f}M params")

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # ── Datasets ──────────────────────────────────────────────────────────────
    trn_ds = CalvinWindowDataset(trn_ep, processor, seed=args.seed)
    val_ds = CalvinWindowDataset(val_ep, processor, seed=0)

    trn_sampler = DistributedSampler(trn_ds, shuffle=True)  if ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp else None

    trn_loader = DataLoader(trn_ds, batch_size=args.batch_size,
                            sampler=trn_sampler,
                            shuffle=(trn_sampler is None),
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            sampler=val_sampler, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)
    if is_main:
        print(f"[data] {len(trn_ds)} train / {len(val_ds)} val windows")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = args.epochs * len(trn_loader)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6)

    # ── Resume state ──────────────────────────────────────────────────────────
    start_epoch  = 1
    best_val_acc = 0.0
    patience_ctr = 0
    results      = {"train_acc": [], "val_acc": [], "seed": args.seed}

    if resuming and is_main:
        saved = json.loads((resume_path / "training_state.json").read_text())
        start_epoch  = saved["epoch"] + 1
        best_val_acc = saved["best_val_acc"]
        patience_ctr = saved["patience_ctr"]
        results      = saved.get("results", results)
        optimizer.load_state_dict(
            torch.load(resume_path / "optimizer.pt", map_location=device))
        scheduler.load_state_dict(
            torch.load(resume_path / "scheduler.pt", map_location=device))
        print(f"[resume] Epoch {start_epoch}  best={best_val_acc*100:.2f}%  patience={patience_ctr}")

    if is_main:
        print(f"\n[train] Epochs {start_epoch}→{args.epochs} | bs={args.batch_size} | lr={args.lr}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        if ddp:
            trn_sampler.set_epoch(epoch)
        model.train()
        trn_loss_sum = 0.0

        for batch in trn_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch["pixel_values"].to(device).to(torch.float16)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                fwd  = model(input_ids=input_ids, attention_mask=attention_mask,
                             pixel_values=pixel_values, labels=labels)
                loss = fwd.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            trn_loss_sum += loss.item()

        val_acc = evaluate(
            model.module if ddp else model,
            processor, val_loader, device)
        elapsed = time.time() - t0

        results["train_acc"].append(0.0)  # not tracked (expensive for 7B model)
        results["val_acc"].append(float(val_acc))

        if is_main:
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"val_acc={val_acc:.4f}  loss={trn_loss_sum/len(trn_loader):.4f}  "
                  f"[{elapsed:.1f}s]")

        if is_main:
            _save_checkpoint(ckpt_dir, model.module if ddp else model,
                             optimizer, scheduler, {
                "epoch":        epoch,
                "best_val_acc": float(best_val_acc),
                "patience_ctr": patience_ctr,
                "results":      results,
            })

        if is_main:
            if val_acc > best_val_acc:
                best_val_acc = float(val_acc)
                patience_ctr = 0
            else:
                patience_ctr += 1

    if is_main:
        results["best_val_acc"] = float(best_val_acc)
        (out / "results.json").write_text(json.dumps(results, indent=2))
        print(f"\n[done] Best val acc: {best_val_acc*100:.2f}%  (random baseline: {100/NUM_ACTIONS:.1f}%)")
        print(f"[done] Results → {out}/results.json")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
