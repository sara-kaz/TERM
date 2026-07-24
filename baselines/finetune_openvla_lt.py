"""
OpenVLA Fine-Tuning for Language-Table (8-bin action classification)
====================================================================
Fine-tunes openvla-7b (LLaMA-2 + SigLIP backbone) using LoRA so it outputs
one of 8 directional action tokens.

The 8-bin vocabulary is mapped to single-word output tokens:
  0→"right"  1→"upright"  2→"up"  3→"upleft"
  4→"left"   5→"downleft" 6→"down" 7→"downright"

Prediction: the model outputs one of these tokens; we map back to bin index
for accuracy computation (same 8-bin metric used by TERM).

LoRA config: r=16, alpha=32, applied to q/v projections of the LLM backbone.
Trained with single A100 (40 GB) or 2×A40 using DDP if WORLD_SIZE>1.

Usage:
    # single GPU
    python -m baselines.finetune_openvla_lt \\
        --data_path /data/language_table_episodes.pkl \\
        --output_dir /checkpoints/openvla_lt_s42 \\
        --seed 42

    # multi-GPU (2× A100)
    torchrun --nproc_per_node=2 -m baselines.finetune_openvla_lt \\
        --data_path /data/language_table_episodes.pkl \\
        --output_dir /checkpoints/openvla_lt_s42 \\
        --seed 42
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.trajectory_dataset import load_episodes


# ─────────────────────────────────────────────────────────────────────────────
# Action vocabulary: bin index → output token string
# ─────────────────────────────────────────────────────────────────────────────

ACTION_TOKENS = {
    0: "right",
    1: "upright",
    2: "up",
    3: "upleft",
    4: "left",
    5: "downleft",
    6: "down",
    7: "downright",
}
TOKEN_TO_BIN = {v: k for k, v in ACTION_TOKENS.items()}
NUM_ACTIONS = 8
IMG_SIZE    = 224
NUM_FRAMES  = 3


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _resize(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


class LTWindowDataset(Dataset):
    """Language-Table windows for OpenVLA: (image, prompt) → action token."""

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
                    "frame": _resize(frames[t]),          # last (most recent) frame
                    "instruction": instr,
                    "action_bin":  int(acts[t]),
                    "action_token": ACTION_TOKENS[int(acts[t])],
                })
        rng.shuffle(self.windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        # OpenVLA prompt format (matches openvla/openvla-7b training format)
        prompt = (
            f"In: What action should the robot take to {w['instruction']}?\n"
            f"Out: {w['action_token']}"
        )
        # Encode image + text with the OpenVLA processor
        inputs = self.processor(
            text=prompt,
            images=w["frame"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=64,
        )
        # For causal LM: shift so the model predicts the action token
        input_ids      = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values   = inputs["pixel_values"].squeeze(0)

        # Labels: -100 everywhere except the last (action) token
        labels = torch.full_like(input_ids, fill_value=-100)
        labels[-1] = input_ids[-1]          # supervise only the action token

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
# Evaluation: greedy decode → map token → bin accuracy
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, processor, dataloader, device, rank=0):
    model.eval()
    correct, total = 0, 0
    for batch in dataloader:
        # Build prompt without the answer token for generation
        # We use teacher-forced logits at the last position for speed
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values   = batch["pixel_values"].to(device)
        gt_bins        = batch["action_bin"].to(device)

        # Trim the last token (the answer) and let the model predict it
        inp_ids_trimmed = input_ids[:, :-1]
        attn_trimmed    = attention_mask[:, :-1]

        out = model(
            input_ids=inp_ids_trimmed,
            attention_mask=attn_trimmed,
            pixel_values=pixel_values,
        )
        # logits over vocabulary at the last position
        last_logits = out.logits[:, -1, :]        # (B, vocab_size)

        # Get token ids for each action word and pick the highest-logit one
        action_token_ids = torch.tensor(
            [processor.tokenizer.encode(t, add_special_tokens=False)[0]
             for t in ACTION_TOKENS.values()],
            device=device,
        )
        # Restrict to action token logits
        action_logits = last_logits[:, action_token_ids]     # (B, 8)
        pred_action_idx = action_logits.argmax(dim=-1)       # (B,) index into ACTION_TOKENS

        correct += (pred_action_idx == gt_bins).sum().item()
        total   += gt_bins.size(0)

    model.train()
    return correct / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",   required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--epochs",      type=int,   default=80)
    p.add_argument("--batch_size",  type=int,   default=8,   help="Per-GPU batch size")
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--val_frac",    type=float, default=0.1)
    p.add_argument("--patience",    type=int,   default=25)
    p.add_argument("--lora_r",      type=int,   default=16)
    p.add_argument("--lora_alpha",  type=int,   default=32)
    p.add_argument("--model_id",    default="openvla/openvla-7b")
    p.add_argument("--local_rank",  type=int,   default=-1)
    return p.parse_args()


def main():
    args = parse_args()

    # ── DDP setup ─────────────────────────────────────────────────────────────
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank   = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank   = 0

    is_main = (rank == 0)
    out = Path(args.output_dir)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── Load data ─────────────────────────────────────────────────────────────
    if is_main:
        print(f"[data] Loading from {args.data_path}")
    all_episodes = load_episodes(args.data_path)
    random.Random(42).shuffle(all_episodes)
    n_val  = max(1, int(len(all_episodes) * args.val_frac))
    val_ep = all_episodes[:n_val]
    trn_ep = all_episodes[n_val:]
    if is_main:
        print(f"[data] {len(trn_ep)} train / {len(val_ep)} val episodes")

    # ── Load processor + model ────────────────────────────────────────────────
    if is_main:
        print(f"[model] Loading {args.model_id} …")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    # 4-bit QLoRA: backbone stored in NF4, compute in bfloat16.
    # GPU footprint: ~4 GB vs ~15 GB for bfloat16 full precision.
    # This lets the job run on any NRP GPU (V100 16 GB, A40, A100, etc.)
    # without requesting a specific node type.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": device},
        trust_remote_code=True,
    )

    # Prepares the frozen 4-bit backbone for gradient flow through LoRA adapters:
    # casts LayerNorm to float32 and enables gradient checkpointing.
    model = prepare_model_for_kbit_training(
        model,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],   # LLaMA-2 attention projections
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    # device_map placed model on GPU; prepare_model_for_kbit_training enables
    # grad checkpointing — no .to(device) and no GradScaler needed.
    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"[lora] Trainable: {trainable/1e6:.1f}M / {total/1e6:.0f}M params")

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # ── Datasets + loaders ────────────────────────────────────────────────────
    trn_ds = LTWindowDataset(trn_ep, processor, seed=args.seed)
    val_ds = LTWindowDataset(val_ep, processor, seed=0)

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

    # ── Optimiser (LoRA params only, with cosine LR) ─────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = args.epochs * len(trn_loader)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    patience_ctr = 0
    results = {"train_acc": [], "val_acc": [], "seed": args.seed}

    if is_main:
        print(f"\n[train] {args.epochs} epochs | bs={args.batch_size} | lr={args.lr}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if ddp:
            trn_sampler.set_epoch(epoch)
        model.train()
        trn_correct, trn_total, trn_loss_sum = 0, 0, 0.0

        for batch in trn_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch["pixel_values"].to(device).to(torch.bfloat16)
            labels         = batch["labels"].to(device)
            gt_bins        = batch["action_bin"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out  = model(input_ids=input_ids, attention_mask=attention_mask,
                             pixel_values=pixel_values, labels=labels)
                loss = out.loss

            # bfloat16 has float32 dynamic range — no GradScaler needed.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            trn_loss_sum += loss.item()

        # Validate on main process only
        val_acc = evaluate(
            model.module if ddp else model,
            processor, val_loader, device, rank)
        trn_acc = trn_correct / trn_total if trn_total > 0 else 0.0
        elapsed = time.time() - t0

        results["train_acc"].append(float(trn_acc))
        results["val_acc"].append(float(val_acc))

        if is_main:
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"val_acc={val_acc:.4f}  loss={trn_loss_sum/len(trn_loader):.4f}  "
                  f"[{elapsed:.1f}s]")

        # ── Checkpoint + early stopping ────────────────────────────────────────
        if is_main and val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            save_model = model.module if ddp else model
            save_model.save_pretrained(out / "best_lora")
            processor.save_pretrained(out / "best_lora")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                if is_main:
                    print(f"[early stop] No improvement for {args.patience} epochs.")
                break

    if is_main:
        results["best_val_acc"] = float(best_val_acc)
        (out / "results.json").write_text(json.dumps(results, indent=2))
        print(f"\n[done] Seed {args.seed} | Best val acc: {best_val_acc*100:.2f}%")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
