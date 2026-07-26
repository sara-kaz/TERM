"""
π₀ Backbone (PaliGemma-3B) Fine-Tuning for Language-Table (8-bin classification)
==================================================================================
Fine-tunes the VLM backbone of π₀ (PaliGemma-3B; Black et al. 2024) with LoRA
for discrete 8-bin action classification on Language-Table.

π₀ uses PaliGemma as its vision-language encoder + a flow-matching action expert.
For a direct comparison with TERM's classification accuracy metric we:
  1. Load PaliGemma-3B-pt-224 from HuggingFace (π₀'s exact backbone)
  2. Apply LoRA (r=16, α=32) to q_proj / v_proj of the language model layers
  3. Add a linear classification head on the last-token hidden state
  4. Fine-tune with the identical protocol used for all other baselines

Result: a fair comparison — π₀'s pre-trained vision-language encoder, adapted
to the same 8-bin LT task, same data, same split, same metric as TERM.

Requirements:
  HF_TOKEN env var with access granted at:
  https://huggingface.co/google/paligemma-3b-pt-224

Usage:
    python -m baselines.finetune_pi0_lt \\
        --data_path /data/language_table_episodes.pkl \\
        --output_dir /results \\
        --seed 42 --epochs 10
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as Tv

from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

HF_MODEL_ID = "google/paligemma-3b-pt-224"
IMG_SIZE    = 224
NUM_FRAMES  = 3   # use last frame only (PaliGemma is single-image)
NUM_ACTIONS = 8


# ─────────────────────────────────────────────────────────────────────────────
# Classification wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PiZeroLTPolicy(nn.Module):
    """
    PaliGemma-3B backbone + LoRA + linear classification head.

    Forward:
      pixel_values (B, 3, 224, 224)  — single frame (last of T=3)
      input_ids    (B, S)            — tokenized "direction: <image>" prompt
      attention_mask (B, S)
    Returns:
      logits (B, num_actions)
    """

    def __init__(self, base_model: nn.Module, hidden_size: int, num_actions: int = 8):
        super().__init__()
        self.base       = base_model
        self.classifier = nn.Linear(hidden_size, num_actions)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        pixel_values:   torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.base(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Last hidden state of the final generated token
        last_hidden = out.hidden_states[-1][:, -1, :].float()  # (B, hidden_size)
        return self.classifier(last_hidden)                      # (B, num_actions)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

_RESIZE = Tv.Compose([Tv.Resize((IMG_SIZE, IMG_SIZE))])


def _to_pil(frame: np.ndarray) -> Image.Image:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return _RESIZE(Image.fromarray(frame))


def load_episodes(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_refs(episodes: list, seed: int) -> list:
    rng = random.Random(seed)
    refs = []
    for ep_idx, ep in enumerate(episodes):
        T = len(ep["actions"])
        for t in range(NUM_FRAMES - 1, T):
            refs.append((ep_idx, t))
    rng.shuffle(refs)
    return refs


def make_batch(episodes, refs, processor, device):
    images, texts, acts = [], [], []
    for ep_idx, t in refs:
        ep = episodes[ep_idx]
        # Use the last (most recent) frame only — PaliGemma is single-image
        images.append(_to_pil(ep["frames"][t]))
        texts.append(f"Robot task: {ep['instruction']}. What direction? Answer:")
        acts.append(int(ep["actions"][t]))

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    acts_t = torch.tensor(acts, dtype=torch.long, device=device)
    return inputs, acts_t


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_ckpt(path: Path, model, optimizer, state: dict):
    path.mkdir(parents=True, exist_ok=True)
    model.base.save_pretrained(path / "paligemma_lora")
    torch.save(model.classifier.state_dict(), path / "classifier.pt")
    torch.save(optimizer.state_dict(), path / "opt.pt")
    (path / "state.json").write_text(json.dumps(state, indent=2))


def load_ckpt(path: Path, model, optimizer):
    from peft import PeftModel
    state = json.loads((path / "state.json").read_text())
    # LoRA weights reload via from_pretrained — classifier head loaded separately
    model.classifier.load_state_dict(
        torch.load(path / "classifier.pt", map_location="cpu")
    )
    if (path / "opt.pt").exists():
        optimizer.load_state_dict(torch.load(path / "opt.pt", map_location="cpu"))
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",         required=True)
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--resume_from",       default=None)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--epochs",            type=int,   default=10)
    p.add_argument("--batch_size",        type=int,   default=4)
    p.add_argument("--grad_accum_steps",  type=int,   default=8)  # effective bs=32
    p.add_argument("--lr",                type=float, default=2e-4)
    p.add_argument("--lora_r",            type=int,   default=16)
    p.add_argument("--lora_alpha",        type=int,   default=32)
    p.add_argument("--val_frac",          type=float, default=0.1)
    p.add_argument("--patience",          type=int,   default=5)
    p.add_argument("--num_actions",       type=int,   default=8)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out     = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoint"

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var is required to download google/paligemma-3b-pt-224.\n"
            "Accept the license at https://huggingface.co/google/paligemma-3b-pt-224\n"
            "then set HF_TOKEN=hf_... in your environment / Kubernetes secret."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}", flush=True)
    if device == "cpu":
        raise RuntimeError("No GPU found. PaliGemma-3B fine-tuning requires a GPU.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Load PaliGemma with 4-bit quantization ──────────────────────────────
    print(f"[model] Loading {HF_MODEL_ID} (4-bit) …", flush=True)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = PaliGemmaForConditionalGeneration.from_pretrained(
        HF_MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        token=hf_token,
    )
    base = prepare_model_for_kbit_training(base)

    # Apply LoRA to language model q/v projections
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    base = get_peft_model(base, lora_cfg)
    base.print_trainable_parameters()

    # Get hidden size from config (Gemma-2B hidden = 2048)
    hidden_size = base.config.text_config.hidden_size
    print(f"[model] PaliGemma hidden_size={hidden_size}", flush=True)

    processor = AutoProcessor.from_pretrained(HF_MODEL_ID, token=hf_token)

    model = PiZeroLTPolicy(base, hidden_size, args.num_actions).to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"[data] Loading {args.data_path}", flush=True)
    all_ep = load_episodes(args.data_path)
    random.Random(42).shuffle(all_ep)
    n_val  = max(1, int(len(all_ep) * args.val_frac))
    val_ep = all_ep[:n_val]
    trn_ep = all_ep[n_val:]
    print(f"[data] {len(trn_ep)} train / {len(val_ep)} val episodes", flush=True)

    trn_refs = build_refs(trn_ep, args.seed)
    val_refs = build_refs(val_ep, 0)
    print(f"[data] {len(trn_refs)} train / {len(val_refs)} val windows", flush=True)

    # ── Optimizer (only trainable params) ─────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    start_epoch  = 1
    best_val_acc = 0.0
    patience_ctr = 0
    results      = {"train_acc": [], "val_acc": [], "seed": args.seed,
                    "model": "pi0_paligemma_lora"}

    resume_path = Path(args.resume_from) if args.resume_from else (
        ckpt_dir if (ckpt_dir / "state.json").exists() else None
    )
    if resume_path and (resume_path / "state.json").exists():
        print(f"[resume] Loading from {resume_path}", flush=True)
        saved = load_ckpt(resume_path, model, optimizer)
        start_epoch  = saved["epoch"] + 1
        best_val_acc = saved["best_val_acc"]
        patience_ctr = saved["patience_ctr"]
        results      = saved.get("results", results)
        print(f"[resume] Epoch {start_epoch}  best={best_val_acc*100:.2f}%", flush=True)
    else:
        print("[train] Starting fresh.", flush=True)

    def run_val():
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for s in range(0, len(val_refs), args.batch_size):
                batch = val_refs[s: s + args.batch_size]
                inputs, acts = make_batch(val_ep, batch, processor, device)
                logits = model(**inputs)
                preds_all.extend(logits.argmax(-1).cpu().tolist())
                labels_all.extend(acts.cpu().tolist())
        return float(np.mean(np.array(preds_all) == np.array(labels_all)))

    print(f"\n[train] Epochs {start_epoch}→{args.epochs}  "
          f"lr={args.lr}  batch={args.batch_size}  "
          f"grad_accum={args.grad_accum_steps}  "
          f"effective_bs={args.batch_size * args.grad_accum_steps}",
          flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        random.shuffle(trn_refs)
        losses, accs = [], []
        optimizer.zero_grad()

        step = 0
        for s in range(0, len(trn_refs), args.batch_size):
            batch = trn_refs[s: s + args.batch_size]
            if len(batch) < 2:
                continue
            inputs, acts = make_batch(trn_ep, batch, processor, device)
            logits = model(**inputs)
            loss   = F.cross_entropy(logits, acts, label_smoothing=0.05)
            loss   = loss / args.grad_accum_steps
            loss.backward()

            losses.append(loss.item() * args.grad_accum_steps)
            accs.append((logits.argmax(-1) == acts).float().mean().item())

            step += 1
            if step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Final partial accumulation
        if step % args.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        val_acc = run_val()
        trn_acc = float(np.mean(accs))
        elapsed = time.time() - t0
        results["train_acc"].append(trn_acc)
        results["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"trn_acc={trn_acc:.4f}  val_acc={val_acc:.4f}  "
              f"loss={np.mean(losses):.4f}  [{elapsed:.1f}s]", flush=True)

        save_ckpt(ckpt_dir, model, optimizer, {
            "epoch":        epoch,
            "best_val_acc": float(best_val_acc),
            "patience_ctr": patience_ctr,
            "results":      results,
        })

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"[early stop] No improvement for {args.patience} epochs.")
                break

    results["best_val_acc"] = float(best_val_acc)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] Best val acc: {best_val_acc*100:.2f}%  "
          f"(random baseline: {100/args.num_actions:.1f}%)")
    print(f"[done] Results → {out}/results.json")


if __name__ == "__main__":
    main()
