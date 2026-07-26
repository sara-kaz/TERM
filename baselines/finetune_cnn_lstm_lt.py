"""
CNN-LSTM Recurrent Baseline for Language-Table (8-bin action classification)
=============================================================================
Open-loop recurrent VLA: frozen CLIP vision + bidirectional LSTM over T frames
+ frozen CLIP language embedding. No action history, no reward, no feedback.

Represents the "recurrent VLA without any closed-loop signal" point on the
ablation curve. Compare against VERA-BC (68.3%) and Full VERA (69.0%) to
isolate the contribution of (a) the LLaMA fusion architecture and (b) language
feedback streams.

Usage:
    python -m baselines.finetune_cnn_lstm_lt \\
        --data_path /data/language_table_episodes.pkl \\
        --output_dir /results \\
        --seed 42 --epochs 30
"""

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
import torchvision.transforms as Tv


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class CNNLSTMPolicy(nn.Module):
    """
    Frozen CLIP ViT-B/32 vision encoder + LSTM over T frames + CLIP language.

    Token flow:
      frames (B,T,C,H,W) → CLIP encode_image → (B,T,512)
                         → LSTM  → last hidden (B, d_hidden)
      lang_tokens (B,77) → CLIP encode_text  → (B, 512)
      concat + MLP → (B, num_actions)

    This is the canonical "recurrent VLA without closed-loop feedback" baseline.
    It shares the same CLIP backbone as VERA but replaces:
      - LLaMA bidirectional fusion → LSTM
      - Action narration token (E_act) → absent
      - Experience token (E_exp) → absent
      - Numerical history stream → absent
    """

    def __init__(
        self,
        num_actions: int = 8,
        d_hidden:    int = 256,
        num_layers:  int = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        clip_model, _ = clip.load("ViT-B/32")
        self.clip = clip_model.float()
        for p in self.clip.parameters():
            p.requires_grad = False

        clip_dim = 512

        self.lstm = nn.LSTM(
            clip_dim, d_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Language projection → same dim as LSTM output for clean fusion
        self.lang_proj = nn.Sequential(
            nn.Linear(clip_dim, d_hidden, bias=False),
            nn.LayerNorm(d_hidden),
        )

        self.fusion = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(d_hidden, num_actions)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        frames:      torch.Tensor,  # (B, T, C, H, W)  float, CLIP-normalised
        lang_tokens: torch.Tensor,  # (B, 77) int64
    ) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        with torch.no_grad():
            vis = self.clip.encode_image(frames.view(B * T, C, H, W)).float()
        vis = vis.view(B, T, -1)                          # (B, T, 512)

        lstm_out, _ = self.lstm(vis)                      # (B, T, d_hidden)
        vis_feat = lstm_out[:, -1]                        # last step

        with torch.no_grad():
            lang_raw = self.clip.encode_text(lang_tokens).float()  # (B, 512)
        lang_feat = self.lang_proj(lang_raw)              # (B, d_hidden)

        fused = self.fusion(torch.cat([vis_feat, lang_feat], dim=-1))
        return self.head(fused)                           # (B, num_actions)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE   = 224
NUM_FRAMES = 3

_TRANSFORM = Tv.Compose([
    Tv.Resize((IMG_SIZE, IMG_SIZE)),
    Tv.ToTensor(),
    Tv.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                 std=[0.26862954, 0.26130258, 0.27577711]),
])


def _preprocess(frame: np.ndarray) -> torch.Tensor:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return _TRANSFORM(Image.fromarray(frame))


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


def make_batch(episodes, refs, device):
    frames_list, lang_list, acts = [], [], []
    for ep_idx, t in refs:
        ep = episodes[ep_idx]
        idxs = [max(0, t - (NUM_FRAMES - 1 - i)) for i in range(NUM_FRAMES)]
        frame_stack = torch.stack([_preprocess(ep["frames"][i]) for i in idxs])
        frames_list.append(frame_stack)
        lang_list.append(ep["instruction"])
        acts.append(int(ep["actions"][t]))

    frames_t = torch.stack(frames_list).to(device)             # (B, T, C, H, W)
    lang_tok  = clip.tokenize(lang_list, truncate=True).to(device)
    acts_t    = torch.tensor(acts, dtype=torch.long, device=device)
    return frames_t, lang_tok, acts_t


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_ckpt(path: Path, model, optimizer, scheduler, state: dict):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    torch.save(optimizer.state_dict(), path / "opt.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), path / "sched.pt")
    (path / "state.json").write_text(json.dumps(state, indent=2))


def load_ckpt(path: Path, model, optimizer, scheduler):
    state = json.loads((path / "state.json").read_text())
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
    optimizer.load_state_dict(torch.load(path / "opt.pt", map_location="cpu"))
    if scheduler is not None and (path / "sched.pt").exists():
        scheduler.load_state_dict(torch.load(path / "sched.pt", map_location="cpu"))
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",   required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--resume_from", default=None)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--weight_decay",type=float, default=5e-4)
    p.add_argument("--val_frac",    type=float, default=0.1)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--num_actions", type=int,   default=8)
    p.add_argument("--d_hidden",    type=int,   default=256)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out     = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoint"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}", flush=True)
    if device == "cpu":
        raise RuntimeError("No GPU found. CNN-LSTM training requires a GPU.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

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

    model = CNNLSTMPolicy(
        num_actions=args.num_actions,
        d_hidden=args.d_hidden,
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Trainable params: {n_train:,}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch  = 1
    best_val_acc = 0.0
    patience_ctr = 0
    results      = {"train_acc": [], "val_acc": [], "seed": args.seed,
                    "model": "cnn_lstm"}

    resume_path = Path(args.resume_from) if args.resume_from else (
        ckpt_dir if (ckpt_dir / "state.json").exists() else None
    )
    if resume_path and (resume_path / "state.json").exists():
        print(f"[resume] Loading from {resume_path}", flush=True)
        saved = load_ckpt(resume_path, model, optimizer, scheduler)
        start_epoch  = saved["epoch"] + 1
        best_val_acc = saved["best_val_acc"]
        patience_ctr = saved["patience_ctr"]
        results      = saved.get("results", results)
        print(f"[resume] Epoch {start_epoch}  best={best_val_acc*100:.2f}%  "
              f"patience={patience_ctr}", flush=True)
    else:
        print("[train] Starting fresh.", flush=True)

    def run_val():
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for s in range(0, len(val_refs), args.batch_size):
                batch = val_refs[s: s + args.batch_size]
                fr, lt, ac = make_batch(val_ep, batch, device)
                logits = model(fr, lt)
                preds_all.extend(logits.argmax(-1).cpu().tolist())
                labels_all.extend(ac.cpu().tolist())
        return float(np.mean(np.array(preds_all) == np.array(labels_all)))

    print(f"\n[train] Epochs {start_epoch}→{args.epochs}  "
          f"lr={args.lr}  batch={args.batch_size}  d_hidden={args.d_hidden}",
          flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        random.shuffle(trn_refs)
        losses, accs = [], []

        for s in range(0, len(trn_refs), args.batch_size):
            batch = trn_refs[s: s + args.batch_size]
            if len(batch) < 2:
                continue
            fr, lt, ac = make_batch(trn_ep, batch, device)
            logits = model(fr, lt)
            loss   = F.cross_entropy(logits, ac, label_smoothing=0.05)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            losses.append(loss.item())
            accs.append((logits.argmax(-1) == ac).float().mean().item())

        scheduler.step()
        val_acc = run_val()
        trn_acc = float(np.mean(accs))
        elapsed = time.time() - t0
        results["train_acc"].append(trn_acc)
        results["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"trn_acc={trn_acc:.4f}  val_acc={val_acc:.4f}  "
              f"loss={np.mean(losses):.4f}  [{elapsed:.1f}s]", flush=True)

        save_ckpt(ckpt_dir, model, optimizer, scheduler, {
            "epoch":        epoch,
            "best_val_acc": float(best_val_acc),
            "patience_ctr": patience_ctr,
            "results":      results,
        })

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            patience_ctr = 0
            torch.save(model.state_dict(), out / "best_model.pt")
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
