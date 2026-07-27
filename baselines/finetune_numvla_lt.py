"""
NumericalVLA Baseline for Language-Table (8-bin action classification)
=======================================================================
Same CLIP+LLaMA fusion architecture as VERA but replaces the two language
feedback streams (E_act, E_exp) with a single raw numerical token that
encodes [prev_action_idx / num_actions, prev_reward] as a linear projection.

Purpose: isolates the question "does language verbalization of feedback help
over raw numerical encoding?" Compare:
  NumericalVLA (this) vs. VERA-BC (no feedback at all):  shows numerical history helps
  NumericalVLA (this) vs. Full VERA (language feedback):  shows language > numbers

Inputs at every step:
  - T=3 RGB frames → frozen CLIP ViT-B/32 vision tokens
  - Language instruction → frozen CLIP text token
  - [prev_action_idx/8, prev_reward] → Linear(2, d_model) + RMSNorm → 1 numerical token
  - H=4 history steps (action_idx, action_vec, reward) → history encoder
  Token sequence: [L_instr | N_prev | V_1..V_3 | H_1..H_4 | CLS]

Usage:
    python -m baselines.finetune_numvla_lt \\
        --data_path /data/language_table_episodes.pkl \\
        --output_dir /results \\
        --seed 42 --epochs 80
"""

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
import torchvision.transforms as Tv

# Reuse VERA model components directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.vera_model import (
    RMSNorm, LLaMAFusionTransformer, ViLTModalityEmbedding,
    ActionRewardHistoryEncoder,
)


# ─────────────────────────────────────────────────────────────────────────────
# NumericalVLA Model
# ─────────────────────────────────────────────────────────────────────────────

class NumericalVLAModel(nn.Module):
    """
    VERA architecture with language feedback streams replaced by a raw numerical token.

    Shares CLIP backbone, LLaMA fusion transformer, ViLT modality embeddings,
    ActionRewardHistoryEncoder, and the discrete + regression heads with VERA.
    The ONLY change: E_act and E_exp tokens are replaced by one token from
    Linear([prev_action_idx/N, prev_reward], d_model) + RMSNorm.

    This tests whether VERA's gain comes from "feedback" (any form) or specifically
    from "language-verbalized feedback."
    """

    def __init__(
        self,
        num_actions:   int   = 8,
        history_len:   int   = 4,
        num_vis_frames:int   = 3,
        fusion_layers: int   = 6,
        fusion_heads:  int   = 8,
        d_model:       int   = 256,
        d_ff_scale:    int   = 4,
        dropout:       float = 0.1,
        action_dim:    int   = 2,
    ):
        super().__init__()
        self.num_actions    = num_actions
        self.history_len    = history_len
        self.num_vis_frames = num_vis_frames
        self.d_model        = d_model
        self.action_dim     = action_dim
        clip_dim            = 512

        # ── Shared frozen CLIP backbone ───────────────────────────────────────
        self.clip_model, _ = clip.load("ViT-B/32")
        self.clip_model    = self.clip_model.float()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # ── CLIP → d_model projections ────────────────────────────────────────
        self.vis_proj  = nn.Sequential(nn.Linear(clip_dim, d_model, bias=False), RMSNorm(d_model))
        self.lang_proj = nn.Sequential(nn.Linear(clip_dim, d_model, bias=False), RMSNorm(d_model))

        # ── Numerical feedback token (replaces E_act + E_exp) ─────────────────
        # Input: [prev_action_idx / num_actions, prev_reward]  — both in [0,1]
        # This is the maximum information a numerical encoding can convey about
        # the previous step without language verbalization.
        self.num_feedback_proj = nn.Sequential(
            nn.Linear(2, d_model, bias=False),
            RMSNorm(d_model),
        )

        # ── History encoder (identical to VERA Stream 4) ─────────────────────
        self.history_encoder = ActionRewardHistoryEncoder(
            num_actions=num_actions, history_len=history_len,
            d_model=d_model, action_dim=action_dim, dropout=dropout,
            use_temporal_transformer=True,
        )

        # ── ViLT modality-type embeddings: 4 types (no CONSEQUENCE) ──────────
        self.modality_embed = ViLTModalityEmbedding(d_model, num_modalities=4)

        # ── LLaMA fusion transformer ──────────────────────────────────────────
        self.fusion_transformer = LLaMAFusionTransformer(
            d_model=d_model, num_heads=fusion_heads, num_layers=fusion_layers,
            d_ff=d_model * d_ff_scale, dropout=dropout, max_seq_len=512,
        )

        # ── CLS aggregation token ─────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Action heads (identical to VERA) ─────────────────────────────────
        self.action_bin_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model * 2, bias=False), nn.SiLU(), nn.Dropout(dropout),
            RMSNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model, bias=False), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_model, num_actions, bias=False),
        )
        self.action_vec_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2, bias=False), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, action_dim, bias=False), nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(
        self,
        frames:          torch.Tensor,                   # (B, T, C, H, W)
        lang_tokens:     torch.Tensor,                   # (B, 77)
        action_hist:     torch.Tensor,                   # (B, H) int
        reward_hist:     torch.Tensor,                   # (B, H) float
        prev_action_idx: Optional[torch.Tensor] = None,  # (B,) int
        prev_reward:     Optional[torch.Tensor] = None,  # (B,) float
        action_vec_hist: Optional[torch.Tensor] = None,  # (B, H, action_dim)
    ) -> dict:
        B = frames.size(0)

        # 1. Vision tokens
        BT = B * self.num_vis_frames
        flat = frames.view(BT, *frames.shape[2:])
        with torch.no_grad():
            vis_raw = self.clip_model.encode_image(flat).float()
        vis_tokens = self.vis_proj(vis_raw.view(B, self.num_vis_frames, -1))  # (B,T,D)

        # 2. Language token
        with torch.no_grad():
            lang_raw = self.clip_model.encode_text(lang_tokens).float()
        lang_token = self.lang_proj(lang_raw).unsqueeze(1)                   # (B,1,D)

        # 3. Numerical feedback token (replaces E_act + E_exp)
        _pa = (prev_action_idx if prev_action_idx is not None
               else action_hist[:, -1]).float() / self.num_actions
        _pr = (prev_reward if prev_reward is not None
               else reward_hist[:, -1]).float()
        num_input     = torch.stack([_pa, _pr.clamp(0, 1)], dim=-1)          # (B, 2)
        num_token     = self.num_feedback_proj(num_input).unsqueeze(1)        # (B,1,D)

        # 4. History tokens (Stream 4, identical to VERA)
        hist_tokens = self.history_encoder(action_hist, reward_hist, action_vec_hist)

        # 5. Assemble sequence: [L_instr | N_prev | V_1..V_T | H_1..H_H | CLS]
        T = vis_tokens.size(1)
        H = hist_tokens.size(1)
        cls = self.cls_token.expand(B, -1, -1)

        M = ViLTModalityEmbedding
        parts   = [lang_token, num_token, vis_tokens, hist_tokens, cls]
        mod_ids = torch.cat([
            torch.full((1,), M.INSTRUCTION, dtype=torch.long),   # lang
            torch.full((1,), M.ACTION_LANG,  dtype=torch.long),  # numerical token (ACTION_LANG slot)
            torch.full((T,), M.VISION,       dtype=torch.long),  # frames
            torch.full((H,), M.HISTORY,      dtype=torch.long),  # history
            torch.full((1,), M.INSTRUCTION,  dtype=torch.long),  # CLS
        ], dim=0)

        seq = torch.cat(parts, dim=1)                            # (B, S, D)
        seq = self.modality_embed(seq, mod_ids.to(seq.device))

        # 6. Bidirectional LLaMA fusion (no causal mask — same as VERA)
        out = self.fusion_transformer(seq, attn_mask=None)
        cls_feat = out[:, -1]                                    # (B, D)

        logits = self.action_bin_head(cls_feat)
        action_vec = self.action_vec_head(cls_feat)

        return {"logits": logits, "action_vec": action_vec, "cls_features": cls_feat}


# ─────────────────────────────────────────────────────────────────────────────
# Data  (same format as VERA training)
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE   = 224
NUM_FRAMES = 3
HISTORY_LEN = 4

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


def make_batch(episodes, refs, num_actions, device, action_dim=2):
    frames_l, lang_l, acts_l = [], [], []
    act_hist_l, rew_hist_l, vec_hist_l = [], [], []
    prev_act_l, prev_rew_l = [], []

    for ep_idx, t in refs:
        ep = episodes[ep_idx]
        T_ep = len(ep["actions"])

        # Frame stack
        idxs = [max(0, t - (NUM_FRAMES - 1 - i)) for i in range(NUM_FRAMES)]
        frames_l.append(torch.stack([_preprocess(ep["frames"][i]) for i in idxs]))
        lang_l.append(ep["instruction"])
        acts_l.append(int(ep["actions"][t]))

        # History (H steps before t)
        ah, rh, vh = [], [], []
        for h in range(HISTORY_LEN, 0, -1):
            t_h = t - h
            if t_h >= 0:
                ah.append(int(ep["actions"][t_h]))
                r = float(ep.get("rewards", [0.0] * T_ep)[t_h]) if "rewards" in ep else 0.0
                rh.append(r)
                vecs = ep.get("action_vecs", ep.get("action_vectors", None))
                if vecs is not None:
                    vh.append(np.array(vecs[t_h], dtype=np.float32)[:action_dim])
                else:
                    vh.append(np.zeros(action_dim, dtype=np.float32))
            else:
                ah.append(num_actions)   # padding index
                rh.append(0.0)
                vh.append(np.zeros(action_dim, dtype=np.float32))

        act_hist_l.append(ah)
        rew_hist_l.append(rh)
        vec_hist_l.append(vh)

        # Previous step
        tp = t - 1
        if tp >= 0:
            prev_act_l.append(int(ep["actions"][tp]))
            prev_rew_l.append(
                float(ep.get("rewards", [0.0] * T_ep)[tp]) if "rewards" in ep else 0.0
            )
        else:
            prev_act_l.append(num_actions)
            prev_rew_l.append(0.0)

    frames_t   = torch.stack(frames_l).to(device)
    lang_tok   = clip.tokenize(lang_l, truncate=True).to(device)
    acts_t     = torch.tensor(acts_l, dtype=torch.long, device=device)
    act_hist_t = torch.tensor(act_hist_l, dtype=torch.long, device=device)
    rew_hist_t = torch.tensor(rew_hist_l, dtype=torch.float32, device=device)
    vec_hist_t = torch.tensor(np.array(vec_hist_l), dtype=torch.float32, device=device)
    prev_act_t = torch.tensor(prev_act_l, dtype=torch.long, device=device)
    prev_rew_t = torch.tensor(prev_rew_l, dtype=torch.float32, device=device)

    return (frames_t, lang_tok, acts_t,
            act_hist_t, rew_hist_t, vec_hist_t,
            prev_act_t, prev_rew_t)


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
    p.add_argument("--epochs",      type=int,   default=80)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--weight_decay",type=float, default=5e-4)
    p.add_argument("--val_frac",    type=float, default=0.1)
    p.add_argument("--patience",    type=int,   default=25)
    p.add_argument("--num_actions", type=int,   default=8)
    p.add_argument("--action_dim",  type=int,   default=2,
                   help="Action vector dim: 2 for LT, 7 for CALVIN")
    p.add_argument("--closed_loop_dropout", type=float, default=0.35)
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
        raise RuntimeError("No GPU found. NumericalVLA training requires a GPU.")

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

    model = NumericalVLAModel(num_actions=args.num_actions,
                             action_dim=args.action_dim).to(device)
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
                    "model": "numerical_vla"}

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
                fr, lt, ac, ah, rh, vh, pa, pr = make_batch(
                    val_ep, batch, args.num_actions, device, args.action_dim)
                out_d = model(fr, lt, ah, rh, pa, pr, vh)
                preds_all.extend(out_d["logits"].argmax(-1).cpu().tolist())
                labels_all.extend(ac.cpu().tolist())
        return float(np.mean(np.array(preds_all) == np.array(labels_all)))

    print(f"\n[train] Epochs {start_epoch}→{args.epochs}  "
          f"lr={args.lr}  batch={args.batch_size}  "
          f"cl_dropout={args.closed_loop_dropout}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        random.shuffle(trn_refs)
        losses, accs = [], []

        for s in range(0, len(trn_refs), args.batch_size):
            batch = trn_refs[s: s + args.batch_size]
            if len(batch) < 2:
                continue
            fr, lt, ac, ah, rh, vh, pa, pr = make_batch(
                trn_ep, batch, args.num_actions, device, args.action_dim)

            # Closed-loop dropout: zero out history fraction of time
            if random.random() < args.closed_loop_dropout:
                ah = torch.full_like(ah, args.num_actions)
                rh = torch.zeros_like(rh)
                vh = torch.zeros_like(vh)
                pa = torch.full_like(pa, args.num_actions)
                pr = torch.zeros_like(pr)

            out_d  = model(fr, lt, ah, rh, pa, pr, vh)
            loss   = F.cross_entropy(out_d["logits"], ac, label_smoothing=0.05)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            losses.append(loss.item())
            accs.append((out_d["logits"].argmax(-1) == ac).float().mean().item())

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
