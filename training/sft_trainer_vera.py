"""
VERA Supervised Fine-Tuning (Behavioural Cloning) Trainer
==========================================================
Phase 1: Train the VERA model on expert demonstrations.

Loss = cross-entropy(logits, target)
     + alignment_loss_coef * contrastive_alignment_loss(instr_emb, action_lang_emb, reward)

The contrastive alignment loss teaches the model that successful actions
(positive reward) should be semantically close to the task instruction in
CLIP's shared embedding space.

Usage
-----
  python -m training.sft_trainer_vera --config configs/config.yaml
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.vera_model import VERAModel
from data.trajectory_dataset import (
    TrajectoryDataset, load_episodes, make_random_episodes,
    load_language_table, load_calvin,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> str:
    spec = cfg["training"].get("device", "auto")
    if spec == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return spec


def build_dataloaders(cfg: dict, device: str):
    data_cfg  = cfg["data"]
    ep_path   = data_cfg.get("episodes_path")
    dataset_type = data_cfg.get("dataset_type", "pkl").lower()  # "pkl"|"language_table"|"calvin"

    if ep_path and Path(ep_path).exists():
        if dataset_type == "language_table":
            episodes = load_language_table(ep_path)
            print(f"[data] Loaded {len(episodes)} Language-Table episodes from {ep_path}")
        elif dataset_type == "calvin":
            split = data_cfg.get("calvin_split", "training")
            print(
                f"[data] Loading CALVIN episodes from {ep_path!r} (split={split!r}) — "
                "disk I/O, may take several minutes on first run…",
                flush=True,
            )
            episodes = load_calvin(ep_path, split=split)
            print(
                f"[data] Loaded {len(episodes)} CALVIN episodes from {ep_path} ({split})",
                flush=True,
            )
            dagger_pkl = data_cfg.get("dagger_episodes_pkl")
            if dagger_pkl and Path(dagger_pkl).exists():
                import pickle
                with open(dagger_pkl, "rb") as f:
                    dagger_eps = pickle.load(f)
                episodes = list(episodes) + list(dagger_eps)
                print(
                    f"[data] Merged {len(dagger_eps)} DAgger episodes from {dagger_pkl} "
                    f"(total {len(episodes)})",
                    flush=True,
                )
        else:
            episodes = load_episodes(ep_path)
            print(f"[data] Loaded {len(episodes)} episodes from {ep_path}")
    else:
        allow_synth = bool(data_cfg.get("allow_synthetic", False))
        if dataset_type == "calvin" and not allow_synth:
            msg = [
                "[data] CALVIN expected but data.episodes_path is missing or does not exist:",
                f"  got: {ep_path!r}",
                "",
                "  Required layout:  <episodes_path>/training/episode_*.npz",
                "                     <episodes_path>/validation/…",
                "  Pass the directory that contains training/ and validation/ (often …/task_D_D).",
                "",
            ]
            if ep_path:
                p = Path(ep_path)
                par = p.parent
                if par.is_dir():
                    msg.append(f"  Parent exists: {par}")
                    try:
                        names = sorted(x.name for x in par.iterdir())
                        preview = names[:25]
                        msg.append("  Contents: " + ", ".join(preview))
                        if len(names) > 25:
                            msg.append(f"  … and {len(names) - 25} more entries")
                    except OSError as e:
                        msg.append(f"  (could not list parent: {e})")
                if par.is_dir() and (par / "training").is_dir() and not p.is_dir():
                    msg.append("")
                    msg.append(f"  Hint: try --calvin_path {par}  (training/ is next to task_D_D?)")
            msg.extend([
                "",
                "  Download:  bash scripts/download_calvin_task_d_d.sh \"$HOME/calvin_task_D\"",
                "  Find data:  find \"$HOME\" -maxdepth 5 -type d -name training 2>/dev/null | head",
            ])
            raise SystemExit("\n".join(msg))
        if dataset_type == "language_table" and not allow_synth:
            raise SystemExit(
                "[data] Language-Table expected but data.episodes_path is missing or invalid:\n"
                f"  got: {ep_path!r}\n"
                "Convert/build the TFDS export and set episodes_path to that directory.\n"
                "Or set data.allow_synthetic: true only for smoke tests."
            )
        print("[data] No dataset found — generating synthetic episodes.")
        episodes = make_random_episodes(
            num_episodes=data_cfg.get("synthetic_episodes", 200),
            ep_len=data_cfg.get("ep_len", 30),
            num_actions=cfg["model"]["num_actions"],
            action_dim=cfg["model"].get("action_dim", 4),
        )

    if not episodes:
        split = data_cfg.get("calvin_split", "training")
        extra = ""
        if dataset_type == "calvin" and ep_path and Path(ep_path).exists():
            tp = Path(ep_path) / split
            n_npz = len(list(tp.glob("episode_*.npz"))) if tp.is_dir() else 0
            extra = (
                f"\n  Looked under: {tp}"
                f"\n  episode_*.npz files: {n_npz}"
            )
        raise SystemExit(
            f"[data] Loaded 0 episodes (dataset_type={dataset_type!r}, path={ep_path!r}).{extra}\n"
            "For CALVIN, episodes_path must be the task_D_D root containing "
            "training/ and validation/, with episode_*.npz under training/."
        )

    # ── Episode-level train/val split (fixed seed = reproducible across runs) ──
    # Window-level random_split leaks data: neighbouring windows from the same
    # episode can end up in both splits, inflating val_acc when resuming from
    # a checkpoint (model already trained on adjacent windows).
    # Splitting episodes first and building two separate datasets fully prevents
    # cross-split episode leakage and gives consistent metrics across restarts.
    val_frac   = cfg["training"].get("val_fraction", 0.1)
    split_seed = cfg["training"].get("split_seed", 42)   # fixed across runs
    rng = random.Random(split_seed)
    ep_indices = list(range(len(episodes)))
    rng.shuffle(ep_indices)
    n_val_ep  = max(1, int(len(episodes) * val_frac))
    val_idx   = ep_indices[:n_val_ep]
    train_idx = ep_indices[n_val_ep:]
    val_episodes   = [episodes[i] for i in sorted(val_idx)]
    train_episodes = [episodes[i] for i in sorted(train_idx)]
    print(f"[data] Episode split (seed={split_seed}): "
          f"{len(train_episodes)} train / {len(val_episodes)} val episodes", flush=True)

    ds_kwargs = dict(
        history_len=cfg["model"]["history_len"],
        num_vis_frames=cfg["model"]["num_vis_frames"],
        num_actions=cfg["model"]["num_actions"],
        action_dim=cfg["model"].get("action_dim", 4),
        img_size=cfg["data"].get("img_size", 224),
        device=device,
        chunk_size=cfg["model"].get("chunk_size", 1),
        proprio_dim=cfg["model"].get("proprio_dim", 0),
        use_gripper_cam=cfg["data"].get("use_gripper_cam", False),
        crop_factor=cfg["data"].get("crop_factor", 1.0),
    )
    stats_path = cfg["data"].get("proprio_stats_path")
    if stats_path and Path(stats_path).exists():
        from data.calvin_utils import load_proprio_stats
        mean, std = load_proprio_stats(stats_path)
        ds_kwargs["proprio_mean"] = mean
        ds_kwargs["proprio_std"] = std
    print("[data] Building train dataset…", flush=True)
    train_ds = TrajectoryDataset(train_episodes, **ds_kwargs)
    print("[data] Building val dataset…", flush=True)
    val_ds   = TrajectoryDataset(val_episodes,   **ds_kwargs)

    kw = dict(
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"].get("num_workers", 2),
        pin_memory=(device != "cpu"),
    )
    seed = int(cfg["training"].get("seed", 42))
    train_gen = torch.Generator().manual_seed(seed + 1)
    val_gen = torch.Generator().manual_seed(seed + 2)
    return (
        DataLoader(train_ds, shuffle=True, generator=train_gen, **kw),
        DataLoader(val_ds,   shuffle=False, generator=val_gen, **kw),
    )


def load_pretrained_lenient(model: VERAModel, state_dict: dict) -> tuple:
    """Load matching shapes only (legacy checkpoints may differ in action_bin_head)."""
    model_sd = model.state_dict()
    loaded, skipped = 0, []
    for k, v in state_dict.items():
        if k not in model_sd or model_sd[k].shape != v.shape:
            skipped.append(k)
            continue
        model_sd[k] = v
        loaded += 1
    model.load_state_dict(model_sd)
    return loaded, skipped


def build_model(cfg: dict) -> VERAModel:
    vera_cfg = cfg.get("vera", {})
    return VERAModel(
        num_actions=cfg["model"]["num_actions"],
        history_len=cfg["model"]["history_len"],
        num_vis_frames=cfg["model"]["num_vis_frames"],
        fusion_layers=cfg["model"].get("fusion_layers", 6),
        fusion_heads=cfg["model"].get("fusion_heads", 8),
        d_model=cfg["model"].get("d_model", 256),
        d_ff_scale=cfg["model"].get("d_ff_scale", 4),
        dropout=cfg["model"].get("dropout", 0.1),
        freeze_clip=cfg["model"].get("freeze_clip", True),
        unfreeze_clip_vision=cfg["model"].get("unfreeze_clip_vision", False),
        use_lang_feedback=vera_cfg.get("use_lang_feedback", True),
        use_action_lang_feedback=vera_cfg.get("use_action_lang_feedback"),
        use_temporal_history=vera_cfg.get("use_temporal_history", True),
        use_reward_gate=vera_cfg.get("use_reward_gate", True),
        use_consequence_token=vera_cfg.get("use_consequence_token", True),
        action_dim=cfg["model"].get("action_dim", 4),   # 4=MetaWorld, 2=Language-Table, 7=CALVIN
        action_vocab=vera_cfg.get("action_vocab"),      # None → use built-in vocabulary
        proprio_dim=cfg["model"].get("proprio_dim", 0),
        chunk_size=cfg["model"].get("chunk_size", 1),
        use_step_conditioning=vera_cfg.get("use_step_conditioning", False),
        step_horizon=vera_cfg.get(
            "step_horizon", cfg.get("rl", {}).get("max_episode_steps", 96)
        ),
    )


# ── Training / validation loop ────────────────────────────────────────────────

def run_epoch(
    model:        VERAModel,
    loader:       DataLoader,
    optimizer:    Optional[torch.optim.Optimizer],
    criterion:    nn.Module,
    device:       str,
    is_train:     bool,
    grad_clip:    float = 1.0,
    align_coef:   float = 0.1,
    reg_coef:     float = 0.5,
    ce_coef:      float = 1.0,
    batch_log_interval: int = 200,
    closed_loop_dropout: float = 0.0,
    pad_action: int = 0,
) -> dict:
    """
    One full pass over *loader*.

    Losses
    ------
    ce       — cross-entropy on discrete action logits          (always)
               When chunk_size > 1, CE is averaged over all K chunk steps:
               ce = mean CE over {t, t+1, ..., t+K-1}.  This gives K×
               the supervised signal and enforces temporal consistency
               (inspired by π0 / GR-1 action chunking).
    align    — dual reward-weighted InfoNCE (experience + reasoning)
    reg      — MSE between predicted action_vec and target_vec  (only when
               the dataset provides continuous action targets, e.g. CALVIN)

    Diagnostics (logged but not optimised)
    ---------------------------------------
    cos_exp  — mean cosine(instr_emb, action_lang_emb)   ∈ [-1, 1]
    cos_rsn  — mean cosine(instr_emb, consequence_emb)   ∈ [-1, 1]

    If either alignment stream is disabled (ablation flags) the corresponding
    cosine value is reported as 0.0 with no contribution to cos_n.
    """
    model.train() if is_train else model.eval()

    total_loss = total_ce = total_align = total_reg = 0.0
    total_correct = total_samples = 0
    total_cos_exp = total_cos_rsn = 0.0
    cos_exp_n = cos_rsn_n = 0

    # Detect chunk_size from model (1 = disabled, no overhead)
    K = getattr(model, "chunk_size", 1)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        n_batches = len(loader)
        tag = "train" if is_train else "val"
        t_batch0 = time.perf_counter()
        t_epoch_wall = t_batch0
        for bi, batch in enumerate(loader):
            if bi == 0:
                print(
                    f"[sft_vera] {tag}: first batch ready after {time.perf_counter() - t_batch0:.1f}s "
                    f"(DataLoader collate + workers). {n_batches} batches this epoch.",
                    flush=True,
                )
                t_h2d = time.perf_counter()

            nb = device.startswith("cuda")
            frames      = batch["frames"].to(device, non_blocking=nb)
            lang_tokens = batch["lang_tokens"].to(device, non_blocking=nb)
            action_hist = batch["action_hist"].to(device, non_blocking=nb)
            reward_hist = batch["reward_hist"].to(device, non_blocking=nb)
            target      = batch["target"].to(device, non_blocking=nb)

            # Low-level action vector history — present for Language-Table / CALVIN,
            # None for dummy / discrete-only datasets (model degrades gracefully)
            avh = batch.get("action_vec_hist")
            action_vec_hist = avh.to(device, non_blocking=nb) if isinstance(avh, torch.Tensor) else None

            # State delta for Stream 3b consequence verbalization.
            # Provided by Language-Table loader as a pseudo dist-to-goal change
            # derived from consecutive reward differences.  For legacy / synthetic
            # episodes the dataset returns 0.0 scalars, which map to the
            # "stationary" branch in verbalize_consequence (still informative).
            sd = batch.get("state_delta")
            state_delta = sd.to(device, non_blocking=nb) if isinstance(sd, torch.Tensor) else None

            ro = batch.get("robot_obs")
            robot_obs = ro.to(device, non_blocking=nb) if isinstance(ro, torch.Tensor) else None

            si = batch.get("step_idx")
            step_idx = si.to(device, non_blocking=nb) if isinstance(si, torch.Tensor) else None

            # ── Per-batch reward normalisation ────────────────────────────────
            # Normalise reward_hist to [0, 1] so that:
            #   (a) the reward gate MLP sees a useful dynamic range, and
            #   (b) the InfoNCE exponential weights exp(5·r) have full contrast.
            # We use a running-max normalisation anchored at the batch maximum
            # (not mean-std) so that zero-reward steps stay at 0.
            r_max = reward_hist.max().clamp(min=1e-6)
            reward_hist_norm = (reward_hist / r_max).clamp(0.0, 1.0)

            # Closed-loop augmentation: zero history (matches rollout drift).
            if is_train and closed_loop_dropout > 0.0 and torch.rand(1).item() < closed_loop_dropout:
                action_hist = torch.full_like(action_hist, pad_action)
                reward_hist_norm = torch.zeros_like(reward_hist_norm)
                if action_vec_hist is not None:
                    action_vec_hist = torch.zeros_like(action_vec_hist)
                if isinstance(state_delta, torch.Tensor):
                    state_delta = torch.zeros_like(state_delta)

            out = model(frames, lang_tokens, action_hist, reward_hist_norm,
                        state_delta=state_delta,
                        action_vec_hist=action_vec_hist,
                        robot_obs=robot_obs,
                        step_idx=step_idx)
            if bi == 0 and device.startswith("cuda"):
                torch.cuda.synchronize()
            if bi == 0:
                print(
                    f"[sft_vera] {tag}: batch0 H2D+forward {time.perf_counter() - t_h2d:.1f}s",
                    flush=True,
                )
                t_bwd = time.perf_counter()

            logits = out["logits"]                          # (B, A) — step t only

            # ── Cross-entropy loss (with optional action chunking) ────────────
            # K=1: standard single-step CE on logits (B, A) vs target (B,).
            # K>1: CE averaged over all K chunk steps — K× supervision signal.
            #   logits_chunk (B, K, A) vs target_chunk (B, K)
            #   Reshape to (B*K, A) / (B*K,) so nn.CrossEntropyLoss works.
            # Primary head always supervised; chunk head is auxiliary when K>1.
            ce = criterion(logits, target)
            if K > 1 and "logits_chunk" in out:
                target_chunk = batch.get("target_chunk")
                if target_chunk is not None:
                    target_chunk = target_chunk.to(device)
                    B_sz, A = logits.shape
                    ce_chunk = criterion(
                        out["logits_chunk"].view(B_sz * K, A),
                        target_chunk.view(B_sz * K),
                    )
                    ce = 0.5 * ce + 0.5 * ce_chunk

            # ── Dual contrastive alignment loss ───────────────────────────────
            align = torch.tensor(0.0, device=device)
            if (out["instr_emb"] is not None
                    and align_coef > 0
                    and is_train):
                # Use normalised rewards so exp(5·r) gives full contrast [1, 148]
                prev_reward_norm = reward_hist_norm[:, -1]  # most recent (normalised)
                align = model.compute_alignment_loss(
                    out["instr_token"],
                    out.get("action_lang_token"),
                    prev_reward_norm,
                    out.get("consequence_token"),
                )

            # ── Continuous action regression loss ─────────────────────────────
            # Only activated when the dataset supplies expert continuous actions.
            # MSE on Tanh-bounded predictions → no explicit range clipping needed.
            reg = torch.tensor(0.0, device=device)
            tvec = batch.get("target_vec")
            if (isinstance(tvec, torch.Tensor)
                    and out.get("action_vec") is not None
                    and reg_coef > 0):
                reg = F.mse_loss(out["action_vec"], tvec.to(device))

            loss = ce_coef * ce + align_coef * align + reg_coef * reg

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
                optimizer.step()
                if bi == 0 and device.startswith("cuda"):
                    torch.cuda.synchronize()
                if bi == 0:
                    print(
                        f"[sft_vera] {tag}: batch0 backward+optimizer "
                        f"{time.perf_counter() - t_bwd:.1f}s "
                        f"(batch0 total {time.perf_counter() - t_batch0:.1f}s)",
                        flush=True,
                    )
            elif bi == 0:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                print(
                    f"[sft_vera] {tag}: batch0 val total {time.perf_counter() - t_batch0:.1f}s "
                    "(no backward)",
                    flush=True,
                )

            # ── Alignment cosine diagnostic (no gradient) ─────────────────────
            # Tracks whether the alignment loss is actually pulling embeddings
            # closer to the instruction over training.  Rising values = working.
            if out.get("alignment_score") is not None:
                total_cos_exp += out["alignment_score"].detach().mean().item()
                cos_exp_n += 1
            if out.get("consequence_score") is not None:
                total_cos_rsn += out["consequence_score"].detach().mean().item()
                cos_rsn_n += 1

            B = target.size(0)
            total_loss    += loss.item()  * B
            total_ce      += ce.item()    * B
            total_align   += align.item() * B
            total_reg     += reg.item()   * B
            total_correct += (logits.argmax(-1) == target).sum().item()
            total_samples += B

            if (
                batch_log_interval > 0
                and (bi + 1) % batch_log_interval == 0
                and (bi + 1) < n_batches
            ):
                wall = time.perf_counter() - t_epoch_wall
                pct = 100.0 * (bi + 1) / n_batches
                rloss = total_loss / max(total_samples, 1)
                print(
                    f"[sft_vera] {tag}: batch {bi + 1}/{n_batches} ({pct:.1f}%) "
                    f"wall {wall:.0f}s  running_loss {rloss:.4f}",
                    flush=True,
                )

    n = max(total_samples, 1)
    return {
        "loss":       total_loss  / n,
        "ce_loss":    total_ce    / n,
        "align_loss": total_align / n,
        "reg_loss":   total_reg   / n,
        "accuracy":   total_correct / n,
        # Alignment cosine diagnostics — averaged over batches (not samples)
        # so they reflect mean per-batch similarity regardless of batch size.
        "cos_exp":    total_cos_exp / max(cos_exp_n, 1),
        "cos_rsn":    total_cos_rsn / max(cos_rsn_n, 1),
    }


# ── Main training function ────────────────────────────────────────────────────

def train(cfg: dict, resume_from: Optional[str] = None):
    """
    Train VERA with optional warm-restart from a saved checkpoint.

    resume_from : path to a .pt file produced by this trainer.
                  If provided, loads model weights (+ optimizer/scheduler
                  states when available) and continues from the saved epoch.
                  The existing sft_vera_log.json is preserved and appended to.
    """
    seed = int(cfg["training"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = resolve_device(cfg)
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    print(f"[sft_vera] device = {device} | seed = {seed}", flush=True)

    train_loader, val_loader = build_dataloaders(cfg, device)
    model = build_model(cfg).to(device)
    print(f"[model] {model.param_summary()}", flush=True)

    # Cast to float/int — YAML round-trips scientific notation (e.g. 1e-4) as
    # strings in PyYAML 6.x, which causes TypeError in torch.optim.AdamW.
    t_cfg = cfg["training"]
    lr           = float(t_cfg["lr"])
    weight_decay = float(t_cfg.get("weight_decay", 1e-4))
    total_epochs = int(t_cfg["epochs"])

    # ── Param-group-aware optimizer ──────────────────────────────────────────
    # When CLIP vision is unfrozen (unfreeze_clip_vision=True), use a much
    # smaller LR for those params to avoid destroying pretrained features.
    # clip_vision_lr defaults to 5% of the main LR if not specified.
    clip_vis_params = [p for p in model.clip_model.visual.parameters()
                       if p.requires_grad]
    if clip_vis_params:
        clip_vision_lr  = float(t_cfg.get("clip_vision_lr", lr * 0.05))
        clip_vis_ids    = {id(p) for p in clip_vis_params}
        non_clip_params = [p for p in model.parameters()
                           if p.requires_grad and id(p) not in clip_vis_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": non_clip_params, "lr": lr,            "weight_decay": weight_decay},
                {"params": clip_vis_params,  "lr": clip_vision_lr, "weight_decay": weight_decay * 0.1},
            ],
            betas=(0.9, 0.98),
        )
        print(f"[opt] 2 param groups — backbone {len(non_clip_params)} params "
              f"lr={lr:.2e}, clip-vision {len(clip_vis_params)} params "
              f"lr={clip_vision_lr:.2e}")
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),      # slightly higher β₂ for transformer training
        )

    # Warmup + cosine annealing: warmup for 5% of total steps, then cosine decay
    warmup_epochs = max(1, int(total_epochs * 0.05))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_epochs
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=float(t_cfg.get("lr_min", 1e-6)),
            ),
        ],
        milestones=[warmup_epochs],
    )

    criterion   = nn.CrossEntropyLoss(label_smoothing=float(t_cfg.get("label_smoothing", 0.05)))
    align_coef  = float(cfg.get("vera", {}).get("alignment_loss_coef", 0.1))
    reg_coef    = float(cfg.get("vera", {}).get("regression_loss_coef", 0.5))
    ce_coef     = float(t_cfg.get("ce_loss_coef", 1.0))
    grad_clip   = float(t_cfg.get("grad_clip", 1.0))
    out_dir     = Path(t_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    patience         = int(t_cfg.get("early_stopping_patience", 10))
    patience_counter = 0
    log, best_val_acc = [], 0.0
    best_val_reg = float("inf")
    ckpt_metric = str(t_cfg.get("checkpoint_metric", "val_acc")).lower()
    cl_dropout = float(t_cfg.get("closed_loop_dropout", 0.0))
    pad_action = int(cfg["model"]["num_actions"])
    start_epoch = 0

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        loaded, skipped = load_pretrained_lenient(model, ckpt["model_state"])
        print(f"[resume] Loaded {loaded} tensors; skipped {len(skipped)} "
              f"(e.g. {skipped[:4]}{'…' if len(skipped) > 4 else ''})")
        finetune = bool(t_cfg.get("finetune", False))
        if finetune:
            print(f"[resume] Finetune mode — fresh optimizer, epochs 1–{total_epochs}, "
                  f"lr={t_cfg.get('lr')}")
        else:
            start_epoch = int(ckpt.get("epoch", 0))
            best_val_acc = float(ckpt.get("val_acc", 0.0))
            if "optimizer_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                    print("[resume] Optimizer state restored.")
                except ValueError as e:
                    print(f"[resume] Optimizer not restored ({e}).")
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            else:
                for _ in range(start_epoch):
                    scheduler.step()
            log_path = out_dir / "sft_vera_log.json"
            if log_path.exists():
                with open(log_path) as f:
                    log = json.load(f)
                log = [r for r in log if r["epoch"] <= start_epoch]
            for r in reversed(log):
                if r.get("val_acc", 0.0) < best_val_acc:
                    patience_counter += 1
                else:
                    break
            print(f"[resume] Continue epoch {start_epoch+1}/{total_epochs}  "
                  f"best_val_acc={best_val_acc:.4f}")

    for epoch in range(start_epoch + 1, total_epochs + 1):
        t0 = time.time()
        _log_b = int(t_cfg.get("log_every_batches", 200))
        if epoch == start_epoch + 1:
            nbt = len(train_loader)
            print(
                f"[sft_vera] Beginning epoch {epoch} — {nbt} train batches; "
                f"progress every {_log_b} batches, then one summary line per epoch.",
                flush=True,
            )

        train_m = run_epoch(
            model, train_loader, optimizer, criterion,
            device, is_train=True,
            grad_clip=grad_clip, align_coef=align_coef, reg_coef=reg_coef,
            ce_coef=ce_coef,
            batch_log_interval=_log_b,
            closed_loop_dropout=cl_dropout,
            pad_action=pad_action,
        )
        val_m = run_epoch(
            model, val_loader, None, criterion,
            device, is_train=False,
            align_coef=0.0, reg_coef=reg_coef, ce_coef=ce_coef,
            batch_log_interval=_log_b,
            pad_action=pad_action,
        )
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            "epoch":           epoch,
            "train_loss":      round(train_m["loss"],      4),
            "train_acc":       round(train_m["accuracy"],  4),
            "train_align":     round(train_m["align_loss"],4),
            "train_reg":       round(train_m["reg_loss"],  4),
            "train_cos_exp":   round(train_m["cos_exp"],   4),
            "train_cos_rsn":   round(train_m["cos_rsn"],   4),
            "val_loss":        round(val_m["loss"],        4),
            "val_acc":         round(val_m["accuracy"],    4),
            "val_reg":         round(val_m["reg_loss"],    4),
            "val_cos_exp":     round(val_m["cos_exp"],     4),
            "val_cos_rsn":     round(val_m["cos_rsn"],     4),
            "lr":              round(scheduler.get_last_lr()[0], 8),
            "time_s":          round(elapsed, 1),
        }
        log.append(row)
        print(f"Epoch {epoch:3d}/{total_epochs} | "
              f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} "
              f"align {row['train_align']:.4f} reg {row['train_reg']:.4f} | "
              f"cos_exp {row['train_cos_exp']:+.3f} cos_rsn {row['train_cos_rsn']:+.3f} | "
              f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.3f} | "
              f"lr {row['lr']:.2e} | {elapsed:.1f}s",
              flush=True)

        # ── Incremental log write (survives disconnection) ────────────────────
        # Write after every epoch so no progress is lost if Colab disconnects.
        with open(out_dir / "sft_vera_log.json", "w") as f:
            json.dump(log, f, indent=2)

        # ── Save best checkpoint (metric: val_acc or val_reg for embodied training) ─
        improved = False
        if ckpt_metric == "val_reg" and val_m["reg_loss"] > 0:
            if val_m["reg_loss"] < best_val_reg:
                best_val_reg = val_m["reg_loss"]
                best_val_acc = max(best_val_acc, val_m["accuracy"])
                improved = True
                metric_msg = f"val_reg={best_val_reg:.5f}"
        elif val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            improved = True
            metric_msg = f"val_acc={best_val_acc:.3f}"

        if improved:
            patience_counter = 0
            torch.save({
                "epoch":            epoch,
                "model_state":      model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "scheduler_state":  scheduler.state_dict(),
                "val_acc":          best_val_acc,
                "val_reg":          best_val_reg,
                "cfg":              cfg,
            }, out_dir / "best_sft_vera.pt")
            print(f"  ✓ best checkpoint saved ({metric_msg})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[sft_vera] Early stop at epoch {epoch} "
                      f"(no improvement for {patience} epochs). "
                      f"Best val acc: {best_val_acc:.3f}")
                break

        # ── Periodic snapshot ─────────────────────────────────────────────────
        if epoch % cfg["training"].get("save_every", 10) == 0:
            torch.save({
                "epoch":            epoch,
                "model_state":      model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "scheduler_state":  scheduler.state_dict(),
                "cfg":              cfg,
            }, out_dir / f"sft_vera_epoch{epoch:04d}.pt")

    print(f"\n[sft_vera] Done. Best val acc: {best_val_acc:.3f}")


# Alias used by run_ablations.py and run_calvin_ablations.py
sft_train = train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume_from", default=None,
                        help="Path to a best_sft_vera.pt checkpoint to resume from.")
    parser.add_argument("--calvin_path", default=None,
                        help="Override data.episodes_path for CALVIN training.")
    parser.add_argument("--episodes_path", default=None,
                        help="Override data.episodes_path (Language-Table or CALVIN).")
    parser.add_argument("--output_dir", default=None,
                        help="Override training.output_dir.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.calvin_path:
        cfg["data"]["episodes_path"] = args.calvin_path
        cfg["data"]["dataset_type"] = "calvin"
    if args.episodes_path:
        cfg["data"]["episodes_path"] = args.episodes_path
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    train(cfg, resume_from=args.resume_from)
