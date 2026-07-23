"""
Online RL Fine-Tuning Trainer (REINFORCE with baseline)
========================================================
Phase 2: roll out the policy in an environment, collect (s, a, r) trajectories,
and update the model using REINFORCE with a learned value baseline to reduce
variance.

Algorithm
---------
  For each RL epoch:
    1. Roll out `num_rollouts` episodes using the current policy (no grad).
    2. Compute discounted returns G_t for each step.
    3. Normalize returns (mean/std) for variance reduction.
    4. Policy gradient loss: -log π(a|s) * (G_t - V(s))      [REINFORCE]
       Value baseline loss:  (V(s) - G_t)^2                  [MSE]
    5. KL penalty against BC checkpoint to prevent forgetting.
    6. Update model + value head with combined loss.

The environment must expose a Gym-compatible interface:
    obs  = env.reset()   -> dict with "frame" (H,W,3) and "instruction" str
    obs, reward, done, info = env.step(action_idx)

To swap in a real robot or different sim, just subclass BaseEnv in envs/.
"""

import argparse
import json
import os
from collections import deque
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from models.vla_model import RLConditionedVLA
from data.trajectory_dataset import make_random_episodes


# ── Value baseline head ───────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """Small MLP that predicts a scalar state-value from the CLS token."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, cls_features: torch.Tensor) -> torch.Tensor:
        """cls_features: (B, D) → values: (B,)"""
        return self.net(cls_features).squeeze(-1)


# ── Rollout buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Accumulates one batch of episode transitions for policy-gradient updates."""

    def __init__(self):
        self.frames:       List[torch.Tensor] = []
        self.lang_tokens:  List[torch.Tensor] = []
        self.action_hists: List[torch.Tensor] = []
        self.reward_hists: List[torch.Tensor] = []
        self.actions:      List[int]           = []
        self.rewards:      List[float]         = []
        self.dones:        List[bool]          = []

    def add(self, frame, lang_tok, act_hist, rew_hist, action, reward, done):
        self.frames.append(frame)
        self.lang_tokens.append(lang_tok)
        self.action_hists.append(act_hist)
        self.reward_hists.append(rew_hist)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def compute_returns(self, gamma: float = 0.99) -> torch.Tensor:
        """Compute discounted returns, reset at episode boundaries."""
        returns = []
        G = 0.0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0.0
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns


# ── Rollout collector ─────────────────────────────────────────────────────────

def collect_rollout(
    model: RLConditionedVLA,
    env,
    cfg: dict,
    device: str,
    tokenizer_cache: dict,
) -> RolloutBuffer:
    """Run the policy for `max_steps` steps and return a filled buffer."""
    buf = RolloutBuffer()
    history_len   = cfg["model"]["history_len"]
    num_vis_frames = cfg["model"]["num_vis_frames"]
    num_actions   = cfg["model"]["num_actions"]
    img_size      = cfg["data"].get("img_size", 224)

    import torchvision.transforms as Tv
    from PIL import Image as PImage

    transform = Tv.Compose([
        Tv.Resize((img_size, img_size)),
        Tv.ToTensor(),
        Tv.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                     std= [0.26862954, 0.26130258, 0.27577711]),
    ])

    obs = env.reset()
    frame_queue  = deque(maxlen=num_vis_frames)
    action_queue = deque([num_actions] * history_len, maxlen=history_len)  # padded
    reward_queue = deque([0.0]         * history_len, maxlen=history_len)

    # Tokenize instruction
    instruction = obs["instruction"]
    if instruction not in tokenizer_cache:
        tokenizer_cache[instruction] = clip.tokenize([instruction])[0]
    lang_tok = tokenizer_cache[instruction]

    done = False
    step = 0
    max_steps = cfg["rl"].get("max_episode_steps", 50)

    while not done and step < max_steps:
        # Build frame tensor
        raw_frame = obs["frame"]  # (H, W, 3) uint8
        frame_t = transform(PImage.fromarray(raw_frame))   # (3, H, W)
        frame_queue.append(frame_t)

        # Pad if not enough frames yet
        pad_needed = num_vis_frames - len(frame_queue)
        padded_frames = [torch.zeros_like(frame_t)] * pad_needed + list(frame_queue)
        frames_tensor = torch.stack(padded_frames).unsqueeze(0).to(device)  # (1, T, 3, H, W)

        lang_tensor   = lang_tok.unsqueeze(0).to(device)
        act_hist_t    = torch.tensor(list(action_queue), dtype=torch.long).unsqueeze(0).to(device)
        rew_hist_t    = torch.tensor(list(reward_queue), dtype=torch.float32).unsqueeze(0).to(device)

        # Sample action from policy (stochastic)
        model.eval()
        with torch.no_grad():
            logits = model(frames_tensor, lang_tensor, act_hist_t, rew_hist_t)
        probs  = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).item()

        obs, reward, done, _ = env.step(action)

        buf.add(
            frame=frames_tensor.squeeze(0).cpu(),        # (T, 3, H, W)
            lang_tok=lang_tok,
            act_hist=act_hist_t.squeeze(0).cpu(),        # (H,)
            rew_hist=rew_hist_t.squeeze(0).cpu(),        # (H,)
            action=action,
            reward=reward,
            done=done,
        )

        action_queue.append(action)
        reward_queue.append(reward)
        step += 1

    return buf


# ── RL update step ────────────────────────────────────────────────────────────

def rl_update(
    model: RLConditionedVLA,
    value_head: ValueHead,
    buf: RolloutBuffer,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    device: str,
    bc_model: Optional[RLConditionedVLA] = None,
) -> dict:
    """Single REINFORCE + value-baseline update over buffer transitions."""
    model.train()
    value_head.train()

    returns = buf.compute_returns(gamma=cfg["rl"].get("gamma", 0.99)).to(device)

    # Stack batch
    frames      = torch.stack(buf.frames).to(device)         # (N, T, 3, H, W)
    lang_tokens = torch.stack(buf.lang_tokens).to(device)    # (N, 77)
    act_hist    = torch.stack(buf.action_hists).to(device)   # (N, H)
    rew_hist    = torch.stack(buf.reward_hists).to(device)   # (N, H)
    actions     = torch.tensor(buf.actions, dtype=torch.long, device=device)

    # Forward pass — need CLS features for value head
    # Patch model to expose CLS token temporarily
    B = frames.size(0)
    vis_tokens  = model.encode_frames(frames)
    lang_proj   = model.encode_language(lang_tokens)
    hist_tokens = model.history_encoder(act_hist, rew_hist)
    cls         = model.cls_token.expand(B, -1, -1)
    # CLS goes LAST — with causal mask, last position attends to all previous tokens.
    sequence    = torch.cat([lang_proj, vis_tokens, hist_tokens, cls], dim=1)
    seq_len     = sequence.size(1)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device), diagonal=1
    ).bool()
    out         = model.transformer(sequence, mask=causal_mask)
    cls_feat    = out[:, -1, :]                               # (B, D) — last token
    logits      = model.action_bin_head(cls_feat)             # (B, A)

    # Value estimate and advantage
    values    = value_head(cls_feat.detach())                 # (B,)
    advantage = (returns - values.detach())

    # Policy gradient loss (REINFORCE)
    log_probs     = F.log_softmax(logits, dim=-1)
    chosen_logp   = log_probs[torch.arange(B), actions]
    policy_loss   = -(chosen_logp * advantage).mean()

    # Value baseline loss
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus (exploration)
    entropy = -(F.softmax(logits, dim=-1) * log_probs).sum(-1).mean()
    entropy_coef = cfg["rl"].get("entropy_coef", 0.01)

    # KL penalty against BC model to prevent catastrophic forgetting
    kl_loss = torch.tensor(0.0, device=device)
    if bc_model is not None:
        bc_model.eval()
        with torch.no_grad():
            bc_logits = bc_model(frames, lang_tokens, act_hist, rew_hist)
        bc_probs = F.softmax(bc_logits, dim=-1)
        kl_loss  = F.kl_div(
            F.log_softmax(logits, dim=-1),
            bc_probs,
            reduction="batchmean",
        )

    kl_coef = cfg["rl"].get("kl_coef", 0.1)
    vf_coef = cfg["rl"].get("vf_coef", 0.5)

    total_loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy + kl_coef * kl_loss

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(value_head.parameters()),
        cfg["rl"].get("grad_clip", 1.0),
    )
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss":  value_loss.item(),
        "entropy":     entropy.item(),
        "kl_loss":     kl_loss.item(),
        "total_loss":  total_loss.item(),
        "mean_return": returns.mean().item(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def rl_train(cfg: dict):
    import yaml
    from envs.sim_env import SimEnv   # swap to RealEnv for hardware

    device = cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rl_trainer] device = {device}")

    # Load BC checkpoint
    out_dir   = Path(cfg["training"]["output_dir"])
    bc_ckpt   = out_dir / "best_sft.pt"
    model     = RLConditionedVLA(
        num_actions=cfg["model"]["num_actions"],
        history_len=cfg["model"]["history_len"],
        fusion_layers=cfg["model"].get("fusion_layers", 6),
        fusion_heads=cfg["model"].get("fusion_heads", 8),
        dropout=cfg["model"].get("dropout", 0.1),
        freeze_clip=cfg["model"].get("freeze_clip", True),
    ).to(device)

    if bc_ckpt.exists():
        ckpt = torch.load(bc_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[rl_trainer] Loaded BC checkpoint from {bc_ckpt}")
        # Frozen reference model for KL penalty
        bc_model = RLConditionedVLA(
            num_actions=cfg["model"]["num_actions"],
            history_len=cfg["model"]["history_len"],
        ).to(device)
        bc_model.load_state_dict(ckpt["model_state"])
        for p in bc_model.parameters():
            p.requires_grad = False
    else:
        print("[rl_trainer] No BC checkpoint found — training from scratch (not recommended).")
        bc_model = None

    value_head = ValueHead(embed_dim=256).to(device)

    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, model.parameters()))
        + list(value_head.parameters()),
        lr=cfg["rl"].get("lr", 1e-5),
        weight_decay=cfg["rl"].get("weight_decay", 1e-4),
    )

    env = SimEnv(cfg)
    tokenizer_cache = {}
    log = []
    rl_out_dir = out_dir / "rl"
    rl_out_dir.mkdir(parents=True, exist_ok=True)

    best_return = -float("inf")
    for epoch in range(1, cfg["rl"]["epochs"] + 1):
        epoch_returns = []

        for rollout_i in range(cfg["rl"].get("num_rollouts", 4)):
            buf = collect_rollout(model, env, cfg, device, tokenizer_cache)
            metrics = rl_update(model, value_head, buf, optimizer, cfg, device, bc_model)
            epoch_returns.append(metrics["mean_return"])
            buf.clear()

        mean_ret = float(np.mean(epoch_returns))
        row = {"epoch": epoch, "mean_return": round(mean_ret, 4), **{k: round(v, 5) for k, v in metrics.items()}}
        log.append(row)
        print(f"RL Epoch {epoch:3d} | mean_return {mean_ret:.4f} | "
              f"policy_loss {metrics['policy_loss']:.4f} | "
              f"entropy {metrics['entropy']:.4f} | "
              f"kl {metrics['kl_loss']:.4f}")

        if mean_ret > best_return:
            best_return = mean_ret
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, rl_out_dir / "best_rl.pt")
            print(f"  -> saved best RL checkpoint (mean_return={best_return:.4f})")

        if epoch % cfg["rl"].get("save_every", 20) == 0:
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, rl_out_dir / f"rl_epoch{epoch:04d}.pt")

    with open(rl_out_dir / "rl_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"[rl_trainer] Done. Best mean return: {best_return:.4f}")


if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rl_train(cfg)
