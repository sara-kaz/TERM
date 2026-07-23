"""
VERA Online RL Fine-Tuning Trainer
====================================
Phase 2: roll out the VERA policy in an environment and optimise using
REINFORCE with a value baseline + KL regularisation against the BC checkpoint.

The RL update extends the SFT loss with:
  L_total = -E[log π(a|s) · Â_t]          ← REINFORCE (policy gradient)
           + 0.5 · (V(s) - G_t)²           ← value baseline (MSE)
           - entropy_coef · H[π]           ← entropy bonus  (exploration)
           + kl_coef    · KL(π ∥ π_BC)    ← KL penalty     (no forgetting)

Key adaptation for VLLA:
  The rollout collector now maintains a prev_action_idx and prev_reward
  that are passed to model.forward(), feeding Feedback Channel A (semantic)
  at every rollout step.

Usage
-----
  python -m training.rl_trainer_vera --config configs/config.yaml
"""

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from models.vera_model import VERAModel, RMSNorm
from envs.sim_env import make_env


# ── Value head ────────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    """
    Scalar state-value estimate from the CLS token representation.

    Uses RMSNorm (matching VERA's backbone) rather than LayerNorm to keep
    normalisation consistent with the policy model's internal activations.
    SiLU matches the SwiGLU gating used throughout the LLaMA fusion stack.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1, bias=False),
        )

    def forward(self, cls_features: torch.Tensor) -> torch.Tensor:
        """cls_features: (B, D) → values: (B,)"""
        return self.net(cls_features).squeeze(-1)


# ── Rollout buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores one batch of episode transitions for a policy-gradient update."""

    def __init__(self):
        self.frames:           List[torch.Tensor]          = []
        self.lang_tokens:      List[torch.Tensor]          = []
        self.action_hists:     List[torch.Tensor]          = []
        self.reward_hists:     List[torch.Tensor]          = []
        self.action_vec_hists: List[Optional[torch.Tensor]]= []  # (H, action_dim) or None
        self.prev_actions:     List[torch.Tensor]          = []
        self.prev_rewards_fb:  List[torch.Tensor]          = []
        self.state_deltas:     List[torch.Tensor]          = []
        self.actions:          List[int]                   = []
        self.rewards:          List[float]                 = []
        self.dones:            List[bool]                  = []

    def add(self, frame, lang_tok, act_hist, rew_hist, act_vec_hist,
            prev_a, prev_r, state_delta, action, reward, done):
        self.frames.append(frame)
        self.lang_tokens.append(lang_tok)
        self.action_hists.append(act_hist)
        self.reward_hists.append(rew_hist)
        self.action_vec_hists.append(act_vec_hist)   # may be None
        self.prev_actions.append(prev_a)
        self.prev_rewards_fb.append(prev_r)
        self.state_deltas.append(state_delta)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def extend(self, other: "RolloutBuffer"):
        """Merge another buffer into this one (for multi-rollout accumulation)."""
        self.frames.extend(other.frames)
        self.lang_tokens.extend(other.lang_tokens)
        self.action_hists.extend(other.action_hists)
        self.reward_hists.extend(other.reward_hists)
        self.action_vec_hists.extend(other.action_vec_hists)
        self.prev_actions.extend(other.prev_actions)
        self.prev_rewards_fb.extend(other.prev_rewards_fb)
        self.state_deltas.extend(other.state_deltas)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.dones.extend(other.dones)

    def clear(self):
        self.__init__()

    def compute_returns(self, gamma: float = 0.99) -> torch.Tensor:
        """Discounted returns with episode boundary resets, then standardised."""
        G, returns = 0.0, []
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0.0
            G = r + gamma * G
            returns.insert(0, G)
        ret = torch.tensor(returns, dtype=torch.float32)
        if ret.std() > 1e-8:
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)
        return ret


# ── Rollout collector ─────────────────────────────────────────────────────────

def collect_rollout(
    model:            VERAModel,
    env,
    cfg:              dict,
    device:           str,
    tokenizer_cache:  dict,
) -> RolloutBuffer:
    """Execute the policy for one episode and store all transitions."""
    import torchvision.transforms as Tv
    from PIL import Image as PILImage

    buf            = RolloutBuffer()
    history_len    = cfg["model"]["history_len"]
    num_vis_frames = cfg["model"]["num_vis_frames"]
    num_actions    = cfg["model"]["num_actions"]
    img_size       = cfg["data"].get("img_size", 224)
    max_steps      = cfg["rl"].get("max_episode_steps", 50)

    transform = Tv.Compose([
        Tv.Resize((img_size, img_size)),
        Tv.ToTensor(),
        Tv.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                     std= [0.26862954, 0.26130258, 0.27577711]),
    ])

    action_dim   = cfg["model"].get("action_dim", 4)
    null_vec     = np.zeros(action_dim, dtype=np.float32)  # padding vector at t=0

    obs          = env.reset()
    frame_q      = deque(maxlen=num_vis_frames)
    action_q     = deque([num_actions] * history_len, maxlen=history_len)
    reward_q     = deque([0.0]         * history_len, maxlen=history_len)
    action_vec_q = deque([null_vec.copy() for _ in range(history_len)],
                         maxlen=history_len)   # (H, action_dim) rolling buffer
    prev_action  = num_actions
    prev_rew_fb  = 0.0

    # Tokenise instruction
    inst = obs["instruction"]
    if inst not in tokenizer_cache:
        tokenizer_cache[inst] = clip.tokenize([inst])[0]
    lang_tok = tokenizer_cache[inst]

    done, step = False, 0
    while not done and step < max_steps:
        # Build frame tensor
        frame_t  = transform(PILImage.fromarray(obs["frame"]))              # (3,H,W)
        frame_q.append(frame_t)
        pad      = num_vis_frames - len(frame_q)
        frames_t = torch.stack([torch.zeros_like(frame_t)] * pad + list(frame_q))

        frames_in   = frames_t.unsqueeze(0).to(device)
        lang_in     = lang_tok.unsqueeze(0).to(device)
        act_hist_in = torch.tensor(list(action_q),     dtype=torch.long).unsqueeze(0).to(device)
        rew_hist_in = torch.tensor(list(reward_q),     dtype=torch.float32).unsqueeze(0).to(device)
        prev_a_in   = torch.tensor([prev_action],      dtype=torch.long).to(device)
        prev_r_in   = torch.tensor([prev_rew_fb],      dtype=torch.float32).to(device)

        # Low-level action vector history — (1, H, action_dim)
        vec_hist_np  = np.stack(list(action_vec_q), axis=0)                # (H, action_dim)
        act_vec_in   = torch.tensor(vec_hist_np, dtype=torch.float32).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            out = model(frames_in, lang_in, act_hist_in, rew_hist_in,
                        prev_a_in, prev_r_in,
                        action_vec_hist=act_vec_in)
        action = torch.multinomial(F.softmax(out["logits"], dim=-1), 1).item()

        obs, reward, done, info = env.step(action)

        # Extract signed distance delta and the executed continuous action vector
        _delta      = info.get("dist_delta") if isinstance(info, dict) else None
        state_delta_t = torch.tensor(
            [_delta if _delta is not None else 0.0], dtype=torch.float32
        )

        # The env may expose the executed low-level vector (MetaWorld codebook entry).
        # Fall back to zeros if not available (BabyAI, dummy).
        executed_vec = info.get("action_vector", null_vec) if isinstance(info, dict) else null_vec
        executed_vec = np.asarray(executed_vec, dtype=np.float32).flatten()
        # Truncate or zero-pad to action_dim
        if len(executed_vec) >= action_dim:
            executed_vec = executed_vec[:action_dim]
        else:
            executed_vec = np.concatenate(
                [executed_vec, np.zeros(action_dim - len(executed_vec), dtype=np.float32)]
            )

        buf.add(
            frame        = frames_t.cpu(),
            lang_tok     = lang_tok,
            act_hist     = act_hist_in.squeeze(0).cpu(),
            rew_hist     = rew_hist_in.squeeze(0).cpu(),
            act_vec_hist = act_vec_in.squeeze(0).cpu(),     # (H, action_dim)
            prev_a       = prev_a_in.squeeze(0).cpu(),
            prev_r       = prev_r_in.squeeze(0).cpu(),
            state_delta  = state_delta_t.cpu(),
            action       = action,
            reward       = reward,
            done         = done,
        )

        action_q.append(action)
        reward_q.append(reward)
        action_vec_q.append(executed_vec)
        prev_action = action
        prev_rew_fb = reward
        step += 1

    return buf


# ── RL update step ─────────────────────────────────────────────────────────────

def rl_update(
    model:      VERAModel,
    value_head: ValueHead,
    buf:        RolloutBuffer,
    optimizer:  torch.optim.Optimizer,
    cfg:        dict,
    device:     str,
    bc_model:   Optional[VERAModel] = None,
) -> dict:
    """Single REINFORCE + value-baseline update over one rollout buffer."""
    model.train()
    value_head.train()

    returns = buf.compute_returns(gamma=cfg["rl"].get("gamma", 0.99)).to(device)

    frames       = torch.stack(buf.frames).to(device)           # (N, T, 3, H, W)
    lang_tokens  = torch.stack(buf.lang_tokens).to(device)      # (N, 77)
    act_hist     = torch.stack(buf.action_hists).to(device)     # (N, H)
    rew_hist     = torch.stack(buf.reward_hists).to(device)     # (N, H)
    prev_actions = torch.stack(buf.prev_actions).to(device)     # (N, 1) or (N,)
    prev_rewards = torch.stack(buf.prev_rewards_fb).to(device)  # (N, 1) or (N,)
    state_deltas = torch.stack(buf.state_deltas).to(device)     # (N, 1)
    actions      = torch.tensor(buf.actions, dtype=torch.long, device=device)

    # Stack low-level action vector history.
    # buf.action_vec_hists entries are (H, action_dim) tensors (never None at this
    # point — collect_rollout always stores a zero tensor when the env doesn't
    # expose a codebook vector).  We guard with a None-check anyway for safety.
    if any(v is None for v in buf.action_vec_hists):
        _action_dim  = cfg["model"].get("action_dim", 4)
        _history_len = act_hist.size(1)
        act_vec_hist = torch.zeros(
            len(buf.action_vec_hists), _history_len, _action_dim, device=device
        )
        for _i, _v in enumerate(buf.action_vec_hists):
            if _v is not None:
                act_vec_hist[_i] = _v.to(device)
    else:
        act_vec_hist = torch.stack(buf.action_vec_hists).to(device)  # (N, H, action_dim)

    # Flatten (N,1) → (N,)
    prev_actions = prev_actions.view(-1).long()
    prev_rewards = prev_rewards.view(-1).float()
    state_deltas = state_deltas.view(-1).float()

    # Forward pass (with grad)
    out = model(frames, lang_tokens, act_hist, rew_hist, prev_actions, prev_rewards,
                state_delta=state_deltas, action_vec_hist=act_vec_hist)
    logits     = out["logits"]                                  # (N, A)
    cls_feat   = out["cls_features"]                           # (N, D)

    # Value estimate (no gradient back through policy for value loss)
    values    = value_head(cls_feat.detach())                   # (N,)
    advantage = returns - values.detach()                       # Â_t = G_t - V(s_t)

    # REINFORCE policy gradient
    log_probs   = F.log_softmax(logits, dim=-1)
    chosen_logp = log_probs[torch.arange(len(actions)), actions]
    policy_loss = -(chosen_logp * advantage).mean()

    # Value baseline MSE
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus
    probs   = F.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(-1).mean()

    # KL penalty against BC checkpoint (prevents catastrophic forgetting)
    kl_loss = torch.tensor(0.0, device=device)
    if bc_model is not None:
        bc_model.eval()
        with torch.no_grad():
            bc_out    = bc_model(frames, lang_tokens, act_hist, rew_hist,
                                 prev_actions, prev_rewards,
                                 state_delta=state_deltas,
                                 action_vec_hist=act_vec_hist)
            bc_probs  = F.softmax(bc_out["logits"], dim=-1)
        kl_loss = F.kl_div(log_probs, bc_probs, reduction="batchmean")

    vf_coef      = cfg["rl"].get("vf_coef", 0.5)
    entropy_coef = cfg["rl"].get("entropy_coef", 0.01)
    # Use warmed-up KL coef if set by the training loop, else fall back to config value
    kl_coef      = cfg["rl"].get("_kl_coef_effective", cfg["rl"].get("kl_coef", 0.1))

    total_loss = (policy_loss
                  + vf_coef      * value_loss
                  - entropy_coef * entropy
                  + kl_coef      * kl_loss)

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(
        list(p for p in model.parameters() if p.requires_grad)
        + list(value_head.parameters()),
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

    device  = cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["training"]["output_dir"])
    print(f"[rl_vera] device = {device}")

    vera_cfg = cfg.get("vera", {})

    def make_model():
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
            use_lang_feedback=vera_cfg.get("use_lang_feedback", True),
            use_action_lang_feedback=vera_cfg.get("use_action_lang_feedback"),
            use_temporal_history=vera_cfg.get("use_temporal_history", True),
            use_reward_gate=vera_cfg.get("use_reward_gate", True),
            use_consequence_token=vera_cfg.get("use_consequence_token", True),
            action_dim=cfg["model"].get("action_dim", 4),   # 4=MetaWorld, 2=Language-Table, 7=CALVIN
            action_vocab=vera_cfg.get("action_vocab"),      # None → use built-in vocabulary
            proprio_dim=cfg["model"].get("proprio_dim", 0),
            chunk_size=cfg["model"].get("chunk_size", 1),
        )

    model = make_model().to(device)

    # Load BC checkpoint
    bc_ckpt_path = out_dir / "best_sft_vera.pt"
    bc_model     = None
    if bc_ckpt_path.exists():
        ckpt = torch.load(bc_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[rl_vera] Loaded BC checkpoint from {bc_ckpt_path}")
        bc_model = make_model().to(device)
        bc_model.load_state_dict(ckpt["model_state"])
        for p in bc_model.parameters():
            p.requires_grad = False
    else:
        print("[rl_vera] Warning: no BC checkpoint — training RL from scratch.")

    value_head = ValueHead(d_model=cfg["model"].get("d_model", 256)).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(value_head.parameters()),
        lr=cfg["rl"].get("lr", 1e-5),
        weight_decay=cfg["rl"].get("weight_decay", 1e-4),
    )

    num_epochs = cfg["rl"]["epochs"]
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=cfg["rl"].get("lr", 1e-5) * 0.1
    )

    # KL warmup: skip KL penalty for the first N epochs so the policy can first
    # adapt to the simulation domain before regularisation pulls it back toward
    # the SFT checkpoint (which gets 0% task success in sim).
    kl_warmup_epochs = cfg["rl"].get("kl_warmup_epochs", 20)

    env             = make_env(cfg)
    tokenizer_cache = {}
    rl_out_dir      = out_dir / "rl"
    rl_out_dir.mkdir(parents=True, exist_ok=True)

    log, best_return, best_success = [], -float("inf"), -1.0
    cumulative_steps = 0
    max_ep_steps     = cfg["rl"].get("max_episode_steps", 50)
    num_rollouts     = cfg["rl"].get("num_rollouts", 4)
    success_thr      = cfg.get("eval", {}).get("success_threshold", 1.0)

    for epoch in range(1, num_epochs + 1):
        epoch_returns, epoch_successes, epoch_lengths = [], [], []

        # ── Collect ALL rollouts into one combined buffer, then do ONE update ──
        # Previously: one update per episode → high-variance gradients from ~50 steps.
        # Now: one update from all num_rollouts episodes → ~200 steps, much lower variance.
        combined_buf = RolloutBuffer()
        for _ in range(num_rollouts):
            buf      = collect_rollout(model, env, cfg, device, tokenizer_cache)
            ep_steps = len(buf.actions)
            cumulative_steps += ep_steps
            ep_return = sum(buf.rewards)
            epoch_returns.append(ep_return)
            epoch_successes.append(int(ep_return >= success_thr))
            epoch_lengths.append(ep_steps)
            combined_buf.extend(buf)
            buf.clear()

        # Apply KL warmup: scale kl_coef linearly from 0 → full over kl_warmup_epochs
        warmup_scale = min(1.0, (epoch - 1) / max(1, kl_warmup_epochs))
        base_kl      = cfg["rl"].get("kl_coef", 0.1)
        cfg["rl"]["_kl_coef_effective"] = base_kl * warmup_scale

        metrics = rl_update(model, value_head, combined_buf, optimizer, cfg, device, bc_model)
        combined_buf.clear()

        scheduler.step()

        mean_ret     = float(np.mean(epoch_returns))
        mean_success = float(np.mean(epoch_successes))
        mean_len     = float(np.mean(epoch_lengths))

        row = {
            "epoch":            epoch,
            "cumulative_steps": cumulative_steps,
            "mean_return":      round(mean_ret, 4),
            "success_rate":     round(mean_success, 4),
            "mean_ep_length":   round(mean_len, 2),
            "kl_warmup_scale":  round(warmup_scale, 3),
            "lr":               round(scheduler.get_last_lr()[0], 8),
            **{k: round(v, 5) for k, v in metrics.items()},
        }
        log.append(row)

        print(f"RL Epoch {epoch:3d} | steps {cumulative_steps:7d} | "
              f"return {mean_ret:.4f} | success {mean_success*100:.1f}% | "
              f"policy {metrics['policy_loss']:.4f} | "
              f"entropy {metrics['entropy']:.4f} | "
              f"kl {metrics['kl_loss']:.4f} (scale={warmup_scale:.2f}) | "
              f"lr {scheduler.get_last_lr()[0]:.2e}")

        # Save best-return checkpoint
        if mean_ret > best_return:
            best_return = mean_ret
            torch.save({
                "epoch":            epoch,
                "cumulative_steps": cumulative_steps,
                "model_state":      model.state_dict(),
                "success_rate":     mean_success,
            }, rl_out_dir / "best_rl_vera.pt")
            print(f"  ✓ best-return checkpoint saved (return={best_return:.4f}, "
                  f"success={mean_success*100:.1f}% @ {cumulative_steps} steps)")

        # Save best-success checkpoint separately — this is what the paper reports
        if mean_success > best_success:
            best_success = mean_success
            torch.save({
                "epoch":            epoch,
                "cumulative_steps": cumulative_steps,
                "model_state":      model.state_dict(),
                "success_rate":     mean_success,
                "mean_return":      mean_ret,
            }, rl_out_dir / "best_success_vera.pt")
            print(f"  ★ best-success checkpoint saved (success={best_success*100:.1f}% "
                  f"@ epoch {epoch}, {cumulative_steps} steps)")

        if epoch % cfg["rl"].get("save_every", 20) == 0:
            torch.save({
                "epoch":            epoch,
                "cumulative_steps": cumulative_steps,
                "model_state":      model.state_dict(),
                "success_rate":     mean_success,
            }, rl_out_dir / f"rl_vera_epoch{epoch:04d}.pt")

        # Flush log every 5 epochs so results survive Colab disconnects
        if epoch % 5 == 0:
            with open(rl_out_dir / "rl_vera_log.json", "w") as f:
                json.dump(log, f, indent=2)

    with open(rl_out_dir / "rl_vera_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Compact sample-efficiency CSV for plotting
    csv_lines = ["epoch,cumulative_steps,mean_return,success_rate,lr"]
    for row in log:
        csv_lines.append(
            f"{row['epoch']},{row['cumulative_steps']},"
            f"{row['mean_return']},{row['success_rate']},{row.get('lr','')}"
        )
    (rl_out_dir / "sample_efficiency.csv").write_text("\n".join(csv_lines))

    print(f"[rl_vera] Done. Best return: {best_return:.4f} | "
          f"Best success: {best_success*100:.1f}% | "
          f"Total env steps: {cumulative_steps:,}")


if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rl_train(cfg)
