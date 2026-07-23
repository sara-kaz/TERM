"""
VLLA Evaluation Script
======================
Runs the trained VLLA policy for N episodes and reports:
  - Mean episode return  ± std
  - Success rate
  - Mean episode length
  - Action distribution histogram
  - Per-step entropy (policy confidence over time)
  - Sample efficiency: return vs cumulative env steps

Usage
-----
  python -m evaluation.evaluate_vera \\
      --config configs/config.yaml \\
      --checkpoint checkpoints/rl/best_rl_vera.pt \\
      --episodes 100 \\
      --seeds 3

Ablation comparison (runs all ablations back-to-back):
  python -m evaluation.evaluate_vera \\
      --config configs/config.yaml \\
      --checkpoint checkpoints/best_sft_vera.pt \\
      --ablation-table
"""

import argparse
import json
import copy
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import clip

from models.vera_model import VERAModel
from envs.sim_env import SimEnv


# ── helpers ───────────────────────────────────────────────────────────────────

def build_vera_from_cfg(cfg: dict, device: str) -> VERAModel:
    vera_cfg = cfg.get("vera", {})
    return VERAModel(
        num_actions           = cfg["model"]["num_actions"],
        history_len           = cfg["model"]["history_len"],
        num_vis_frames        = cfg["model"].get("num_vis_frames", 3),
        fusion_layers         = cfg["model"].get("fusion_layers", 6),
        fusion_heads          = cfg["model"].get("fusion_heads", 8),
        d_model               = cfg["model"].get("d_model", 256),
        d_ff_scale            = cfg["model"].get("d_ff_scale", 4),
        dropout               = 0.0,          # no dropout at eval
        freeze_clip           = cfg["model"].get("freeze_clip", True),
        unfreeze_clip_vision  = cfg["model"].get("unfreeze_clip_vision", False),
        use_lang_feedback     = vera_cfg.get("use_lang_feedback", True),
        use_action_lang_feedback = vera_cfg.get("use_action_lang_feedback"),
        use_temporal_history  = vera_cfg.get("use_temporal_history", True),
        use_reward_gate       = vera_cfg.get("use_reward_gate", True),
        use_consequence_token = vera_cfg.get("use_consequence_token", True),
        action_dim            = cfg["model"].get("action_dim", 4),
        action_vocab          = vera_cfg.get("action_vocab"),
        proprio_dim           = cfg["model"].get("proprio_dim", 0),
        chunk_size            = cfg["model"].get("chunk_size", 1),
        use_step_conditioning = vera_cfg.get("use_step_conditioning", False),
        step_horizon          = vera_cfg.get(
            "step_horizon", cfg.get("rl", {}).get("max_episode_steps", 96)
        ),
    ).to(device)


def load_checkpoint(model: VERAModel, checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_checkpoint] missing keys ({len(missing)}): {missing[:6]}{'…' if len(missing) > 6 else ''}", flush=True)
    if unexpected:
        print(f"[load_checkpoint] unexpected keys ({len(unexpected)}): {unexpected[:6]}{'…' if len(unexpected) > 6 else ''}", flush=True)
    model.eval()


def _make_transform(img_size: int):
    import torchvision.transforms as Tv
    return Tv.Compose([
        Tv.Resize((img_size, img_size)),
        Tv.ToTensor(),
        Tv.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                     std= [0.26862954, 0.26130258, 0.27577711]),
    ])


# ── single-run evaluation ─────────────────────────────────────────────────────

def evaluate_once(
    model:          VERAModel,
    cfg:            dict,
    num_episodes:   int  = 50,
    deterministic:  bool = True,
    seed:           int  = 0,
) -> dict:
    """
    Roll out the VLLA policy for `num_episodes` episodes.

    Returns a dict with:
      mean_return, std_return, success_rate, mean_length,
      action_dist, mean_entropy, returns_per_ep, lengths_per_ep,
      entropy_per_ep (list of per-episode mean step entropy)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = next(model.parameters()).device
    transform = _make_transform(cfg["data"].get("img_size", 224))

    from PIL import Image as PILImage
    env           = SimEnv(cfg)
    history_len   = cfg["model"]["history_len"]
    num_vis       = cfg["model"].get("num_vis_frames", 3)
    num_actions   = cfg["model"]["num_actions"]
    max_steps     = cfg["rl"].get("max_episode_steps", 50)
    success_thr   = cfg.get("eval", {}).get("success_threshold", 1.0)
    tok_cache     = {}

    ep_returns, ep_lengths, ep_successes, ep_entropies = [], [], [], []
    all_action_counts = Counter()

    for ep in range(num_episodes):
        obs         = env.reset()
        frame_q     = deque(maxlen=num_vis)
        action_q    = deque([num_actions] * history_len, maxlen=history_len)
        reward_q    = deque([0.0]         * history_len, maxlen=history_len)
        ep_r_max    = 1e-6
        skip_reward = False
        prev_delta  = 0.0

        instr = obs["instruction"]
        if instr not in tok_cache:
            tok_cache[instr] = clip.tokenize([instr])[0]

        total_r, step = 0.0, 0
        step_entropies = []
        done = False

        while not done and step < max_steps:
            frame_t = transform(PILImage.fromarray(obs["frame"]))
            frame_q.append(frame_t)
            pad = num_vis - len(frame_q)
            frames_t = torch.stack([torch.zeros_like(frame_t)] * pad + list(frame_q))

            frames_in   = frames_t.unsqueeze(0).to(device)
            lang_in     = tok_cache[instr].unsqueeze(0).to(device)
            act_hist_in = torch.tensor(list(action_q), dtype=torch.long).unsqueeze(0).to(device)
            rew_arr = np.array(list(reward_q), dtype=np.float32)
            r_max = max(float(rew_arr.max()), ep_r_max, 1e-6)
            rew_hist_in = torch.tensor(
                np.clip(rew_arr / r_max, 0.0, 1.0), dtype=torch.float32
            ).unsqueeze(0).to(device)
            delta_in    = torch.tensor([prev_delta],  dtype=torch.float32).to(device)

            with torch.no_grad():
                out    = model(frames_in, lang_in, act_hist_in, rew_hist_in,
                               state_delta=delta_in)
                logits = out["logits"]                          # (1, A)
                probs  = F.softmax(logits, dim=-1)
                ent    = -(probs * probs.log().clamp(min=-20)).sum(-1).item()
                step_entropies.append(ent)

            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                action = torch.multinomial(probs, 1).item()

            obs, reward, done, info = env.step(action)
            _delta = info.get("dist_delta", 0.0) if isinstance(info, dict) else 0.0

            total_r += reward
            action_q.append(action)
            if not skip_reward:
                ep_r_max = max(ep_r_max, float(reward), 1e-6)
                reward_q.append(float(np.clip(reward / ep_r_max, 0.0, 1.0)))
            skip_reward = False
            all_action_counts[action] += 1
            prev_delta  = _delta
            step       += 1

        ep_returns.append(total_r)
        ep_lengths.append(step)
        ep_successes.append(int(total_r >= success_thr))
        ep_entropies.append(float(np.mean(step_entropies)) if step_entropies else 0.0)

    env.close()

    return {
        "mean_return":    float(np.mean(ep_returns)),
        "std_return":     float(np.std(ep_returns)),
        "success_rate":   float(np.mean(ep_successes)),
        "mean_length":    float(np.mean(ep_lengths)),
        "mean_entropy":   float(np.mean(ep_entropies)),
        "action_dist":    dict(sorted(all_action_counts.items())),
        "returns_per_ep": ep_returns,
        "lengths_per_ep": ep_lengths,
        "successes":      ep_successes,
    }


# ── multi-seed evaluation ─────────────────────────────────────────────────────

def evaluate_multi_seed(
    cfg:             dict,
    checkpoint_path: str,
    num_episodes:    int = 50,
    seeds:           int = 3,
    deterministic:   bool = True,
) -> dict:
    """
    Runs evaluate_once over `seeds` different random seeds and aggregates.
    Returns: mean ± std across seeds for the key metrics.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_vera_from_cfg(cfg, device)
    load_checkpoint(model, checkpoint_path, device)

    all_results = []
    for s in range(seeds):
        print(f"  Seed {s+1}/{seeds} ...", end=" ", flush=True)
        r = evaluate_once(model, cfg, num_episodes=num_episodes,
                          deterministic=deterministic, seed=s * 42)
        all_results.append(r)
        print(f"return={r['mean_return']:.3f}  success={r['success_rate']*100:.1f}%")

    keys = ["mean_return", "success_rate", "mean_length", "mean_entropy"]
    aggregated = {}
    for k in keys:
        vals = [r[k] for r in all_results]
        aggregated[f"{k}_mean"] = float(np.mean(vals))
        aggregated[f"{k}_std"]  = float(np.std(vals))
    aggregated["num_seeds"]    = seeds
    aggregated["num_episodes"] = num_episodes
    aggregated["per_seed"]     = all_results
    return aggregated


# ── ablation comparison table ─────────────────────────────────────────────────

ABLATION_PRESETS = {
    "Full VLLA (ours)": {},
    "A — no lang feedback":     {"use_action_lang_feedback": False, "use_consequence_token": False},
    "B — no temporal history":  {"use_temporal_history": False},
    "C — no reward gate":       {"use_reward_gate": False},
    "D — no alignment loss":    {},   # handled via alignment_loss_coef=0 in training, not eval
    "F — action token only":    {"use_consequence_token": False},
    "G — consequence only (no action lang)": {"use_action_lang_feedback": False, "use_consequence_token": True},
    "E — minimal (no lang, no hist)": {
        "use_action_lang_feedback": False,
        "use_consequence_token": False,
        "use_temporal_history": False,
    },
}


def run_ablation_table(
    cfg:             dict,
    checkpoint_path: str,
    num_episodes:    int = 50,
    seeds:           int = 3,
):
    """
    Loads the same checkpoint for each ablation preset and evaluates.
    Prints a markdown table and saves results to JSON.
    """
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    for name, overrides in ABLATION_PRESETS.items():
        print(f"\n── Ablation: {name} ──")
        ablation_cfg = copy.deepcopy(cfg)
        for k, v in overrides.items():
            ablation_cfg["vera"][k] = v

        model = build_vera_from_cfg(ablation_cfg, device)
        # Load checkpoint — some weights may not match ablation (expected: missing keys OK)
        try:
            load_checkpoint(model, checkpoint_path, device)
        except RuntimeError as e:
            print(f"  [warn] partial load: {e}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state, strict=False)
            model.eval()

        seed_returns, seed_successes, seed_entropies = [], [], []
        for s in range(seeds):
            r = evaluate_once(model, ablation_cfg, num_episodes=num_episodes,
                              deterministic=True, seed=s * 42)
            seed_returns.append(r["mean_return"])
            seed_successes.append(r["success_rate"])
            seed_entropies.append(r["mean_entropy"])
            print(f"  Seed {s+1}: return={r['mean_return']:.3f}  "
                  f"success={r['success_rate']*100:.1f}%")

        results[name] = {
            "return":  f"{np.mean(seed_returns):.3f} ± {np.std(seed_returns):.3f}",
            "success": f"{np.mean(seed_successes)*100:.1f} ± {np.std(seed_successes)*100:.1f}%",
            "entropy": f"{np.mean(seed_entropies):.3f} ± {np.std(seed_entropies):.3f}",
            "raw": {
                "returns":   seed_returns,
                "successes": seed_successes,
                "entropies": seed_entropies,
            },
        }

    # ── print markdown table ──
    print("\n\n## Ablation Results\n")
    header = f"| {'Method':<40} | {'Mean Return':>14} | {'Success Rate':>16} | {'Policy Entropy':>15} |"
    sep    = f"|{'-'*42}|{'-'*16}|{'-'*18}|{'-'*17}|"
    print(header)
    print(sep)
    for name, vals in results.items():
        marker = " *" if name.startswith("Full") else ""
        print(f"| {name+marker:<40} | {vals['return']:>14} | {vals['success']:>16} | {vals['entropy']:>15} |")

    print("\n* = proposed method")
    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import yaml
    parser = argparse.ArgumentParser(description="Evaluate a VLLA checkpoint")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--episodes",    type=int, default=50)
    parser.add_argument("--seeds",       type=int, default=3,
                        help="Number of random seeds to average over")
    parser.add_argument("--stochastic",  action="store_true",
                        help="Sample from policy instead of argmax")
    parser.add_argument("--ablation-table", action="store_true",
                        help="Run all ablation presets and print comparison table")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.checkpoint).parent
    deterministic = not args.stochastic

    if args.ablation_table:
        print("Running ablation comparison table …")
        results = run_ablation_table(cfg, args.checkpoint,
                                     num_episodes=args.episodes, seeds=args.seeds)
        out_path = out_dir / "ablation_table.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")

    else:
        print(f"Evaluating VLLA over {args.seeds} seeds × {args.episodes} episodes …")
        results = evaluate_multi_seed(cfg, args.checkpoint,
                                      num_episodes=args.episodes,
                                      seeds=args.seeds,
                                      deterministic=deterministic)

        print("\n── Evaluation Results ──────────────────────────────")
        for k in ["mean_return", "success_rate", "mean_length", "mean_entropy"]:
            mu  = results[f"{k}_mean"]
            std = results[f"{k}_std"]
            print(f"  {k:<20}: {mu:.4f} ± {std:.4f}")

        out_path = out_dir / "vera_eval_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
