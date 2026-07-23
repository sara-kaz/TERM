"""
Evaluation Script
=================
Runs the trained VLA policy for N episodes and reports:
  - Mean episode return
  - Mean episode length
  - Success rate (episodes where total return > success_threshold)
  - Action distribution histogram

Usage
-----
  python -m evaluation.evaluate --config configs/config.yaml --checkpoint checkpoints/best_rl.pt
  python -m evaluation.evaluate --config configs/config.yaml --checkpoint checkpoints/best_sft.pt --use-sft
"""

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import clip

from models.vla_model import RLConditionedVLA
from envs.sim_env import SimEnv


def evaluate(cfg: dict, checkpoint_path: str, num_episodes: int = 20, deterministic: bool = True):
    device = cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

    model = RLConditionedVLA(
        num_actions=cfg["model"]["num_actions"],
        history_len=cfg["model"]["history_len"],
        fusion_layers=cfg["model"].get("fusion_layers", 6),
        fusion_heads=cfg["model"].get("fusion_heads", 8),
        dropout=0.0,
        freeze_clip=True,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[eval] Loaded checkpoint: {checkpoint_path}")

    import torchvision.transforms as Tv
    from PIL import Image as PImage

    img_size = cfg["data"].get("img_size", 224)
    transform = Tv.Compose([
        Tv.Resize((img_size, img_size)),
        Tv.ToTensor(),
        Tv.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                     std= [0.26862954, 0.26130258, 0.27577711]),
    ])

    env            = SimEnv(cfg)
    history_len    = cfg["model"]["history_len"]
    num_vis_frames = cfg["model"]["num_vis_frames"]
    num_actions    = cfg["model"]["num_actions"]
    max_steps      = cfg["rl"].get("max_episode_steps", 50)
    success_thresh = cfg.get("eval", {}).get("success_threshold", 1.0)
    tok_cache      = {}

    ep_returns, ep_lengths, successes = [], [], []
    all_actions = Counter()

    for ep in range(num_episodes):
        obs          = env.reset()
        frame_queue  = deque(maxlen=num_vis_frames)
        action_queue = deque([num_actions] * history_len, maxlen=history_len)
        reward_queue = deque([0.0]         * history_len, maxlen=history_len)

        instr = obs["instruction"]
        if instr not in tok_cache:
            tok_cache[instr] = clip.tokenize([instr])[0]

        total_reward = 0.0
        step = 0
        done = False

        while not done and step < max_steps:
            frame_t = transform(PImage.fromarray(obs["frame"]))
            frame_queue.append(frame_t)

            pad_needed    = num_vis_frames - len(frame_queue)
            padded_frames = [torch.zeros_like(frame_t)] * pad_needed + list(frame_queue)
            frames_tensor = torch.stack(padded_frames).unsqueeze(0).to(device)
            lang_tensor   = tok_cache[instr].unsqueeze(0).to(device)
            act_hist_t    = torch.tensor(list(action_queue), dtype=torch.long).unsqueeze(0).to(device)
            rew_hist_t    = torch.tensor(list(reward_queue), dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(frames_tensor, lang_tensor, act_hist_t, rew_hist_t)

            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                action = torch.multinomial(F.softmax(logits, dim=-1), 1).item()

            obs, reward, done, info = env.step(action)
            total_reward += reward
            action_queue.append(action)
            reward_queue.append(reward)
            all_actions[action] += 1
            step += 1

        ep_returns.append(total_reward)
        ep_lengths.append(step)
        successes.append(int(total_reward >= success_thresh))
        print(f"  Episode {ep+1:3d} | return={total_reward:.2f} | steps={step} | {'SUCCESS' if successes[-1] else 'fail'}")

    env.close()

    results = {
        "num_episodes":   num_episodes,
        "mean_return":    float(np.mean(ep_returns)),
        "std_return":     float(np.std(ep_returns)),
        "mean_length":    float(np.mean(ep_lengths)),
        "success_rate":   float(np.mean(successes)),
        "action_dist":    dict(sorted(all_actions.items())),
    }

    print("\n── Evaluation Results ──────────────────────────────")
    print(f"  Episodes     : {results['num_episodes']}")
    print(f"  Mean Return  : {results['mean_return']:.3f} ± {results['std_return']:.3f}")
    print(f"  Mean Length  : {results['mean_length']:.1f} steps")
    print(f"  Success Rate : {results['success_rate']*100:.1f}%")
    print(f"  Action Dist  : {results['action_dist']}")

    out_path = Path(checkpoint_path).parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Results saved to {out_path}")
    return results


if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes",   type=int, default=20)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, args.checkpoint, num_episodes=args.episodes, deterministic=not args.stochastic)
