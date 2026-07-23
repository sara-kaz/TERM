#!/usr/bin/env python3
"""
Official Language-Table simulation task-success evaluation for a VERA checkpoint.

Mirrors language_table/eval/main.py: 5 task families × N episodes, success = env.succeeded.

Example (seed 123 checkpoint):
  CUDA_VISIBLE_DEVICES=0 python scripts/run_language_table_rollout_eval.py \\
      --checkpoint checkpoints/Language_table/seed123/best_sft_vera.pt

Smoke test:
  python scripts/run_language_table_rollout_eval.py --checkpoint ... --num_episodes 2 --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _reward_factories():
    from language_table.environments.rewards import block2absolutelocation
    from language_table.environments.rewards import block2block
    from language_table.environments.rewards import block2block_relative_location
    from language_table.environments.rewards import block2relativelocation
    from language_table.environments.rewards import separate_blocks

    return {
        "blocktoblock": block2block.BlockToBlockReward,
        "blocktoabsolutelocation": block2absolutelocation.BlockToAbsoluteLocationReward,
        "blocktoblockrelativelocation": (
            block2block_relative_location.BlockToBlockRelativeLocationReward
        ),
        "blocktorelativelocation": block2relativelocation.BlockToRelativeLocationReward,
        "separate": separate_blocks.SeparateBlocksReward,
    }


def _make_env(reward_factory, seed: int):
    from language_table.environments import blocks
    from language_table.environments import language_table

    return language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=reward_factory,
        control_frequency=10.0,
        seed=seed,
    )


_ORACLE_WARNED = False


def _try_valid_reset(env, policy_seed: int, max_attempts: int = 20):
    """Reset until oracle can motion-plan (same idea as language_table/eval/main.py)."""
    global _ORACLE_WARNED
    try:
        from language_table.environments.oracles import push_oracle_rrt_slowdown

        oracle_policy = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(
            env, use_ee_planner=True
        )
        for attempt in range(max_attempts):
            obs = env.reset()
            try:
                raw_state = env.compute_state()
                if oracle_policy.get_plan(raw_state):
                    return obs
            except Exception:
                pass
            env.seed(policy_seed + attempt + 1)
    except ImportError:
        if not _ORACLE_WARNED:
            print(
                "[lt_eval] tf_agents not installed — using plain env.reset() "
                "(pip install tf_agents for oracle-valid inits)",
                flush=True,
            )
            _ORACLE_WARNED = True
    return env.reset()


def run_episode(env, policy, ep_seed: int, max_steps: int) -> dict:
    import numpy as np

    env.seed(ep_seed)
    obs = _try_valid_reset(env, ep_seed)
    policy.reset()

    prev_r = 0.0
    steps = 0
    total_r = 0.0
    done = False

    while not done and steps < max_steps:
        action = policy.step(obs)
        obs, reward, done, _info = env.step(action)
        r = float(reward)
        delta = float(np.clip(-(r - prev_r) * 3.0, -0.5, 0.5))
        policy.note_step_reward(r, delta)
        total_r += r
        prev_r = r
        steps += 1

    success = bool(getattr(env, "succeeded", False))
    return {
        "success": success,
        "steps": steps,
        "total_reward": total_r,
    }


def main():
    parser = argparse.ArgumentParser(description="Language-Table rollout eval for VERA")
    parser.add_argument("--checkpoint", required=True, help="best_sft_vera.pt path")
    parser.add_argument(
        "--language_table_root",
        default=os.environ.get(
            "LANGUAGE_TABLE_ROOT",
            str(Path(__file__).resolve().parent.parent / "vendor" / "language-table"),
        ),
        help="Clone of google-research/language-table (for PYTHONPATH)",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Episodes per task family (official LT eval = 50)",
    )
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--eval_seed", type=int, default=123, help="Base eval seed (training seed)")
    parser.add_argument("--eval_log_dir", default=None)
    parser.add_argument(
        "--action_mode",
        choices=("hybrid", "discrete", "regression"),
        default="hybrid",
    )
    parser.add_argument(
        "--action_magnitude",
        type=float,
        default=0.05,
        help="Planar delta scale (training data median ‖Δ‖≈0.039)",
    )
    parser.add_argument(
        "--action_hold_steps",
        type=int,
        default=4,
        help="Repeat each predicted delta N sim steps (matches chunk_size)",
    )
    parser.add_argument(
        "--crop_factor",
        type=float,
        default=0.95,
        help="Central crop before resize (Language-Table eval default)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Grid-search action_mode/magnitude/hold on blocktoblock, then run full eval",
    )
    parser.add_argument(
        "--tune_episodes",
        type=int,
        default=8,
        help="Episodes per tune candidate (blocktoblock only)",
    )
    parser.add_argument("--smoke", action="store_true", help="Only blocktoblock, 2 episodes")
    args = parser.parse_args()

    lt_root = Path(args.language_table_root).resolve()
    if not (lt_root / "language_table").is_dir():
        sys.exit(
            f"[lt_eval] language_table package not found at {lt_root}\n"
            "  git clone https://github.com/google-research/language-table.git "
            "vendor/language-table"
        )

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(lt_root))
    sys.path.insert(0, str(repo_root))

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import torch
    from evaluation.vera_language_table_policy import VERALanguageTablePolicy

    if torch.cuda.is_available():
        device = f"cuda:{args.device}"
    else:
        device = "cpu"

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.is_file():
        sys.exit(f"[lt_eval] checkpoint not found: {ckpt}")

    log_dir = Path(args.eval_log_dir) if args.eval_log_dir else ckpt.parent / "lt_rollout_eval"
    log_dir.mkdir(parents=True, exist_ok=True)

    rewards = _reward_factories()

    action_mode = args.action_mode
    action_magnitude = args.action_magnitude
    action_hold_steps = args.action_hold_steps

    if args.tune:
        print("[lt_eval] ── Tuning on blocktoblock (quick grid) ──", flush=True)
        tune_candidates = [
            ("hybrid", 0.04, 4),
            ("hybrid", 0.05, 4),
            ("hybrid", 0.06, 4),
            ("hybrid", 0.08, 4),
            ("regression", 0.05, 4),
            ("regression", 0.08, 4),
            ("discrete", 0.05, 4),
            ("discrete", 0.08, 1),
        ]
        best = {"rate": -1.0, "cfg": None, "returns": -1.0}
        rf = rewards["blocktoblock"]
        for mode, mag, hold in tune_candidates:
            pol = VERALanguageTablePolicy(
                checkpoint=str(ckpt),
                device=device,
                action_mode=mode,
                action_magnitude=mag,
                action_hold_steps=hold,
                crop_factor=args.crop_factor,
            )
            succ, ret_sum = 0, 0.0
            for ep in range(args.tune_episodes):
                ep_seed = args.eval_seed * 1000 + ep
                env = _make_env(rf, seed=ep_seed)
                try:
                    res = run_episode(
                        env, pol, ep_seed=ep_seed, max_steps=args.max_episode_steps
                    )
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass
                succ += int(res["success"])
                ret_sum += res["total_reward"]
            rate = succ / max(args.tune_episodes, 1)
            print(
                f"  tune mode={mode:10s} mag={mag:.2f} hold={hold}  "
                f"success={rate*100:.1f}%  mean_return={ret_sum/args.tune_episodes:.3f}",
                flush=True,
            )
            if rate > best["rate"] or (rate == best["rate"] and ret_sum > best["returns"]):
                best = {
                    "rate": rate,
                    "returns": ret_sum,
                    "cfg": (mode, mag, hold),
                }
        if best["cfg"]:
            action_mode, action_magnitude, action_hold_steps = best["cfg"]
            print(
                f"[lt_eval] Best tune: mode={action_mode} mag={action_magnitude} "
                f"hold={action_hold_steps}  success={best['rate']*100:.1f}%",
                flush=True,
            )
        tune_path = log_dir / "tune_results.json"
        tune_path.write_text(
            json.dumps(
                {
                    "best": {
                        "action_mode": action_mode,
                        "action_magnitude": action_magnitude,
                        "action_hold_steps": action_hold_steps,
                        "success_rate": best["rate"],
                    },
                    "candidates": [
                        {"mode": m, "mag": g, "hold": h} for m, g, h in tune_candidates
                    ],
                },
                indent=2,
            )
        )

    policy = VERALanguageTablePolicy(
        checkpoint=str(ckpt),
        device=device,
        action_mode=action_mode,
        action_magnitude=action_magnitude,
        action_hold_steps=action_hold_steps,
        crop_factor=args.crop_factor,
    )
    if args.smoke:
        rewards = {"blocktoblock": rewards["blocktoblock"]}
        args.num_episodes = min(args.num_episodes, 2)

    print(
        f"[lt_eval] checkpoint={ckpt}\n"
        f"[lt_eval] device={device}  eval_seed={args.eval_seed}  "
        f"episodes_per_task={args.num_episodes}  max_steps={args.max_episode_steps}",
        flush=True,
    )

    results = {}
    per_episode = {}
    t0 = time.time()

    for task_name, reward_factory in rewards.items():
        successes = 0
        ep_logs = []
        print(f"\n[lt_eval] ── {task_name} ──", flush=True)

        for ep in range(args.num_episodes):
            ep_seed = args.eval_seed * 1000 + ep
            env = _make_env(reward_factory, seed=ep_seed)
            try:
                ep_res = run_episode(
                    env, policy, ep_seed=ep_seed, max_steps=args.max_episode_steps
                )
            finally:
                try:
                    env.close()
                except Exception:
                    pass

            successes += int(ep_res["success"])
            ep_logs.append(ep_res)
            status = "SUCCESS" if ep_res["success"] else "fail"
            print(
                f"  ep {ep+1:3d}/{args.num_episodes}: {status}  "
                f"steps={ep_res['steps']}  return={ep_res['total_reward']:.3f}",
                flush=True,
            )

        rate = successes / max(args.num_episodes, 1)
        results[task_name] = {
            "successes": successes,
            "episodes": args.num_episodes,
            "task_success_rate": rate,
        }
        per_episode[task_name] = ep_logs
        print(f"  => {task_name} task success: {rate * 100:.1f}% ({successes}/{args.num_episodes})")

    overall_success = sum(v["successes"] for v in results.values())
    overall_eps = sum(v["episodes"] for v in results.values())
    overall_rate = overall_success / max(overall_eps, 1)

    summary = {
        "checkpoint": str(ckpt),
        "eval_seed": args.eval_seed,
        "num_episodes_per_task": args.num_episodes,
        "max_episode_steps": args.max_episode_steps,
        "action_mode": action_mode,
        "action_magnitude": action_magnitude,
        "action_hold_steps": action_hold_steps,
        "crop_factor": args.crop_factor,
        "per_task": results,
        "overall_task_success_rate": overall_rate,
        "overall_successes": overall_success,
        "overall_episodes": overall_eps,
        "wall_time_s": round(time.time() - t0, 1),
        "per_episode": per_episode,
    }

    out_path = log_dir / "vera_language_table_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n[lt_eval] ── Language-Table task success (official 5-task benchmark) ──", flush=True)
    for task_name, vals in results.items():
        print(
            f"  {task_name:<32} {vals['task_success_rate'] * 100:6.2f}%  "
            f"({vals['successes']}/{vals['episodes']})",
            flush=True,
        )
    print(
        f"\n  OVERALL                         {overall_rate * 100:6.2f}%  "
        f"({overall_success}/{overall_eps})",
        flush=True,
    )
    print(f"\n[lt_eval] Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
