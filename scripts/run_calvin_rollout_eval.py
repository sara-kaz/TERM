#!/usr/bin/env python3
"""
Official CALVIN long-horizon rollout evaluation for a trained VERA checkpoint.

Uses mees/calvin evaluate_policy (1000 sequences by default) with VERACalvinPolicy.

Example (full VERA seed 123 peak checkpoint):
  CUDA_VISIBLE_DEVICES=0 python scripts/run_calvin_rollout_eval.py \\
      --checkpoint checkpoints/calvin_core6_ltdev/full_vera/seed123/best_sft_vera.pt \\
      --dataset_path "$HOME/calvin_task_D/task_D_D" \\
      --calvin_root ~/work/calvin

Smoke test (10 sequences):
  python scripts/run_calvin_rollout_eval.py ... --num_sequences 10 --debug
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="CALVIN rollout eval for VERA")
    parser.add_argument("--checkpoint", required=True, help="best_sft_vera.pt path")
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="CALVIN task_D_D root (must contain validation/)",
    )
    parser.add_argument(
        "--calvin_root",
        default=os.environ.get("CALVIN_ROOT", str(Path.home() / "work" / "calvin")),
        help="Clone of https://github.com/mees/calvin (with calvin_env submodule)",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=1000,
        help="Number of evaluation chains (official CALVIN = 1000)",
    )
    parser.add_argument("--eval_log_dir", default=None, help="Output dir for results.json")
    parser.add_argument("--debug", action="store_true", help="Visualize first subtasks")
    parser.add_argument(
        "--action_mode",
        choices=("hybrid", "discrete", "regression", "continuous"),
        default="hybrid",
        help="continuous=full 7-DoF regression (best for embodied); "
        "hybrid=14-way argmax + regression scale; "
        "discrete=argmax only; regression=legacy tanh head",
    )
    parser.add_argument(
        "--action_magnitude",
        type=float,
        default=0.45,
        help="Peak |rel_action| on dominant DoF for discrete/hybrid modes (training scale ~0.3-0.5)",
    )
    parser.add_argument(
        "--action_hold_steps",
        type=int,
        default=30,
        help="Repeat each predicted action N steps before re-inference (CALVIN MCIL uses 30)",
    )
    parser.add_argument(
        "--discrete_actions",
        action="store_true",
        help="Deprecated: same as --action_mode discrete",
    )
    parser.add_argument(
        "--reset_history_each_step",
        action="store_true",
        help="Ignore model action/reward history each step (closer to 58%% offline, not 97%%)",
    )
    parser.add_argument(
        "--lang_goal",
        choices=("task_key", "natural"),
        default="task_key",
        help="task_key=subtask id (matches CALVIN lang_ann training); "
        "natural=validation.yaml sentence (train/eval mismatch)",
    )
    args = parser.parse_args()

    calvin_root = Path(args.calvin_root).resolve()
    agent_dir = calvin_root / "calvin_models" / "calvin_agent"
    if not agent_dir.is_dir():
        sys.exit(
            f"[calvin] calvin_agent not found at {agent_dir}\n"
            "  git clone --recurse-submodules https://github.com/mees/calvin.git\n"
            "  cd calvin && git submodule update --init --recursive"
        )

    val_dir = Path(args.dataset_path) / "validation"
    if not val_dir.is_dir():
        sys.exit(f"[calvin] Missing validation split: {val_dir}")

    sys.path.insert(0, str(calvin_root / "calvin_models"))
    sys.path.insert(0, str(calvin_root / "calvin_env"))
    vera_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(vera_root))

    from data.calvin_utils import setup_calvin_egl

    setup_calvin_egl(args.device, str(calvin_root))

    # pyhash is optional for eval; use a minimal FNV-1 32-bit stub if missing.
    import types
    try:
        import pyhash  # noqa: F401
    except ImportError:
        import sys as _sys

        class _Fnv1_32:
            def __call__(self, data):
                h = 2166136261
                for b in str(data).encode("utf-8"):
                    h = (h ^ b) * 16777619
                    h &= 0xFFFFFFFF
                return h

        _sys.modules["pyhash"] = types.SimpleNamespace(fnv1_32=lambda: _Fnv1_32())

    import torch
    from pytorch_lightning import seed_everything

    from evaluation.vera_calvin_policy import VERACalvinPolicy
    from calvin_agent.evaluation import evaluate_policy as ep

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"[calvin_eval] device={device}  sequences={args.num_sequences}", flush=True)
    print(f"[calvin_eval] checkpoint={args.checkpoint}", flush=True)
    print(f"[calvin_eval] dataset={args.dataset_path}", flush=True)

    seed_everything(0, workers=True)

    action_mode = "discrete" if args.discrete_actions else args.action_mode

    ep.NUM_SEQUENCES = int(args.num_sequences)
    model = VERACalvinPolicy(
        checkpoint=args.checkpoint,
        device=device,
        action_mode=action_mode,
        action_magnitude=args.action_magnitude,
        action_hold_steps=args.action_hold_steps,
        reset_history_each_step=args.reset_history_each_step,
        use_regression=(action_mode != "discrete"),
    )

    # Refresh reward history after each env step (lang-feedback streams).
    _orig_rollout = ep.rollout

    def _rollout_with_reward(env, mdl, task_oracle, subtask, val_annotations, plans, debug):
        import time as _time
        from termcolor import colored

        if debug:
            print(f"{subtask} ", end="")
            _time.sleep(0.5)
        obs = env.get_obs()
        if args.lang_goal == "task_key":
            lang_goal = subtask
        else:
            lang_goal = val_annotations[subtask][0]
        mdl.reset()
        start_info = env.get_info()

        for step in range(ep.EP_LEN):
            action = mdl.step(obs, lang_goal)
            obs, reward, done, info = env.step(action)
            if hasattr(mdl, "note_step_reward"):
                mdl.note_step_reward(float(reward), 0.0)
            if debug:
                img = env.render(mode="rgb_array")
                ep.join_vis_lang(img, lang_goal)
            if step == 0:
                ep.collect_plan(mdl, plans, subtask)

            current_task_info = task_oracle.get_task_info_for_set(
                start_info, info, {subtask}
            )
            if len(current_task_info) > 0:
                if debug:
                    print(colored("success", "green"), end=" ")
                return True
        if debug:
            print(colored("fail", "red"), end=" ")
        return False

    ep.rollout = _rollout_with_reward

    # Drop tactile camera (needs full tacto stack); VERA uses rgb_static only.
    from calvin_env.envs.play_table_env import get_env as calvin_get_env

    def make_env_no_tactile(dataset_root):
        val_folder = str(Path(dataset_root) / "validation")
        obs_space = {
            "rgb_obs": ["rgb_static", "rgb_gripper"],
            "depth_obs": [],
        }
        return calvin_get_env(val_folder, obs_space=obs_space, show_gui=False)

    env = make_env_no_tactile(args.dataset_path)

    log_dir = args.eval_log_dir
    if log_dir is None:
        ckpt = Path(args.checkpoint)
        log_dir = str(ckpt.parent / "calvin_rollout_eval")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = ep.evaluate_policy(
        model,
        env,
        epoch="vera",
        eval_log_dir=str(log_dir),
        debug=args.debug,
        create_plan_tsne=False,
    )
    elapsed = time.time() - t0

    from calvin_agent.evaluation.utils import count_success

    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "num_sequences": args.num_sequences,
        "avg_chain_length": float(sum(results) / max(len(results), 1)),
        "chain_success_rate": chain_sr,
        "elapsed_sec": elapsed,
        "action_mode": action_mode,
        "action_magnitude": args.action_magnitude,
        "action_hold_steps": args.action_hold_steps,
        "lang_goal": args.lang_goal,
        "reset_history_each_step": args.reset_history_each_step,
    }
    out_json = log_dir / "vera_calvin_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("\n[calvin_eval] ── CALVIN chain success (official protocol) ──", flush=True)
    print(f"  Avg successful chain length: {summary['avg_chain_length']:.3f}", flush=True)
    for k, v in chain_sr.items():
        print(f"  {k} subtask(s) in a row: {v * 100:.2f}%", flush=True)
    print(f"  Saved: {log_dir / 'results.json'}", flush=True)
    print(f"  Summary: {out_json}", flush=True)
    print(f"  Wall time: {elapsed / 60:.1f} min", flush=True)

    env.close()


if __name__ == "__main__":
    main()
