#!/usr/bin/env python3
"""
Collect DAgger-style episodes: roll policy from demo inits, label with expert rel_actions.

Saves checkpoints/calvin_dagger/seed123/dagger_episodes.pkl for finetune.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _make_env(split_folder: Path, calvin_root: Path, device: int, obs_space: dict):
    """Fresh CALVIN sim (EGL must be re-inited after any prior env.close())."""
    from data.calvin_utils import setup_calvin_egl
    from calvin_env.envs.play_table_env import get_env as calvin_get_env

    setup_calvin_egl(device, str(calvin_root))
    return calvin_get_env(str(split_folder), obs_space=obs_space, show_gui=False)


def _close_env(env) -> None:
    if env is None:
        return
    try:
        env.close()
    except Exception:
        pass
    del env
    gc.collect()


def _save_pkl(path: Path, episodes: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(episodes, f)
    print(f"[dagger] saved {len(episodes)} episodes -> {path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--calvin_root", default=os.environ.get("CALVIN_ROOT", str(Path.home() / "work/calvin")))
    p.add_argument("--out_pkl", default=str(ROOT / "checkpoints/calvin_dagger/seed123/dagger_episodes.pkl"))
    p.add_argument("--splits", default="validation,training", help="Comma-separated CALVIN splits")
    p.add_argument("--max_trials", type=int, default=800, help="Max episodes per split")
    p.add_argument(
        "--unique_tasks_only",
        action="store_true",
        help="Collect at most one episode per task id (legacy; limits to ~34/split)",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--action_mode", default="hybrid", choices=("continuous", "hybrid"))
    p.add_argument("--action_magnitude", type=float, default=0.45)
    p.add_argument("--action_hold_steps", type=int, default=1)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    calvin_root = Path(args.calvin_root)
    sys.path.insert(0, str(calvin_root / "calvin_models"))
    sys.path.insert(0, str(calvin_root / "calvin_env"))
    sys.path.insert(0, str(ROOT))

    try:
        import pyhash  # noqa: F401
    except ImportError:
        sys.modules["pyhash"] = types.SimpleNamespace(
            fnv1_32=lambda: (lambda d: sum(str(d).encode()) % (2**32))
        )

    from omegaconf import OmegaConf
    import hydra
    from calvin_agent.evaluation.evaluate_policy import EP_LEN
    from data.calvin_utils import sanitize_rel_action_for_calvin
    from evaluation.vera_calvin_policy import VERACalvinPolicy, discretise_rel_action

    conf_dir = calvin_root / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    obs_space = {"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []}

    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    policy = VERACalvinPolicy(
        checkpoint=args.checkpoint,
        device=device,
        action_mode=args.action_mode,
        action_hold_steps=args.action_hold_steps,
        action_magnitude=args.action_magnitude,
    )

    collected: list = []
    out = Path(args.out_pkl)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)

    for split in splits:
        split_folder = Path(args.dataset_path) / split
        lang_path = split_folder / "lang_annotations" / "auto_lang_ann.npy"
        if not lang_path.exists():
            print(f"[dagger] skip split {split}: no lang annotations")
            continue

        ann = np.load(lang_path, allow_pickle=True).item()
        indx = ann["info"]["indx"]
        tasks = ann["language"]["task"]
        pairs = [(i, t) for i, t in zip(indx, tasks) if t in val_annotations]
        rng.shuffle(pairs)

        env = None
        seen: set = set()
        n_split = 0

        try:
            env = _make_env(split_folder, calvin_root, args.device, obs_space)

            for (start, end), task in pairs:
                if args.unique_tasks_only:
                    if task in seen:
                        continue
                    seen.add(task)
                if n_split >= args.max_trials:
                    break

                expert_actions = []
                robot0 = scene0 = None
                ok_frames = True
                for idx in range(int(start), int(end) + 1):
                    ep_path = split_folder / f"episode_{idx:07d}.npz"
                    if not ep_path.exists():
                        ok_frames = False
                        break
                    d = np.load(ep_path, allow_pickle=True)
                    if robot0 is None:
                        robot0 = d["robot_obs"]
                        scene0 = d["scene_obs"]
                    expert_actions.append(
                        sanitize_rel_action_for_calvin(
                            np.asarray(d["rel_actions"], dtype=np.float32).flatten()[:7]
                        )
                    )
                if not ok_frames or len(expert_actions) < 4:
                    continue

                try:
                    env.reset(robot_obs=robot0, scene_obs=scene0)
                except Exception as exc:
                    print(f"[dagger] reset failed ({task}): {exc} — recreating sim", flush=True)
                    _close_env(env)
                    env = _make_env(split_folder, calvin_root, args.device, obs_space)
                    env.reset(robot_obs=robot0, scene_obs=scene0)

                policy.reset()
                frames, gripper_frames, action_vecs, proprio = [], [], [], []
                obs = env.get_obs()
                lang_goal = task

                for step in range(min(EP_LEN, len(expert_actions))):
                    static = np.asarray(obs["rgb_obs"]["rgb_static"], dtype=np.uint8)
                    grip = np.asarray(obs["rgb_obs"]["rgb_gripper"], dtype=np.uint8)
                    frames.append(static)
                    gripper_frames.append(grip)
                    ro = obs.get("robot_obs")
                    if ro is not None:
                        proprio.append(np.asarray(ro, dtype=np.float32).flatten())

                    action_vecs.append(expert_actions[step].copy())
                    rel = policy.step(obs, lang_goal)
                    obs, reward, done, info = env.step(rel)
                    if hasattr(policy, "note_step_reward"):
                        policy.note_step_reward(float(reward), 0.0)

                if len(frames) < 4:
                    continue

                actions_idx = np.array(
                    [discretise_rel_action(a) for a in action_vecs], dtype=np.int64
                )
                ep_dict = {
                    "frames": np.stack(frames),
                    "gripper_frames": np.stack(gripper_frames),
                    "instruction": task,
                    "actions": actions_idx,
                    "rewards": np.zeros(len(frames), dtype=np.float32),
                    "action_vectors": np.stack(action_vecs).astype(np.float32),
                }
                if proprio:
                    ep_dict["robot_obs"] = np.stack(proprio).astype(np.float32)
                collected.append(ep_dict)
                n_split += 1

        finally:
            _close_env(env)

        print(
            f"[dagger] split={split}: collected {n_split} episodes (total {len(collected)})",
            flush=True,
        )
        _save_pkl(out, collected)

    print(f"Done — {len(collected)} DAgger episodes -> {out}", flush=True)


if __name__ == "__main__":
    main()
