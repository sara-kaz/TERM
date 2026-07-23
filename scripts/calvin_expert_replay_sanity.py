#!/usr/bin/env python3
"""Replay stored expert rel_actions in CALVIN sim (upper bound; should be >> 0%)."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
from collections import defaultdict


def main():
    root = Path(__file__).resolve().parents[1]
    calvin_root = Path(os.environ.get("CALVIN_ROOT", Path.home() / "work/calvin"))
    data_root = Path(os.environ.get("CALVIN_DATA", Path.home() / "calvin_task_D/task_D_D"))
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(calvin_root / "calvin_models"))
    sys.path.insert(0, str(calvin_root / "calvin_env"))
    try:
        import pyhash  # noqa: F401
    except ImportError:
        sys.modules["pyhash"] = types.SimpleNamespace(
            fnv1_32=lambda: (lambda d: sum(d) % (2**32))
        )

    from data.calvin_utils import sanitize_rel_action_for_calvin, setup_calvin_egl

    setup_calvin_egl(0, str(calvin_root))

    from omegaconf import OmegaConf
    import hydra
    from calvin_agent.evaluation.evaluate_policy import rollout, EP_LEN
    from calvin_env.envs.play_table_env import get_env as calvin_get_env

    val_folder = data_root / "validation"
    lang_path = val_folder / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(lang_path, allow_pickle=True).item()
    indx = ann["info"]["indx"]
    tasks = ann["language"]["task"]

    obs_space = {"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []}
    env = calvin_get_env(str(val_folder), obs_space=obs_space, show_gui=False)

    conf_dir = calvin_root / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    # Pick first episode per unique task (up to 10 tasks)
    seen = set()
    trials = []
    valid_tasks = set(val_annotations.keys())
    for (start, end), task in zip(indx, tasks):
        if task in seen or task not in valid_tasks:
            continue
        seen.add(task)
        actions = []
        robot0 = scene0 = None
        ok_frames = True
        for idx in range(int(start), int(end) + 1):
            p = val_folder / f"episode_{idx:07d}.npz"
            if not p.exists():
                ok_frames = False
                break
            d = np.load(p, allow_pickle=True)
            if robot0 is None:
                robot0 = d["robot_obs"]
                scene0 = d["scene_obs"]
            actions.append(
                sanitize_rel_action_for_calvin(
                    np.asarray(d["rel_actions"], dtype=np.float32).flatten()[:7]
                )
            )
        if ok_frames and actions:
            trials.append((task, robot0, scene0, actions))
        if len(trials) >= 12:
            break

    class ExpertReplay:
        def __init__(self, seq):
            self.seq = seq
            self.i = 0
            self._last = seq[-1].copy() if seq else sanitize_rel_action_for_calvin(np.zeros(7))

        def reset(self):
            self.i = 0

        def step(self, obs, goal):
            if self.i < len(self.seq):
                self._last = self.seq[self.i]
                self.i += 1
                return self._last.copy()
            # Rollout runs EP_LEN steps; hold last expert action (never send gripper=0).
            return self._last.copy()

    n_ok = 0
    for task, robot_obs, scene_obs, actions in trials:
        policy = ExpertReplay(actions[:EP_LEN])
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        ok = rollout(env, policy, task_oracle, task, val_annotations, defaultdict(list), debug=False)
        n_ok += int(ok)
        print(f"  {task}: expert_replay_success={ok}  steps={len(actions)}")

    print(f"\nExpert replay sanity: {n_ok}/{len(trials)} subtasks succeeded")
    print("(If this is ~0%, sim/task setup is broken; if high, policy is the bottleneck.)")
    env.close()


if __name__ == "__main__":
    main()
