#!/usr/bin/env python3
"""Replay expert rel_actions in CALVIN sim to verify env + task oracle (upper bound)."""

import os
import sys
import types
from pathlib import Path

import numpy as np

def main():
    calvin_root = Path(os.environ.get("CALVIN_ROOT", Path.home() / "work/calvin"))
    data_root = Path(os.environ.get("CALVIN_DATA", Path.home() / "calvin_task_D/task_D_D"))
    sys.path.insert(0, str(calvin_root / "calvin_models"))
    sys.path.insert(0, str(calvin_root / "calvin_env"))
    try:
        import pyhash  # noqa: F401
    except ImportError:
        sys.modules["pyhash"] = types.SimpleNamespace(
            fnv1_32=lambda: (lambda d: sum(d) % (2**32))
        )

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_VISIBLE_DEVICES", "0")

    from omegaconf import OmegaConf
    import hydra
    from calvin_agent.evaluation.multistep_sequences import get_sequences
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
    from calvin_agent.evaluation.evaluate_policy import rollout
    from calvin_env.envs.play_table_env import get_env as calvin_get_env

    val_folder = data_root / "validation"
    obs_space = {"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []}
    env = calvin_get_env(str(val_folder), obs_space=obs_space, show_gui=False)

    conf_dir = calvin_root / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    # Build frame index -> rel_action map from validation npz
    frame_to_action = {}
    for npz in val_folder.glob("episode_*.npz"):
        idx = int(npz.stem.split("_")[1])
        d = np.load(npz, allow_pickle=True)
        frame_to_action[idx] = np.asarray(d["rel_actions"], dtype=np.float32).flatten()[:7]

    class OracleReplay:
        def __init__(self, actions):
            self.actions = actions
            self.i = 0

        def reset(self):
            self.i = 0

        def step(self, obs, goal):
            if self.i < len(self.actions):
                a = self.actions[self.i]
                self.i += 1
                return a.astype(np.float32)
            return np.zeros(7, dtype=np.float32)

    sequences = get_sequences(5)
    n_ok = 0
    for initial_state, eval_sequence in sequences:
        subtask = eval_sequence[0]
        # pick any validation frame index present (proxy expert)
        if not frame_to_action:
            break
        expert = OracleReplay([frame_to_action[k] for k in sorted(frame_to_action)[:360]])
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        ok = rollout(env, expert, task_oracle, subtask, val_annotations, {}, debug=False)
        n_ok += int(ok)
        print(f"  subtask={subtask}  oracle_replay_success={ok}")

    print(f"Oracle sanity: {n_ok}/{len(sequences)} first subtasks succeeded (random expert frames)")
    env.close()


if __name__ == "__main__":
    main()
