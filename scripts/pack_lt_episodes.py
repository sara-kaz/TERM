"""
Pack Language-Table episode dirs → flat episodes.pkl

Standalone script with no torch/clip/torchvision dependency.
Used by NRP pods (JAX image has no PyTorch) to prepare the pkl
that finetune_octo_lt.py and finetune_openvla_lt.py read via
data.trajectory_dataset.load_episodes().

Usage:
    python scripts/pack_lt_episodes.py \
        --lt_dir /data/lt_vera \
        --out    /data/language_table_episodes.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def _discretise(action_vec, stop_thresh=1e-3):
    dx, dy = float(action_vec[0]), float(action_vec[1])
    if abs(dx) < stop_thresh and abs(dy) < stop_thresh:
        return -1
    angle = np.arctan2(dy, dx)
    return int(round(angle / (np.pi / 4))) % 8


def _extract_frame(step):
    for key in ("obs", "image", "rgb", "pixels", "frame"):
        val = step.get(key)
        if isinstance(val, np.ndarray) and val.ndim == 3:
            return val.astype(np.uint8)
    for top in ("obs", "observation"):
        container = step.get(top)
        if isinstance(container, dict):
            for sub in ("rgb", "image", "pixels"):
                val = container.get(sub)
                if isinstance(val, np.ndarray) and val.ndim == 3:
                    return val.astype(np.uint8)
    return None


def _extract_instruction(steps):
    for step in steps[:5]:
        for key in ("instruction", "language_instruction", "language", "task"):
            val = step.get(key)
            if val is None:
                continue
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            s = str(val).strip()
            if s:
                return s
    return "complete the task"


def pack(lt_dir: str, out_path: str):
    root = Path(lt_dir)
    episodes = []
    n_stop = 0

    for ep_dir in sorted(root.glob("episode_*")):
        sp = ep_dir / "steps.pkl"
        if not sp.exists():
            continue
        with open(sp, "rb") as f:
            steps = pickle.load(f)
        if len(steps) < 2:
            continue

        instruction = _extract_instruction(steps)
        frames, actions, action_vecs, rewards = [], [], [], []

        for step in steps:
            frame = _extract_frame(step)
            if frame is None:
                continue

            for key in ("action", "action_vec", "effector_delta", "actions"):
                val = step.get(key)
                if val is not None:
                    av = np.asarray(val, dtype=np.float32).flatten()[:2]
                    break
            else:
                av = np.zeros(2, dtype=np.float32)

            bin_idx = _discretise(av)
            if bin_idx == -1:
                n_stop += 1
                continue

            frames.append(frame)
            actions.append(bin_idx)
            action_vecs.append(av)
            rewards.append(float(step.get("reward", 0.0)))

        if len(frames) < 2:
            continue

        rewards_arr = np.array(rewards, dtype=np.float32)
        ep_max = float(rewards_arr.max())
        if ep_max > 1e-6:
            rewards_arr = (rewards_arr / ep_max).astype(np.float32)

        delta_r = np.zeros_like(rewards_arr)
        delta_r[1:] = rewards_arr[1:] - rewards_arr[:-1]
        state_deltas = np.clip(-delta_r * 3.0, -0.5, 0.5).astype(np.float32)

        episodes.append({
            "frames":         np.stack(frames),
            "instruction":    instruction,
            "actions":        np.array(actions, dtype=np.int64),
            "rewards":        rewards_arr,
            "action_vectors": np.stack(action_vecs).astype(np.float32),
            "state_deltas":   state_deltas,
        })

    print(f"[pack] Skipped {n_stop} stop steps")
    print(f"[pack] Packed {len(episodes)} episodes → {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(episodes, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lt_dir", required=True)
    parser.add_argument("--out",    required=True)
    args = parser.parse_args()
    pack(args.lt_dir, args.out)
