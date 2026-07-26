"""
Pack CALVIN task_D_D training split → episodes.pkl

Standalone — no torch/clip dependency (runs in the JAX image).
Output pkl is a list of episode dicts with the same keys as the
Language-Table pkl:
    frames        : (T, H, W, 3)  uint8  — rgb_static 200×200
    instruction   : str
    actions       : (T,)           int64  — 14-bin arm-priority label
    rewards       : (T,)           float32
    action_vectors: (T, 7)         float32 — raw rel_actions
    state_deltas  : (T,)           float32

14-bin arm-priority discretisation (matches VERA CALVIN protocol):
  dominant arm DoF → bins 0–11 (6 DoF × 2 directions)
  gripper open/close → bins 12/13 only when arm motion < arm_thresh

Usage:
    python scripts/pack_calvin_episodes.py \\
        --calvin_path /data/task_D_D \\
        --split       training \\
        --out         /data/calvin_episodes.pkl \\
        --max_eps     4000
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


ARM_THRESH = 0.03  # same as trajectory_dataset.py


def _discretise(rel_action):
    arm = np.asarray(rel_action[:6], dtype=np.float32)
    arm_mag = float(np.max(np.abs(arm)))
    if arm_mag >= ARM_THRESH:
        dom = int(np.argmax(np.abs(arm)))
        return dom * 2 + (0 if arm[dom] >= 0 else 1)
    return 12 if float(rel_action[6]) >= 0.0 else 13


def pack(calvin_path: str, split: str, out_path: str, max_eps: int):
    root = Path(calvin_path) / split

    lang_ann_path = root / "lang_annotations" / "auto_lang_ann.npy"
    if not lang_ann_path.exists():
        raise FileNotFoundError(f"Language annotations not found: {lang_ann_path}")

    lang_ann = np.load(lang_ann_path, allow_pickle=True).item()
    indx  = lang_ann["info"]["indx"]    # list of (start, end) frame indices
    tasks = lang_ann["language"]["task"] # list of task strings

    episode_files = sorted(root.glob("episode_*.npz"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz in {root}")

    available = {int(f.stem.split("_")[1]): f for f in episode_files}
    print(f"[pack] {len(episode_files)} frames available, {len(indx)} annotated episodes")

    episodes = []
    skipped  = 0
    for ep_i, (seg, task_str) in enumerate(zip(indx, tasks)):
        if max_eps and len(episodes) >= max_eps:
            break
        start, end = int(seg[0]), int(seg[1])
        frame_indices = range(start, end + 1)

        frames, actions, action_vecs, rewards = [], [], [], []
        ok = True
        for idx in frame_indices:
            if idx not in available:
                ok = False
                break
            try:
                data = np.load(available[idx], allow_pickle=True)
            except Exception:
                ok = False
                break
            rgb = data.get("rgb_static", None)
            if rgb is None:
                ok = False
                break
            rel = np.asarray(
                data.get("rel_actions", np.zeros(7, dtype=np.float32)),
                dtype=np.float32,
            ).flatten()[:7]
            frames.append(np.asarray(rgb, dtype=np.uint8))
            actions.append(_discretise(rel))
            action_vecs.append(rel)
            rewards.append(float(data.get("done", 0)))

        if not ok or len(frames) < 2:
            skipped += 1
            continue

        av = np.stack(action_vecs).astype(np.float32)
        state_deltas = np.zeros(len(av), dtype=np.float32)
        for i in range(1, len(av)):
            state_deltas[i] = -float(np.linalg.norm(av[i] - av[i - 1]))

        episodes.append({
            "frames":          np.stack(frames),
            "instruction":     task_str,
            "actions":         np.array(actions, dtype=np.int64),
            "rewards":         np.array(rewards, dtype=np.float32),
            "action_vectors":  av,
            "state_deltas":    state_deltas,
        })

        if len(episodes) % 100 == 0:
            print(f"  Packed {len(episodes)} episodes …", flush=True)

    print(f"[pack] Skipped {skipped} incomplete episodes")
    print(f"[pack] Packed {len(episodes)} episodes → {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(episodes, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin_path", required=True,
                        help="Root dir that contains training/ and validation/")
    parser.add_argument("--split",       default="training",
                        choices=["training", "validation"])
    parser.add_argument("--out",         required=True)
    parser.add_argument("--max_eps",     type=int, default=0,
                        help="Cap on episodes (0 = all)")
    args = parser.parse_args()
    pack(args.calvin_path, args.split, args.out, args.max_eps)
