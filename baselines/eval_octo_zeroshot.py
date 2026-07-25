"""
Zero-shot Octo evaluation on Language-Table (8-bin action accuracy).
No fine-tuning — loads pretrained octo-small, samples continuous actions,
discretises them with the same atan2 binning used to build the dataset.
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

if not hasattr(jax.random, "KeyArray"):
    jax.random.KeyArray = jax.Array
if not hasattr(jax.random, "PRNGKeyArray"):
    jax.random.PRNGKeyArray = jax.Array

from octo.model.octo_model import OctoModel
from PIL import Image

IMG_SIZE   = 256
NUM_BINS   = 8
STOP_THRESH = 1e-3


def discretise(action_2d):
    dx, dy = float(action_2d[0]), float(action_2d[1])
    if abs(dx) < STOP_THRESH and abs(dy) < STOP_THRESH:
        return -1   # stop step — skip
    angle = np.arctan2(dy, dx)
    return int(round(angle / (np.pi / 4))) % NUM_BINS


def preprocess(frame: np.ndarray) -> np.ndarray:
    img = Image.fromarray(frame).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def load_episodes(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",  required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--val_frac",   type=float, default=0.1)
    p.add_argument("--octo_ckpt",  default="hf://rail-berkeley/octo-small")
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[data] Loading {args.data_path}")
    all_eps = load_episodes(args.data_path)
    random.Random(42).shuffle(all_eps)
    n_val   = max(1, int(len(all_eps) * args.val_frac))
    val_eps = all_eps[:n_val]
    print(f"[data] {len(val_eps)} val episodes")

    print(f"[octo] Loading {args.octo_ckpt} ...")
    model = OctoModel.load_pretrained(args.octo_ckpt)
    key   = jax.random.PRNGKey(42)
    print("[octo] Loaded.")

    correct, total, skipped = 0, 0, 0

    for ep_idx, ep in enumerate(val_eps):
        T = len(ep["actions"])
        for t in range(T):
            frame    = preprocess(ep["frames"][t])
            obs      = {"image_primary": frame[None, None],
                       "timestep_pad_mask": jnp.ones((1, 1), dtype=bool)}
            task     = model.create_tasks(texts=[ep["instruction"]])
            pad_mask = jnp.ones((1, 1), dtype=bool)
            key, rng = jax.random.split(key)

            actions = model.sample_actions(obs, task, rng=rng)
            # shape: (1, horizon, action_dim) — take first horizon step, first 2 dims
            act_xy   = np.array(actions[0, 0, :2])
            pred_bin = discretise(act_xy)

            if pred_bin == -1:
                skipped += 1
                continue

            gt_bin = int(ep["actions"][t])
            correct += int(pred_bin == gt_bin)
            total   += 1

        if (ep_idx + 1) % 25 == 0:
            acc = correct / max(total, 1) * 100
            print(f"  [{ep_idx+1}/{len(val_eps)}]  acc={acc:.2f}%  "
                  f"({correct}/{total}, {skipped} skipped)")

    acc = correct / total if total > 0 else 0.0
    results = {
        "model":    "octo-small",
        "mode":     "zero_shot",
        "val_acc":  acc,
        "correct":  correct,
        "total":    total,
        "skipped":  skipped,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] Zero-shot Octo accuracy: {acc*100:.2f}%  ({correct}/{total})")
    print(f"       ({skipped} near-zero-action steps skipped)")


if __name__ == "__main__":
    main()
