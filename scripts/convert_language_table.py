"""
Convert Language-Table RLDS dataset → VERA episode format
==========================================================
Language-Table (Lynch et al. 2023) is distributed as TFRecord / RLDS files.
This script converts them to the `root/episode_XXX/steps.pkl` layout that
`data.trajectory_dataset.load_language_table()` expects.

Prerequisites
-------------
  pip install tensorflow tensorflow-datasets

Download
--------
  The Language-Table dataset is available on the Open X-Embodiment dataset
  repository (https://robotics-transformer-x.github.io/) and via TFDS:

    import tensorflow_datasets as tfds
    ds = tfds.load("language_table", data_dir="./language_table_tfds")

  Or download the raw TFRecord files from:
    https://github.com/google-research/language-table

Usage
-----
  python scripts/convert_language_table.py \
      --tfds_dir  ./language_table_tfds \
      --out_dir   ./language_table_vera \
      --split     train \
      --max_eps   5000

Output layout
-------------
  ./language_table_vera/
    episode_000000/
      steps.pkl    ← list of step dicts: {obs, action, reward, instruction}
    episode_000001/
      steps.pkl
    ...

Each steps.pkl is a list of dicts:
  {
    "obs":         {"rgb": np.ndarray (H,W,3) uint8},
    "action":      np.ndarray (2,) float32   ← [Δx, Δy] in [-1, 1]
    "reward":      float
    "instruction": str
  }

Then set in config.yaml:
  data:
    episodes_path: ./language_table_vera
    dataset_type:  language_table
  model:
    action_dim:    2
    num_actions:   8
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def convert(tfds_dir: str, out_dir: str, split: str, max_eps: int):
    try:
        import tensorflow as tf          # noqa: F401
        import tensorflow_datasets as tfds
    except ImportError:
        raise SystemExit(
            "tensorflow and tensorflow-datasets are required.\n"
            "  pip install tensorflow tensorflow-datasets"
        )

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[convert] Loading Language-Table/{split} from {tfds_dir} ...")
    ds = tfds.load(
        "language_table",
        split=split,
        data_dir=tfds_dir,
        with_info=False,
    )

    ep_idx = 0
    for episode in ds:
        if ep_idx >= max_eps:
            break

        steps = []
        steps_data = episode["steps"]

        # Extract instruction (same for all steps in an episode)
        instruction = ""
        for step in steps_data:
            obs    = step["observation"]
            action = step["action"]
            reward = float(step["reward"].numpy())

            # Language-Table obs keys: "rgb", "instruction"
            frame = obs.get("rgb", obs.get("image", None))
            if frame is None:
                continue
            frame = frame.numpy().astype(np.uint8)

            instr_bytes = obs.get("instruction", b"complete the task")
            if hasattr(instr_bytes, "numpy"):
                instr_bytes = instr_bytes.numpy()
            if isinstance(instr_bytes, bytes):
                instr_bytes = instr_bytes.decode("utf-8", errors="replace")
            instruction = str(instr_bytes)

            # Action: [Δx, Δy] ∈ [-1, 1]²
            act = action.numpy().astype(np.float32).flatten()[:2]

            steps.append({
                "obs":         {"rgb": frame},
                "action":      act,
                "reward":      reward,
                "instruction": instruction,
            })

        if len(steps) < 2:
            continue

        ep_dir = out_root / f"episode_{ep_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        with open(ep_dir / "steps.pkl", "wb") as f:
            pickle.dump(steps, f)

        ep_idx += 1
        if ep_idx % 100 == 0:
            print(f"  Converted {ep_idx} episodes ...")

    print(f"\n✓ Done: {ep_idx} episodes saved to {out_dir}")
    print(f"\nNext step — update configs/config.yaml:")
    print(f"  data:")
    print(f"    episodes_path: {out_dir}")
    print(f"    dataset_type:  language_table")
    print(f"  model:")
    print(f"    action_dim:    2")
    print(f"    num_actions:   8")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Language-Table RLDS → VERA episode format"
    )
    parser.add_argument("--tfds_dir",  required=True,
                        help="Directory containing the TFDS Language-Table data")
    parser.add_argument("--out_dir",   required=True,
                        help="Output directory for converted episodes")
    parser.add_argument("--split",     default="train",
                        help="Dataset split: train | test (default: train)")
    parser.add_argument("--max_eps",   type=int, default=5000,
                        help="Maximum number of episodes to convert (default: 5000)")
    args = parser.parse_args()
    convert(args.tfds_dir, args.out_dir, args.split, args.max_eps)


if __name__ == "__main__":
    main()
