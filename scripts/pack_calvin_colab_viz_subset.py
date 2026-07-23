#!/usr/bin/env python3
"""
Pack ~1 GB of CALVIN D→D for Colab online visualization (no full 170 GB download).

Copies language-annotated episodes from validation (default) or training:
  - episode_XXXXXXX.npz  (rgb_static, rgb_gripper, rel_actions, robot_obs, …)
  - lang_annotations/auto_lang_ann.npy  (trimmed to selected episodes only)

Output layout matches load_calvin():
  calvin_colab_viz_1gb/
    validation/   (or training/)
      episode_*.npz
      lang_annotations/auto_lang_ann.npy

Usage (on machine that already has task_D_D):
  python scripts/pack_calvin_colab_viz_subset.py \\
      --calvin_path ~/calvin_task_D/task_D_D \\
      --split validation \\
      --target_bytes 1_000_000_000 \\
      --out calvin_colab_viz_1gb

  tar -czf calvin_colab_viz_1gb.tar.gz calvin_colab_viz_1gb
  # Upload tar.gz to Google Drive; in Colab:
  # !tar -xzf /content/drive/MyDrive/calvin_colab_viz_1gb.tar.gz -C /content
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _episode_byte_size(available: dict, start: int, end: int) -> int:
    total = 0
    for idx in range(int(start), int(end) + 1):
        f = available.get(idx)
        if f is not None and f.is_file():
            total += f.stat().st_size
    return total


def _select_episodes(
    indx: list,
    tasks: list,
    available: dict,
    target_bytes: int,
    max_episodes: int | None,
) -> list[int]:
    """Greedy pack by episode index until target_bytes (language order)."""
    sizes = []
    for i, (s, e) in enumerate(indx):
        sizes.append((i, _episode_byte_size(available, s, e)))
    chosen: list[int] = []
    used = 0
    for i, b in sizes:
        if max_episodes is not None and len(chosen) >= max_episodes:
            break
        if used + b > target_bytes and chosen:
            break
        if b <= 0:
            continue
        chosen.append(i)
        used += b
    return chosen, used


def pack(
    calvin_path: Path,
    split: str,
    out_dir: Path,
    target_bytes: int,
    max_episodes: int | None,
) -> dict:
    src = calvin_path / split
    if not src.is_dir():
        raise FileNotFoundError(f"Missing split: {src}")

    lang_path = src / "lang_annotations" / "auto_lang_ann.npy"
    if not lang_path.is_file():
        raise FileNotFoundError(f"Missing {lang_path}")

    ann = np.load(lang_path, allow_pickle=True).item()
    indx = ann["info"]["indx"]
    tasks = ann["language"]["task"]

    available = {int(f.stem.split("_")[1]): f for f in src.glob("episode_*.npz")}
    chosen, used_bytes = _select_episodes(
        indx, tasks, available, target_bytes, max_episodes
    )

    dst = out_dir / split
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "lang_annotations").mkdir(parents=True, exist_ok=True)

    frame_indices: set[int] = set()
    new_indx = []
    new_tasks = []
    copied_files = 0

    for i in chosen:
        s, e = indx[i]
        new_indx.append((int(s), int(e)))
        new_tasks.append(tasks[i])
        for idx in range(int(s), int(e) + 1):
            src_f = available.get(idx)
            if src_f is None:
                continue
            dst_f = dst / src_f.name
            if not dst_f.exists():
                shutil.copy2(src_f, dst_f)
                copied_files += 1
            frame_indices.add(idx)

    sub_ann = {
        "language": {"task": new_tasks, "ann": ann["language"].get("ann", [])[: len(new_tasks)]},
        "info": {"indx": new_indx},
    }
    # Keep only annotation lists aligned with chosen episodes when present
    if "ann" in ann["language"] and len(ann["language"]["ann"]) == len(indx):
        sub_ann["language"]["ann"] = [ann["language"]["ann"][i] for i in chosen]

    np.save(dst / "lang_annotations" / "auto_lang_ann.npy", sub_ann, allow_pickle=True)

    manifest = {
        "source": str(calvin_path.resolve()),
        "split": split,
        "target_bytes": target_bytes,
        "packed_bytes": used_bytes,
        "num_language_episodes": len(chosen),
        "num_frame_npz": len(frame_indices),
        "num_npz_copied": copied_files,
        "episode_indices": chosen,
        "sample_tasks": new_tasks[:5],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[pack] {split}: {len(chosen)} language episodes, "
          f"{len(frame_indices)} frames, {used_bytes/1e9:.3f} GB → {dst}")
    return manifest


def main():
    p = argparse.ArgumentParser(description="Pack ~1GB CALVIN subset for Colab viz")
    p.add_argument("--calvin_path", required=True, help="task_D_D root")
    p.add_argument("--split", default="validation", choices=("validation", "training"))
    p.add_argument("--out", default="calvin_colab_viz_1gb")
    p.add_argument("--target_bytes", type=int, default=1_000_000_000)
    p.add_argument("--max_episodes", type=int, default=None)
    args = p.parse_args()

    pack(
        Path(args.calvin_path),
        args.split,
        Path(args.out),
        args.target_bytes,
        args.max_episodes,
    )


if __name__ == "__main__":
    main()
