"""CALVIN proprioception stats and normalization (matches MCIL-style z-score)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def setup_calvin_egl(cuda_index: int = 0, calvin_root: Optional[str] = None) -> int:
    """
    Map PyBullet headless EGL to a CUDA device index (within CUDA_VISIBLE_DEVICES).

    Do not set EGL_VISIBLE_DEVICES manually before calling this — CUDA and EGL
    device indices often differ on multi-GPU hosts.
    """
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.pop("EGL_VISIBLE_DEVICES", None)
    if calvin_root:
        env_path = str(Path(calvin_root) / "calvin_env")
        if env_path not in sys.path:
            sys.path.insert(0, env_path)
    from calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id

    try:
        egl_id = int(get_egl_device_id(cuda_index))
    except EglDeviceNotFoundError:
        egl_id = cuda_index
    os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
    return egl_id


def compute_robot_obs_stats(
    calvin_root: str,
    split: str = "training",
    max_frames: int = 50000,
    proprio_dim: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    root = Path(calvin_root) / split
    files = sorted(root.glob("episode_*.npz"))
    acc = []
    for i, p in enumerate(files):
        if i >= max_frames:
            break
        d = np.load(p, allow_pickle=True)
        ro = np.asarray(d.get("robot_obs", np.zeros(proprio_dim)), dtype=np.float32).flatten()
        if len(ro) >= proprio_dim:
            acc.append(ro[:proprio_dim])
    if not acc:
        mean = np.zeros(proprio_dim, dtype=np.float32)
        std = np.ones(proprio_dim, dtype=np.float32)
    else:
        arr = np.stack(acc)
        mean = arr.mean(0).astype(np.float32)
        std = arr.std(0).astype(np.float32)
        std = np.maximum(std, 1e-6)
    return mean, std


def save_proprio_stats(path: str, mean: np.ndarray, std: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2)
    )


def load_proprio_stats(path: str) -> Tuple[np.ndarray, np.ndarray]:
    d = json.loads(Path(path).read_text())
    return np.asarray(d["mean"], dtype=np.float32), np.asarray(d["std"], dtype=np.float32)


def sanitize_rel_action_for_calvin(
    rel: np.ndarray,
    *,
    deadzone: float = 0.02,
) -> np.ndarray:
    """
    Clip 7-DoF rel_actions for CALVIN env.step.

    PyBullet gripper must be exactly -1 or 1 (0.0 triggers assertion in robot.py).
    """
    rel = np.asarray(rel, dtype=np.float32).flatten()[:7].copy()
    if rel.size < 7:
        rel = np.pad(rel, (0, 7 - rel.size))
    rel = np.clip(rel, -1.0, 1.0)
    for i in range(6):
        if abs(rel[i]) < deadzone:
            rel[i] = 0.0
    rel[6] = 1.0 if rel[6] >= 0.0 else -1.0
    return rel


def normalize_robot_obs(
    ro: np.ndarray,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    proprio_dim: int = 15,
) -> np.ndarray:
    ro = np.asarray(ro, dtype=np.float32).flatten()
    if len(ro) > proprio_dim:
        ro = ro[:proprio_dim]
    elif len(ro) < proprio_dim:
        ro = np.concatenate([ro, np.zeros(proprio_dim - len(ro), dtype=np.float32)])
    if mean is not None and std is not None:
        ro = (ro - mean) / std
    return ro.astype(np.float32)
