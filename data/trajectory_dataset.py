"""
TrajectoryDataset
=================
Loads robot trajectories stored as a list of episode dicts.

Episode format (minimum required):
  episode = {
      "frames"      : np.ndarray  (T_ep, H, W, 3)   uint8
      "instruction" : str
      "actions"     : np.ndarray  (T_ep,)            int64   discrete index
      "rewards"     : np.ndarray  (T_ep,)            float32
  }

Episode format (with low-level action vectors — recommended):
  episode = {
      ...same as above...
      "action_vectors": np.ndarray (T_ep, action_dim) float32
                        The actual continuous robot command executed at each step.
                        action_dim depends on dataset / robot:
                          MetaWorld      : 4   [Δx, Δy, Δz, gripper]
                          Language-Table : 2   [Δx, Δy]  (planar EE delta)
                          CALVIN         : 7   [x,y,z,roll,pitch,yaw,gripper]
                        If absent, action_vec_hist is returned as None and the
                        VERA model's history encoder falls back to discrete-only.
      "state_deltas":   np.ndarray (T_ep,) float32
                        Signed change in distance-to-goal at each step
                        (negative = closer, positive = farther).
                        Used by VERA's ConsequenceLanguageEncoder (Stream 3b) to
                        produce rich direction×magnitude×reward consequence strings.
                        For Language-Table, approximated from consecutive reward diffs.
                        If absent, state_delta defaults to 0.0 (stationary fallback).
  }

Each __getitem__ returns one training window centred on timestep t:
  frames         : (num_vis_frames, 3, 224, 224)
  lang_tokens    : (77,)
  action_hist    : (history_len,)              int64   — discrete indices
  reward_hist    : (history_len,)              float32
  action_vec_hist: (history_len, action_dim)   float32  OR  None
  state_delta    : ()                          float32  — signed dist change at t
  target         : int                         — discrete action label at t
  target_vec     : (action_dim,)               float32  — continuous action at t (OR None)

Dataset loaders
---------------
  load_episodes(path)                   — generic .pkl
  load_language_table(root)             — Language-Table (Lynch et al. 2023)
  load_calvin(root, split)              — CALVIN (Mees et al. 2022)
  make_random_episodes(...)             — synthetic, for unit tests
"""

import pickle
import random
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import clip
from PIL import Image
import torchvision.transforms as T


# ── Synthetic episodes (unit testing) ─────────────────────────────────────────

def make_random_episodes(
    num_episodes: int  = 50,
    ep_len:       int  = 30,
    num_actions:  int  = 8,
    action_dim:   int  = 4,
    img_size:     int  = 64,
) -> List[Dict]:
    """
    Generate random synthetic episodes for pipeline testing.
    Includes action_vectors so the full VERA history encoder is exercised.
    """
    instructions = [
        "pick up the red cube",
        "move to the left side",
        "push the block forward",
        "grasp the cylinder",
    ]
    episodes = []
    for _ in range(num_episodes):
        T_ep = random.randint(ep_len // 2, ep_len)
        episodes.append({
            "frames":         np.random.randint(0, 255, (T_ep, img_size, img_size, 3), dtype=np.uint8),
            "instruction":    random.choice(instructions),
            "actions":        np.random.randint(0, num_actions, (T_ep,), dtype=np.int64),
            "rewards":        np.random.uniform(-1.0, 1.0, (T_ep,)).astype(np.float32),
            "action_vectors": np.random.uniform(-1.0, 1.0, (T_ep, action_dim)).astype(np.float32),
        })
    return episodes


# ── Language-Table loader (Lynch et al., 2023) ────────────────────────────────

def _lt_extract_frame(step: dict) -> Optional[np.ndarray]:
    """
    Try every known key/sub-key layout used by Language-Table pkl exports.

    Supported layouts (in priority order):
      step["obs"]           — numpy array (H, W, 3)          [synthetic / our format]
      step["obs"]["rgb"]    — nested dict with rgb key
      step["observation"]["rgb"]  — RLDS-converted pkl
      step["observation"]["image"]
      step["image"] / step["rgb"] / step["pixels"]  — flat
    """
    # 1. Direct numpy array under common top-level keys
    for key in ("obs", "image", "rgb", "pixels", "frame"):
        val = step.get(key)
        if isinstance(val, np.ndarray) and val.ndim == 3:
            return val.astype(np.uint8)

    # 2. Nested dict under "obs" or "observation"
    for top in ("obs", "observation"):
        container = step.get(top)
        if isinstance(container, dict):
            for sub in ("rgb", "image", "pixels", "agentview_rgb"):
                val = container.get(sub)
                if isinstance(val, np.ndarray) and val.ndim == 3:
                    return val.astype(np.uint8)

    return None


def _lt_extract_action(step: dict) -> np.ndarray:
    """
    Extract the 2-DoF action vector [Δx, Δy] from a step dict.

    Handles:
      step["action"]           — numpy array or list  (most formats)
      step["action_vec"]
      step["effector_delta"]
    """
    for key in ("action", "action_vec", "effector_delta", "actions"):
        val = step.get(key)
        if val is not None:
            arr = np.asarray(val, dtype=np.float32).flatten()
            if arr.size >= 2:
                return arr[:2]
    return np.zeros(2, dtype=np.float32)


def _normalize_instruction_text(val) -> str:
    """Decode LT/CALVIN instruction fields (str, bytes, padded uint8 arrays, …)."""
    if val is None:
        return "complete the task"
    if isinstance(val, str):
        return val.strip() or "complete the task"
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace").strip() or "complete the task"
    if isinstance(val, np.ndarray):
        flat = val.flatten()
        if flat.size == 0:
            return "complete the task"
        # Padded ASCII code arrays (common in LT exports)
        if flat.dtype in (np.uint8, np.int8, np.int16, np.int32, np.int64):
            codes = [int(c) for c in flat if int(c) != 0]
            if codes:
                try:
                    return bytes(codes).decode("utf-8", errors="replace").strip()
                except (ValueError, OverflowError):
                    return "".join(chr(c) for c in codes if 0 < c < 0x110000).strip()
        if flat.dtype.kind in "US":
            s = str(flat.item()).strip()
            return s or "complete the task"
    return str(val).strip() or "complete the task"


def _lt_extract_instruction(steps: list, ep_meta: Optional[dict] = None) -> str:
    """
    Extract the language instruction for an episode.

    Priority:
      1. Episode-level metadata dict (ep_meta keys: instruction, language_instruction, task)
      2. First step with a non-empty instruction key
      3. Fallback generic string
    """
    if ep_meta is not None:
        for key in ("instruction", "language_instruction", "language", "task"):
            val = ep_meta.get(key)
            if val is not None and (not isinstance(val, str) or val):
                if isinstance(val, str) and not val.strip():
                    continue
                return _normalize_instruction_text(val)

    for step in steps[:5]:   # look in first 5 steps
        for key in ("instruction", "language_instruction", "language", "task"):
            val = step.get(key)
            if val is not None and (not isinstance(val, str) or val):
                if isinstance(val, str) and not val.strip():
                    continue
                return _normalize_instruction_text(val)

    return "complete the task"


def central_crop_uint8(frame: np.ndarray, crop_factor: float = 0.95) -> np.ndarray:
    """Central crop matching Language-Table official eval (factor=0.95)."""
    if crop_factor is None or crop_factor >= 1.0:
        return frame
    h, w = frame.shape[:2]
    nh = max(1, int(round(h * crop_factor)))
    nw = max(1, int(round(w * crop_factor)))
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    return frame[y0 : y0 + nh, x0 : x0 + nw]


def _lt_discretise(action_vec: np.ndarray, stop_thresh: float = 1e-3) -> int:
    """
    Discretise a 2-DoF [Δx, Δy] action into one of 8 directional bins.

    Bins (arctan2 sectors of π/4 each, 0 = East/right):
      0=right, 1=up-right, 2=up, 3=up-left,
      4=left,  5=down-left, 6=down, 7=down-right

    Near-zero actions (‖action‖ < stop_thresh) map to the closest previous
    direction to avoid polluting the label distribution with random zeros.
    Returns -1 for stop steps so the caller can decide to skip or keep them.
    """
    dx, dy = float(action_vec[0]), float(action_vec[1])
    if abs(dx) < stop_thresh and abs(dy) < stop_thresh:
        return -1  # stop / no-op step
    angle = np.arctan2(dy, dx)                       # [-π, π]
    return int(round(angle / (np.pi / 4))) % 8       # 8 equal sectors


def inspect_language_table(root: str, n: int = 3) -> None:
    """
    Print the structure of the first n episodes so you can verify the format.
    Call this once before training to confirm the loader reads your data correctly.
    """
    root = Path(root)
    dirs = sorted(root.glob("episode_*"))[:n]
    if not dirs:
        print(f"[LT inspect] No episode_* dirs found in {root}")
        return
    for ep_dir in dirs:
        sp = ep_dir / "steps.pkl"
        if not sp.exists():
            print(f"  {ep_dir.name}: no steps.pkl"); continue
        steps = pickle.load(open(sp, "rb"))
        print(f"\n{ep_dir.name}  ({len(steps)} steps)")
        s0 = steps[0]
        print(f"  step keys: {list(s0.keys())}")
        for k, v in s0.items():
            if isinstance(v, np.ndarray):
                print(f"    {k}: ndarray shape={v.shape} dtype={v.dtype} "
                      f"range=[{v.min():.3f}, {v.max():.3f}]")
            elif isinstance(v, dict):
                print(f"    {k}: dict keys={list(v.keys())}")
                for kk, vv in v.items():
                    if isinstance(vv, np.ndarray):
                        print(f"      {kk}: ndarray shape={vv.shape} dtype={vv.dtype} "
                              f"range=[{vv.min():.3f}, {vv.max():.3f}]")
                    else:
                        print(f"      {kk}: {type(vv).__name__} = {str(vv)[:60]}")
            else:
                print(f"    {k}: {type(v).__name__} = {str(v)[:60]}")
        frame = _lt_extract_frame(s0)
        action = _lt_extract_action(s0)
        instr  = _lt_extract_instruction(steps)
        print(f"  → frame  : {'OK shape=' + str(frame.shape) if frame is not None else 'NOT FOUND'}")
        print(f"  → action : {action}  bin={_lt_discretise(action)}")
        print(f"  → instr  : {instr[:80]}")


def load_language_table(root: str, skip_stop_steps: bool = True) -> List[Dict]:
    """
    Load Language-Table episodes from the standard directory layout:

      root/
        episode_000/
          steps.pkl   — list of step dicts (any of the common LT pkl formats)
        episode_001/
          ...

    Handles all known pkl export formats from the Language-Table dataset:
      • Our synthetic format : step["obs"] = np.array, step["instruction"]
      • RLDS-converted pkl   : step["observation"]["rgb"], episode-level instruction
      • Flat format          : step["rgb"] / step["image"] / step["pixels"]

    Action format: 2-DoF planar end-effector delta [Δx, Δy].
    Near-zero / stop steps are skipped by default (skip_stop_steps=True) to
    prevent random-direction label noise from corrupting the discrete targets.

    Returns list of episode dicts compatible with TrajectoryDataset.
    """
    root = Path(root)
    episodes = []
    n_stop_skipped = 0

    for ep_dir in sorted(root.glob("episode_*")):
        steps_path = ep_dir / "steps.pkl"
        if not steps_path.exists():
            continue
        with open(steps_path, "rb") as f:
            raw = pickle.load(f)

        # Support both list-of-steps and {steps: [...], instruction: "..."} formats
        ep_meta = None
        if isinstance(raw, dict):
            ep_meta = raw
            steps   = raw.get("steps", [])
        elif isinstance(raw, list):
            steps = raw
        else:
            continue

        if not steps:
            continue

        instruction = _lt_extract_instruction(steps, ep_meta)

        frames      = []
        actions     = []
        rewards     = []
        action_vecs = []

        for step in steps:
            frame = _lt_extract_frame(step)
            if frame is None:
                continue

            action_vec = _lt_extract_action(step)
            action_idx = _lt_discretise(action_vec)

            # Skip near-zero (stop) steps — they produce random labels
            if skip_stop_steps and action_idx == -1:
                n_stop_skipped += 1
                continue
            if action_idx == -1:
                action_idx = 0   # keep stop → default bin if user disables skip

            frames.append(frame)
            actions.append(action_idx)
            rewards.append(float(step.get("reward", 0.0)))
            # Normalise action_vec to [-1, 1] if needed (some exports use mm/s)
            av = action_vec[:2].copy()
            max_abs = np.abs(av).max()
            if max_abs > 1.0:
                av = av / max_abs
            action_vecs.append(av)

        if len(frames) < 2:
            continue

        rewards_arr = np.array(rewards, dtype=np.float32)

        # ── Normalise LT rewards from [0, ~0.2] → [0, 1] ─────────────────────
        # Language-Table shaped rewards top out around 0.2 (goal reached).
        # Leaving them raw causes two problems:
        #   1. The reward gate sigmoid(MLP(r)) ≈ 0.5 for all steps (no range).
        #   2. verbalize_consequence's "high" threshold (≥1.0) is never reached.
        #   3. InfoNCE exponential weights exp(5·r) give only 2.7× contrast vs
        #      148× when r ∈ [0,1].
        # Dividing by the episode max (or 0.2 if flat) maps rewards to [0,1]
        # and makes all three mechanisms discriminative.
        ep_max = float(rewards_arr.max())
        if ep_max > 1e-6:
            rewards_arr = (rewards_arr / ep_max).astype(np.float32)
        # (episodes with all-zero rewards stay at 0 — no division by zero)

        # ── Pseudo state_delta from consecutive reward differences ────────────
        # Language-Table rewards are shaped ∈ [0, 1] after normalisation above.
        # Δr > 0 → agent moved closer to goal → negative delta_dist by convention.
        #
        # Scale factor 3.0 maps a typical Δr ≈ 0.05 to δd ≈ -0.15, which
        # falls in the "significantly" magnitude bin of verbalize_consequence.
        # Clipping at ±0.5 avoids outliers from noisy reward signals.
        delta_r    = np.zeros_like(rewards_arr)
        delta_r[1:] = rewards_arr[1:] - rewards_arr[:-1]
        state_deltas = np.clip(-delta_r * 3.0, -0.5, 0.5).astype(np.float32)

        episodes.append({
            "frames":         np.stack(frames),
            "instruction":    instruction,
            "actions":        np.array(actions,  dtype=np.int64),
            "rewards":        rewards_arr,
            "action_vectors": np.stack(action_vecs).astype(np.float32),
            "state_deltas":   state_deltas,          # (T,) signed dist-to-goal proxy
        })

    if n_stop_skipped:
        print(f"[Language-Table] Skipped {n_stop_skipped} near-zero (stop) steps")
    print(f"[Language-Table] Loaded {len(episodes)} episodes from {root}")
    return episodes


# ── CALVIN loader (Mees et al., 2022) ─────────────────────────────────────────

def load_calvin(root: str, split: str = "training") -> List[Dict]:
    """
    Load CALVIN episodes from the standard layout:

      root/
        training/
          episode_0000000.npz
          episode_0000001.npz
          ...
          lang_annotations/
            auto_lang_ann.npy   — {language: {task: list[str]}, info: {indx: list}}
        validation/
          ...

    Each .npz contains keys: rgb_static, rgb_gripper, rel_actions, robot_obs,
    scene_obs, done.

    Action format: 7-DoF [x, y, z, roll, pitch, yaw, gripper] ∈ [-1, 1]⁷
    (action_dim = 7).

    Returns list of episode dicts compatible with TrajectoryDataset.
    """
    root      = Path(root) / split
    episodes  = []

    # Load language annotations if available
    lang_ann_path = root / "lang_annotations" / "auto_lang_ann.npy"
    lang_ann      = None
    if lang_ann_path.exists():
        lang_ann = np.load(lang_ann_path, allow_pickle=True).item()
        # lang_ann["language"]["task"] — list of task strings per indexed segment
        # lang_ann["info"]["indx"]     — list of (start, end) episode indices

    episode_files = sorted(root.glob("episode_*.npz"))
    if not episode_files:
        print(f"[CALVIN] No .npz files found in {root}. Check dataset path.")
        return episodes

    # Build index of available frame numbers for fast lookup
    available = {int(f.stem.split("_")[1]): f for f in episode_files}

    def _discretise(rel_action, arm_thresh: float = 0.03):
        """
        14-bin discretisation for CALVIN with arm-priority.

        Critical detail:
        CALVIN rel_actions use gripper ∈ {-1, +1} almost every step. If gripper is
        checked first, labels collapse to classes 12/13 and arm classes 0..11 never
        train, yielding high offline acc but no embodied arm control.

        We therefore prioritize dominant arm motion when it is non-trivial, and emit
        gripper classes only when arm motion is near zero.
        """
        arm = np.asarray(rel_action[:6], dtype=np.float32)
        arm_mag = float(np.max(np.abs(arm)))
        if arm_mag >= arm_thresh:
            dom = int(np.argmax(np.abs(arm)))
            return dom * 2 + (0 if arm[dom] >= 0 else 1)
        return 12 if float(rel_action[6]) >= 0.0 else 13

    def _load_episode(frame_indices, task_str):
        """Load one annotated episode from a list of absolute frame indices."""
        frames, gripper_frames, actions_idx, rewards, action_vecs, proprio = [], [], [], [], [], []
        for idx in frame_indices:
            if idx not in available:
                return None           # frame not downloaded — skip episode
            try:
                data = np.load(available[idx], allow_pickle=True)
            except Exception:
                return None           # corrupted file — skip episode
            frame = data.get("rgb_static", None)
            if frame is None:
                return None
            gf = data.get("rgb_gripper", None)
            if gf is not None:
                gripper_frames.append(np.asarray(gf, dtype=np.uint8))
            rel_action = np.asarray(
                data.get("rel_actions", np.zeros(7, dtype=np.float32)),
                dtype=np.float32,
            ).flatten()[:7]
            frames.append(np.asarray(frame, dtype=np.uint8))
            actions_idx.append(_discretise(rel_action))
            rewards.append(float(data.get("done", 0)))
            action_vecs.append(rel_action)
            ro = data.get("robot_obs", None)
            if ro is not None:
                proprio.append(np.asarray(ro, dtype=np.float32).flatten())
        if len(frames) < 2:
            return None
        av = np.stack(action_vecs).astype(np.float32)
        # Pseudo state_delta for Stream 3b: negative change in action magnitude.
        state_deltas = np.zeros(len(av), dtype=np.float32)
        for i in range(1, len(av)):
            state_deltas[i] = -float(np.linalg.norm(av[i] - av[i - 1]))
        ep_dict = {
            "frames":         np.stack(frames),
            "instruction":    task_str,
            "actions":        np.array(actions_idx, dtype=np.int64),
            "rewards":        np.array(rewards,     dtype=np.float32),
            "action_vectors": av,
            "state_deltas":   state_deltas,
        }
        if proprio:
            ep_dict["robot_obs"] = np.stack(proprio).astype(np.float32)
        if gripper_frames and len(gripper_frames) == len(frames):
            ep_dict["gripper_frames"] = np.stack(gripper_frames)
        return ep_dict

    # ── Resolve episode boundaries from annotation sources ────────────────────

    ep_se_path = root / "ep_start_end_ids.npy"

    if lang_ann is not None:
        # Primary: language annotations with task strings
        indx  = lang_ann["info"]["indx"]       # [(start_frame, end_frame), ...]
        tasks = lang_ann["language"]["task"]   # [str, ...]
        print(f"[CALVIN] Using lang_annotations ({len(indx)} episodes)")

    elif ep_se_path.exists():
        # Fallback: ep_start_end_ids.npy — absolute frame index pairs, no text
        ep_se = np.load(ep_se_path)            # (N, 2) int array
        indx  = [(int(s), int(e)) for s, e in ep_se]
        tasks = ["complete the manipulation task"] * len(indx)
        print(f"[CALVIN] No lang_annotations — using ep_start_end_ids.npy "
              f"({len(indx)} episodes, generic instruction)")

    else:
        indx, tasks = [], []

    # ── Load annotated episodes ────────────────────────────────────────────────
    if indx:
        skipped = 0
        for (start, end), task_str in zip(indx, tasks):
            ep = _load_episode(list(range(start, end + 1)), task_str)
            if ep is None:
                skipped += 1
            else:
                episodes.append(ep)
        if skipped:
            print(f"[CALVIN] Skipped {skipped} episodes (frames not downloaded)")

    # ── Last-resort fallback: no annotation files at all ─────────────────────
    if not indx and not episodes:
        print("[CALVIN] No annotation files found — treating each NPZ as a 1-step episode.")
        for frame_idx, npz_path in sorted(available.items()):
            data  = np.load(npz_path, allow_pickle=True)
            frame = data.get("rgb_static", None)
            if frame is None:
                continue
            rel_action = np.asarray(
                data.get("rel_actions", np.zeros(7, dtype=np.float32)),
                dtype=np.float32,
            ).flatten()[:7]
            episodes.append({
                "frames":         np.stack([np.asarray(frame, dtype=np.uint8)] * 2),
                "instruction":    "complete the manipulation task",
                "actions":        np.array([_discretise(rel_action), _discretise(rel_action)],
                                           dtype=np.int64),
                "rewards":        np.array([float(data.get("done", 0))] * 2,
                                           dtype=np.float32),
                "action_vectors": np.stack([rel_action] * 2).astype(np.float32),
            })

    print(f"[CALVIN] Loaded {len(episodes)} episodes from {root}")
    return episodes


# ── Generic loader ─────────────────────────────────────────────────────────────

def load_episodes(path: str) -> List[Dict]:
    """Load episodes from a .pkl file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_episodes(episodes: List[Dict], path: str) -> None:
    """Save episodes to a .pkl file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(episodes, f)


# ── Dataset ────────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """
    Sliding-window dataset over robot episodes.

    Handles episodes with or without 'action_vectors'. When present,
    action_vec_hist is returned as a (history_len, action_dim) tensor and
    passed to VERA's history encoder for richer low-level conditioning.
    When absent, action_vec_hist is None and the encoder falls back to
    discrete-only history (fully backward compatible).

    When chunk_size > 1 (action chunking, π0 / GR-1 style), __getitem__
    also returns `target_chunk` of shape (chunk_size,) containing the
    discrete action labels for timesteps t, t+1, …, t+K-1.  Steps beyond
    the episode boundary are padded by repeating the last valid action.
    The trainer uses all K targets to compute CE loss, giving K× the
    supervision signal and enforcing temporal consistency across steps.
    """

    def __init__(
        self,
        episodes:        List[Dict],
        history_len:     int  = 4,
        num_vis_frames:  int  = 3,
        num_actions:     int  = 8,
        action_dim:      int  = 4,
        img_size:        int  = 224,
        clip_model_name: str  = "ViT-B/32",
        device:          str  = "cpu",
        chunk_size:      int  = 1,    # action chunking window K (1 = disabled)
        proprio_dim:     int  = 0,    # CALVIN robot_obs (15); 0 = disabled
        use_gripper_cam: bool = False,
        proprio_mean:    Optional[np.ndarray] = None,
        proprio_std:     Optional[np.ndarray] = None,
        crop_factor:     float = 1.0,  # <1.0 = central crop (Language-Table eval match)
    ):
        self.history_len    = history_len
        self.num_vis_frames = num_vis_frames
        self.num_actions    = num_actions
        self.action_dim     = action_dim
        self.chunk_size     = max(1, chunk_size)
        self.proprio_dim    = int(proprio_dim)
        self.use_gripper_cam = bool(use_gripper_cam)
        self.proprio_mean   = proprio_mean
        self.proprio_std    = proprio_std
        self.crop_factor    = float(crop_factor)
        self.pad_action     = num_actions   # "no history" discrete padding index

        # NOTE: Do not call clip.load here — VERAModel already loads CLIP. A duplicate
        # load (train + val datasets) added minutes of startup and GPU memory for no use.

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275,  0.40821073],
                        std= [0.26862954, 0.26130258, 0.27577711]),
        ])

        # Check whether any episode has action_vectors
        self.has_action_vecs = any("action_vectors" in ep for ep in episodes)
        self.has_proprio = self.proprio_dim > 0 and any("robot_obs" in ep for ep in episodes)
        self.has_gripper = self.use_gripper_cam and any("gripper_frames" in ep for ep in episodes)
        if self.use_gripper_cam and not self.has_gripper:
            print("[TrajectoryDataset] use_gripper_cam=True but no gripper_frames in episodes.")

        # Validate action_dim consistency
        if self.has_action_vecs:
            for ep in episodes:
                if "action_vectors" in ep:
                    ep_dim = ep["action_vectors"].shape[1]
                    if ep_dim != action_dim:
                        print(
                            f"[TrajectoryDataset] Warning: episode action_dim={ep_dim} "
                            f"but dataset action_dim={action_dim}. "
                            f"Truncating/padding to {action_dim}."
                        )
                    break

        # Build flat (episode_idx, timestep) index.
        # Require each episode to have at least history_len + 1 steps so that
        # __getitem__ never produces an out-of-bounds action_hist window.
        min_len = history_len + 1
        self.index: List[tuple] = []
        self.episodes = episodes
        skipped = 0
        for ep_i, ep in enumerate(episodes):
            T_ep = len(ep["actions"])
            if T_ep < min_len:
                skipped += 1
                continue
            for t in range(1, T_ep):
                self.index.append((ep_i, t))
        if skipped:
            print(f"[TrajectoryDataset] Skipped {skipped} episodes shorter than "
                  f"history_len+1={min_len} steps.")

        # Pre-tokenize all unique instructions (truncate=True for CLIP 77-token cap)
        unique_inst  = [_normalize_instruction_text(ep["instruction"]) for ep in episodes]
        unique_inst  = list(dict.fromkeys(unique_inst))  # preserve order, dedupe
        tokens       = clip.tokenize(unique_inst, truncate=True)
        self._token_cache = {inst: tokens[i] for i, inst in enumerate(unique_inst)}
        for ep in episodes:
            ep["instruction"] = _normalize_instruction_text(ep["instruction"])

        print(
            f"[TrajectoryDataset] {len(episodes)} episodes, "
            f"{len(self.index)} windows, "
            f"action_vectors={'yes' if self.has_action_vecs else 'no'}, "
            f"proprio={'yes' if self.has_proprio else 'no'}, "
            f"gripper={'yes' if self.has_gripper else 'no'}, "
            f"action_dim={action_dim}, "
            f"chunk_size={self.chunk_size}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        ep_i, t = self.index[idx]
        ep      = self.episodes[ep_i]

        # ── Language tokens ───────────────────────────────────────────────────
        lang_tokens = self._token_cache[ep["instruction"]]   # (77,)

        # ── Visual frames: last num_vis_frames up to and including t ─────────
        # Must match closed-loop rollout: policy decides action[t] from obs[t].
        start_vis  = max(0, t - self.num_vis_frames + 1)
        raw_frames = ep["frames"][start_vis : t + 1]
        pad_needed = self.num_vis_frames - len(raw_frames)
        if pad_needed > 0:
            pad = np.zeros_like(raw_frames[0:1]).repeat(pad_needed, axis=0)
            raw_frames = np.concatenate([pad, raw_frames], axis=0)
        vis_list = []
        for f in raw_frames:
            if self.crop_factor < 1.0:
                f = central_crop_uint8(f, self.crop_factor)
            vis_list.append(self.transform(Image.fromarray(f)))

        if self.has_gripper:
            raw_g = ep["gripper_frames"][start_vis : t + 1]
            if pad_needed > 0:
                pad_g = np.zeros_like(raw_g[0:1]).repeat(pad_needed, axis=0)
                raw_g = np.concatenate([pad_g, raw_g], axis=0)
            for f in raw_g:
                if self.crop_factor < 1.0:
                    f = central_crop_uint8(f, self.crop_factor)
                vis_list.extend([self.transform(Image.fromarray(f))])

        frames = torch.stack(vis_list)   # (T_vis, 3, H, W)  T=3 or 6

        # ── Discrete action + reward history ──────────────────────────────────
        start_h      = max(0, t - self.history_len)
        hist_actions = ep["actions"][start_h:t].tolist()
        hist_rewards = ep["rewards"][start_h:t].tolist()
        pad_h        = self.history_len - len(hist_actions)
        hist_actions = [self.pad_action] * pad_h + hist_actions
        hist_rewards = [0.0]            * pad_h + hist_rewards

        action_hist = torch.tensor(hist_actions, dtype=torch.long)
        reward_hist = torch.tensor(hist_rewards, dtype=torch.float32)

        # ── Low-level continuous action vectors ───────────────────────────────
        action_vec_hist = None
        if "action_vectors" in ep:
            raw_vecs = ep["action_vectors"][start_h:t]   # (<= H, action_dim)
            ep_dim   = raw_vecs.shape[1] if len(raw_vecs) > 0 else self.action_dim

            # Truncate or zero-pad to self.action_dim
            if ep_dim > self.action_dim:
                raw_vecs = raw_vecs[:, :self.action_dim]
            elif ep_dim < self.action_dim:
                pad_cols = np.zeros((len(raw_vecs), self.action_dim - ep_dim), dtype=np.float32)
                raw_vecs = np.concatenate([raw_vecs, pad_cols], axis=1)

            # Left-pad timestep dimension with zero vectors
            if pad_h > 0:
                zero_pad = np.zeros((pad_h, self.action_dim), dtype=np.float32)
                raw_vecs = np.concatenate([zero_pad, raw_vecs], axis=0)

            action_vec_hist = torch.tensor(raw_vecs, dtype=torch.float32)
            # shape: (history_len, action_dim)

        # ── Target action (label for discrete classifier) ─────────────────────
        target = int(ep["actions"][t])

        # ── Action chunk targets (for action chunking with chunk_size > 1) ─────
        # Returns the K consecutive discrete labels starting at timestep t.
        # Steps that exceed the episode boundary are padded by repeating the last
        # valid action (boundary padding — avoids the "stop" padding index which
        # would introduce erroneous label noise near episode ends).
        K     = self.chunk_size
        T_ep  = len(ep["actions"])
        chunk_end   = min(t + K, T_ep)
        chunk_acts  = ep["actions"][t:chunk_end].tolist()     # 1..K valid actions
        if len(chunk_acts) < K:                               # pad to K
            chunk_acts = chunk_acts + [chunk_acts[-1]] * (K - len(chunk_acts))
        target_chunk = torch.tensor(chunk_acts, dtype=torch.long)  # (K,)

        # ── State delta for Stream 3b consequence verbalization ──────────────
        # Signed distance-to-goal change at timestep t.  Negative → got closer.
        # For Language-Table episodes this is approximated from consecutive
        # reward differences (computed in load_language_table).
        # For other datasets / synthetic episodes it defaults to 0.0, which maps
        # to the "stationary" branch in verbalize_consequence — still informative
        # when combined with the reward bucket (e.g. "stationary + high reward"
        # vs "stationary + no reward" are two distinct strings).
        state_delta_val = 0.0
        if "state_deltas" in ep:
            state_delta_val = float(ep["state_deltas"][t])

        # ── Target continuous action vector (label for regression head) ────────
        # This is the expert's executed action at timestep t — the same step
        # whose discrete index is in `target`.  Truncated/padded to action_dim.
        target_vec = None
        if "action_vectors" in ep:
            tv = ep["action_vectors"][t].astype(np.float32)   # (ep_dim,)
            if len(tv) > self.action_dim:
                tv = tv[:self.action_dim]
            elif len(tv) < self.action_dim:
                tv = np.concatenate(
                    [tv, np.zeros(self.action_dim - len(tv), dtype=np.float32)]
                )
            target_vec = torch.tensor(tv, dtype=torch.float32)   # (action_dim,)

        robot_obs_t = None
        if self.has_proprio and "robot_obs" in ep:
            from data.calvin_utils import normalize_robot_obs
            ro = normalize_robot_obs(
                ep["robot_obs"][t], self.proprio_mean, self.proprio_std, self.proprio_dim
            )
            robot_obs_t = torch.tensor(ro, dtype=torch.float32)

        out = {
            "frames":          frames,           # (num_vis_frames, 3, H, W)
            "lang_tokens":     lang_tokens,      # (77,)
            "action_hist":     action_hist,      # (history_len,)
            "reward_hist":     reward_hist,      # (history_len,)
            "action_vec_hist": action_vec_hist,  # (history_len, action_dim) or None
            "state_delta":     torch.tensor(state_delta_val, dtype=torch.float32),  # scalar
            "step_idx":        torch.tensor(t, dtype=torch.long),  # rollout phase (0-based)
            "target":          torch.tensor(target, dtype=torch.long),
            "target_chunk":    target_chunk,     # (chunk_size,) discrete labels t..t+K-1
            "target_vec":      target_vec,       # (action_dim,) or None
        }
        if robot_obs_t is not None:
            out["robot_obs"] = robot_obs_t
        return out
