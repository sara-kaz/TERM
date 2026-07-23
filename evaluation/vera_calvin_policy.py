"""
VERA policy adapter for the official CALVIN evaluation protocol.

Implements CalvinBaseModel.step(obs, goal) for evaluate_policy.py rollouts.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Literal, Optional

from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image as PILImage

from evaluation.evaluate_vera import build_vera_from_cfg, load_checkpoint

ActionMode = Literal["hybrid", "discrete", "regression", "continuous"]

# Typical |rel_actions| on CALVIN D (dominant DoF); p90 ~ 0.3–0.5
DEFAULT_ACTION_MAGNITUDE = 0.45
DEFAULT_ACTION_HOLD_STEPS = 30  # MCIL replan_freq; use 1 for continuous mode
DEFAULT_GRIPPER_THRESH = 0.25


def postprocess_arm_action(rel: np.ndarray, deadzone: float = 0.02) -> np.ndarray:
    """Clip + deadzone on arm DoF only (gripper set separately)."""
    rel = np.asarray(rel, dtype=np.float32).flatten()[:7].copy()
    if rel.size < 7:
        rel = np.pad(rel, (0, 7 - rel.size))
    rel[:6] = np.clip(rel[:6], -1.0, 1.0)
    for i in range(6):
        if abs(rel[i]) < deadzone:
            rel[i] = 0.0
    return rel


def gripper_from_logits(logits: torch.Tensor) -> float:
    """Open (12) vs close (13) — independent of arm argmax."""
    g = logits.squeeze(0) if logits.dim() > 1 else logits
    return 1.0 if float(g[12]) >= float(g[13]) else -1.0


def arm_discrete_idx(logits: torch.Tensor) -> int:
    """Argmax over arm motion classes 0–11 (excludes gripper classes)."""
    arm = logits.squeeze(0)[:12] if logits.dim() > 1 else logits[:12]
    return int(arm.argmax().item())


def postprocess_rel_action(
    rel: np.ndarray,
    gripper_thresh: float = DEFAULT_GRIPPER_THRESH,
    deadzone: float = 0.02,
) -> np.ndarray:
    """CALVIN-style 7-DoF rel_actions from regression head (clip + gripper binarize)."""
    from data.calvin_utils import sanitize_rel_action_for_calvin

    rel = np.asarray(rel, dtype=np.float32).flatten()[:7]
    rel = np.clip(rel, -1.0, 1.0)
    if rel[6] > gripper_thresh:
        rel[6] = 1.0
    elif rel[6] < -gripper_thresh:
        rel[6] = -1.0
    else:
        # Keep last sign when regression is ambiguous; default open (1.0).
        rel[6] = 1.0 if rel[6] >= 0.0 else -1.0
    for i in range(6):
        if abs(rel[i]) < deadzone:
            rel[i] = 0.0
    return sanitize_rel_action_for_calvin(rel, deadzone=0.0)


def discretise_rel_action(rel_action: np.ndarray, arm_thresh: float = 0.03) -> int:
    """Match CALVIN arm-priority discretisation used in training loader."""
    rel_action = np.asarray(rel_action, dtype=np.float32).flatten()[:7]
    arm = rel_action[:6]
    if float(np.max(np.abs(arm))) >= arm_thresh:
        dom = int(np.argmax(np.abs(arm)))
        return dom * 2 + (0 if arm[dom] >= 0 else 1)
    return 12 if float(rel_action[6]) >= 0.0 else 13


def rel_action_from_discrete(idx: int, magnitude: float = DEFAULT_ACTION_MAGNITUDE) -> np.ndarray:
    """Map discrete class → single-axis normalized rel_actions (CALVIN style)."""
    a = np.zeros(7, dtype=np.float32)
    if idx == 12:
        a[6] = 1.0
    elif idx == 13:
        a[6] = -1.0
    else:
        dom = idx // 2
        sign = 1.0 if idx % 2 == 0 else -1.0
        a[dom] = sign * float(magnitude)
    return np.clip(a, -1.0, 1.0)


class VERACalvinPolicy:
    """
    CalvinBaseModel-compatible policy (duck-typed; no import of calvin_agent here).

    action_mode
    -----------
    hybrid (default, best for rollout):
        14-way argmax → single-axis motion (matches 97% offline classifier).
        Regression head refines magnitude on the active DoF when signs agree.
    discrete:
        argmax only with ``action_magnitude``.
    regression:
        tanh regression head only (legacy; gripper forced ±1).
    continuous (recommended for embodied eval):
        full 7-DoF regression vector every step (matches expert rel_actions).
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda:0",
        use_regression: bool = True,
        action_mode: ActionMode = "hybrid",
        action_magnitude: float = DEFAULT_ACTION_MAGNITUDE,
        action_hold_steps: int = DEFAULT_ACTION_HOLD_STEPS,
        reset_history_each_step: bool = False,
    ):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = ckpt.get("cfg")
        if self.cfg is None:
            raise ValueError(f"No cfg in checkpoint {checkpoint}")

        if isinstance(device, int):
            device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.device = device

        if action_mode == "continuous":
            self.action_mode: ActionMode = "continuous"
        elif action_mode == "regression":
            self.action_mode = "regression"
        elif action_mode == "discrete" or not use_regression:
            self.action_mode = "discrete"
        else:
            self.action_mode = "hybrid"

        self.action_magnitude = float(action_magnitude)
        self.action_hold_steps = max(1, int(action_hold_steps))
        self.reset_history_each_step = bool(reset_history_each_step)

        self.model = build_vera_from_cfg(self.cfg, self.device)
        load_checkpoint(self.model, checkpoint, self.device)

        m = self.cfg["model"]
        self.history_len = int(m["history_len"])
        self.num_vis = int(m.get("num_vis_frames", 3))
        self.num_actions = int(m["num_actions"])
        self.action_dim = int(m.get("action_dim", 7))
        self.pad_action = self.num_actions
        self.proprio_dim = int(m.get("proprio_dim", 0))
        self.use_proprio = self.proprio_dim > 0
        vera_cfg = self.cfg.get("vera", {})
        self.use_step_conditioning = bool(vera_cfg.get("use_step_conditioning", False))

        self.num_static_vis = int(m.get("num_vis_frames", 3))
        self.use_gripper_cam = bool(self.cfg.get("data", {}).get("use_gripper_cam", False))
        self.proprio_mean = self.proprio_std = None
        stats_path = self.cfg.get("data", {}).get("proprio_stats_path")
        if stats_path:
            from data.calvin_utils import load_proprio_stats
            sp = Path(stats_path)
            if not sp.is_absolute():
                sp = Path(__file__).resolve().parents[1] / stats_path
            if sp.exists():
                self.proprio_mean, self.proprio_std = load_proprio_stats(str(sp))

        import torchvision.transforms as Tv

        img_size = self.cfg["data"].get("img_size", 224)
        self.transform = Tv.Compose([
            Tv.Resize((img_size, img_size)),
            Tv.ToTensor(),
            Tv.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

        self._reset_internal()
        self._held_action: Optional[np.ndarray] = None
        self._hold_count = 0

        print(
            f"[VERACalvinPolicy] mode={self.action_mode}  "
            f"magnitude={self.action_magnitude:.2f}  hold_steps={self.action_hold_steps}  "
            f"reset_hist={self.reset_history_each_step}  device={self.device}",
            flush=True,
        )

    def _reset_internal(self):
        self.frame_q: Deque = deque(maxlen=self.num_static_vis)
        self.gripper_q: Deque = deque(maxlen=self.num_static_vis) if self.use_gripper_cam else None
        self.action_q: Deque[int] = deque(
            [self.pad_action] * self.history_len, maxlen=self.history_len
        )
        self.reward_q: Deque[float] = deque(
            [0.0] * self.history_len, maxlen=self.history_len
        )
        self.vec_q: Deque[np.ndarray] = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.vec_q.append(np.zeros(self.action_dim, dtype=np.float32))

        self.prev_action = self.pad_action
        self.prev_reward = 0.0
        self.prev_delta = 0.0
        self.ep_reward_max = 1e-6
        self._skip_next_reward = False
        self.lang_tokens: Optional[torch.Tensor] = None
        self.current_goal: Optional[str] = None
        self._held_action = None
        self._hold_count = 0
        self._ep_step = 0

    def reset(self):
        """Called at the start of each CALVIN subtask."""
        self._reset_internal()

    def _tokenize_goal(self, goal: str) -> torch.Tensor:
        if goal != self.current_goal or self.lang_tokens is None:
            self.current_goal = goal
            self.lang_tokens = clip.tokenize([goal], truncate=True)[0]
        return self.lang_tokens

    def _step_kw(self, device: str) -> dict:
        if not self.use_step_conditioning:
            return {}
        return {
            "step_idx": torch.tensor(
                [self._ep_step], dtype=torch.long, device=device
            )
        }

    def _extract_gripper(self, obs: dict) -> np.ndarray:
        rgb = obs.get("rgb_obs", {})
        frame = rgb.get("rgb_gripper") if isinstance(rgb, dict) else None
        if frame is None:
            raise KeyError("Expected obs['rgb_obs']['rgb_gripper']")
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def _stack_frames(self, frame_t: torch.Tensor) -> torch.Tensor:
        """Build (T,3,H,W) with optional gripper half — matches training."""
        self.frame_q.append(frame_t)
        pad_s = self.num_static_vis - len(self.frame_q)
        static = [torch.zeros_like(frame_t)] * pad_s + list(self.frame_q)
        if self.use_gripper_cam and self.gripper_q is not None:
            g = self.gripper_q[-1] if self.gripper_q else torch.zeros_like(frame_t)
            pad_g = self.num_static_vis - len(self.gripper_q)
            grip = [torch.zeros_like(g)] * pad_g + list(self.gripper_q)
            return torch.stack(static + grip)
        return torch.stack(static)

    def _extract_frame(self, obs: dict) -> np.ndarray:
        rgb = obs.get("rgb_obs", {})
        if isinstance(rgb, dict) and "rgb_static" in rgb:
            frame = rgb["rgb_static"]
        else:
            raise KeyError(
                f"Expected obs['rgb_obs']['rgb_static'], got keys {list(obs.keys())}"
            )
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def _rel_from_outputs(self, out: dict) -> np.ndarray:
        logits = out["logits"]
        reg = out.get("action_vec")
        reg_np = None
        if reg is not None:
            reg_np = reg.squeeze(0).detach().cpu().numpy().astype(np.float32)
        full_idx = int(logits.argmax(dim=-1).item())
        grip = gripper_from_logits(logits)

        # Gripper-only expert labels (12/13): execute gripper, no spurious arm motion.
        if full_idx >= 12:
            return rel_action_from_discrete(full_idx, self.action_magnitude)

        if self.action_mode == "discrete":
            rel = rel_action_from_discrete(full_idx, self.action_magnitude)
            rel[6] = grip
            return rel

        if self.action_mode == "continuous" and reg is not None:
            rel = reg_np.copy()
            rel = postprocess_arm_action(rel)
            rel[6] = 1.0 if float(rel[6]) >= 0.0 else -1.0
            return rel

        if self.action_mode == "regression" and reg is not None:
            rel = reg_np.copy()
            rel = postprocess_arm_action(rel)
            rel[6] = 1.0 if float(rel[6]) >= 0.0 else -1.0
            return rel

        # hybrid: arm structure from discrete 0–11 + regression magnitude + factored gripper
        idx = full_idx if full_idx < 12 else arm_discrete_idx(logits)
        rel = rel_action_from_discrete(idx, self.action_magnitude)
        if reg is not None and idx < 12:
            dom = idx // 2
            if abs(reg_np[dom]) > 0.02 and np.sign(reg_np[dom]) == np.sign(rel[dom]):
                mag = float(np.clip(abs(reg_np[dom]) * self.action_magnitude,
                                    self.action_magnitude * 0.35, 1.0))
                rel[dom] = np.sign(rel[dom]) * mag
        if reg is not None:
            rel[6] = 1.0 if float(reg_np[6]) >= 0.0 else -1.0
        else:
            rel[6] = grip
        return rel

    def step(self, obs: dict, goal: str) -> np.ndarray:
        if self.reset_history_each_step:
            self._reset_internal()
        if self._held_action is not None and self._hold_count < self.action_hold_steps:
            self._hold_count += 1
            self._skip_next_reward = True
            self._ep_step += 1
            return self._held_action.copy()

        frame = self._extract_frame(obs)
        frame_t = self.transform(PILImage.fromarray(frame))
        if self.use_gripper_cam and self.gripper_q is not None:
            try:
                g = self._extract_gripper(obs)
                self.gripper_q.append(self.transform(PILImage.fromarray(g)))
            except KeyError:
                self.gripper_q.append(torch.zeros_like(frame_t))
        frames_t = self._stack_frames(frame_t)

        lang = self._tokenize_goal(str(goal))
        device = self.device

        frames_in = frames_t.unsqueeze(0).to(device)
        lang_in = lang.unsqueeze(0).to(device)
        act_hist_in = torch.tensor(
            list(self.action_q), dtype=torch.long, device=device
        ).unsqueeze(0)
        rew = np.array(list(self.reward_q), dtype=np.float32)
        r_max = max(float(rew.max()), self.ep_reward_max, 1e-6)
        rew_norm = torch.tensor(
            np.clip(rew / r_max, 0.0, 1.0), dtype=torch.float32, device=device
        ).unsqueeze(0)
        delta_in = torch.tensor([self.prev_delta], dtype=torch.float32, device=device)

        avh = None
        if len(self.vec_q) == self.history_len:
            avh = torch.tensor(
                np.stack(list(self.vec_q)), dtype=torch.float32, device=device
            ).unsqueeze(0)

        robot_in = None
        if self.use_proprio:
            ro = obs.get("robot_obs")
            if ro is not None:
                from data.calvin_utils import normalize_robot_obs
                ro = normalize_robot_obs(
                    ro, self.proprio_mean, self.proprio_std, self.proprio_dim
                )
                robot_in = torch.tensor(ro, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            out = self.model(
                frames_in,
                lang_in,
                act_hist_in,
                rew_norm,
                state_delta=delta_in,
                action_vec_hist=avh,
                robot_obs=robot_in,
                **self._step_kw(device),
            )

        rel = self._rel_from_outputs(out)
        disc = discretise_rel_action(rel)
        self.action_q.append(disc)
        self.vec_q.append(rel.copy())
        self.prev_action = disc

        from data.calvin_utils import sanitize_rel_action_for_calvin

        rel = sanitize_rel_action_for_calvin(rel)
        self._held_action = rel.copy()
        self._hold_count = 0
        self._ep_step += 1
        return rel

    def note_step_reward(self, reward: float, delta: float = 0.0):
        """Append reward aligned with the last replanned action (matches SFT trainer)."""
        if self._skip_next_reward:
            self._skip_next_reward = False
            return
        r = float(reward)
        self.ep_reward_max = max(self.ep_reward_max, r, 1e-6)
        norm_r = float(np.clip(r / self.ep_reward_max, 0.0, 1.0))
        self.reward_q.append(norm_r)
        self.prev_reward = norm_r
        self.prev_delta = float(delta)
