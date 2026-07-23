"""
VERA policy adapter for the official Language-Table simulation benchmark.

Maps VERA discrete (+ optional regression) outputs to 2-DoF EE deltas in [-0.1, 0.1].
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Literal, Optional

import numpy as np
import torch
import clip
from PIL import Image as PILImage

from data.trajectory_dataset import _lt_discretise
from evaluation.evaluate_vera import build_vera_from_cfg, load_checkpoint

ActionMode = Literal["hybrid", "discrete", "regression"]
# Training data median ‖Δ‖ ≈ 0.039; p90 ≈ 0.084 (see language_table_vera stats).
DEFAULT_ACTION_MAGNITUDE = 0.05
DEFAULT_ACTION_HOLD_STEPS = 4
DEFAULT_CROP_FACTOR = 0.95


def lt_action_from_discrete(idx: int, magnitude: float = DEFAULT_ACTION_MAGNITUDE) -> np.ndarray:
    """Map 8-bin index → planar delta (matches training discretisation)."""
    angle = idx * (np.pi / 4)
    return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * float(magnitude)


def central_crop_uint8(frame: np.ndarray, crop_factor: float = DEFAULT_CROP_FACTOR) -> np.ndarray:
    """Match Language-Table eval CentralCropImageWrapper (factor=0.95)."""
    if crop_factor is None or crop_factor >= 1.0:
        return frame
    h, w = frame.shape[:2]
    nh = max(1, int(round(h * crop_factor)))
    nw = max(1, int(round(w * crop_factor)))
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    return frame[y0 : y0 + nh, x0 : x0 + nw]


class VERALanguageTablePolicy:
    """Closed-loop policy for Language-Table PyBullet env."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda:0",
        action_mode: ActionMode = "hybrid",
        action_magnitude: float = DEFAULT_ACTION_MAGNITUDE,
        action_hold_steps: int = DEFAULT_ACTION_HOLD_STEPS,
        crop_factor: float = DEFAULT_CROP_FACTOR,
    ):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = ckpt.get("cfg")
        if self.cfg is None:
            raise ValueError(f"No cfg in checkpoint {checkpoint}")

        if isinstance(device, int):
            device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.action_mode = action_mode
        self.action_magnitude = float(action_magnitude)
        self.action_hold_steps = max(1, int(action_hold_steps))
        data_cfg = self.cfg.get("data", {})
        self.crop_factor = float(data_cfg.get("crop_factor", crop_factor))

        self.model = build_vera_from_cfg(self.cfg, self.device)
        load_checkpoint(self.model, checkpoint, self.device)
        self.chunk_size = int(getattr(self.model, "chunk_size", 1))

        m = self.cfg["model"]
        self.history_len = int(m["history_len"])
        self.num_vis = int(m.get("num_vis_frames", 3))
        self.num_actions = int(m["num_actions"])
        self.action_dim = int(m.get("action_dim", 2))
        self.pad_action = self.num_actions
        vera_cfg = self.cfg.get("vera", {})
        self.use_step_conditioning = bool(vera_cfg.get("use_step_conditioning", False))

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
            f"[VERALanguageTablePolicy] mode={self.action_mode}  "
            f"magnitude={self.action_magnitude:.3f}  hold_steps={self.action_hold_steps}  "
            f"crop={self.crop_factor}  chunk_size={self.chunk_size}  device={self.device}",
            flush=True,
        )

    def _reset_internal(self):
        self.frame_q: Deque = deque(maxlen=self.num_vis)
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
        self.current_instruction: Optional[str] = None
        self._held_action = None
        self._hold_count = 0
        self._ep_step = 0

    def reset(self):
        """Called at the start of each Language-Table episode."""
        self._reset_internal()

    def _tokenize_instruction(self, instruction: str) -> torch.Tensor:
        if instruction != self.current_instruction or self.lang_tokens is None:
            self.current_instruction = instruction
            self.lang_tokens = clip.tokenize([instruction], truncate=True)[0]
        return self.lang_tokens

    def _step_kw(self, device: str) -> dict:
        if not self.use_step_conditioning:
            return {}
        return {
            "step_idx": torch.tensor(
                [self._ep_step], dtype=torch.long, device=device
            )
        }

    @staticmethod
    def decode_instruction(obs: dict) -> str:
        instr = obs.get("instruction")
        if instr is None:
            return "complete the task"
        try:
            from language_table.environments import language_table as lt_env

            return lt_env.LanguageTable.decode_instruction(np.asarray(instr))
        except Exception:
            if isinstance(instr, (bytes, bytearray)):
                return instr.decode("utf-8", errors="replace")
            return str(instr)

    def _extract_frame(self, obs: dict) -> np.ndarray:
        frame = obs.get("rgb")
        if frame is None:
            raise KeyError(f"Expected obs['rgb'], got keys {list(obs.keys())}")
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = central_crop_uint8(frame, self.crop_factor)
        return frame

    def _reward_hist_tensor(self, device: str) -> torch.Tensor:
        rew = np.array(list(self.reward_q), dtype=np.float32)
        r_max = max(float(rew.max()), self.ep_reward_max, 1e-6)
        rew_norm = np.clip(rew / r_max, 0.0, 1.0)
        return torch.tensor(rew_norm, dtype=torch.float32, device=device).unsqueeze(0)

    def _planar_from_outputs(self, out: dict) -> np.ndarray:
        logits = out.get("logits")
        if logits is None:
            raise KeyError("model output missing 'logits'")

        reg = out.get("action_vec")
        if self.action_mode == "regression" and reg is not None:
            rel = reg.squeeze(0).detach().cpu().numpy().astype(np.float32)[:2]
            return np.clip(rel, -1.0, 1.0) * self.action_magnitude

        idx = int(logits.argmax(dim=-1).item())

        if self.action_mode == "discrete":
            return lt_action_from_discrete(idx, self.action_magnitude)

        # hybrid: discrete direction + regression magnitude (training uses both heads)
        rel = lt_action_from_discrete(idx, self.action_magnitude)
        if reg is not None:
            reg_np = reg.squeeze(0).detach().cpu().numpy().astype(np.float32)[:2]
            reg_np = np.clip(reg_np, -1.0, 1.0) * self.action_magnitude
            if np.linalg.norm(reg_np) > 1e-4:
                if np.dot(rel, reg_np) > 0:
                    rel = reg_np
                else:
                    # Opposite signs: trust discrete direction, scale by |reg|
                    rel = rel / max(np.linalg.norm(rel), 1e-6) * min(
                        np.linalg.norm(reg_np), self.action_magnitude
                    )
        return np.clip(rel, -0.1, 0.1)

    def _infer_action(self, obs: dict) -> np.ndarray:
        """Single forward pass → one planar action (matches chunk_size=1 training)."""
        frame = self._extract_frame(obs)
        frame_t = self.transform(PILImage.fromarray(frame))
        self.frame_q.append(frame_t)
        pad = self.num_vis - len(self.frame_q)
        frames_t = torch.stack([torch.zeros_like(frame_t)] * pad + list(self.frame_q))

        instruction = self.decode_instruction(obs)
        lang = self._tokenize_instruction(instruction)
        device = self.device

        frames_in = frames_t.unsqueeze(0).to(device)
        lang_in = lang.unsqueeze(0).to(device)
        act_hist_in = torch.tensor(
            list(self.action_q), dtype=torch.long, device=device
        ).unsqueeze(0)
        rew_hist_in = self._reward_hist_tensor(device)
        delta_in = torch.tensor([self.prev_delta], dtype=torch.float32, device=device)

        avh = None
        if len(self.vec_q) == self.history_len:
            avh = torch.tensor(
                np.stack(list(self.vec_q)), dtype=torch.float32, device=device
            ).unsqueeze(0)

        with torch.no_grad():
            out = self.model(
                frames_in,
                lang_in,
                act_hist_in,
                rew_hist_in,
                state_delta=delta_in,
                action_vec_hist=avh,
                **self._step_kw(device),
            )

        logits_chunk = out.get("logits_chunk")
        if logits_chunk is not None and logits_chunk.ndim == 3:
            sub = {"logits": logits_chunk[:, 0, :], "action_vec": out.get("action_vec")}
        else:
            sub = out
        return self._planar_from_outputs(sub)

    def step(self, obs: dict) -> np.ndarray:
        if self._held_action is not None and self._hold_count < self.action_hold_steps:
            self._hold_count += 1
            self._skip_next_reward = True
            self._ep_step += 1
            return self._held_action.copy()

        planar = self._infer_action(obs)

        disc = _lt_discretise(planar)
        if disc >= 0:
            self.action_q.append(disc)
            self.vec_q.append(planar.copy())
            self.prev_action = disc

        self._held_action = planar.copy()
        self._hold_count = 0
        self._ep_step += 1
        return planar

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
