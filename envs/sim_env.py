"""
SimEnv — Gym-compatible simulation environment wrapper
======================================================
Wraps any OpenAI Gym / Gymnasium environment and exposes the
VLA-friendly interface:

    obs  = env.reset()   -> {"frame": np.ndarray (H,W,3), "instruction": str}
    obs, reward, done, info = env.step(action_idx)

Supports:
  - Any Gym env that returns pixel observations (render_mode="rgb_array")
  - FrankaKitchen, MiniGrid, Meta-World, and custom envs via the adapter pattern
  - Domain randomization hooks for sim-to-real transfer

For real robot use, swap `SimEnv` for `RealEnv` defined at the bottom of this file.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, Optional, List


# ── Base interface ─────────────────────────────────────────────────────────────

class BaseEnv:
    """Minimal interface every environment must implement."""

    def reset(self) -> Dict[str, Any]:
        """Returns {"frame": np.ndarray (H,W,3) uint8, "instruction": str}"""
        raise NotImplementedError

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Returns (obs, reward, done, info)."""
        raise NotImplementedError

    def close(self):
        pass


# ── Synthetic dummy env (no Gym required) ─────────────────────────────────────

class RandomDummyEnv(BaseEnv):
    """
    Minimal dummy environment for unit-testing the training pipeline
    without installing Gym/MuJoCo.

    Actions: 0..num_actions-1
    Reward:  +1 if action matches a hidden target, else 0
    Done:    after max_steps
    """

    INSTRUCTIONS = [
        "pick up the red cube",
        "move to the left side",
        "push the block forward",
        "grasp the cylinder and place it on the shelf",
    ]

    def __init__(self, num_actions: int = 8, max_steps: int = 30, img_size: int = 64):
        self.num_actions = num_actions
        self.max_steps   = max_steps
        self.img_size    = img_size
        self._step       = 0
        self._target     = 0
        self._instruction = ""

    def reset(self) -> Dict[str, Any]:
        self._step        = 0
        self._target      = np.random.randint(0, self.num_actions)
        self._instruction = np.random.choice(self.INSTRUCTIONS)
        return {
            "frame":       np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8),
            "instruction": self._instruction,
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        self._step += 1
        reward = 1.0 if action == self._target else -0.1
        done   = (self._step >= self.max_steps) or (action == self._target)
        obs    = {
            "frame":       np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8),
            "instruction": self._instruction,
        }
        return obs, reward, done, {"target": self._target}


# ── Gym wrapper ────────────────────────────────────────────────────────────────

class SimEnv(BaseEnv):
    """
    Wraps a Gym environment.  The env must support render_mode="rgb_array".

    If the Gym env cannot be imported (not installed), falls back to
    RandomDummyEnv so the rest of the pipeline can still be tested.

    Supported envs (examples):
      "MiniGrid-Empty-5x5-v0"       — needs pip install minigrid
      "FrankaKitchen-v1"            — needs pip install gym-robotics
      "FetchReach-v2"               — needs pip install gym-robotics
      "CartPole-v1"                 — needs pip install gymnasium

    Set env_id: "dummy" to always use RandomDummyEnv.
    """

    # Map action counts to Gym discrete action spaces
    ACTION_MAPS: Dict[str, List[int]] = {}

    def __init__(self, cfg: dict):
        env_cfg     = cfg.get("env", {})
        env_id      = env_cfg.get("env_id", "dummy")
        num_actions = cfg["model"]["num_actions"]
        img_size    = cfg["data"].get("img_size", 64)

        self._gym_env = None
        self._instruction = env_cfg.get("instruction", "complete the task")
        self._img_size    = img_size

        if env_id == "dummy":
            self._dummy = RandomDummyEnv(
                num_actions=num_actions,
                max_steps=cfg["rl"].get("max_episode_steps", 50),
                img_size=img_size,
            )
            return

        try:
            import gymnasium as gym
            self._gym_env = gym.make(env_id, render_mode="rgb_array")
            print(f"[SimEnv] Loaded Gym env: {env_id}")
        except Exception as e:
            print(f"[SimEnv] Could not load '{env_id}': {e}. Falling back to dummy env.")
            self._dummy = RandomDummyEnv(
                num_actions=num_actions,
                max_steps=cfg["rl"].get("max_episode_steps", 50),
                img_size=img_size,
            )

        # Domain randomization settings
        self._domain_rand = env_cfg.get("domain_randomization", False)
        self._noise_std   = env_cfg.get("obs_noise_std", 5.0)

    def reset(self) -> Dict[str, Any]:
        if self._gym_env is None:
            return self._dummy.reset()
        self._gym_env.reset()
        frame = self._render_frame()
        return {"frame": frame, "instruction": self._instruction}

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        if self._gym_env is None:
            return self._dummy.step(action)
        _, reward, terminated, truncated, info = self._gym_env.step(action)
        done  = terminated or truncated
        frame = self._render_frame()
        return {"frame": frame, "instruction": self._instruction}, float(reward), done, info

    def _render_frame(self) -> np.ndarray:
        frame = self._gym_env.render()                              # (H, W, 3)
        if self._domain_rand:
            noise = np.random.normal(0, self._noise_std, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if frame.shape[:2] != (self._img_size, self._img_size):
            from PIL import Image
            frame = np.array(Image.fromarray(frame).resize((self._img_size, self._img_size)))
        return frame

    def close(self):
        if self._gym_env is not None:
            self._gym_env.close()


# ── MetaWorld wrapper ──────────────────────────────────────────────────────────

class MetaWorldEnv(BaseEnv):
    """
    Wraps a Meta-World ML1 / MT50 task and exposes the VLA interface.

    Key additions over the plain Gym wrapper:
      • `info["dist_delta"]` — signed change in end-effector–to–goal Euclidean
        distance between consecutive steps. Positive = moved away, negative = closer.
        This populates Stream 3b (consequence language encoder) with a real signal.
      • Action discretisation — MetaWorld has a continuous 4-DoF action space.
        We project it to `num_actions` discrete bins via a fixed codebook built
        from a uniform grid over [-1, 1]^4.
      • Language instruction — generated from the task name automatically.

    Install: pip install metaworld

    Example config:
      env:
        env_id: metaworld-reach-v2    # any metaworld task name
        num_actions: 8                # discrete bins
    """

    # Task name → human-readable instruction
    _TASK_INSTRUCTIONS: Dict[str, str] = {
        "reach-v2":          "move the robot arm to reach the target position",
        "push-v2":           "push the puck to the target location",
        "pick-place-v2":     "pick up the object and place it at the target",
        "door-open-v2":      "grasp the door handle and open the door",
        "drawer-close-v2":   "push the drawer closed",
        "drawer-open-v2":    "pull the drawer open",
        "button-press-v2":   "press the button down",
        "peg-insert-side-v2":"insert the peg into the hole from the side",
        "window-open-v2":    "slide the window open",
        "window-close-v2":   "slide the window closed",
    }

    def __init__(self, cfg: dict):
        import metaworld
        env_cfg      = cfg.get("env", {})
        task_name    = env_cfg.get("env_id", "reach-v2").replace("metaworld-", "")
        num_actions  = cfg["model"]["num_actions"]
        self._img    = cfg["data"].get("img_size", 224)

        ml1 = metaworld.ML1(task_name)
        self._env = ml1.train_classes[task_name]()
        task      = np.random.choice(ml1.train_tasks)
        self._env.set_task(task)

        self._task_name  = task_name
        self._num_actions = num_actions
        self._instruction = self._TASK_INSTRUCTIONS.get(
            task_name, f"complete the {task_name.replace('-', ' ')} task"
        )

        # Build discrete action codebook: num_actions centroids in [-1,1]^4
        self._action_dim  = 4    # MetaWorld: xyz delta + gripper
        self._codebook    = self._build_codebook(num_actions)
        self._prev_dist   = None

        print(f"[MetaWorldEnv] Task: {task_name} | "
              f"Actions: {num_actions} discrete bins | "
              f"Instruction: \"{self._instruction}\"")

    def _build_codebook(self, num_actions: int) -> np.ndarray:
        """
        Simple codebook: evenly space `num_actions` points across the
        action space principal directions. Each action moves along a
        different axis or combination.
        """
        rng     = np.linspace(-1, 1, max(2, int(np.ceil(num_actions ** 0.25))))
        grid    = np.array(np.meshgrid(rng, rng, rng, rng)).T.reshape(-1, 4)
        idx     = np.linspace(0, len(grid) - 1, num_actions, dtype=int)
        return grid[idx].astype(np.float32)

    def _dist_to_goal(self) -> float:
        obs  = self._env._get_obs()
        hand = obs[:3]
        goal = self._env._get_pos_goal()
        return float(np.linalg.norm(hand - goal))

    def reset(self) -> Dict[str, Any]:
        self._env.reset()
        self._prev_dist = self._dist_to_goal()
        frame = self._render(self._img)
        return {"frame": frame, "instruction": self._instruction}

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        cont_action = self._codebook[action_idx % self._num_actions]
        _, reward, done, info = self._env.step(cont_action)
        frame = self._render(self._img)

        curr_dist  = self._dist_to_goal()
        dist_delta = float(curr_dist - self._prev_dist)   # negative = closer
        self._prev_dist = curr_dist

        # Expose the executed codebook vector so the RL rollout collector can
        # feed it into Stream 4 (ActionRewardHistoryEncoder) at the next step.
        # Shape: (action_dim,) = (4,) for MetaWorld [Δx, Δy, Δz, gripper].
        info["dist_delta"]    = dist_delta
        info["action_vector"] = cont_action.copy()   # numpy array (4,)
        obs = {"frame": frame, "instruction": self._instruction}
        return obs, float(reward), bool(done), info

    def _render(self, size: int) -> np.ndarray:
        frame = self._env.render(offscreen=True)
        if frame.shape[:2] != (size, size):
            from PIL import Image
            frame = np.array(Image.fromarray(frame).resize((size, size)))
        return frame

    def close(self):
        self._env.close()


# ── BabyAI / MiniGrid wrapper ─────────────────────────────────────────────────

class BabyAIEnv(BaseEnv):
    """
    Wraps a BabyAI / MiniGrid environment for cheap language-grounded RL.

    Why BabyAI for VLLA?
      • Procedurally generated language instructions ("go to the red ball")
      • Discrete action space (7 actions) — maps cleanly to verbalization
      • Dense enough episodes (H~100 steps) to exercise the history encoder
      • dist_delta = L1 distance change to goal object, exposable per step
      • Fast to run: 1000 episodes in <1min on CPU

    Install: pip install minigrid

    Config:
      env:
        env_id: babyai-GoToLocal-v0
        num_actions: 7

    Exposed info keys:
      dist_delta : signed change in Manhattan distance to target object
                   (negative = closer, positive = farther)
      mission    : the full language instruction string
    """

    # 7 MiniGrid primitive actions (matches minigrid.core.actions.Actions)
    ACTION_NAMES = [
        "turn left",
        "turn right",
        "move forward",
        "pick up the object",
        "drop the object",
        "toggle the door or switch",
        "done",
    ]

    def __init__(self, cfg: dict):
        env_cfg    = cfg.get("env", {})
        env_id     = env_cfg.get("env_id", "BabyAI-GoToLocal-v0")
        self._img  = cfg["data"].get("img_size", 224)

        try:
            import gymnasium as gym
            self._env = gym.make(env_id, render_mode="rgb_array")
            print(f"[BabyAIEnv] Loaded: {env_id}")
        except Exception as e:
            raise RuntimeError(f"BabyAI env '{env_id}' could not be loaded: {e}") from e

        self._instruction  = ""
        self._target_pos   = None
        self._prev_dist    = None

    def reset(self) -> Dict[str, Any]:
        obs, _ = self._env.reset()
        self._instruction = obs.get("mission", "complete the task")
        self._target_pos  = self._find_target()
        self._prev_dist   = self._agent_dist_to_target()
        frame = self._render()
        return {"frame": frame, "instruction": self._instruction}

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        obs, reward, terminated, truncated, info = self._env.step(action_idx)
        done = terminated or truncated

        self._instruction = obs.get("mission", self._instruction)
        self._target_pos  = self._find_target()
        curr_dist         = self._agent_dist_to_target()
        dist_delta        = float(curr_dist - self._prev_dist)
        self._prev_dist   = curr_dist

        info["dist_delta"] = dist_delta
        frame = self._render()
        return {"frame": frame, "instruction": self._instruction}, float(reward), done, info

    def _find_target(self) -> Optional[Tuple[int, int]]:
        """Return grid position of the mission target object if findable."""
        try:
            grid = self._env.unwrapped.grid
            for i in range(grid.width):
                for j in range(grid.height):
                    cell = grid.get(i, j)
                    if cell is not None and cell.type not in ("wall", "floor", "door"):
                        return (i, j)
        except Exception:
            pass
        return None

    def _agent_dist_to_target(self) -> float:
        if self._target_pos is None:
            return 0.0
        agent_pos = self._env.unwrapped.agent_pos
        return float(abs(agent_pos[0] - self._target_pos[0])
                   + abs(agent_pos[1] - self._target_pos[1]))

    def _render(self) -> np.ndarray:
        frame = self._env.render()   # (H, W, 3)
        if frame.shape[:2] != (self._img, self._img):
            from PIL import Image
            frame = np.array(Image.fromarray(frame).resize((self._img, self._img)))
        return frame

    def close(self):
        self._env.close()


# ── Language-Table wrapper ────────────────────────────────────────────────────

class LanguageTableEnv(BaseEnv):
    """
    Language-Table PyBullet environment wrapper for VERA RL training.

    Adapts the dm_env/TimeStep interface from DeepMind's language_table package
    into the VERA BaseEnv interface:
      reset() → {"frame": np.ndarray (H,W,3) uint8, "instruction": str}
      step(action_idx: int) → (obs, reward, done, info)

    Install:
      pip install git+https://github.com/google-deepmind/language_table.git
      pip install pybullet dm-env

    Config keys used:
      model.num_actions          : must be 8 (arctan2 directional bins)
      model.action_dim           : must be 2 ([Δx, Δy])
      env.action_magnitude       : continuous step size (default 0.03)
      rl.max_episode_steps       : episode horizon
    """

    def __init__(self, cfg: dict):
        from language_table.environments import language_table as lt_lib
        from language_table.environments.rewards import (
            block2absoluteposition,
        )

        env_cfg   = cfg.get("env", {})
        self._max = cfg["rl"].get("max_episode_steps", 50)
        self._mag = float(env_cfg.get("action_magnitude", 0.03))
        self._step_count = 0

        self._env = lt_lib.LanguageTable(
            block_mode    = lt_lib.LanguageTableBlockMode.SEPARATE,
            reward_factory= block2absoluteposition.BlockToAbsolutePositionReward,
            seed          = int(env_cfg.get("seed", 0)),
        )
        self._instruction = "push the block to the target"
        self._prev_obs    = None

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _decode_instruction(raw) -> str:
        if raw is None:
            return "push the block to the target"
        if isinstance(raw, (bytes, bytearray)):
            return raw.decode("utf-8", errors="replace").strip() or "push the block to the target"
        if isinstance(raw, np.ndarray):
            flat = raw.flatten()
            if flat.dtype in (np.uint8, np.int8):
                codes = [int(c) for c in flat if int(c) != 0]
                try:
                    return bytes(codes).decode("utf-8", errors="replace").strip()
                except Exception:
                    pass
        return str(raw).strip() or "push the block to the target"

    def _discrete_to_continuous(self, idx: int) -> np.ndarray:
        angle = idx * (np.pi / 4)
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * self._mag

    def _obs_from_ts(self, ts) -> Dict[str, Any]:
        obs_dict = ts.observation if hasattr(ts, "observation") else ts
        frame = np.asarray(obs_dict.get("rgb", np.zeros((180, 320, 3), dtype=np.uint8)),
                           dtype=np.uint8)
        raw_instr = obs_dict.get("instruction", None)
        if raw_instr is not None:
            self._instruction = self._decode_instruction(raw_instr)
        return {"frame": frame, "instruction": self._instruction}

    # ── interface ─────────────────────────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        ts = self._env.reset()
        self._step_count = 0
        return self._obs_from_ts(ts)

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        cont = self._discrete_to_continuous(action_idx)
        ts   = self._env.step(cont)
        self._step_count += 1
        reward = float(ts.reward) if ts.reward is not None else 0.0
        done   = bool(ts.last()) or self._step_count >= self._max
        obs    = self._obs_from_ts(ts)
        info   = {"dist_delta": 0.0, "action_vector": cont.copy()}
        return obs, reward, done, info

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


# ── Minimal PyBullet push env (no language_table package required) ─────────────

class MinimalPushEnv(BaseEnv):
    """
    Pure-PyBullet block-pushing environment.
    Fallback used when google-deepmind/language_table cannot be installed.

    Mimics Language-Table's interface exactly:
    - One colored block on a flat plane
    - Random target direction chosen each episode
    - 8-bin directional push actions (same arctan2 sectors as LT)
    - Shaped reward: progress toward target + 1.0 success bonus at dist < 0.05m

    Only dependency: pybullet (pip install pybullet).
    """

    INSTRUCTIONS = [
        "push the block to the right",
        "push the block up and to the right",
        "push the block upward",
        "push the block up and to the left",
        "push the block to the left",
        "push the block down and to the left",
        "push the block downward",
        "push the block down and to the right",
    ]

    def __init__(self, cfg: dict):
        import pybullet as p
        import pybullet_data
        self._p   = p
        self._pbd = pybullet_data

        env_cfg   = cfg.get("env", {})
        self._max = cfg["rl"].get("max_episode_steps", 50)
        self._mag = float(env_cfg.get("action_magnitude", 0.03))
        self._img = cfg["data"].get("img_size", 224)

        self._client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)

        self._block_id    = None
        self._step_count  = 0
        self._prev_dist   = 1.0
        self._target_pos  = np.zeros(2, dtype=np.float32)
        self._instruction = "push the block to the right"

        self._setup_scene()

        # Precompute camera matrices once
        self._view_mat = p.computeViewMatrix(
            cameraEyePosition   = [0, 0, 0.8],
            cameraTargetPosition= [0, 0, 0],
            cameraUpVector      = [1, 0, 0],
            physicsClientId     = self._client,
        )
        self._proj_mat = p.computeProjectionMatrixFOV(
            fov=70, aspect=1.0, nearVal=0.01, farVal=5.0,
            physicsClientId=self._client,
        )
        print("[MinimalPushEnv] Pure-PyBullet push env ready (language_table not installed).")

    def _setup_scene(self):
        p, cid = self._p, self._client
        p.resetSimulation(physicsClientId=cid)
        p.setGravity(0, 0, -9.8, physicsClientId=cid)
        p.loadURDF("plane.urdf", physicsClientId=cid)

        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03], physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03],
                                   rgbaColor=[1.0, 0.5, 0.0, 1.0], physicsClientId=cid)
        self._block_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0.0, 0.0, 0.03],
            physicsClientId=cid,
        )

    def _block_xy(self) -> np.ndarray:
        pos, _ = self._p.getBasePositionAndOrientation(self._block_id, physicsClientId=self._client)
        return np.array(pos[:2], dtype=np.float32)

    def _render_frame(self) -> np.ndarray:
        p, cid = self._p, self._client
        _, _, rgb, _, _ = p.getCameraImage(
            self._img, self._img,
            self._view_mat, self._proj_mat,
            physicsClientId=cid,
        )
        return np.array(rgb, dtype=np.uint8)[:, :, :3]

    def reset(self) -> Dict[str, Any]:
        p, cid = self._p, self._client

        bx = float(np.random.uniform(-0.12, 0.12))
        by = float(np.random.uniform(-0.12, 0.12))
        p.resetBasePositionAndOrientation(self._block_id, [bx, by, 0.03], [0, 0, 0, 1],
                                          physicsClientId=cid)
        for _ in range(5):
            p.stepSimulation(physicsClientId=cid)

        dir_idx = int(np.random.randint(0, 8))
        self._instruction = self.INSTRUCTIONS[dir_idx]

        angle = dir_idx * (np.pi / 4)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        self._target_pos = self._block_xy() + direction * 0.15

        self._step_count = 0
        self._prev_dist  = float(np.linalg.norm(self._block_xy() - self._target_pos))

        return {"frame": self._render_frame(), "instruction": self._instruction}

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        p, cid = self._p, self._client

        angle = action_idx * (np.pi / 4)
        force_scale = self._mag * 80.0
        p.applyExternalForce(
            self._block_id, -1,
            [np.cos(angle) * force_scale, np.sin(angle) * force_scale, 0.0],
            [0.0, 0.0, 0.0],
            p.WORLD_FRAME,
            physicsClientId=cid,
        )
        for _ in range(12):
            p.stepSimulation(physicsClientId=cid)

        block_pos  = self._block_xy()
        dist       = float(np.linalg.norm(block_pos - self._target_pos))
        reward     = float(self._prev_dist - dist)           # positive = got closer
        dist_delta = dist - self._prev_dist                  # negative = closer
        self._prev_dist = dist
        self._step_count += 1

        success = dist < 0.05
        if success:
            reward += 1.0

        done       = success or self._step_count >= self._max
        action_vec = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * self._mag
        info       = {"dist_delta": dist_delta, "action_vector": action_vec, "success": success}

        return {"frame": self._render_frame(), "instruction": self._instruction}, reward, done, info

    def close(self):
        try:
            self._p.disconnect(physicsClientId=self._client)
        except Exception:
            pass


# ── factory ────────────────────────────────────────────────────────────────────

def make_env(cfg: dict) -> BaseEnv:
    """
    Build the correct environment from config.

    env_id routing:
      "dummy"                  → RandomDummyEnv   (no install required)
      "language_table" / "lt"  → LanguageTableEnv, falls back to MinimalPushEnv
      "babyai-*" / "BabyAI-*" → BabyAIEnv        (pip install minigrid)
      "metaworld-*"            → MetaWorldEnv      (pip install metaworld)
      anything else            → SimEnv (Gymnasium wrapper)
    """
    env_id = cfg.get("env", {}).get("env_id", "dummy")
    if env_id == "dummy":
        return SimEnv(cfg)
    if env_id.lower() in ("language_table", "lt", "languagetable"):
        try:
            return LanguageTableEnv(cfg)
        except ImportError:
            print("[make_env] language_table package not installed; "
                  "using MinimalPushEnv (pure PyBullet).")
            return MinimalPushEnv(cfg)
    if env_id.lower().startswith(("babyai", "minigrid")):
        return BabyAIEnv(cfg)
    if env_id.lower().startswith("metaworld"):
        return MetaWorldEnv(cfg)
    return SimEnv(cfg)


# ── Real robot environment stub ────────────────────────────────────────────────

class RealEnv(BaseEnv):
    """
    Stub for a real robot interface.
    Replace the body of each method with your robot SDK calls.

    Expected hardware interface:
      - camera: returns BGR frame from cv2 (converted to RGB here)
      - robot:  accepts discrete action index, returns done flag + reward signal
    """

    def __init__(self, cfg: dict):
        self._instruction = cfg.get("env", {}).get("instruction", "complete the task")
        self._img_size    = cfg["data"].get("img_size", 224)
        # TODO: initialize camera, robot arm SDK, reward sensor here

    def reset(self) -> Dict[str, Any]:
        # TODO: move robot to home position, reset sensors
        frame = self._capture_frame()
        return {"frame": frame, "instruction": self._instruction}

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict]:
        # TODO: send `action` to robot hardware
        # reward = read from force/torque sensor or task completion logic
        reward = 0.0
        done   = False
        frame  = self._capture_frame()
        return {"frame": frame, "instruction": self._instruction}, reward, done, {}

    def _capture_frame(self) -> np.ndarray:
        # TODO: replace with actual camera capture
        # import cv2
        # ret, frame = self._cap.read()
        # frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self._img_size, self._img_size))
        return np.zeros((self._img_size, self._img_size, 3), dtype=np.uint8)
