"""ACP value-based reward wrapper for online RL.

Intercepts dual-camera images from ManiSkill sensor_data + render(),
computes V(s) using the ACP value model, and provides TD-shaped rewards
r(s, s') = (V(s') - V(s)) * reward_scale to replace simulator dense reward.

This is potential-based reward shaping (Ng et al., 1999) which preserves the
optimal policy while providing a learned dense signal from the ACP value model.

Wrapper ordering: raw_env -> DualCameraRewardWrapper -> FlattenRGBDObservationWrapper
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import gymnasium as gym
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _np(x: Any) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def _get_render_frame(
    env: gym.Env,
    num_envs: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Get render camera frame (second viewpoint), resized to target resolution.

    Args:
        env: ManiSkill environment (must have render_mode="rgb_array").
        num_envs: Number of parallel environments.
        height: Target image height.
        width: Target image width.

    Returns:
        (num_envs, height, width, 3) uint8 array. Zeros on failure.
    """
    fallback = np.zeros((num_envs, height, width, 3), dtype=np.uint8)
    try:
        render_out = env.render()
        if isinstance(render_out, torch.Tensor):
            render_out = render_out.cpu().numpy()
        if render_out is None:
            return fallback
        if render_out.ndim == 4:
            rgb_render = render_out.astype(np.uint8)
        else:
            # Single env: broadcast to (N, H, W, 3)
            rgb_render = np.stack([render_out] * num_envs).astype(np.uint8)
        # Resize if resolution does not match target
        if rgb_render.shape[1] != height or rgb_render.shape[2] != width:
            from PIL import Image as PILImage
            resized = np.zeros((num_envs, height, width, 3), dtype=np.uint8)
            for i in range(num_envs):
                resized[i] = np.asarray(
                    PILImage.fromarray(rgb_render[i]).resize(
                        (width, height), PILImage.BILINEAR
                    )
                )
            rgb_render = resized
        return rgb_render
    except Exception as e:
        logger.warning(f"Failed to get render frame: {e}")
        return fallback


def _extract_base_camera(
    obs: dict,
    num_envs: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Extract base_camera RGB from raw ManiSkill obs (before flattening).

    Args:
        obs: Raw ManiSkill observation dict with sensor_data.
        num_envs: Number of parallel environments.
        height: Target image height.
        width: Target image width.

    Returns:
        (num_envs, height, width, 3) uint8 array.
    """
    sensor_data = obs.get("sensor_data", {})
    base_rgb = sensor_data.get("base_camera", {}).get("rgb")

    if base_rgb is not None:
        rgb_base = _np(base_rgb).astype(np.uint8)  # (N, H, W, 3)
        # Resize if needed
        if rgb_base.shape[1] != height or rgb_base.shape[2] != width:
            from PIL import Image as PILImage
            resized = np.zeros((num_envs, height, width, 3), dtype=np.uint8)
            for i in range(num_envs):
                resized[i] = np.asarray(
                    PILImage.fromarray(rgb_base[i]).resize(
                        (width, height), PILImage.BILINEAR
                    )
                )
            rgb_base = resized
        return rgb_base
    else:
        return np.zeros((num_envs, height, width, 3), dtype=np.uint8)


@dataclass
class ACPRewardConfig:
    """Configuration for ACP reward wrapper."""

    checkpoint_path: str = "checkpoints/vlaw/acp/v3_so/best.safetensors"
    """ACP value model checkpoint (safetensors format)."""

    camera_height: int = 128
    """Target camera image height for ACP model."""

    camera_width: int = 128
    """Target camera image width for ACP model."""

    task_instruction: str = "Pick up the peg and lift it upright."
    """Task instruction text for the ACP Gemma encoder."""

    reward_scale: float = 100.0
    """Multiplier for ACP rewards. V values are in [-1, 0], per-step diffs
    are O(0.005-0.05), so scale ~100 brings them to O(0.5-5.0) range."""

    reward_shaping: Literal["td", "potential"] = "td"
    """Reward shaping mode:
    - 'td': r = (V(s') - V(s)) * scale  (difference-based, zero at steady state)
    - 'potential': r = V(s') * scale  (absolute value, continuous signal at success)
    """

    reward_clip: float = 5.0
    """Clip ACP reward to [-clip, +clip]. 0 = no clipping.
    v5 validated: clip=5 yields SAE=70% (AWSC best)."""

    grasp_bonus: float = 0.0
    """Per-step bonus when gripper is grasping the target object. 0=disabled.
    Added for v6 PLD/DSRL SAE experiments. Requires sim env with is_grasping() API."""

    device: str = "cuda:1"
    """Device for ACP model. Default cuda:1 to separate from RL training GPU."""

    dtype: str = "bfloat16"
    """ACP model dtype (bfloat16 for efficiency)."""


class ACPRewardComputer:
    """Stateful ACP value model that computes TD-shaped rewards.

    Wraps ``ManiSkillValueModel`` from the ACP module. Caches V(s_t) from the
    previous step and computes ``r = (V(s_{t+1}) - V(s_t)) * reward_scale``.

    Args:
        config: ACP reward configuration.
    """

    def __init__(self, config: ACPRewardConfig) -> None:
        self.config = config
        self._value_model: Optional[Any] = None
        self._prev_values: Optional[np.ndarray] = None
        self._num_envs: int = 0
        self._loaded = False
        self._last_details: dict[str, np.ndarray | float | str] = {}

    def _ensure_loaded(self) -> None:
        """Lazy-load the ACP value model on first use."""
        if self._loaded:
            return
        from rlft.acp.config import ValueModelConfig
        from rlft.acp.value_model import ManiSkillValueModel

        cfg = ValueModelConfig(
            task_instruction=self.config.task_instruction,
            dtype=self.config.dtype,
        )
        self._value_model = ManiSkillValueModel(cfg, device=self.config.device)
        self._value_model.load(self.config.checkpoint_path)
        self._loaded = True
        logger.info(
            f"ACPRewardComputer loaded model from {self.config.checkpoint_path} "
            f"on {self.config.device}"
        )

    def _predict_values(
        self, rgb_base: np.ndarray, rgb_render: np.ndarray
    ) -> np.ndarray:
        """Run ACP value model inference on dual-camera images.

        Args:
            rgb_base: (N, H, W, 3) uint8 base camera images.
            rgb_render: (N, H, W, 3) uint8 render camera images.

        Returns:
            (N,) float32 predicted values in [-1, 0].
        """
        self._ensure_loaded()

        N = rgb_base.shape[0]
        # Stack cameras: (N, 2, 3, H, W) — ACP expects CHW float
        base_chw = torch.from_numpy(rgb_base).permute(0, 3, 1, 2).float()  # (N, 3, H, W)
        render_chw = torch.from_numpy(rgb_render).permute(0, 3, 1, 2).float()  # (N, 3, H, W)
        images = torch.stack([base_chw, render_chw], dim=1)  # (N, 2, 3, H, W)
        image_mask = torch.ones(N, 2, dtype=torch.bool)

        values = self._value_model.predict_values(images, image_mask)  # (N,)
        return values.cpu().numpy().astype(np.float32)

    def reset(self, num_envs: int) -> None:
        """Reset all cached values (called on env.reset()).

        Args:
            num_envs: Number of parallel environments.
        """
        self._num_envs = num_envs
        self._prev_values = None

    def reset_env(self, env_indices: np.ndarray) -> None:
        """Clear cached values for specific envs that auto-reset.

        After calling this, the next ``compute_reward()`` will return 0 reward
        for these envs and re-prime their cached values.

        Args:
            env_indices: 1D array of env indices that were reset.
        """
        if self._prev_values is not None and len(env_indices) > 0:
            # Set prev_values to NaN to mark as needing re-prime
            self._prev_values[env_indices] = np.nan

    def compute_reward(
        self, rgb_base: np.ndarray, rgb_render: np.ndarray
    ) -> np.ndarray:
        """Compute ACP reward based on configured shaping mode.

        TD mode:        r = (V(s') - V(s)) * reward_scale
        Potential mode: r = V(s') * reward_scale

        On first call (prev_values is None), returns zeros and primes the cache.
        For envs that were reset (prev_values is NaN), returns 0 and re-primes.

        Args:
            rgb_base: (N, H, W, 3) uint8 base camera images.
            rgb_render: (N, H, W, 3) uint8 render camera images.

        Returns:
            (N,) float32 rewards.
        """
        N = rgb_base.shape[0]
        current_values = self._predict_values(rgb_base, rgb_render)
        prev_values = None if self._prev_values is None else self._prev_values.copy()
        if prev_values is None:
            prev_for_delta = np.full(N, np.nan, dtype=np.float32)
        else:
            prev_for_delta = prev_values.copy()
        nan_mask = np.isnan(prev_for_delta)
        if self.config.reward_shaping == "potential":
            raw_reward = current_values * self.config.reward_scale
            if prev_values is None:
                self._prev_values = current_values.copy()
                reward = np.zeros(N, dtype=np.float32)
            else:
                reward = raw_reward.copy()
                if nan_mask.any():
                    reward[nan_mask] = 0.0
                self._prev_values = current_values.copy()
        else:
            if prev_values is None:
                self._prev_values = current_values.copy()
                reward = np.zeros(N, dtype=np.float32)
                raw_reward = np.zeros(N, dtype=np.float32)
            else:
                raw_reward = (current_values - prev_for_delta) * self.config.reward_scale
                reward = raw_reward.copy()
                if nan_mask.any():
                    reward[nan_mask] = 0.0
                self._prev_values = current_values.copy()

        clipped_reward = reward.copy()
        if self.config.reward_clip > 0:
            clipped_reward = np.clip(clipped_reward, -self.config.reward_clip, self.config.reward_clip)

        td_delta = np.zeros(N, dtype=np.float32)
        valid_prev = ~nan_mask
        if prev_values is not None and valid_prev.any():
            td_delta[valid_prev] = current_values[valid_prev] - prev_for_delta[valid_prev]

        self._last_details = {
            "current_values": current_values.astype(np.float32, copy=True),
            "prev_values": prev_for_delta.astype(np.float32, copy=True),
            "td_delta": td_delta,
            "raw_reward": raw_reward.astype(np.float32, copy=True),
            "reward_before_clip": reward.astype(np.float32, copy=True),
            "reward_after_clip": clipped_reward.astype(np.float32, copy=True),
            "clipped_mask": (np.abs(clipped_reward - reward) > 1e-6).astype(np.float32),
            "reset_mask": nan_mask.astype(np.float32),
            "reward_shaping": self.config.reward_shaping,
            "reward_scale": float(self.config.reward_scale),
            "reward_clip": float(self.config.reward_clip),
        }
        return clipped_reward.astype(np.float32, copy=False)


class DualCameraRewardWrapper(gym.Wrapper):
    """Gymnasium wrapper that replaces sim reward with ACP TD-shaped reward.

    Must be applied **before** ``FlattenRGBDObservationWrapper`` in the wrapper
    chain, because it reads ``obs["sensor_data"]["base_camera"]["rgb"]`` which
    ``FlattenRGBDObservationWrapper`` destructively pops.

    Wrapper ordering::

        raw_env -> DualCameraRewardWrapper -> FlattenRGBDObservationWrapper

    Args:
        env: ManiSkill environment (must have render_mode="rgb_array" for dual camera).
        config: ACP reward configuration.
    """

    def __init__(self, env: gym.Env, config: ACPRewardConfig) -> None:
        super().__init__(env)
        self.config = config
        self.acp_computer = ACPRewardComputer(config)
        self._step_count = 0
        self._num_envs: int = 0
        # Cache unwrapped env for grasp detection
        if config.grasp_bonus > 0:
            self._unwrapped_env = env.unwrapped

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict, dict]:
        obs, info = self.env.reset(seed=seed, options=options)

        # Infer num_envs from observation
        self._num_envs = self._infer_num_envs(obs)
        self.acp_computer.reset(self._num_envs)

        # Prime ACP value cache with initial observation
        rgb_base, rgb_render = self._extract_cameras(obs)
        self.acp_computer.compute_reward(rgb_base, rgb_render)  # returns zeros, primes cache
        details = self.acp_computer._last_details
        info["acp_reset_mask"] = np.ones(self._num_envs, dtype=np.float32)
        if details:
            info["acp_current_value"] = details["current_values"].copy()
            info["acp_prev_value"] = details["prev_values"].copy()

        return obs, info

    def step(
        self, action: Any
    ) -> tuple[dict, Any, Any, Any, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        # Save original sim reward for logging
        if torch.is_tensor(reward):
            sim_reward = reward.cpu().numpy().copy()
        else:
            sim_reward = np.asarray(reward, dtype=np.float32).copy()
        info["sim_reward"] = sim_reward

        # Detect auto-reset envs
        done = terminated | truncated
        done_np = _np(done).astype(bool) if not isinstance(done, np.ndarray) else done.astype(bool)

        # Reset ACP cache for done envs (before computing reward, so they get 0)
        if done_np.any():
            reset_indices = np.where(done_np)[0]
            self.acp_computer.reset_env(reset_indices)

        # Extract dual-camera images and compute ACP reward
        rgb_base, rgb_render = self._extract_cameras(obs)
        base_reward = self.acp_computer.compute_reward(rgb_base, rgb_render).astype(np.float32, copy=True)
        total_reward = base_reward.copy()

        grasp_bonus = np.zeros_like(total_reward, dtype=np.float32)
        is_grasped = np.zeros_like(total_reward, dtype=np.float32)
        if self.config.grasp_bonus > 0:
            is_grasped = self._unwrapped_env.agent.is_grasping(
                self._unwrapped_env.peg
            ).cpu().numpy().astype(np.float32)
            grasp_bonus = self.config.grasp_bonus * is_grasped
            total_reward = total_reward + grasp_bonus

        info["acp_base_reward"] = base_reward
        info["acp_grasp_bonus"] = grasp_bonus
        info["acp_total_reward"] = total_reward.copy()
        info["is_grasping"] = is_grasped
        details = self.acp_computer._last_details
        if details:
            info["acp_current_value"] = details["current_values"].copy()
            info["acp_prev_value"] = details["prev_values"].copy()
            info["acp_td_delta"] = details["td_delta"].copy()
            info["acp_raw_reward"] = details["raw_reward"].copy()
            info["acp_reward_before_clip"] = details["reward_before_clip"].copy()
            info["acp_reward_after_clip"] = details["reward_after_clip"].copy()
            info["acp_reward_clipped"] = details["clipped_mask"].copy()
            info["acp_reset_mask"] = details["reset_mask"].copy()

        # Return ACP reward in the same format as original
        if torch.is_tensor(reward):
            reward = torch.from_numpy(total_reward).to(reward.device, dtype=reward.dtype)
        else:
            reward = total_reward

        return obs, reward, terminated, truncated, info

    def _extract_cameras(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Extract dual-camera images from raw ManiSkill obs.

        Returns:
            (rgb_base, rgb_render): each (N, H, W, 3) uint8.
        """
        H, W = self.config.camera_height, self.config.camera_width

        rgb_base = _extract_base_camera(obs, self._num_envs, H, W)
        rgb_render = _get_render_frame(self.env, self._num_envs, H, W)

        return rgb_base, rgb_render

    def _infer_num_envs(self, obs: dict) -> int:
        """Infer number of parallel environments from observation structure."""
        # Try sensor_data first
        sensor_data = obs.get("sensor_data", {})
        for cam_data in sensor_data.values():
            if isinstance(cam_data, dict) and "rgb" in cam_data:
                return _np(cam_data["rgb"]).shape[0]
        # Try agent obs
        agent = obs.get("agent", {})
        for v in agent.values():
            return _np(v).shape[0]
        # Fallback
        return 1
