"""
BaseFlowEnvWrapper — shared observation-encoding logic for Flow-based
environment wrappers (DSRL, PLD, etc.).

This base class provides:
    * ``__init__`` setup for visual encoder freezing, obs dim calculation,
      obs history buffer, and observation_space construction.
    * ``_init_obs_history``
    * ``_encode_single_frame``
    * ``_encode_and_update_history``
    * ``_get_obs_cond``

Subclasses (``ManiSkillFlowEnvWrapper``, ``ManiSkillResidualEnvWrapper``)
only need to define their own **action_space** and **step()** logic.

NOTE: ``ManiSkillFlowEnvWrapper`` (dsrl_env.py) keeps its own copy of
      these methods for backward compatibility.  Only new wrappers
      (e.g. ``ManiSkillResidualEnvWrapper``) should inherit from this.
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any


class BaseFlowEnvWrapper(gymnasium.Wrapper):
    """Shared observation encoding for flow-policy envs.

    Subclasses MUST:
        1. Call ``super().__init__(env, ...)`` in their ``__init__``.
        2. Set ``self.action_space`` themselves (noise vs residual bounds differ).
        3. Override ``step()`` with their own action-composition logic.

    Args:
        env: ManiSkill3 vectorized environment.
        visual_encoder: Frozen visual encoder (or None for state-only).
        act_steps: Number of real-env steps per RL step (action chunk).
        action_dim: Per-step action dimension (e.g. 7).
        state_dim: Proprioceptive state dimension.
        visual_feature_dim: Output dim of *visual_encoder*.
        obs_horizon: Number of observation frames to stack.
        include_rgb: Whether to encode RGB.
        device: Torch device string.
    """

    def __init__(
        self,
        env,
        visual_encoder: Optional[nn.Module] = None,
        act_steps: int = 8,
        action_dim: int = 7,
        state_dim: int = 25,
        visual_feature_dim: int = 256,
        obs_horizon: int = 2,
        include_rgb: bool = True,
        device: str = "cuda",
    ):
        super().__init__(env)

        self.visual_encoder = visual_encoder
        self.act_steps = act_steps
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.visual_feature_dim = visual_feature_dim
        self.obs_horizon = obs_horizon
        self.include_rgb = include_rgb
        self.device = device

        if self.visual_encoder is not None:
            self.visual_encoder = self.visual_encoder.to(device)
            self.visual_encoder.eval()
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

        # per-frame feature dim
        self.single_obs_dim = state_dim
        if include_rgb and visual_encoder is not None:
            self.single_obs_dim += visual_feature_dim
        self.obs_dim = obs_horizon * self.single_obs_dim

        self.num_envs = getattr(env, "num_envs", 1)

        # ---- observation space (common) ----
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(self.obs_dim, dtype=np.float32),
            high=np.inf * np.ones(self.obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        # NOTE: action_space must be set by subclass.
        self._obs_history: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _init_obs_history(self):
        self._obs_history = torch.zeros(
            self.num_envs, self.obs_horizon, self.single_obs_dim,
            device=self.device, dtype=torch.float32,
        )

    def _encode_single_frame(self, obs) -> torch.Tensor:
        """Encode one frame of raw observations to features."""
        parts = []

        if isinstance(obs, dict):
            # visual
            if self.include_rgb and self.visual_encoder is not None and "rgb" in obs:
                rgb = obs["rgb"]
                if isinstance(rgb, np.ndarray):
                    rgb = torch.from_numpy(rgb).to(self.device)
                else:
                    rgb = rgb.to(self.device)

                B = rgb.shape[0]
                T = rgb.shape[1] if rgb.dim() == 5 else 1

                if rgb.dim() == 5:
                    rgb = rgb.reshape(B * T, *rgb.shape[2:])
                if rgb.dim() == 4 and rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:
                    rgb = rgb.permute(0, 3, 1, 2)
                rgb = rgb.float()
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0

                with torch.no_grad():
                    vfeat = self.visual_encoder(rgb)  # (B*T, visual_dim)
                vfeat = vfeat.reshape(B, T, -1) if T > 1 else vfeat.unsqueeze(1)
                parts.append(vfeat)

            # state
            state = obs.get("state", obs.get("agent", None))
            if state is not None:
                if isinstance(state, np.ndarray):
                    state = torch.from_numpy(state).to(self.device).float()
                else:
                    state = state.to(self.device).float()
                if state.dim() == 2:
                    state = state.unsqueeze(1)
                parts.append(state)

            if not parts:
                B = self.num_envs
                return torch.zeros(B, 1, self.single_obs_dim, device=self.device)

            combined = torch.cat(parts, dim=-1)  # (B, T, D)
            return combined
        else:
            # plain tensor / array
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).to(self.device).float()
            else:
                obs = obs.to(self.device).float()
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)
            return obs

    def _encode_and_update_history(self, obs) -> torch.Tensor:
        """Encode obs, update history, return flat conditioning vector."""
        encoded = self._encode_single_frame(obs)  # (B, T, D)

        # ManiSkill3 FrameStack returns (B, T, D) with T == obs_horizon
        if encoded.shape[1] >= self.obs_horizon:
            # Full history already available from env — overwrite buffer
            self._obs_history = encoded[:, -self.obs_horizon:, :self.single_obs_dim]
        else:
            # Single frame — roll history
            if self._obs_history is None:
                self._init_obs_history()
            self._obs_history = torch.roll(self._obs_history, -1, dims=1)
            self._obs_history[:, -1, :] = encoded[:, -1, :self.single_obs_dim]

        flat = self._obs_history.reshape(self.num_envs, -1)
        self._cached_obs_cond = flat
        return flat

    def _get_obs_cond(self) -> torch.Tensor:
        if hasattr(self, "_cached_obs_cond") and self._cached_obs_cond is not None:
            return self._cached_obs_cond
        if self._obs_history is None:
            self._init_obs_history()
        return self._obs_history.reshape(self.num_envs, -1)
