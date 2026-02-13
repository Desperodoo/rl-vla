"""
DSRL Flow Environment Wrapper for ManiSkill3.

Wraps a ManiSkill3 GPU-vectorized environment so that the *action space
exposed to the RL agent* is the **noise space** of a pretrained ShortCut Flow
policy.  The wrapper decodes noise → real actions internally and executes them
with built-in action chunking.

This is the core mechanism of DSRL-SAC:  SAC operates in noise space
$w \\in [-\\text{mag}, +\\text{mag}]^{T \\times d}$ and the environment
wrapper transparently converts each $w$ to real actions via ODE integration.

Reference: https://github.com/ajwagen/dsrl
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any


class ManiSkillFlowEnvWrapper(gymnasium.Wrapper):
    """GPU-vectorized ManiSkill3 environment with ShortCut Flow decoding.

    **Why this wrapper exists**:  DSRL-SAC treats the *noise* fed to a
    pretrained flow policy as the RL action.  This wrapper:

    1. Exposes a Box action-space of shape ``(act_steps * action_dim,)``
       bounded by ``[-action_magnitude, +action_magnitude]``.
    2. On ``step(w)``: decodes ``w`` via the frozen flow policy, then
       executes the resulting ``act_steps`` real actions one-by-one,
       accumulating reward.
    3. Encodes raw observations (RGB + state) into a flat feature vector
       suitable for an MLP actor/critic.

    Args:
        env: ManiSkill3 vectorized environment (already wrapped with
            ``FlattenRGBDObservationWrapper`` if RGB).
        base_policy: Frozen ``ShortCutFlowWrapper``.
        visual_encoder: Frozen ``PlainConv`` visual encoder (or None).
        action_magnitude: Noise bounds [-mag, +mag].
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
        base_policy,
        visual_encoder: Optional[nn.Module] = None,
        action_magnitude: float = 1.5,
        act_steps: int = 8,
        action_dim: int = 7,
        state_dim: int = 25,
        visual_feature_dim: int = 256,
        obs_horizon: int = 2,
        include_rgb: bool = True,
        device: str = "cuda",
    ):
        super().__init__(env)

        self.base_policy = base_policy
        self.visual_encoder = visual_encoder
        self.action_magnitude = action_magnitude
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

        # ---- spaces ----
        self.action_space = spaces.Box(
            low=-action_magnitude * np.ones(action_dim * act_steps, dtype=np.float32),
            high=action_magnitude * np.ones(action_dim * act_steps, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(self.obs_dim, dtype=np.float32),
            high=np.inf * np.ones(self.obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self._obs_history: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def step(self, action) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Execute one RL step (= *act_steps* real env steps).

        Args:
            action: Noise vector, shape ``(num_envs, act_steps * action_dim)``
                or numpy equivalent.

        Returns:
            (encoded_obs, cumulative_reward, terminated, truncated, info)
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device).float()
        else:
            action = action.to(self.device).float()

        # (B, act_steps, action_dim)
        noise = action.view(-1, self.act_steps, self.action_dim)
        obs_cond = self._get_obs_cond()

        with torch.no_grad():
            real_actions = self.base_policy(
                obs_cond, noise, return_numpy=False, act_steps=self.act_steps,
            )

        # ---- execute action chunk ----
        total_reward = torch.zeros(self.num_envs, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        info: Dict[str, Any] = {}

        # Use actual number of returned actions (may be < act_steps due to
        # temporal offset slicing when pred_horizon == act_steps).
        n_steps = real_actions.shape[1]
        for i in range(n_steps):
            step_action = real_actions[:, i, :]
            obs, rew, term, trunc, step_info = self.env.step(step_action)
            total_reward += rew
            terminated = terminated | term
            truncated = truncated | trunc
            if term.any() or trunc.any():
                info = step_info
                break
        if not info:
            info = step_info  # type: ignore[possibly-undefined]

        # Safety: clear obs_history for envs that just reset so that the
        # manual-roll fallback path (used when FrameStack is absent) does
        # not leak frames from the previous episode.
        done = terminated | truncated
        if done.any() and self._obs_history is not None:
            self._obs_history[done] = 0.0

        encoded_obs = self._encode_and_update_history(obs)
        return encoded_obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[torch.Tensor, Dict]:
        obs, info = self.env.reset(**kwargs)
        self._init_obs_history()
        encoded_obs = self._encode_and_update_history(obs)
        return encoded_obs, info

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
            self._obs_history = encoded[:, -self.obs_horizon :, : self.single_obs_dim]
        else:
            # Single frame — roll history
            if self._obs_history is None:
                self._init_obs_history()
            self._obs_history = torch.roll(self._obs_history, -1, dims=1)
            self._obs_history[:, -1, :] = encoded[:, -1, : self.single_obs_dim]

        flat = self._obs_history.reshape(self.num_envs, -1)
        self._cached_obs_cond = flat
        return flat

    def _get_obs_cond(self) -> torch.Tensor:
        if hasattr(self, "_cached_obs_cond") and self._cached_obs_cond is not None:
            return self._cached_obs_cond
        if self._obs_history is None:
            self._init_obs_history()
        return self._obs_history.reshape(self.num_envs, -1)
