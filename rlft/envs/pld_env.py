"""
PLD Residual Environment Wrapper for ManiSkill3.

Wraps a ManiSkill3 GPU-vectorized environment so that the *action space
exposed to the RL agent* is the **residual action space** of a frozen base
policy.  The wrapper composes residual + base actions internally and executes
them with built-in action chunking.

This is the core mechanism of PLD-SAC (Stage 1 of PLD):
    a_bar = clamp(a_base + a_delta, -1, 1)

where a_base comes from a frozen ShortCut Flow policy and a_delta is the
RL agent's output in [-action_scale, +action_scale].

Inherits observation encoding from ``BaseFlowEnvWrapper``.

Reference:
    - PLD: https://arxiv.org/abs/2511.00091
    - DSRL env wrapper: rlft/envs/dsrl_env.py
"""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any

from rlft.envs.base_flow_env import BaseFlowEnvWrapper


class ManiSkillResidualEnvWrapper(BaseFlowEnvWrapper):
    """GPU-vectorized ManiSkill3 environment with residual action composition.

    Inherits observation encoding (``_encode_single_frame``,
    ``_encode_and_update_history``, ``_get_obs_cond``, ``_init_obs_history``)
    from ``BaseFlowEnvWrapper``.

    **Why this wrapper exists**: PLD-SAC treats the *residual action* added
    to a frozen base policy as the RL action.  This wrapper:

    1. Exposes a Box action-space of shape ``(act_steps * action_dim,)``
       bounded by ``[-action_scale, +action_scale]``.
    2. On ``step(a_delta)``:
       a) Queries the frozen base policy for deterministic base actions.
       b) Computes ``a_bar = clamp(a_base + a_delta, -1, 1)``.
       c) Executes the resulting ``act_steps`` real actions one-by-one,
          accumulating reward.
    3. Encodes raw observations (RGB + state) into a flat feature vector
       suitable for an MLP actor/critic.

    Args:
        env: ManiSkill3 vectorized environment (already wrapped with
            ``FlattenRGBDObservationWrapper`` if RGB).
        base_policy: Frozen ``ShortCutFlowWrapper``.
        visual_encoder: Frozen ``PlainConv`` visual encoder (or None).
        action_scale: Residual action bounds [-ξ, +ξ].
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
        action_scale: float = 0.5,
        act_steps: int = 8,
        action_dim: int = 7,
        state_dim: int = 25,
        visual_feature_dim: int = 256,
        obs_horizon: int = 2,
        include_rgb: bool = True,
        device: str = "cuda",
    ):
        super().__init__(
            env,
            visual_encoder=visual_encoder,
            act_steps=act_steps,
            action_dim=action_dim,
            state_dim=state_dim,
            visual_feature_dim=visual_feature_dim,
            obs_horizon=obs_horizon,
            include_rgb=include_rgb,
            device=device,
        )

        self.base_policy = base_policy
        self.action_scale = action_scale

        # ---- action space (residual bounds) ----
        self.action_space = spaces.Box(
            low=-action_scale * np.ones(action_dim * act_steps, dtype=np.float32),
            high=action_scale * np.ones(action_dim * act_steps, dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def step(self, action) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Execute one RL step (= *act_steps* real env steps).

        Args:
            action: Residual action a_delta, shape ``(num_envs, act_steps * action_dim)``
                or numpy equivalent.

        Returns:
            (encoded_obs, cumulative_reward, terminated, truncated, info)
            info also contains 'combined_action' for storing in buffer.
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device).float()
        else:
            action = action.to(self.device).float()

        # (B, act_steps, action_dim)
        a_delta = action.view(-1, self.act_steps, self.action_dim)
        obs_cond = self._get_obs_cond()

        # ---- Get base policy actions (deterministic, zero noise) ----
        with torch.no_grad():
            # Zero noise → base policy deterministic output
            zero_noise = torch.zeros_like(a_delta)
            a_base = self.base_policy(
                obs_cond, zero_noise, return_numpy=False, act_steps=self.act_steps,
            )
            # a_base may have fewer steps than act_steps when
            # pred_horizon == act_steps and obs_horizon > 1 (the flow model
            # slices from index obs_horizon-1, losing the first frames).
            # Align a_delta to match.
            n_actual = a_base.shape[1]
            if n_actual < self.act_steps:
                a_delta = a_delta[:, :n_actual, :]

        # ---- Compose combined action ----
        a_bar = torch.clamp(a_base + a_delta, -1.0, 1.0)

        # ---- Execute action chunk ----
        total_reward = torch.zeros(self.num_envs, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        info: Dict[str, Any] = {}

        n_steps = a_bar.shape[1]
        for i in range(n_steps):
            step_action = a_bar[:, i, :]
            obs, rew, term, trunc, step_info = self.env.step(step_action)
            total_reward += rew
            terminated = terminated | term
            truncated = truncated | trunc
            if term.any() or trunc.any():
                info = step_info
                break
        if not info:
            info = step_info  # type: ignore[possibly-undefined]

        # Store combined action in info (informational; train_pld stores residual)
        # a_bar may have fewer steps than act_steps (see n_actual above)
        info["combined_action"] = a_bar.reshape(self.num_envs, -1)

        # Safety: clear obs_history for envs that just reset
        done = terminated | truncated
        if done.any() and self._obs_history is not None:
            self._obs_history[done] = 0.0

        encoded_obs = self._encode_and_update_history(obs)
        return encoded_obs, total_reward, terminated, truncated, info

    def step_base_only(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Execute one RL step using ONLY the base policy (zero residual).

        Used for base policy probing: the RL agent doesn't act, only the
        base policy runs.  These steps change the state but are NOT added
        to the replay buffer.

        Returns:
            (encoded_obs, cumulative_reward, terminated, truncated, info)
        """
        zero_residual = torch.zeros(
            self.num_envs, self.act_steps * self.action_dim,
            device=self.device, dtype=torch.float32,
        )
        return self.step(zero_residual)

    def reset(self, **kwargs) -> Tuple[torch.Tensor, Dict]:
        obs, info = self.env.reset(**kwargs)
        self._init_obs_history()
        encoded_obs = self._encode_and_update_history(obs)
        return encoded_obs, info

    # Observation encoding methods (_init_obs_history, _encode_single_frame,
    # _encode_and_update_history, _get_obs_cond) are inherited from
    # BaseFlowEnvWrapper.
