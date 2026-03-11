"""Noisy policy wrappers for ACP data diversity collection.

Two noise models covering the main real-world data distributions:

1. ``OUNoisePolicyWrapper``  (teleop simulation)
   Ornstein-Uhlenbeck correlated noise on top of base policy actions.
   Captures human teleop characteristics: smooth, mean-reverting, with
   occasional deliberate pauses / micro-corrections.

2. ``GaussianNoisePolicyWrapper``  (RL fine-tuning exploration prior)
   i.i.d. Gaussian noise.  Approximates the wide action distribution
   created by high-entropy SAC during early online RL training.

Usage (inside collect_acp_data.py, after loading base policy)::

    from rlft.vlaw.data.noisy_policy import OUNoisePolicyWrapper

    noisy = OUNoisePolicyWrapper(base_policy, action_dim=7)
    trajectories = collector.collect_rollouts(noisy, visual_encoder)
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck noise (teleop simulation)
# ---------------------------------------------------------------------------

class OUNoisePolicyWrapper:
    """Wrap any PolicyProtocol with Ornstein-Uhlenbeck correlated noise.

    OU dynamics:  noise_{t+1} = (1 - theta)*noise_t + sigma*N(0,1)
    Resulting action: clip(base_action + noise_t, -action_clip, action_clip)

    In addition, with probability ``pause_prob`` a near-zero ``hold`` action
    is returned regardless of the policy output, simulating a human pausing
    to assess the situation.

    Args:
        policy: Base policy implementing ``get_actions(obs) -> np.ndarray``.
        action_dim: Dimension of the action space.
        theta: Mean-reversion rate (higher = faster decorrelation).
            Typical human motor noise: 0.10–0.20.
        sigma: Noise amplitude.  Set ~0.05 for light teleop, ~0.12 for heavy.
        action_clip: Hard clip on final action magnitude.
        pause_prob: Per-step probability of replacing action with a near-zero
            hold action (gripper unchanged, delta pose = 0).
        hold_gripper_sigma: Small noise on the gripper dimension during holds
            so the model sees realistic gripper signals.
        rng_seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        policy,
        action_dim: int = 7,
        theta: float = 0.15,
        sigma: float = 0.07,
        action_clip: float = 1.0,
        pause_prob: float = 0.04,
        hold_gripper_sigma: float = 0.02,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._policy = policy
        self._action_dim = action_dim
        self._theta = theta
        self._sigma = sigma
        self._action_clip = action_clip
        self._pause_prob = pause_prob
        self._hold_gripper_sigma = hold_gripper_sigma
        self._rng = np.random.default_rng(rng_seed)
        # OU state shape: (num_envs, action_dim) — initialised lazily
        self._noise: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # PolicyProtocol interface
    # ------------------------------------------------------------------

    def get_actions(self, obs_features: torch.Tensor) -> np.ndarray:
        base: np.ndarray = self._policy.get_actions(obs_features)  # (N, T, action_dim) or (N, action_dim)

        num_envs = base.shape[0]
        if self._noise is None or self._noise.shape[0] != num_envs:
            self._noise = np.zeros((num_envs, self._action_dim), dtype=np.float32)

        # OU update on the *first* action of each chunk (or scalar action)
        # We apply the same noise shift per chunk for simplicity.
        self._noise = (
            (1.0 - self._theta) * self._noise
            + self._sigma * self._rng.standard_normal((num_envs, self._action_dim)).astype(np.float32)
        )

        # Determine which envs are in a "pause" step
        pausing = self._rng.random(num_envs) < self._pause_prob  # (N,)

        noisy = base.copy()
        if noisy.ndim == 3:
            # Action chunk: (N, chunk_len, action_dim)
            noisy += self._noise[:, np.newaxis, :]
            noisy = np.clip(noisy, -self._action_clip, self._action_clip)
            # Pause: zero out delta-pose dims (indices 0-5), keep gripper noisy
            for i in np.where(pausing)[0]:
                noisy[i, :, :6] = 0.0
                noisy[i, :, 6] += (
                    self._hold_gripper_sigma
                    * self._rng.standard_normal(noisy.shape[1]).astype(np.float32)
                )
        else:
            # Single action: (N, action_dim)
            noisy += self._noise
            noisy = np.clip(noisy, -self._action_clip, self._action_clip)
            for i in np.where(pausing)[0]:
                noisy[i, :6] = 0.0
                noisy[i, 6] += self._hold_gripper_sigma * float(self._rng.standard_normal())

        return noisy.astype(np.float32)


# ---------------------------------------------------------------------------
# Gaussian noise (RL fine-tuning exploration prior)
# ---------------------------------------------------------------------------

class GaussianNoisePolicyWrapper:
    """Wrap any PolicyProtocol with i.i.d. Gaussian action noise.

    Approximates the broad action distribution produced by entropy-maximising
    SAC during early online RL training.  Larger ``sigma`` values model the
    random-walk phase; smaller values model a partially-converged policy.

    Args:
        policy: Base policy implementing ``get_actions(obs) -> np.ndarray``.
        action_dim: Dimension of the action space.
        sigma: Noise std.  Suggested range: 0.15 (mild) – 0.40 (aggressive).
        action_clip: Hard clip on final action.
        rng_seed: Optional RNG seed.
    """

    def __init__(
        self,
        policy,
        action_dim: int = 7,
        sigma: float = 0.25,
        action_clip: float = 1.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._policy = policy
        self._action_dim = action_dim
        self._sigma = sigma
        self._action_clip = action_clip
        self._rng = np.random.default_rng(rng_seed)

    def get_actions(self, obs_features: torch.Tensor) -> np.ndarray:
        base: np.ndarray = self._policy.get_actions(obs_features)
        noise = self._rng.standard_normal(base.shape).astype(np.float32) * self._sigma
        return np.clip(base + noise, -self._action_clip, self._action_clip).astype(np.float32)


# ---------------------------------------------------------------------------
# Scaled random policy (distribution boundary / ablation)
# ---------------------------------------------------------------------------

class ScaledRandomPolicy:
    """Purely random actions drawn from N(0, sigma).

    Useful as a lower-bound ablation: an ACP trained on this data cannot
    have learned anything about task-relevant transitions.

    Args:
        action_dim: Dimension of the action space.
        num_envs: Expected batch size (for pre-allocating buffers).
        sigma: Std of random actions.  1.0 = uniform [-1,1]-ish coverage.
        action_clip: Hard clip.
        rng_seed: Optional RNG seed.
    """

    def __init__(
        self,
        action_dim: int = 7,
        num_envs: int = 1,
        sigma: float = 0.8,
        action_clip: float = 1.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._action_dim = action_dim
        self._num_envs = num_envs
        self._sigma = sigma
        self._action_clip = action_clip
        self._rng = np.random.default_rng(rng_seed)

    def get_actions(self, obs_features: torch.Tensor) -> np.ndarray:
        n = obs_features.shape[0] if obs_features is not None else self._num_envs
        actions = self._rng.standard_normal((n, self._action_dim)).astype(np.float32) * self._sigma
        return np.clip(actions, -self._action_clip, self._action_clip)
