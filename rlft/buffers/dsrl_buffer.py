"""
Simple Replay Buffer for DSRL-SAC (Option A).

Since the ``ManiSkillFlowEnvWrapper`` handles action chunking internally,
one ``env.step(w)`` returns a single transition
``(obs, noise_w, cumulative_reward, next_obs, done)``.

This is a *standard MDP* replay buffer — no SMDP discount bookkeeping needed.

Storage layout:
    obs_buffer:       (capacity, obs_dim)    — encoded flat features
    next_obs_buffer:  (capacity, obs_dim)
    action_buffer:    (capacity, noise_dim)  — flat noise vector
    reward_buffer:    (capacity,)            — cumulative reward from chunk
    done_buffer:      (capacity,)
"""

import numpy as np
import torch
from typing import Dict


class DSRLReplayBuffer:
    """Fixed-size replay buffer storing pre-encoded observations.

    All observations are assumed to be **already encoded** to a flat feature
    vector (visual_encoder + obs_history flattened) before being stored.

    Args:
        capacity: Maximum number of transitions.
        obs_dim: Dimension of encoded (flat) observation.
        noise_dim: Dimension of noise action (act_steps * action_dim).
        device: Output device for sampled batches.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        noise_dim: int,
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.noise_dim = noise_dim
        self.device = device

        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity, noise_dim), dtype=np.float32)
        self.reward_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self._size = 0

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ):
        """Add a batch of transitions (one per environment).

        All inputs have shape ``(num_envs, ...)``.
        """
        batch = obs.shape[0]
        for i in range(batch):
            idx = self.ptr
            self.obs_buf[idx] = obs[i]
            self.next_obs_buf[idx] = next_obs[i]
            self.action_buf[idx] = action[i]
            self.reward_buf[idx] = reward[i]
            self.done_buf[idx] = done[i]
            self.ptr = (self.ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def add_single(self, obs, action, reward, next_obs, done):
        """Add a single transition (no batch dim)."""
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.action_buf[idx] = action
        self.reward_buf[idx] = reward
        self.done_buf[idx] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a uniformly random batch."""
        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs_buf[indices]).float().to(self.device),
            "next_obs": torch.from_numpy(self.next_obs_buf[indices]).float().to(self.device),
            "actions": torch.from_numpy(self.action_buf[indices]).float().to(self.device),
            "rewards": torch.from_numpy(self.reward_buf[indices]).float().to(self.device),
            "dones": torch.from_numpy(self.done_buf[indices]).float().to(self.device),
        }
