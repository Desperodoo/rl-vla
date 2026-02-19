"""
PLD Replay Buffer with Offline / Online Mixing.

PLD-SAC (Stage 1) trains with a **mixed** replay strategy:
  - 50 % transitions from the **online** buffer (RL exploration)
  - 50 % transitions from the **offline** buffer (base-policy demos)

Both sub-buffers are standard MDP buffers (same layout as
``DSRLReplayBuffer``), since the ``ManiSkillResidualEnvWrapper`` handles
action chunking internally.

Reference:
    - PLD paper §4.1, Table 5 (UTD ratio = 2, batch_size = 256)
    - DSRL buffer: rlft/buffers/dsrl_buffer.py
"""

import numpy as np
import torch
from typing import Dict, Optional


class PLDReplayBuffer:
    """Fixed-size replay buffer with offline/online mixing for PLD-SAC.

    Stores pre-encoded flat observations.  Provides ``sample_mixed()``
    which draws half from each sub-buffer.

    Args:
        online_capacity:  Max transitions for the online (RL) buffer.
        offline_capacity: Max transitions for the offline (demo) buffer.
        obs_dim:  Dimension of encoded flat observation.
        action_dim: Dimension of the RL action (act_steps * per_step_dim).
        device: Output device for sampled batches.
    """

    def __init__(
        self,
        online_capacity: int,
        offline_capacity: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cuda",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # ---------- Online (RL exploration) ----------
        self._online = _SubBuffer(online_capacity, obs_dim, action_dim)
        # ---------- Offline (base-policy demos) ----------
        self._offline = _SubBuffer(offline_capacity, obs_dim, action_dim)

    # ------------------------------------------------------------------
    # Sizes
    # ------------------------------------------------------------------

    @property
    def online_size(self) -> int:
        return self._online.size

    @property
    def offline_size(self) -> int:
        return self._offline.size

    @property
    def total_size(self) -> int:
        return self._online.size + self._offline.size

    # ------------------------------------------------------------------
    # Adding data
    # ------------------------------------------------------------------

    def add_online(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ):
        """Add a batch of online RL transitions.  All shapes ``(B, ...)``."""
        self._online.add(obs, action, reward, next_obs, done)

    def add_offline(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ):
        """Add a batch of offline (base policy) transitions.  All shapes ``(B, ...)``."""
        self._offline.add(obs, action, reward, next_obs, done)

    def add_online_single(self, obs, action, reward, next_obs, done):
        self._online.add_single(obs, action, reward, next_obs, done)

    def add_offline_single(self, obs, action, reward, next_obs, done):
        self._offline.add_single(obs, action, reward, next_obs, done)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_mixed(
        self,
        batch_size: int,
        online_ratio: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Sample a mixed batch from both sub-buffers.

        If one sub-buffer is still empty the full batch comes from the other.

        Args:
            batch_size: Total number of transitions.
            online_ratio: Fraction of the batch drawn from the online buffer.
        """
        has_online = self._online.size > 0
        has_offline = self._offline.size > 0

        if has_online and has_offline:
            n_online = int(batch_size * online_ratio)
            n_offline = batch_size - n_online
            online_batch = self._online.sample_np(n_online)
            offline_batch = self._offline.sample_np(n_offline)
            merged = _merge_np(online_batch, offline_batch)
        elif has_online:
            merged = self._online.sample_np(batch_size)
        elif has_offline:
            merged = self._offline.sample_np(batch_size)
        else:
            raise RuntimeError("Both sub-buffers are empty; cannot sample.")

        return _to_torch(merged, self.device)

    def sample_online(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample only from the online buffer."""
        return _to_torch(self._online.sample_np(batch_size), self.device)

    def sample_offline(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample only from the offline buffer."""
        return _to_torch(self._offline.sample_np(batch_size), self.device)


# ======================================================================
# Internal helpers
# ======================================================================


class _SubBuffer:
    """Minimal ring buffer backed by preallocated numpy arrays."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def add(self, obs, action, reward, next_obs, done):
        batch = obs.shape[0]
        for i in range(batch):
            self.add_single(obs[i], action[i], reward[i], next_obs[i], done[i])

    def add_single(self, obs, action, reward, next_obs, done):
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.action_buf[idx] = action
        self.reward_buf[idx] = reward
        self.done_buf[idx] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample_np(self, n: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self._size, size=n)
        return {
            "obs": self.obs_buf[idx],
            "next_obs": self.next_obs_buf[idx],
            "actions": self.action_buf[idx],
            "rewards": self.reward_buf[idx],
            "dones": self.done_buf[idx],
        }


def _merge_np(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate([a[k], b[k]], axis=0) for k in a}


def _to_torch(d: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).float().to(device) for k, v in d.items()}
