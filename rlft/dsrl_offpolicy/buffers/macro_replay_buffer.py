"""
Macro Replay Buffer for DSRL Off-Policy

Stores SMDP macro-step transitions for off-policy SAC training.
Each transition represents act_horizon steps in the original environment.

Key differences from MacroRolloutBuffer (on-policy):
1. Fixed-capacity ring buffer (circular) instead of per-rollout storage
2. Random sampling instead of sequential iteration
3. No GAE computation (SAC uses TD learning)
4. Stores full transitions: (s, w, R, γ^τ, done, s')

References:
- On-policy MacroRolloutBuffer: dsrl/macro_rollout_buffer.py
- RLPD online_buffer.py for replay buffer patterns
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


class MacroReplayBuffer:
    """Replay buffer for macro-step SMDP transitions.
    
    Stores off-policy transitions where each step represents act_horizon
    primitive steps. Uses ring buffer for efficient memory management.
    
    Transition format: (obs_cond, latent_w, reward, discount, done, next_obs_cond)
    
    Args:
        capacity: Maximum number of transitions to store
        obs_dim: Dimension of observation conditioning
        pred_horizon: Prediction horizon of flow policy
        action_dim: Action dimension
        device: Device to store tensors on
    """
    
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        pred_horizon: int,
        action_dim: int,
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.device = device
        
        # Ring buffer storage (allocated on CPU, moved to GPU on sample)
        # Using numpy for efficient storage, convert to tensor on sample
        self.obs_cond = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.latent_w = np.zeros((capacity, pred_horizon, action_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)  # Cumulative chunk reward
        self.discount_factor = np.zeros(capacity, dtype=np.float32)  # γ^τ
        self.done = np.zeros(capacity, dtype=np.float32)  # Macro-step done
        self.next_obs_cond = np.zeros((capacity, obs_dim), dtype=np.float32)
        
        # Ring buffer pointers
        self._ptr = 0  # Next write position
        self._size = 0  # Current size (capped at capacity)
    
    def add(
        self,
        obs_cond: np.ndarray,
        latent_w: np.ndarray,
        reward: float,
        discount_factor: float,
        done: float,
        next_obs_cond: np.ndarray,
    ):
        """
        Add a single transition to the buffer.
        
        Args:
            obs_cond: (obs_dim,) observation conditioning
            latent_w: (pred_horizon, action_dim) latent steering
            reward: Cumulative chunk reward
            discount_factor: γ^effective_length
            done: Done flag (0 or 1)
            next_obs_cond: (obs_dim,) next observation conditioning
        """
        idx = self._ptr
        
        self.obs_cond[idx] = obs_cond
        self.latent_w[idx] = latent_w
        self.reward[idx] = reward
        self.discount_factor[idx] = discount_factor
        self.done[idx] = done
        self.next_obs_cond[idx] = next_obs_cond
        
        # Update ring buffer pointers
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
    
    def add_batch(
        self,
        obs_cond: np.ndarray,
        latent_w: np.ndarray,
        reward: np.ndarray,
        discount_factor: np.ndarray,
        done: np.ndarray,
        next_obs_cond: np.ndarray,
    ):
        """
        Add a batch of transitions to the buffer.
        
        Args:
            obs_cond: (B, obs_dim) observation conditioning
            latent_w: (B, pred_horizon, action_dim) latent steering
            reward: (B,) cumulative chunk rewards
            discount_factor: (B,) discount factors
            done: (B,) done flags
            next_obs_cond: (B, obs_dim) next observation conditioning
        """
        B = obs_cond.shape[0]
        
        for i in range(B):
            self.add(
                obs_cond=obs_cond[i],
                latent_w=latent_w[i],
                reward=reward[i],
                discount_factor=discount_factor[i],
                done=done[i],
                next_obs_cond=next_obs_cond[i],
            )
    
    def add_batch_from_torch(
        self,
        obs_cond: torch.Tensor,
        latent_w: torch.Tensor,
        reward: torch.Tensor,
        discount_factor: torch.Tensor,
        done: torch.Tensor,
        next_obs_cond: torch.Tensor,
    ):
        """
        Add a batch of transitions from PyTorch tensors.
        
        Args:
            obs_cond: (B, obs_dim) observation conditioning
            latent_w: (B, pred_horizon, action_dim) latent steering
            reward: (B,) cumulative chunk rewards
            discount_factor: (B,) discount factors
            done: (B,) done flags
            next_obs_cond: (B, obs_dim) next observation conditioning
        """
        self.add_batch(
            obs_cond=obs_cond.cpu().numpy(),
            latent_w=latent_w.cpu().numpy(),
            reward=reward.cpu().numpy(),
            discount_factor=discount_factor.cpu().numpy(),
            done=done.cpu().numpy(),
            next_obs_cond=next_obs_cond.cpu().numpy(),
        )
    
    def sample(
        self,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dict with tensors on self.device:
                - obs_cond: (B, obs_dim)
                - latent_w: (B, pred_horizon, action_dim)
                - reward: (B,)
                - discount_factor: (B,)
                - done: (B,)
                - next_obs_cond: (B, obs_dim)
        """
        if self._size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Random indices
        indices = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "obs_cond": torch.from_numpy(self.obs_cond[indices]).to(self.device),
            "latent_w": torch.from_numpy(self.latent_w[indices]).to(self.device),
            "reward": torch.from_numpy(self.reward[indices]).to(self.device),
            "discount_factor": torch.from_numpy(self.discount_factor[indices]).to(self.device),
            "done": torch.from_numpy(self.done[indices]).to(self.device),
            "next_obs_cond": torch.from_numpy(self.next_obs_cond[indices]).to(self.device),
        }
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self._size
    
    @property
    def size(self) -> int:
        """Return current buffer size."""
        return self._size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self._size >= batch_size
    
    def reset(self):
        """Clear the buffer."""
        self._ptr = 0
        self._size = 0


class MixedMacroReplayBuffer:
    """Mixed replay buffer combining online and offline data.
    
    Samples from both online (new) and offline (pretrained) buffers
    with configurable mixing ratio.
    
    Args:
        online_buffer: MacroReplayBuffer for online data
        offline_buffer: MacroReplayBuffer for offline data (preloaded)
        online_ratio: Ratio of online samples (0.0 to 1.0)
    """
    
    def __init__(
        self,
        online_buffer: MacroReplayBuffer,
        offline_buffer: Optional[MacroReplayBuffer] = None,
        online_ratio: float = 0.5,
    ):
        self.online_buffer = online_buffer
        self.offline_buffer = offline_buffer
        self.online_ratio = online_ratio
    
    def sample(
        self,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a mixed batch from online and offline buffers.
        
        Args:
            batch_size: Total number of samples
            
        Returns:
            Dict with mixed batch of transitions
        """
        # If no offline buffer, sample only from online
        if self.offline_buffer is None or self.offline_buffer.size == 0:
            return self.online_buffer.sample(batch_size)
        
        # If online buffer empty, sample only from offline
        if self.online_buffer.size == 0:
            return self.offline_buffer.sample(batch_size)
        
        # Mixed sampling
        online_size = int(batch_size * self.online_ratio)
        offline_size = batch_size - online_size
        
        online_batch = self.online_buffer.sample(online_size)
        offline_batch = self.offline_buffer.sample(offline_size)
        
        # Concatenate batches
        return {
            key: torch.cat([online_batch[key], offline_batch[key]], dim=0)
            for key in online_batch.keys()
        }
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.online_buffer.is_ready(batch_size) or (
            self.offline_buffer is not None and self.offline_buffer.is_ready(batch_size)
        )
