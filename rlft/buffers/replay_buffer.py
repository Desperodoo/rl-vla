"""
Online Replay Buffer for Off-Policy RL.

Implements:
- OnlineReplayBuffer: Pre-encoded observation replay buffer
- OnlineReplayBufferRaw: Raw observation replay buffer (for encoder fine-tuning)
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from .smdp import compute_smdp_rewards


class OnlineReplayBuffer:
    """Online Replay Buffer with pre-encoded observations.
    
    For use when visual encoder is frozen and observations can be pre-encoded.
    
    Args:
        capacity: Maximum number of transitions
        num_envs: Number of parallel environments
        obs_dim: Dimension of encoded observations
        action_dim: Dimension of action space
        action_horizon: Length of action chunks
        gamma: Discount factor
        device: Device for output tensors
    """
    
    def __init__(
        self,
        capacity: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        action_horizon: int = 8,
        gamma: float = 0.99,
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.gamma = gamma
        self.device = device
        
        # Preallocate buffers
        self.obs_buffer = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buffer = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_buffer = np.zeros((capacity, action_horizon, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(capacity, dtype=np.float32)
        self.done_buffer = np.zeros(capacity, dtype=np.float32)
        self.cumulative_reward_buffer = np.zeros(capacity, dtype=np.float32)
        self.chunk_done_buffer = np.zeros(capacity, dtype=np.float32)
        self.discount_factor_buffer = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self._size = 0
    
    @property
    def size(self) -> int:
        return self._size
    
    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        cumulative_reward: np.ndarray,
        chunk_done: np.ndarray,
        discount_factor: np.ndarray,
    ):
        """Store transitions from all environments."""
        batch_size = action.shape[0]
        
        for i in range(batch_size):
            idx = self.ptr
            
            self.obs_buffer[idx] = obs[i]
            self.next_obs_buffer[idx] = next_obs[i]
            self.action_buffer[idx] = action[i]
            self.reward_buffer[idx] = reward[i]
            self.done_buffer[idx] = done[i]
            self.cumulative_reward_buffer[idx] = cumulative_reward[i]
            self.chunk_done_buffer[idx] = chunk_done[i]
            self.discount_factor_buffer[idx] = discount_factor[i]
            
            self.ptr = (self.ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch."""
        indices = np.random.choice(self._size, size=batch_size, replace=True)
        
        return {
            "obs_features": torch.from_numpy(self.obs_buffer[indices]).float().to(self.device),
            "next_obs_features": torch.from_numpy(self.next_obs_buffer[indices]).float().to(self.device),
            "actions": torch.from_numpy(self.action_buffer[indices]).float().to(self.device),
            "reward": torch.from_numpy(self.reward_buffer[indices]).float().to(self.device),
            "done": torch.from_numpy(self.done_buffer[indices]).float().to(self.device),
            "cumulative_reward": torch.from_numpy(self.cumulative_reward_buffer[indices]).float().to(self.device),
            "chunk_done": torch.from_numpy(self.chunk_done_buffer[indices]).float().to(self.device),
            "discount_factor": torch.from_numpy(self.discount_factor_buffer[indices]).float().to(self.device),
        }


class OnlineReplayBufferRaw:
    """Online Replay Buffer storing raw observations (RGB + state).
    
    Unlike pre-encoding buffers, this buffer stores raw observations to:
    - Support visual encoder fine-tuning (gradients flow through encoder)
    - Enable data augmentation on stored images
    - Maintain format compatibility with OfflineRLDataset
    
    Args:
        capacity: Maximum number of transitions
        num_envs: Number of parallel environments
        state_dim: Dimension of state observations
        action_dim: Dimension of action space
        action_horizon: Length of action chunks
        obs_horizon: Number of observation frames to stack
        include_rgb: Whether to store RGB observations
        rgb_shape: Shape of RGB images (H, W, C)
        gamma: Discount factor
        device: Device for output tensors
    """
    
    def __init__(
        self,
        capacity: int,
        num_envs: int,
        state_dim: int,
        action_dim: int,
        action_horizon: int = 8,
        obs_horizon: int = 2,
        include_rgb: bool = False,
        rgb_shape: Tuple[int, int, int] = (128, 128, 3),
        gamma: float = 0.99,
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.include_rgb = include_rgb
        self.rgb_shape = rgb_shape
        self.gamma = gamma
        self.device = device
        
        # Preallocate state buffers
        self.state_buffer = np.zeros(
            (capacity, obs_horizon, state_dim), dtype=np.float32
        )
        self.next_state_buffer = np.zeros(
            (capacity, obs_horizon, state_dim), dtype=np.float32
        )
        
        # RGB buffers (optional, stored as uint8 in NCHW format)
        if include_rgb:
            H, W, C = rgb_shape
            self.rgb_buffer = np.zeros(
                (capacity, obs_horizon, C, H, W), dtype=np.uint8
            )
            self.next_rgb_buffer = np.zeros(
                (capacity, obs_horizon, C, H, W), dtype=np.uint8
            )
        else:
            self.rgb_buffer = None
            self.next_rgb_buffer = None
        
        # Action and SMDP buffers
        self.action_buffer = np.zeros(
            (capacity, action_horizon, action_dim), dtype=np.float32
        )
        self.reward_buffer = np.zeros(capacity, dtype=np.float32)
        self.done_buffer = np.zeros(capacity, dtype=np.float32)
        self.cumulative_reward_buffer = np.zeros(capacity, dtype=np.float32)
        self.chunk_done_buffer = np.zeros(capacity, dtype=np.float32)
        self.discount_factor_buffer = np.zeros(capacity, dtype=np.float32)
        self.effective_length_buffer = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self._size = 0
    
    @property
    def size(self) -> int:
        return self._size
    
    def store(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: Dict[str, np.ndarray],
        done: np.ndarray,
        cumulative_reward: np.ndarray,
        chunk_done: np.ndarray,
        discount_factor: np.ndarray,
        effective_length: np.ndarray,
    ):
        """Store transitions from all environments."""
        batch_size = action.shape[0]
        
        for i in range(batch_size):
            idx = self.ptr
            
            self.state_buffer[idx] = obs["state"][i]
            self.next_state_buffer[idx] = next_obs["state"][i]
            
            if self.include_rgb and "rgb" in obs:
                rgb = obs["rgb"][i]
                next_rgb = next_obs["rgb"][i]
                if hasattr(rgb, 'cpu'):
                    rgb = rgb.cpu().numpy()
                if hasattr(next_rgb, 'cpu'):
                    next_rgb = next_rgb.cpu().numpy()
                if rgb.ndim == 4 and rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:
                    rgb = np.transpose(rgb, (0, 3, 1, 2))
                if next_rgb.ndim == 4 and next_rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:
                    next_rgb = np.transpose(next_rgb, (0, 3, 1, 2))
                if rgb.dtype != np.uint8:
                    rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8)
                if next_rgb.dtype != np.uint8:
                    next_rgb = (next_rgb * 255).astype(np.uint8) if next_rgb.max() <= 1 else next_rgb.astype(np.uint8)
                self.rgb_buffer[idx] = rgb
                self.next_rgb_buffer[idx] = next_rgb
            
            self.action_buffer[idx] = action[i]
            self.reward_buffer[idx] = reward[i]
            self.done_buffer[idx] = done[i]
            self.cumulative_reward_buffer[idx] = cumulative_reward[i]
            self.chunk_done_buffer[idx] = chunk_done[i]
            self.discount_factor_buffer[idx] = discount_factor[i]
            self.effective_length_buffer[idx] = effective_length[i]
            
            self.ptr = (self.ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a random batch of raw observations."""
        indices = np.random.choice(self._size, size=batch_size, replace=True)
        
        obs = {"state": torch.from_numpy(self.state_buffer[indices]).float().to(self.device)}
        next_obs = {"state": torch.from_numpy(self.next_state_buffer[indices]).float().to(self.device)}
        
        if self.include_rgb and self.rgb_buffer is not None:
            obs["rgb"] = torch.from_numpy(self.rgb_buffer[indices]).to(self.device)
            next_obs["rgb"] = torch.from_numpy(self.next_rgb_buffer[indices]).to(self.device)
        
        return {
            "observations": obs,
            "next_observations": next_obs,
            "actions": torch.from_numpy(self.action_buffer[indices]).float().to(self.device),
            "actions_for_q": torch.from_numpy(self.action_buffer[indices]).float().to(self.device),
            "reward": torch.from_numpy(self.reward_buffer[indices]).float().to(self.device),
            "done": torch.from_numpy(self.done_buffer[indices]).float().to(self.device),
            "cumulative_reward": torch.from_numpy(self.cumulative_reward_buffer[indices]).float().to(self.device),
            "chunk_done": torch.from_numpy(self.chunk_done_buffer[indices]).float().to(self.device),
            "discount_factor": torch.from_numpy(self.discount_factor_buffer[indices]).float().to(self.device),
            "effective_length": torch.from_numpy(self.effective_length_buffer[indices]).float().to(self.device),
            "is_demo": torch.zeros(batch_size, dtype=torch.bool, device=self.device),
        }
    
    def sample_mixed(
        self,
        batch_size: int,
        offline_dataset,
        online_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        """Sample mixed batch from online buffer and offline dataset."""
        if offline_dataset is None or self._size == 0:
            if self._size > 0:
                return self.sample(batch_size)
            elif offline_dataset is not None:
                return self._sample_offline(offline_dataset, batch_size)
            else:
                raise RuntimeError("Both buffers are empty")
        
        n_online = int(batch_size * online_ratio)
        n_offline = batch_size - n_online
        
        online_batch = self.sample(n_online) if n_online > 0 else None
        offline_batch = self._sample_offline(offline_dataset, n_offline) if n_offline > 0 else None
        
        if online_batch is None:
            return offline_batch
        if offline_batch is None:
            return online_batch
        
        combined = {}
        combined["observations"] = {
            "state": torch.cat([online_batch["observations"]["state"], 
                               offline_batch["observations"]["state"]], dim=0)
        }
        combined["next_observations"] = {
            "state": torch.cat([online_batch["next_observations"]["state"],
                               offline_batch["next_observations"]["state"]], dim=0)
        }
        
        if "rgb" in online_batch["observations"]:
            combined["observations"]["rgb"] = torch.cat([
                online_batch["observations"]["rgb"],
                offline_batch["observations"]["rgb"]
            ], dim=0)
            combined["next_observations"]["rgb"] = torch.cat([
                online_batch["next_observations"]["rgb"],
                offline_batch["next_observations"]["rgb"]
            ], dim=0)
        
        for key in ["actions", "actions_for_q", "reward", "done", 
                    "cumulative_reward", "chunk_done", "discount_factor", "effective_length", "is_demo"]:
            if key in online_batch and key in offline_batch:
                combined[key] = torch.cat([online_batch[key], offline_batch[key]], dim=0)
        
        return combined
    
    def _sample_offline(self, dataset, batch_size: int) -> Dict[str, Any]:
        """Sample from offline dataset."""
        indices = np.random.choice(len(dataset), size=batch_size, replace=True)
        items = [dataset[i] for i in indices]
        
        obs_state = torch.stack([item["observations"]["state"] for item in items], dim=0)
        next_obs_state = torch.stack([item["next_observations"]["state"] for item in items], dim=0)
        
        obs = {"state": obs_state.to(self.device)}
        next_obs = {"state": next_obs_state.to(self.device)}
        
        if self.include_rgb and "rgb" in items[0]["observations"]:
            obs["rgb"] = torch.stack([item["observations"]["rgb"] for item in items], dim=0).to(self.device)
            next_obs["rgb"] = torch.stack([item["next_observations"]["rgb"] for item in items], dim=0).to(self.device)
        
        actions_for_q = torch.stack([item["actions_for_q"] for item in items], dim=0).to(self.device)
        
        return {
            "observations": obs,
            "next_observations": next_obs,
            "actions": actions_for_q,
            "actions_for_q": actions_for_q,
            "reward": torch.stack([item["rewards"] for item in items], dim=0).to(self.device),
            "done": torch.stack([item["dones"] for item in items], dim=0).to(self.device),
            "cumulative_reward": torch.stack([item["cumulative_reward"] for item in items], dim=0).to(self.device),
            "chunk_done": torch.stack([item["chunk_done"] for item in items], dim=0).to(self.device),
            "discount_factor": torch.stack([item["discount_factor"] for item in items], dim=0).to(self.device),
            "effective_length": torch.stack([item["effective_length"] for item in items], dim=0).to(self.device),
            "is_demo": torch.ones(batch_size, dtype=torch.bool, device=self.device),
        }
