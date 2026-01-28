"""
Success-Filtered Replay Buffer for Online RL.

Implements a replay buffer that can filter samples based on episode success,
supporting Policy-Critic data separation in AWSC-style algorithms.

Key Features:
- Track episode success for each stored transition
- Filter samples by success status for policy training
- Keep all samples for critic training (full data utilization)
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List


class SuccessReplayBuffer:
    """Replay Buffer with Success-based Filtering.
    
    Extends basic replay buffer with:
    - Episode success tracking per transition
    - Configurable success filtering for policy training
    - Full data access for critic training (no filtering)
    
    This enables Policy-Critic data separation:
    - Critic sees all data (learns from failures and successes)
    - Policy only learns from successful or high-advantage samples
    
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
        min_success_ratio: Minimum ratio of successful samples in policy batch
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
        min_success_ratio: float = 0.3,
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
        self.min_success_ratio = min_success_ratio
        
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
        
        # Success tracking buffers
        self.success_buffer = np.zeros(capacity, dtype=np.float32)  # Episode success flag
        self.advantage_buffer = np.zeros(capacity, dtype=np.float32)  # Cached advantage
        
        self.ptr = 0
        self._size = 0
        
        # Track success/fail indices for efficient sampling
        self._success_indices = []
        self._fail_indices = []
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def num_success(self) -> int:
        return len(self._success_indices)
    
    @property
    def num_fail(self) -> int:
        return len(self._fail_indices)
    
    @property
    def success_rate(self) -> float:
        if self._size == 0:
            return 0.0
        return self.num_success / self._size
    
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
        success: Optional[np.ndarray] = None,
    ):
        """Store transitions with optional success labels.
        
        Args:
            obs: Current observations dict
            action: Action chunks
            reward: Single-step rewards
            next_obs: Next observations dict
            done: Done flags
            cumulative_reward: SMDP cumulative rewards
            chunk_done: SMDP done flags
            discount_factor: SMDP discount factors
            effective_length: SMDP effective lengths
            success: Episode success flags (0 or 1). If None, derived from done.
        """
        batch_size = action.shape[0]
        
        for i in range(batch_size):
            idx = self.ptr
            
            # Remove old index from tracking lists
            if self._size == self.capacity:
                if idx in self._success_indices:
                    self._success_indices.remove(idx)
                if idx in self._fail_indices:
                    self._fail_indices.remove(idx)
            
            # Store data
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
            
            # Store success label
            if success is not None:
                success_val = success[i]
            else:
                # Default: success = done (for environments where done = success)
                success_val = done[i]
            self.success_buffer[idx] = success_val
            
            # Update tracking lists
            if success_val > 0.5:
                self._success_indices.append(idx)
            else:
                self._fail_indices.append(idx)
            
            self.ptr = (self.ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
    
    def update_advantages(self, indices: np.ndarray, advantages: np.ndarray):
        """Update cached advantage values for given indices.
        
        This allows storing Q-computed advantages for later filtering.
        """
        self.advantage_buffer[indices] = advantages
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a random batch (all data, for critic training)."""
        indices = np.random.choice(self._size, size=batch_size, replace=True)
        return self._get_batch(indices)
    
    def sample_policy(
        self,
        batch_size: int,
        success_only: bool = False,
        min_success_ratio: Optional[float] = None,
        use_advantage_filter: bool = False,
        advantage_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Sample batch for policy training with optional success filtering.
        
        Args:
            batch_size: Number of samples
            success_only: If True, sample only from successful episodes
            min_success_ratio: Minimum ratio of successful samples (overrides default)
            use_advantage_filter: If True, filter by cached advantage
            advantage_threshold: Minimum advantage for samples
        
        Returns:
            Batch dictionary with filtered samples
        """
        if min_success_ratio is None:
            min_success_ratio = self.min_success_ratio
        
        if success_only and len(self._success_indices) >= batch_size:
            # Sample only successful transitions
            indices = np.random.choice(self._success_indices, size=batch_size, replace=True)
        elif min_success_ratio > 0 and len(self._success_indices) > 0:
            # Mix of success and other samples
            n_success = max(1, int(batch_size * min_success_ratio))
            n_other = batch_size - n_success
            
            success_indices = np.random.choice(
                self._success_indices, 
                size=min(n_success, len(self._success_indices)),
                replace=True
            )
            
            # Sample remaining from all data
            other_indices = np.random.choice(self._size, size=n_other, replace=True)
            
            indices = np.concatenate([success_indices, other_indices])
        else:
            # Standard sampling
            indices = np.random.choice(self._size, size=batch_size, replace=True)
        
        # Optional advantage filtering
        if use_advantage_filter and self.advantage_buffer is not None:
            advantages = self.advantage_buffer[indices]
            mask = advantages >= advantage_threshold
            if mask.sum() < batch_size // 4:
                # If too few samples pass filter, use all
                pass
            else:
                # Re-sample from high-advantage samples
                high_adv_indices = indices[mask]
                if len(high_adv_indices) >= batch_size:
                    indices = np.random.choice(high_adv_indices, size=batch_size, replace=True)
        
        return self._get_batch(indices)
    
    def sample_mixed(
        self,
        batch_size: int,
        offline_dataset,
        online_ratio: float = 0.5,
        policy_mode: bool = False,
        success_only: bool = False,
    ) -> Dict[str, Any]:
        """Sample mixed batch from online buffer and offline dataset.
        
        Args:
            batch_size: Total batch size
            offline_dataset: Offline RL dataset
            online_ratio: Ratio of online samples
            policy_mode: If True, use policy filtering for online samples
            success_only: If True and policy_mode, sample only successful online transitions
        """
        if offline_dataset is None or self._size == 0:
            if self._size > 0:
                if policy_mode:
                    return self.sample_policy(batch_size, success_only=success_only)
                return self.sample(batch_size)
            elif offline_dataset is not None:
                return self._sample_offline(offline_dataset, batch_size)
            else:
                raise RuntimeError("Both buffers are empty")
        
        n_online = int(batch_size * online_ratio)
        n_offline = batch_size - n_online
        
        if policy_mode and n_online > 0:
            online_batch = self.sample_policy(n_online, success_only=success_only)
        else:
            online_batch = self.sample(n_online) if n_online > 0 else None
        offline_batch = self._sample_offline(offline_dataset, n_offline) if n_offline > 0 else None
        
        if online_batch is None:
            return offline_batch
        if offline_batch is None:
            return online_batch
        
        # Merge batches
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
                    "cumulative_reward", "chunk_done", "discount_factor", 
                    "effective_length", "is_demo", "success"]:
            if key in online_batch and key in offline_batch:
                combined[key] = torch.cat([online_batch[key], offline_batch[key]], dim=0)
        
        return combined
    
    def _get_batch(self, indices: np.ndarray) -> Dict[str, Any]:
        """Convert indices to batch dictionary."""
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
            "is_demo": torch.zeros(len(indices), dtype=torch.bool, device=self.device),
            "success": torch.from_numpy(self.success_buffer[indices]).float().to(self.device),
        }
    
    def _sample_offline(self, dataset, batch_size: int) -> Dict[str, Any]:
        """Sample from offline dataset (assumed to be successful demonstrations)."""
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
            "success": torch.ones(batch_size, dtype=torch.float32, device=self.device),  # Demos are successful
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        return {
            "size": self._size,
            "capacity": self.capacity,
            "fill_ratio": self._size / self.capacity,
            "num_success": self.num_success,
            "num_fail": self.num_fail,
            "success_rate": self.success_rate,
            "avg_reward": float(self.reward_buffer[:self._size].mean()) if self._size > 0 else 0.0,
            "avg_cumulative_reward": float(self.cumulative_reward_buffer[:self._size].mean()) if self._size > 0 else 0.0,
        }
