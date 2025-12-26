"""
Replay Buffer Module for RLPD with SMDP support.

Implements:
- BaseRolloutBuffer: Base class with common SMDP computation
- RolloutBuffer: Lightweight buffer for SMDP chunk reward computation
- RolloutBufferPPO: Full PPO rollout buffer with GAE
- OnlineReplayBufferRaw: Vectorized replay buffer storing raw observations
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Any, List


# ============================================================================
# Base Rollout Buffer with SMDP Support
# ============================================================================

class BaseRolloutBuffer:
    """Base class for rollout buffers with SMDP action chunk support.
    
    Provides common functionality for computing SMDP cumulative rewards
    used by both off-policy (RolloutBuffer) and on-policy (RolloutBufferPPO) training.
    
    The SMDP (Semi-Markov Decision Process) formulation handles action chunks:
    - cumulative_reward: R_t^(τ) = Σ_{i=0}^{τ-1} γ^i r_{t+i}
    - chunk_done: 1 if episode ends within chunk
    - discount_factor: γ^τ for Bellman bootstrapping
    - effective_length: actual steps executed (may be < act_horizon if episode ends)
    
    Args:
        num_envs: Number of parallel environments
        gamma: Discount factor
        action_horizon: Action chunk length
    """
    
    def __init__(
        self,
        num_envs: int,
        gamma: float = 0.99,
        action_horizon: int = 8,
    ):
        self.num_envs = num_envs
        self.gamma = gamma
        self.action_horizon = action_horizon
    
    def reset(self):
        """Reset buffer state."""
        raise NotImplementedError
    
    @staticmethod
    def compute_smdp_rewards_static(
        rewards: np.ndarray,
        dones: np.ndarray,
        gamma: float,
        num_envs: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute SMDP cumulative rewards for collected chunks (static method).
        
        Efficient numpy-based implementation for vectorized computation.
        
        Args:
            rewards: (num_steps, num_envs) step rewards
            dones: (num_steps, num_envs) done flags
            gamma: Discount factor
            num_envs: Number of parallel environments
            
        Returns:
            cumulative_reward: (num_envs,) cumulative discounted reward
            chunk_done: (num_envs,) whether episode ended within chunk
            discount_factor: (num_envs,) γ^τ
            effective_length: (num_envs,) actual steps executed
        """
        num_steps = rewards.shape[0]
        
        cumulative_reward = np.zeros(num_envs, dtype=np.float32)
        chunk_done = np.zeros(num_envs, dtype=np.float32)
        effective_length = np.ones(num_envs, dtype=np.float32) * num_steps
        
        # Track which envs are still running
        still_running = np.ones(num_envs, dtype=bool)
        
        discount = 1.0
        for step_idx in range(num_steps):
            step_rewards = rewards[step_idx]
            step_dones = dones[step_idx]
            
            # Accumulate rewards for running envs
            cumulative_reward += discount * step_rewards * still_running
            
            # Check for episode termination
            terminated = step_dones > 0.5
            newly_done = terminated & still_running
            
            # Record effective length for newly terminated envs
            effective_length = np.where(newly_done, step_idx + 1, effective_length)
            chunk_done = np.where(newly_done, 1.0, chunk_done)
            
            # Update running status
            still_running = still_running & ~terminated
            
            discount *= gamma
        
        # Compute discount factor: γ^τ
        discount_factor = gamma ** effective_length
        
        return cumulative_reward, chunk_done, discount_factor, effective_length


# ============================================================================
# Online Replay Buffer (Raw Observations)
# ============================================================================

class OnlineReplayBufferRaw:
    """
    Online Replay Buffer storing raw observations (RGB + state).
    
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
        
        # Total buffer size
        self.total_capacity = capacity
        
        # Preallocate state buffers
        self.state_buffer = np.zeros(
            (self.total_capacity, obs_horizon, state_dim), dtype=np.float32
        )
        self.next_state_buffer = np.zeros(
            (self.total_capacity, obs_horizon, state_dim), dtype=np.float32
        )
        
        # RGB buffers (optional, stored as uint8 in NCHW format to match offline)
        if include_rgb:
            H, W, C = rgb_shape
            # Store in NCHW format: (capacity, obs_horizon, C, H, W)
            self.rgb_buffer = np.zeros(
                (self.total_capacity, obs_horizon, C, H, W), dtype=np.uint8
            )
            self.next_rgb_buffer = np.zeros(
                (self.total_capacity, obs_horizon, C, H, W), dtype=np.uint8
            )
        else:
            self.rgb_buffer = None
            self.next_rgb_buffer = None
        
        # Action and SMDP buffers
        self.action_buffer = np.zeros(
            (self.total_capacity, action_horizon, action_dim), dtype=np.float32
        )
        self.reward_buffer = np.zeros(self.total_capacity, dtype=np.float32)
        self.done_buffer = np.zeros(self.total_capacity, dtype=np.float32)
        self.cumulative_reward_buffer = np.zeros(self.total_capacity, dtype=np.float32)
        self.chunk_done_buffer = np.zeros(self.total_capacity, dtype=np.float32)
        self.discount_factor_buffer = np.zeros(self.total_capacity, dtype=np.float32)
        self.effective_length_buffer = np.zeros(self.total_capacity, dtype=np.float32)
        
        # Ring buffer pointer
        self.ptr = 0
        self._size = 0
    
    @property
    def size(self) -> int:
        """Return total number of stored transitions."""
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
        """
        Store transitions from all environments.
        
        Args:
            obs: Dict with 'state' (num_envs, obs_horizon, state_dim) 
                 and optionally 'rgb' (num_envs, obs_horizon, H, W, C)
            action: (num_envs, action_horizon, action_dim)
            reward: (num_envs,)
            next_obs: Same format as obs
            done: (num_envs,)
            cumulative_reward: (num_envs,)
            chunk_done: (num_envs,)
            discount_factor: (num_envs,)
            effective_length: (num_envs,)
        """
        batch_size = action.shape[0]
        
        for i in range(batch_size):
            idx = self.ptr
            
            # State observations
            self.state_buffer[idx] = obs["state"][i]
            self.next_state_buffer[idx] = next_obs["state"][i]
            
            # RGB observations (store as uint8 in NCHW format)
            if self.include_rgb and "rgb" in obs:
                rgb = obs["rgb"][i]
                next_rgb = next_obs["rgb"][i]
                # Handle tensor input
                if hasattr(rgb, 'cpu'):
                    rgb = rgb.cpu().numpy()
                if hasattr(next_rgb, 'cpu'):
                    next_rgb = next_rgb.cpu().numpy()
                # Convert NHWC to NCHW if needed: (T, H, W, C) -> (T, C, H, W)
                if rgb.ndim == 4 and rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:
                    rgb = np.transpose(rgb, (0, 3, 1, 2))
                if next_rgb.ndim == 4 and next_rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:
                    next_rgb = np.transpose(next_rgb, (0, 3, 1, 2))
                # Convert to uint8 if needed
                if rgb.dtype != np.uint8:
                    rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8)
                if next_rgb.dtype != np.uint8:
                    next_rgb = (next_rgb * 255).astype(np.uint8) if next_rgb.max() <= 1 else next_rgb.astype(np.uint8)
                self.rgb_buffer[idx] = rgb
                self.next_rgb_buffer[idx] = next_rgb
            
            # Actions and SMDP fields
            self.action_buffer[idx] = action[i]
            self.reward_buffer[idx] = reward[i]
            self.done_buffer[idx] = done[i]
            self.cumulative_reward_buffer[idx] = cumulative_reward[i]
            self.chunk_done_buffer[idx] = chunk_done[i]
            self.discount_factor_buffer[idx] = discount_factor[i]
            self.effective_length_buffer[idx] = effective_length[i]
            
            # Update pointer
            self.ptr = (self.ptr + 1) % self.total_capacity
            self._size = min(self._size + 1, self.total_capacity)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """
        Sample a random batch of raw observations.
        
        Returns:
            Dict with 'observations', 'next_observations' (dicts), 
            'actions', and SMDP fields as tensors.
            Includes 'is_demo' field (all False for online data).
        """
        if self._size == 0:
            raise RuntimeError("Cannot sample from empty buffer")
        
        indices = np.random.choice(self._size, size=batch_size, replace=True)
        
        # Build observation dicts
        obs = {
            "state": torch.from_numpy(self.state_buffer[indices]).float().to(self.device)
        }
        next_obs = {
            "state": torch.from_numpy(self.next_state_buffer[indices]).float().to(self.device)
        }
        
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
            "is_demo": torch.zeros(batch_size, dtype=torch.bool, device=self.device),  # Online data is not demo
        }
    
    def sample_mixed(
        self,
        batch_size: int,
        offline_dataset,  # OfflineRLDataset or similar with __getitem__
        online_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Sample mixed batch from online buffer and offline dataset.
        
        Args:
            batch_size: Total batch size
            offline_dataset: Offline dataset with __getitem__ and __len__
            online_ratio: Fraction from online buffer
            
        Returns:
            Combined batch with raw observations
        """
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
        
        # Combine batches
        combined = {}
        
        # Combine observation dicts
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
        
        # Combine other fields (including is_demo)
        for key in ["actions", "actions_for_q", "reward", "done", 
                    "cumulative_reward", "chunk_done", "discount_factor", "effective_length", "is_demo"]:
            if key in online_batch and key in offline_batch:
                combined[key] = torch.cat([online_batch[key], offline_batch[key]], dim=0)
        
        return combined
    
    def _sample_offline(self, dataset, batch_size: int) -> Dict[str, Any]:
        """Sample from offline dataset.
        
        Returns batch with is_demo=True for all samples.
        """
        indices = np.random.choice(len(dataset), size=batch_size, replace=True)
        
        # Collect items
        items = [dataset[i] for i in indices]
        
        # Stack into batches
        obs_state = torch.stack([item["observations"]["state"] for item in items], dim=0)
        next_obs_state = torch.stack([item["next_observations"]["state"] for item in items], dim=0)
        
        obs = {"state": obs_state.to(self.device)}
        next_obs = {"state": next_obs_state.to(self.device)}
        
        if self.include_rgb and "rgb" in items[0]["observations"]:
            obs["rgb"] = torch.stack([item["observations"]["rgb"] for item in items], dim=0).to(self.device)
            next_obs["rgb"] = torch.stack([item["next_observations"]["rgb"] for item in items], dim=0).to(self.device)
        
        # Use actions_for_q (act_horizon) as primary actions for consistency with online buffer
        actions_for_q = torch.stack([item["actions_for_q"] for item in items], dim=0).to(self.device)
        
        return {
            "observations": obs,
            "next_observations": next_obs,
            "actions": actions_for_q,  # Use act_horizon actions for mixed sampling
            "actions_for_q": actions_for_q,
            "reward": torch.stack([item["rewards"] for item in items], dim=0).to(self.device),
            "done": torch.stack([item["dones"] for item in items], dim=0).to(self.device),
            "cumulative_reward": torch.stack([item["cumulative_reward"] for item in items], dim=0).to(self.device),
            "chunk_done": torch.stack([item["chunk_done"] for item in items], dim=0).to(self.device),
            "discount_factor": torch.stack([item["discount_factor"] for item in items], dim=0).to(self.device),
            "effective_length": torch.stack([item["effective_length"] for item in items], dim=0).to(self.device),
            "is_demo": torch.ones(batch_size, dtype=torch.bool, device=self.device),  # Offline data is demo
        }


# ============================================================================
# SMDP Chunk Collector for Reward Computation
# ============================================================================

class SMDPChunkCollector(BaseRolloutBuffer):
    """Lightweight collector for SMDP cumulative reward computation.
    
    Only stores rewards and dones during action chunk execution.
    Does NOT store observations or actions - those are managed separately
    by the replay buffer and policy.
    
    Args:
        num_envs: Number of parallel environments
        gamma: Discount factor for SMDP reward computation
        action_horizon: Expected action chunk length (for reference)
    """
    
    def __init__(
        self,
        num_envs: int,
        gamma: float = 0.99,
        action_horizon: int = 8,
    ):
        super().__init__(num_envs=num_envs, gamma=gamma, action_horizon=action_horizon)
        self.reset()
    
    def reset(self):
        """Reset buffers for new chunk collection."""
        self.reward_list: List[np.ndarray] = []
        self.done_list: List[np.ndarray] = []
    
    def add(self, reward: np.ndarray, done: np.ndarray):
        """Add reward and done for a single environment step."""
        self.reward_list.append(reward.copy())
        self.done_list.append(done.copy())
    
    def compute_smdp_rewards(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute SMDP cumulative rewards for the collected chunk.
        
        Returns:
            cumulative_reward: (num_envs,) cumulative discounted reward
            chunk_done: (num_envs,) whether episode ended within chunk
            discount_factor: (num_envs,) γ^τ
            effective_length: (num_envs,) actual steps executed
        """
        if len(self.reward_list) == 0:
            raise RuntimeError("No steps collected")
        
        rewards = np.stack(self.reward_list, axis=0)  # (num_steps, num_envs)
        dones = np.stack(self.done_list, axis=0)      # (num_steps, num_envs)
        
        return BaseRolloutBuffer.compute_smdp_rewards_static(
            rewards, dones, self.gamma, self.num_envs
        )


# ============================================================================
# PPO-style Rollout Buffer with GAE
# ============================================================================

class RolloutBufferPPO(BaseRolloutBuffer):
    """Vectorized rollout buffer for on-policy RL (PPO/ReinFlow).
    
    Follows the official ManiSkill PPO implementation pattern with
    (num_steps, num_envs) shaped tensors for efficient vectorized computation.
    
    Key features:
    - Pre-allocated PyTorch tensors for GPU efficiency
    - Stores value estimates and log probabilities for PPO
    - Computes GAE (Generalized Advantage Estimation)
    - Stores x_chain for accurate log_prob computation in flow-based policies
    
    Args:
        num_steps: Number of rollout steps per update
        num_envs: Number of parallel environments
        obs_dim: Observation dimension
        pred_horizon: Action prediction horizon
        act_dim: Action dimension
        num_inference_steps: Number of denoising steps (K) for x_chain storage
        gamma: Discount factor for GAE
        gae_lambda: GAE lambda parameter
        normalize_advantages: Whether to normalize advantages
        device: Device for tensors
    """
    
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        pred_horizon: int,
        act_dim: int,
        num_inference_steps: int = 8,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
        device: str = "cuda",
    ):
        super().__init__(num_envs=num_envs, gamma=gamma, action_horizon=pred_horizon)
        
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.num_inference_steps = num_inference_steps
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.device = device
        
        # Pre-allocate tensors (num_steps, num_envs, ...)
        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, pred_horizon, act_dim), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        
        # x_chain storage: (num_steps, num_envs, K+1, pred_horizon, act_dim)
        # Essential for accurate log_prob computation in PPO updates
        self.x_chains = torch.zeros(
            (num_steps, num_envs, num_inference_steps + 1, pred_horizon, act_dim), 
            device=device
        )
        
        # For handling episode boundaries
        self.final_values = torch.zeros((num_steps, num_envs), device=device)
        
        # Computed after rollout
        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None
        
        self.ptr = 0
    
    def reset(self):
        """Reset buffer pointer for new rollout."""
        self.ptr = 0
        self.final_values.zero_()
        self.advantages = None
        self.returns = None
    
    def add(
        self,
        obs: torch.Tensor,           # (num_envs, obs_dim)
        actions: torch.Tensor,       # (num_envs, pred_horizon, act_dim)
        rewards: torch.Tensor,       # (num_envs,)
        values: torch.Tensor,        # (num_envs,)
        log_probs: torch.Tensor,     # (num_envs,)
        dones: torch.Tensor,         # (num_envs,)
        x_chain: Optional[torch.Tensor] = None,  # (num_envs, K+1, pred_horizon, act_dim)
    ):
        """Add a step of transitions for all environments."""
        if self.ptr >= self.num_steps:
            return
        
        # Detach all tensors to prevent gradient graph accumulation
        self.obs[self.ptr] = obs.detach()
        self.actions[self.ptr] = actions.detach()
        self.rewards[self.ptr] = rewards.detach()
        self.values[self.ptr] = values.detach().flatten()
        self.log_probs[self.ptr] = log_probs.detach()
        self.dones[self.ptr] = dones.detach()
        
        if x_chain is not None:
            self.x_chains[self.ptr] = x_chain.detach()
        
        self.ptr += 1
    
    def set_final_values(self, step: int, env_mask: torch.Tensor, final_vals: torch.Tensor):
        """Set bootstrap values for episodes that ended at this step."""
        self.final_values[step, env_mask] = final_vals.detach().flatten()
    
    def compute_returns_and_advantages(
        self,
        next_value: torch.Tensor,    # (num_envs,)
        next_done: torch.Tensor,     # (num_envs,)
    ):
        """Compute GAE advantages and returns."""
        with torch.no_grad():
            advantages = torch.zeros_like(self.rewards)
            lastgaelam = torch.zeros(self.num_envs, device=self.device)
            
            for t in reversed(range(self.ptr)):
                if t == self.ptr - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value.flatten()
                else:
                    next_not_done = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                
                # Handle episode boundaries using final_values
                real_next_values = next_not_done * nextvalues + self.final_values[t]
                
                # Standard GAE
                delta = self.rewards[t] + self.gamma * real_next_values - self.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * next_not_done * lastgaelam
            
            self.advantages = advantages[:self.ptr]
            self.returns = (advantages + self.values)[:self.ptr]
            
            # Normalize advantages if enabled
            if self.normalize_advantages:
                adv_mean = self.advantages.mean()
                adv_std = self.advantages.std()
                if adv_std > 1e-8:
                    self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
    
    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """Get mini-batches for PPO updates."""
        if self.advantages is None:
            raise RuntimeError("Must call compute_returns_and_advantages first")
        
        # Flatten (num_steps, num_envs, ...) -> (num_steps * num_envs, ...)
        n = self.ptr * self.num_envs
        K = self.num_inference_steps
        
        # Detach all tensors
        b_obs = self.obs[:self.ptr].reshape(n, -1).detach()
        b_actions = self.actions[:self.ptr].reshape(n, self.pred_horizon, self.act_dim).detach()
        b_log_probs = self.log_probs[:self.ptr].reshape(n).detach()
        b_values = self.values[:self.ptr].reshape(n).detach()
        b_advantages = self.advantages.reshape(n).detach()
        b_returns = self.returns.reshape(n).detach()
        b_x_chains = self.x_chains[:self.ptr].reshape(n, K + 1, self.pred_horizon, self.act_dim).detach()
        
        indices = torch.randperm(n, device=self.device) if shuffle else torch.arange(n, device=self.device)
        
        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            
            batch = {
                "obs": b_obs[batch_indices],
                "actions": b_actions[batch_indices],
                "log_probs": b_log_probs[batch_indices],
                "values": b_values[batch_indices],
                "advantages": b_advantages[batch_indices],
                "returns": b_returns[batch_indices],
                "x_chain": b_x_chains[batch_indices],
            }
            batches.append(batch)
        
        return batches


# ============================================================================
# Success Replay Buffer for Storing Successful Episodes
# ============================================================================

class SuccessReplayBuffer:
    """
    Buffer for storing successful (reward > threshold) online trajectories.
    
    This buffer stores raw observations like OnlineReplayBufferRaw, but only
    for successful episodes. It can be used for:
    - Curriculum learning (gradually use more online success data)
    - Self-improving from successful exploration
    - Data-efficient online RL with success filtering
    
    Currently implemented for storage only (not used in training loop).
    Future use: combine with OnlineReplayBufferRaw for improved data quality.
    
    Args:
        capacity: Maximum number of transitions
        state_dim: Dimension of state observations
        action_dim: Dimension of action space
        action_horizon: Length of action chunks
        obs_horizon: Number of observation frames to stack
        include_rgb: Whether to store RGB observations
        rgb_shape: Shape of RGB images (H, W, C)
        success_threshold: Reward threshold for success (default: 1.0 for sparse reward)
        device: Device for output tensors
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        action_horizon: int = 8,
        obs_horizon: int = 2,
        include_rgb: bool = False,
        rgb_shape: Tuple[int, int, int] = (128, 128, 3),
        success_threshold: float = 1.0,
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.include_rgb = include_rgb
        self.rgb_shape = rgb_shape
        self.success_threshold = success_threshold
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
        
        # Action and reward buffers
        self.action_buffer = np.zeros(
            (capacity, action_horizon, action_dim), dtype=np.float32
        )
        self.reward_buffer = np.zeros(capacity, dtype=np.float32)
        self.done_buffer = np.zeros(capacity, dtype=np.float32)
        self.cumulative_reward_buffer = np.zeros(capacity, dtype=np.float32)
        self.chunk_done_buffer = np.zeros(capacity, dtype=np.float32)
        self.discount_factor_buffer = np.zeros(capacity, dtype=np.float32)
        self.effective_length_buffer = np.zeros(capacity, dtype=np.float32)
        
        # Ring buffer pointer
        self.ptr = 0
        self._size = 0
        self._total_successes = 0  # Track total successful episodes stored
    
    @property
    def size(self) -> int:
        """Return total number of stored transitions."""
        return self._size
    
    @property
    def total_successes(self) -> int:
        """Return total number of successful episodes stored (including overwrites)."""
        return self._total_successes
    
    def store_if_success(
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
        success: np.ndarray,  # (num_envs,) success flags from env
    ) -> int:
        """
        Store transitions only for successful episodes.
        
        Args:
            obs: Dict with 'state' and optionally 'rgb'
            action: (num_envs, action_horizon, action_dim)
            reward: (num_envs,)
            next_obs: Same format as obs
            done: (num_envs,)
            cumulative_reward: (num_envs,)
            chunk_done: (num_envs,)
            discount_factor: (num_envs,)
            effective_length: (num_envs,)
            success: (num_envs,) whether episode was successful
            
        Returns:
            Number of successful transitions stored
        """
        # Find successful episodes
        success_mask = success > 0.5
        n_success = success_mask.sum()
        
        if n_success == 0:
            return 0
        
        success_indices = np.where(success_mask)[0]
        
        for env_idx in success_indices:
            idx = self.ptr
            
            # State observations
            self.state_buffer[idx] = obs["state"][env_idx]
            self.next_state_buffer[idx] = next_obs["state"][env_idx]
            
            # RGB observations
            if self.include_rgb and "rgb" in obs:
                rgb = obs["rgb"][env_idx]
                next_rgb = next_obs["rgb"][env_idx]
                # Handle tensor input
                if hasattr(rgb, 'cpu'):
                    rgb = rgb.cpu().numpy()
                if hasattr(next_rgb, 'cpu'):
                    next_rgb = next_rgb.cpu().numpy()
                # Convert NHWC to NCHW if needed
                if rgb.ndim == 4 and rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:
                    rgb = np.transpose(rgb, (0, 3, 1, 2))
                if next_rgb.ndim == 4 and next_rgb.shape[-1] in [1, 3, 4, 6, 9, 12]:
                    next_rgb = np.transpose(next_rgb, (0, 3, 1, 2))
                # Convert to uint8 if needed
                if rgb.dtype != np.uint8:
                    rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8)
                if next_rgb.dtype != np.uint8:
                    next_rgb = (next_rgb * 255).astype(np.uint8) if next_rgb.max() <= 1 else next_rgb.astype(np.uint8)
                self.rgb_buffer[idx] = rgb
                self.next_rgb_buffer[idx] = next_rgb
            
            # Actions and SMDP fields
            self.action_buffer[idx] = action[env_idx]
            self.reward_buffer[idx] = reward[env_idx]
            self.done_buffer[idx] = done[env_idx]
            self.cumulative_reward_buffer[idx] = cumulative_reward[env_idx]
            self.chunk_done_buffer[idx] = chunk_done[env_idx]
            self.discount_factor_buffer[idx] = discount_factor[env_idx]
            self.effective_length_buffer[idx] = effective_length[env_idx]
            
            # Update pointer
            self.ptr = (self.ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
            self._total_successes += 1
        
        return int(n_success)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """
        Sample a random batch from success buffer.
        
        Returns batch with is_demo=False (they are online successes, not demos).
        """
        if self._size == 0:
            raise RuntimeError("Cannot sample from empty success buffer")
        
        indices = np.random.choice(self._size, size=batch_size, replace=True)
        
        # Build observation dicts
        obs = {
            "state": torch.from_numpy(self.state_buffer[indices]).float().to(self.device)
        }
        next_obs = {
            "state": torch.from_numpy(self.next_state_buffer[indices]).float().to(self.device)
        }
        
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
            "is_demo": torch.zeros(batch_size, dtype=torch.bool, device=self.device),  # Online success, not demo
            "is_success": torch.ones(batch_size, dtype=torch.bool, device=self.device),  # All are successes
        }
        
        return batches
    
    def __len__(self) -> int:
        return self.ptr * self.num_envs
