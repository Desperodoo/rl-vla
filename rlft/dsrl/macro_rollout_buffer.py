"""
Macro-Step Rollout Buffer for DSRL Stage 2

Stores macro-step MDP transitions for PPO training.
Each transition represents act_horizon steps in the original environment.

Key features:
1. Stores (obs_cond, w, log_prob, R_chunk, done_macro, obs_next)
2. Supports GAE computation over macro-steps
3. Configurable capacity and batch sampling
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


class MacroRolloutBuffer:
    """Rollout buffer for macro-step MDP.
    
    Stores transitions where each step represents act_horizon primitive steps.
    Used for PPO training in DSRL Stage 2.
    
    Supports vectorized environments with shape (num_steps, num_envs, ...).
    
    Args:
        capacity: Maximum number of macro-steps per environment
        obs_dim: Dimension of observation conditioning
        pred_horizon: Prediction horizon of flow policy
        action_dim: Action dimension
        gamma: Base discount factor
        gae_lambda: GAE lambda parameter
        act_horizon: Action execution horizon (for gamma_macro)
        num_envs: Number of parallel environments
        device: Device to store tensors on
    """
    
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        pred_horizon: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        act_horizon: int = 8,
        num_envs: int = 1,
        device: str = "cuda",
    ):
        self.capacity = capacity  # Number of steps per env
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.act_horizon = act_horizon
        self.device = device
        
        # Macro-step discount: gamma^act_horizon (default, used when discount_factor not provided)
        self.gamma_macro = gamma ** act_horizon
        
        # Storage tensors: (num_steps, num_envs, ...)
        self.obs_cond = torch.zeros(capacity, num_envs, obs_dim, device=device)
        self.latent_w = torch.zeros(capacity, num_envs, pred_horizon, action_dim, device=device)
        self.log_prob = torch.zeros(capacity, num_envs, device=device)
        self.reward = torch.zeros(capacity, num_envs, device=device)  # Cumulative chunk reward
        self.done = torch.zeros(capacity, num_envs, device=device)  # Macro-step done
        self.value = torch.zeros(capacity, num_envs, device=device)  # V(s) for GAE
        self.discount_factor = torch.zeros(capacity, num_envs, device=device)  # Per-step discount factor
        
        # Next obs for value bootstrap
        self.next_obs_cond = torch.zeros(capacity, num_envs, obs_dim, device=device)
        
        # Computed after rollout
        self.advantage = torch.zeros(capacity, num_envs, device=device)
        self.returns = torch.zeros(capacity, num_envs, device=device)
        
        self.step_ptr = 0  # Current step index
        self.gae_computed = False
    
    def reset(self):
        """Reset buffer for new rollout."""
        self.step_ptr = 0
        self.gae_computed = False
    
    def store_step(
        self,
        obs_cond: torch.Tensor,
        latent_w: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        next_obs_cond: torch.Tensor,
        discount_factor: Optional[torch.Tensor] = None,
    ):
        """
        Store a batch of macro-step transitions (one step, all envs).
        
        Args:
            obs_cond: (num_envs, obs_dim) observation conditioning
            latent_w: (num_envs, pred_horizon, action_dim) latent
            log_prob: (num_envs,) log probabilities
            reward: (num_envs,) cumulative chunk rewards
            done: (num_envs,) done flags
            value: (num_envs,) V(s) estimates
            next_obs_cond: (num_envs, obs_dim) next observations
            discount_factor: (num_envs,) per-step discount factors (gamma^effective_length)
                If None, uses gamma^act_horizon for all envs
        """
        if self.step_ptr >= self.capacity:
            raise RuntimeError(f"Buffer overflow: step_ptr={self.step_ptr} >= capacity={self.capacity}")
        
        idx = self.step_ptr
        self.obs_cond[idx] = obs_cond
        self.latent_w[idx] = latent_w
        # Squeeze tensors to ensure (num_envs,) shape (handle (B,1) -> (B,))
        self.log_prob[idx] = log_prob.squeeze(-1) if log_prob.dim() > 1 else log_prob
        self.reward[idx] = reward.squeeze(-1) if reward.dim() > 1 else reward
        self.done[idx] = done.squeeze(-1) if done.dim() > 1 else done
        self.value[idx] = value.squeeze(-1) if value.dim() > 1 else value
        self.next_obs_cond[idx] = next_obs_cond
        
        # Store discount factor (use gamma_macro as default)
        if discount_factor is not None:
            df = discount_factor.squeeze(-1) if discount_factor.dim() > 1 else discount_factor
            self.discount_factor[idx] = df
        else:
            self.discount_factor[idx] = self.gamma_macro
        
        self.step_ptr += 1
    
    # Alias for backward compatibility
    def store_batch(self, *args, **kwargs):
        """Alias for store_step (backward compatibility)."""
        return self.store_step(*args, **kwargs)
    
    def compute_gae(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
    ):
        """
        Compute GAE advantages and returns for the current rollout.
        
        Uses per-step discount_factor (gamma^effective_length) for correct
        handling of early termination within action chunks.
        
        Args:
            last_value: (num_envs,) Value estimates for the final next state
            last_done: (num_envs,) Done flags for the final state (MUST be real done status)
        """
        num_steps = self.step_ptr
        
        # Ensure inputs are on the right device and shape
        if last_value.dim() == 0:
            last_value = last_value.unsqueeze(0).expand(self.num_envs)
        elif last_value.dim() > 1:
            last_value = last_value.squeeze(-1)  # (B, 1) -> (B,)
        if last_done.dim() == 0:
            last_done = last_done.unsqueeze(0).expand(self.num_envs)
        elif last_done.dim() > 1:
            last_done = last_done.squeeze(-1)
        
        last_value = last_value.squeeze()  # Ensure (num_envs,)
        last_done = last_done.squeeze()    # (num_envs,)
        
        # Vectorized GAE computation over all environments
        gae = torch.zeros(self.num_envs, device=self.device)
        
        for t in reversed(range(num_steps)):
            # Get per-step discount factor (gamma^effective_length)
            gamma_t = self.discount_factor[t]  # (num_envs,)
            
            if t == num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.done[t + 1]  # (num_envs,)
                next_value = self.value[t + 1]  # (num_envs,)
            
            # TD error with per-step discount
            delta = (
                self.reward[t] + 
                gamma_t * next_non_terminal * next_value - 
                self.value[t]
            )
            
            # GAE accumulation with per-step discount
            gae = delta + gamma_t * self.gae_lambda * next_non_terminal * gae
            
            self.advantage[t] = gae
            self.returns[t] = gae + self.value[t]
        
        self.gae_computed = True
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """
        Get all stored data for training, flattened across envs.
        
        Returns:
            Dict with keys: obs_cond, latent_w, log_prob, advantage, returns
            All tensors have shape (num_steps * num_envs, ...)
        """
        assert self.gae_computed, "Call compute_gae() before get_all()"
        
        num_steps = self.step_ptr
        
        # Flatten (num_steps, num_envs, ...) -> (num_steps * num_envs, ...)
        return {
            "obs_cond": self.obs_cond[:num_steps].reshape(-1, self.obs_dim),
            "latent_w": self.latent_w[:num_steps].reshape(-1, self.pred_horizon, self.action_dim),
            "log_prob": self.log_prob[:num_steps].reshape(-1),
            "advantage": self.advantage[:num_steps].reshape(-1),
            "returns": self.returns[:num_steps].reshape(-1),
            "value": self.value[:num_steps].reshape(-1),
        }
    
    def sample_batch(
        self,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a random batch for training.
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            Dict with sampled data (flattened across envs)
        """
        all_data = self.get_all()
        total_size = all_data["obs_cond"].shape[0]
        
        indices = torch.randperm(total_size, device=self.device)[:batch_size]
        
        return {k: v[indices] for k, v in all_data.items()}
    
    def iterate_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ):
        """
        Iterate over all data in batches.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Yields:
            Dict with batch data
        """
        all_data = self.get_all()
        total_size = all_data["obs_cond"].shape[0]
        
        if shuffle:
            indices = torch.randperm(total_size, device=self.device)
        else:
            indices = torch.arange(total_size, device=self.device)
        
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]
            
            yield {k: v[batch_indices] for k, v in all_data.items()}
