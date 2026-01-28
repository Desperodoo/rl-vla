"""
Rollout Buffer for On-Policy RL (PPO/ReinFlow).

Provides:
- SMDPChunkCollector: Lightweight collector for SMDP reward computation
- RolloutBufferPPO: Full PPO rollout buffer with GAE
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional

from .smdp import compute_smdp_rewards


class SMDPChunkCollector:
    """Lightweight collector for SMDP cumulative reward computation.
    
    Only stores rewards and dones during action chunk execution.
    Does NOT store observations or actions - those are managed separately.
    
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
        self.num_envs = num_envs
        self.gamma = gamma
        self.action_horizon = action_horizon
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
        """Compute SMDP cumulative rewards for the collected chunk."""
        if len(self.reward_list) == 0:
            raise RuntimeError("No steps collected")
        
        rewards = np.stack(self.reward_list, axis=0)
        dones = np.stack(self.done_list, axis=0)
        
        return compute_smdp_rewards(rewards, dones, self.gamma, self.num_envs)


class RolloutBufferPPO:
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
        num_inference_steps: Number of flow integration steps (for x_chain)
        gamma: Discount factor
        gae_lambda: GAE lambda
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
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.num_inference_steps = num_inference_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        # Pre-allocate tensors (num_steps, num_envs, ...)
        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, pred_horizon, act_dim), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        
        # x_chain for flow-based policies: (num_steps, num_envs, K+1, pred_horizon, act_dim)
        K = num_inference_steps
        self.x_chains = torch.zeros(
            (num_steps, num_envs, K + 1, pred_horizon, act_dim), device=device
        )
        
        # Computed after rollout
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
        
        self.step = 0
    
    def reset(self):
        """Reset buffer for new rollout."""
        self.step = 0
    
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        x_chain: Optional[torch.Tensor] = None,
    ):
        """Add a step to the buffer.
        
        Args:
            obs: (num_envs, obs_dim)
            action: (num_envs, pred_horizon, act_dim)
            log_prob: (num_envs,)
            reward: (num_envs,)
            done: (num_envs,)
            value: (num_envs,)
            x_chain: (num_envs, K+1, pred_horizon, act_dim)
        """
        assert self.step < self.num_steps, "Buffer is full"
        
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        
        if x_chain is not None:
            self.x_chains[self.step] = x_chain
        
        self.step += 1
    
    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        last_done: torch.Tensor,
    ):
        """Compute GAE advantages and returns.
        
        Args:
            last_value: (num_envs,) value estimate for final state
            last_done: (num_envs,) done flag for final state
        """
        gae = torch.zeros(self.num_envs, device=self.device)
        
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]
    
    def get_batches(self, batch_size: int, normalize_advantages: bool = True):
        """Generate random batches from the buffer.
        
        Args:
            batch_size: Size of each batch
            normalize_advantages: Whether to normalize advantages
            
        Yields:
            Dict with batch data
        """
        total_size = self.num_steps * self.num_envs
        indices = np.random.permutation(total_size)
        
        # Flatten buffers
        flat_obs = self.obs.view(-1, self.obs_dim)
        flat_actions = self.actions.view(-1, self.pred_horizon, self.act_dim)
        flat_log_probs = self.log_probs.view(-1)
        flat_values = self.values.view(-1)
        flat_advantages = self.advantages.view(-1)
        flat_returns = self.returns.view(-1)
        flat_x_chains = self.x_chains.view(-1, self.num_inference_steps + 1, self.pred_horizon, self.act_dim)
        
        # Normalize advantages
        if normalize_advantages:
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]
            
            yield {
                "obs": flat_obs[batch_indices],
                "actions": flat_actions[batch_indices],
                "log_probs": flat_log_probs[batch_indices],
                "values": flat_values[batch_indices],
                "advantages": flat_advantages[batch_indices],
                "returns": flat_returns[batch_indices],
                "x_chains": flat_x_chains[batch_indices],
            }
