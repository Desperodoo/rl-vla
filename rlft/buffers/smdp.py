"""
SMDP utility functions for action chunking.

The SMDP (Semi-Markov Decision Process) formulation handles action chunks:
- cumulative_reward: R_t^(τ) = Σ_{i=0}^{τ-1} γ^i r_{t+i}
- chunk_done: 1 if episode ends within chunk
- discount_factor: γ^τ for Bellman bootstrapping
- effective_length: actual steps executed (may be < act_horizon if episode ends)
"""

import numpy as np
from typing import Tuple


def compute_smdp_rewards(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    num_envs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute SMDP cumulative rewards for collected chunks.
    
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
