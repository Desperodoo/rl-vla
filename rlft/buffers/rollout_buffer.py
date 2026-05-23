"""
SMDP reward collection utilities.

Provides:
- SMDPChunkCollector: Lightweight collector for SMDP reward computation
"""

import numpy as np
from typing import Tuple, List

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
