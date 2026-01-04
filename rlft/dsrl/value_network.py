"""
Value Network for DSRL Online RL (Stage 2)

Simple MLP-based value function V(s) for PPO-style advantage estimation.
Used in macro-step GAE computation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List


class ValueNetwork(nn.Module):
    """Value network for PPO.
    
    Simple MLP that takes observation features and outputs scalar value.
    Used for macro-step advantage estimation in Stage 2 online RL.
    
    Args:
        obs_dim: Dimension of observation features (flattened)
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Smaller weights for output layer
        final_layer = self.net[-1]
        nn.init.orthogonal_(final_layer.weight, gain=0.01)
    
    def forward(self, obs_features: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.
        
        Args:
            obs_features: (B, obs_dim) observation features
            
        Returns:
            value: (B, 1) state value
        """
        return self.net(obs_features)
