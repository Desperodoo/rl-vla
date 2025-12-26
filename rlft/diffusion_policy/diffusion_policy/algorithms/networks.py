"""
Network architectures for Offline RL algorithms
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

from ..conditional_unet1d import ConditionalUnet1D# Import EnsembleQNetwork from rlpd for unified offline/online Q-network architecture
from ..rlpd.networks import EnsembleQNetwork

class VelocityUNet1D(nn.Module):
    """1D U-Net for velocity field prediction (Flow Matching).
    
    Similar to ConditionalUnet1D but predicts velocity instead of noise.
    Uses FiLM conditioning with observation features.
    
    Args:
        input_dim: Action dimension
        global_cond_dim: Dimension of global conditioning (obs features)
        diffusion_step_embed_dim: Dimension of timestep embedding
        down_dims: Channel dimensions for each downsampling stage
        n_groups: Number of groups for GroupNorm
    """
    
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 64,
        down_dims: List[int] = [64, 128, 256],
        n_groups: int = 8,
    ):
        super().__init__()
        # Reuse ConditionalUnet1D architecture for velocity prediction
        # The only difference is semantic: output is velocity instead of noise
        self.unet = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            n_groups=n_groups,
        )
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sample: (B, pred_horizon, action_dim) noisy action sequence
            timestep: (B,) timestep values in [0, 1] for flow matching
            global_cond: (B, global_cond_dim) or (B, obs_horizon, cond_dim) observation features
            
        Returns:
            velocity: (B, pred_horizon, action_dim) predicted velocity field
        """
        # Flatten global_cond if it's 3D (B, obs_horizon, cond_dim) -> (B, obs_horizon * cond_dim)
        if global_cond.dim() == 3:
            global_cond = global_cond.reshape(global_cond.shape[0], -1)
        
        # Convert continuous timestep [0, 1] to integer for U-Net
        # U-Net internally uses sinusoidal embedding, so we scale to [0, 100]
        timestep_int = (timestep * 100).long()
        return self.unet(sample, timestep_int, global_cond=global_cond)


class DoubleQNetwork(nn.Module):
    """Twin Q-Network with MLP architecture.
    
    Simple MLP-based Q-network following mainstream offline RL methods
    (Diffusion-QL, IDQL, CPQL, etc.). Takes (action_sequence, obs_features) 
    and outputs scalar Q-values.
    
    Architecture: Flatten(action_seq) + obs_cond → MLP → Q-value
    
    Args:
        action_dim: Action dimension
        obs_dim: Dimension of observation features
        action_horizon: Length of action sequence (typically act_horizon)
        hidden_dims: Hidden layer dimensions for MLP
    """
    
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        action_horizon: int = 8,
        hidden_dims: List[int] = [512, 512, 512],
    ):
        super().__init__()
        
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        
        # Input: flattened action sequence + observation features
        input_dim = action_horizon * action_dim + obs_dim
        
        # Build Q1 MLP
        q1_layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            q1_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
            ])
            in_dim = hidden_dim
        q1_layers.append(nn.Linear(in_dim, 1))
        self.q1_net = nn.Sequential(*q1_layers)
        
        # Build Q2 MLP
        q2_layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            q2_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
            ])
            in_dim = hidden_dim
        q2_layers.append(nn.Linear(in_dim, 1))
        self.q2_net = nn.Sequential(*q2_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Output layers with smaller weights for stability
        for net in [self.q1_net, self.q2_net]:
            final_layer = net[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.orthogonal_(final_layer.weight, gain=0.01)
                if final_layer.bias is not None:
                    nn.init.zeros_(final_layer.bias)
    
    def forward(
        self,
        action_seq: torch.Tensor,
        obs_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            action_seq: (B, action_horizon, action_dim) action sequence
            obs_cond: (B, obs_dim) observation features
            
        Returns:
            q1, q2: (B, 1) Q-values from both networks
        """
        B = action_seq.shape[0]
        action_flat = action_seq.reshape(B, -1)
        x = torch.cat([action_flat, obs_cond], dim=-1)
        
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        
        return q1, q2
    
    def q1_forward(
        self,
        action_seq: torch.Tensor,
        obs_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through Q1 network only."""
        B = action_seq.shape[0]
        action_flat = action_seq.reshape(B, -1)
        x = torch.cat([action_flat, obs_cond], dim=-1)
        return self.q1_net(x)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Soft update target network parameters.
    
    θ_target = τ * θ_source + (1 - τ) * θ_target
    """
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)
