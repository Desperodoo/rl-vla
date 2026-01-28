"""
Q-Networks for Reinforcement Learning.

- DoubleQNetwork: Twin Q-network for standard double-Q learning
- EnsembleQNetwork: Ensemble Q with subsample + min for RLPD
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Soft update target network parameters.
    
    θ_target = τ * θ_source + (1 - τ) * θ_target
    """
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


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


class MLPFeatureExtractor(nn.Module):
    """Simple MLP feature extractor.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (if None, use last hidden_dim)
        activation: Activation function class
        layer_norm: Whether to use LayerNorm
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256],
        output_dim: int = None,
        activation: nn.Module = nn.Mish,
        layer_norm: bool = True,
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        
        if output_dim is not None:
            layers.append(nn.Linear(in_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim if output_dim is not None else hidden_dims[-1]
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with orthogonal weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnsembleQNetwork(nn.Module):
    """Ensemble Q-Network with configurable number of Q-networks.
    
    Extends DoubleQNetwork to support num_qs > 2 and subsample + min
    for conservative Q-value estimation (used in RLPD).
    
    Args:
        action_dim: Action dimension
        obs_dim: Dimension of observation features
        action_horizon: Length of action sequence
        hidden_dims: Hidden layer dimensions for each Q-network MLP
        num_qs: Number of Q-networks in ensemble (default: 10)
        num_min_qs: Number of Q-networks to subsample for min (default: 2)
    """
    
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        action_horizon: int = 8,
        hidden_dims: List[int] = [256, 256, 256],
        num_qs: int = 10,
        num_min_qs: int = 2,
    ):
        super().__init__()
        
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.num_qs = num_qs
        self.num_min_qs = num_min_qs
        
        # Input: flattened action sequence + observation features
        input_dim = action_horizon * action_dim + obs_dim
        
        # Build ensemble of Q-networks
        self.q_nets = nn.ModuleList()
        for _ in range(num_qs):
            q_layers = []
            in_dim = input_dim
            for hidden_dim in hidden_dims:
                q_layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Mish(),
                ])
                in_dim = hidden_dim
            q_layers.append(nn.Linear(in_dim, 1))
            self.q_nets.append(nn.Sequential(*q_layers))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for q_net in self.q_nets:
            for module in q_net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            
            # Output layer with smaller weights
            final_layer = q_net[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.orthogonal_(final_layer.weight, gain=0.01)
                if final_layer.bias is not None:
                    nn.init.zeros_(final_layer.bias)
    
    def forward(
        self,
        action_seq: torch.Tensor,
        obs_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through all Q-networks.
        
        Args:
            action_seq: (B, action_horizon, action_dim) action sequence
            obs_cond: (B, obs_dim) observation features
            
        Returns:
            q_values: (num_qs, B, 1) Q-values from all networks
        """
        B = action_seq.shape[0]
        action_flat = action_seq.reshape(B, -1)
        x = torch.cat([action_flat, obs_cond], dim=-1)
        
        q_values = torch.stack([q_net(x) for q_net in self.q_nets], dim=0)
        return q_values  # (num_qs, B, 1)
    
    def get_min_q(
        self,
        action_seq: torch.Tensor,
        obs_cond: torch.Tensor,
        random_subset: bool = True,
    ) -> torch.Tensor:
        """
        Get conservative Q-value estimate using subsample + min.
        
        This is the key mechanism in RLPD for sample-efficient learning:
        randomly sample num_min_qs networks and take the minimum.
        
        Args:
            action_seq: (B, action_horizon, action_dim)
            obs_cond: (B, obs_dim)
            random_subset: Whether to randomly subsample
            
        Returns:
            q_min: (B, 1) Conservative Q-value estimate
        """
        q_all = self.forward(action_seq, obs_cond)  # (num_qs, B, 1)
        
        if random_subset and self.num_min_qs < self.num_qs:
            indices = torch.randperm(self.num_qs, device=q_all.device)[:self.num_min_qs]
            q_subset = q_all[indices]
        else:
            q_subset = q_all
        
        q_min = q_subset.min(dim=0).values  # (B, 1)
        return q_min
    
    def get_mean_q(
        self,
        action_seq: torch.Tensor,
        obs_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Get Q-value estimate using ensemble mean."""
        q_all = self.forward(action_seq, obs_cond)
        return q_all.mean(dim=0)
    
    def get_double_q(
        self,
        action_seq: torch.Tensor,
        obs_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values from first two networks (for compatibility)."""
        q_all = self.forward(action_seq, obs_cond)
        return q_all[0], q_all[1]
