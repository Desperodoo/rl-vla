"""
Latent Q-Network for DSRL Off-Policy

Implements Double Q-network in latent space Q^W(s, w) for SAC-based training.
Includes target networks with soft update for stable TD learning.

Key design choices:
1. Input: (obs_cond, latent_w) -> Q-value (scalar)
2. Double Q architecture to mitigate overestimation
3. Target networks with soft update (τ = 0.005)
4. Support both full latent and act_horizon-only steering modes

References:
- SAC: https://arxiv.org/abs/1812.05905
- DSRL on-policy value_network.py for architecture patterns
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class LatentQNetwork(nn.Module):
    """Single Q-network for Q^W(s, w) in latent space.
    
    Takes observation conditioning and latent steering vector,
    outputs scalar Q-value estimate.
    
    Args:
        obs_dim: Dimension of observation features (flattened)
        pred_horizon: Prediction horizon of flow policy
        action_dim: Action dimension
        hidden_dims: List of hidden layer dimensions
        steer_mode: Which part of latent is steered ("full" or "act_horizon")
        act_horizon: Action execution horizon (used when steer_mode="act_horizon")
    """
    
    def __init__(
        self,
        obs_dim: int,
        pred_horizon: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        steer_mode: str = "full",
        act_horizon: int = 8,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.steer_mode = steer_mode
        self.act_horizon = act_horizon
        
        # Compute latent input dimension
        if steer_mode == "full":
            self.latent_dim = pred_horizon * action_dim
        else:  # "act_horizon"
            self.latent_dim = act_horizon * action_dim
        
        # Input dimension: obs_cond + latent_w
        input_dim = obs_dim + self.latent_dim
        
        # Build MLP
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))  # Output scalar Q-value
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights (aligned with on-policy ValueNetwork)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Smaller weights for output layer
        final_layer = self.net[-1]
        nn.init.orthogonal_(final_layer.weight, gain=0.01)
    
    def forward(
        self,
        obs_cond: torch.Tensor,
        latent_w: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q-value for (observation, latent) pair.
        
        Args:
            obs_cond: (B, obs_dim) flattened observation conditioning
            latent_w: (B, pred_horizon, action_dim) latent steering vector
                      or (B, steer_horizon, action_dim) if steer_mode="act_horizon"
            
        Returns:
            q_value: (B, 1) Q-value estimate
        """
        B = obs_cond.shape[0]
        
        # Flatten latent to (B, latent_dim)
        # Handle both full and steered latent inputs
        if latent_w.dim() == 3:
            if self.steer_mode == "act_horizon" and latent_w.shape[1] == self.pred_horizon:
                # Extract only steered portion
                latent_flat = latent_w[:, :self.act_horizon, :].reshape(B, -1)
            else:
                latent_flat = latent_w.reshape(B, -1)
        else:
            latent_flat = latent_w  # Already flattened
        
        # Concatenate obs and latent
        x = torch.cat([obs_cond, latent_flat], dim=-1)
        
        return self.net(x)


class DoubleLatentQNetwork(nn.Module):
    """Double Q-network for latent space Q^W(s, w).
    
    Maintains two independent Q-networks to reduce overestimation bias.
    Also maintains target networks for stable TD learning.
    
    Args:
        obs_dim: Dimension of observation features
        pred_horizon: Prediction horizon of flow policy
        action_dim: Action dimension
        hidden_dims: Hidden layer dimensions for each Q-network
        steer_mode: "full" or "act_horizon"
        act_horizon: Action execution horizon
        tau: Soft update rate for target networks
    """
    
    def __init__(
        self,
        obs_dim: int,
        pred_horizon: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        steer_mode: str = "full",
        act_horizon: int = 8,
        tau: float = 0.005,
    ):
        super().__init__()
        
        self.tau = tau
        
        # Twin Q-networks
        self.q1 = LatentQNetwork(
            obs_dim=obs_dim,
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            steer_mode=steer_mode,
            act_horizon=act_horizon,
        )
        self.q2 = LatentQNetwork(
            obs_dim=obs_dim,
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            steer_mode=steer_mode,
            act_horizon=act_horizon,
        )
        
        # Target networks (initialized as copies)
        self.q1_target = LatentQNetwork(
            obs_dim=obs_dim,
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            steer_mode=steer_mode,
            act_horizon=act_horizon,
        )
        self.q2_target = LatentQNetwork(
            obs_dim=obs_dim,
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            steer_mode=steer_mode,
            act_horizon=act_horizon,
        )
        
        # Freeze target networks
        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False
        
        # Sync target with online networks
        self.hard_update_target()
    
    def forward(
        self,
        obs_cond: torch.Tensor,
        latent_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values from both Q-networks.
        
        Args:
            obs_cond: (B, obs_dim) observation conditioning
            latent_w: (B, pred_horizon, action_dim) latent steering
            
        Returns:
            q1: (B, 1) Q-value from Q1
            q2: (B, 1) Q-value from Q2
        """
        q1 = self.q1(obs_cond, latent_w)
        q2 = self.q2(obs_cond, latent_w)
        return q1, q2
    
    def forward_target(
        self,
        obs_cond: torch.Tensor,
        latent_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values from target networks.
        
        Args:
            obs_cond: (B, obs_dim) observation conditioning
            latent_w: (B, pred_horizon, action_dim) latent steering
            
        Returns:
            q1_t: (B, 1) Q-value from target Q1
            q2_t: (B, 1) Q-value from target Q2
        """
        with torch.no_grad():
            q1_t = self.q1_target(obs_cond, latent_w)
            q2_t = self.q2_target(obs_cond, latent_w)
        return q1_t, q2_t
    
    def get_min_q(
        self,
        obs_cond: torch.Tensor,
        latent_w: torch.Tensor,
        use_target: bool = False,
    ) -> torch.Tensor:
        """
        Get minimum Q-value (pessimistic estimate).
        
        Args:
            obs_cond: (B, obs_dim) observation conditioning
            latent_w: (B, pred_horizon, action_dim) latent steering
            use_target: Whether to use target networks
            
        Returns:
            q_min: (B, 1) minimum Q-value
        """
        if use_target:
            q1, q2 = self.forward_target(obs_cond, latent_w)
        else:
            q1, q2 = self.forward(obs_cond, latent_w)
        return torch.min(q1, q2)
    
    def soft_update_target(self):
        """Soft update target networks: θ_t ← τ * θ + (1-τ) * θ_t"""
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def hard_update_target(self):
        """Hard update target networks: θ_t ← θ"""
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
    
    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable parameters (both Q-networks, not targets)."""
        return list(self.q1.parameters()) + list(self.q2.parameters())


class Temperature(nn.Module):
    """Learnable temperature parameter for SAC entropy regularization.
    
    Implements automatic temperature tuning to maintain target entropy.
    
    Args:
        initial_temperature: Initial temperature value
        learnable: Whether temperature is learnable
    """
    
    def __init__(
        self,
        initial_temperature: float = 0.1,
        learnable: bool = True,
    ):
        super().__init__()
        
        self.learnable = learnable
        
        if learnable:
            # Log scale for positivity constraint
            self.log_alpha = nn.Parameter(
                torch.tensor(np.log(initial_temperature), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "log_alpha",
                torch.tensor(np.log(initial_temperature), dtype=torch.float32)
            )
    
    @property
    def alpha(self) -> torch.Tensor:
        """Get temperature value (α = exp(log_α))."""
        return self.log_alpha.exp()
    
    def forward(self) -> torch.Tensor:
        """Return current temperature."""
        return self.alpha
