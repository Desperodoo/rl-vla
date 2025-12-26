"""
Network architectures for RLPD algorithms.

Implements:
- EnsembleQNetwork: Configurable num_qs Q-network ensemble with subsample + min
- DiagGaussianActor: Diagonal Gaussian policy with tanh squashing (for SAC)
- MLPFeatureExtractor: Simple MLP feature extractor
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import copy

from .distributions import SquashedNormal


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Soft update target network parameters.
    
    θ_target = τ * θ_source + (1 - τ) * θ_target
    """
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


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
        output_dim: Optional[int] = None,
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
    for conservative Q-value estimation (used in RLPD sample-efficient configs).
    
    Args:
        action_dim: Action dimension
        obs_dim: Dimension of observation features
        action_horizon: Length of action sequence (for action chunking SMDP)
        hidden_dims: Hidden layer dimensions for each Q-network MLP
        num_qs: Number of Q-networks in ensemble (default: 10 for sample-efficient)
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
            
            # Output layer with smaller weights for stability
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
            action_seq: (B, action_horizon, action_dim) action sequence
            obs_cond: (B, obs_dim) observation features
            random_subset: Whether to randomly subsample (True for training, False for eval)
            
        Returns:
            q_min: (B, 1) Conservative Q-value estimate
        """
        q_all = self.forward(action_seq, obs_cond)  # (num_qs, B, 1)
        
        if random_subset and self.num_min_qs < self.num_qs:
            # Randomly select num_min_qs networks
            indices = torch.randperm(self.num_qs, device=q_all.device)[:self.num_min_qs]
            q_subset = q_all[indices]
        else:
            q_subset = q_all
        
        # Take minimum across ensemble
        q_min = q_subset.min(dim=0).values  # (B, 1)
        return q_min
    
    def get_mean_q(
        self,
        action_seq: torch.Tensor,
        obs_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get Q-value estimate using ensemble mean.
        
        This is used for Actor updates where we want an unbiased Q estimate,
        as opposed to the conservative min estimate used for critic targets.
        
        Reference: RLPD paper (Ball et al., 2023) uses mean for actor.
        
        Args:
            action_seq: (B, action_horizon, action_dim) action sequence
            obs_cond: (B, obs_dim) observation features
            
        Returns:
            q_mean: (B, 1) Mean Q-value across ensemble
        """
        q_all = self.forward(action_seq, obs_cond)  # (num_qs, B, 1)
        q_mean = q_all.mean(dim=0)  # (B, 1)
        return q_mean
    
    def get_double_q(
        self,
        action_seq: torch.Tensor,
        obs_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get Q-values from first two networks (for backward compatibility).
        
        Returns:
            q1, q2: (B, 1) Q-values from first two networks
        """
        q_all = self.forward(action_seq, obs_cond)
        return q_all[0], q_all[1]


class DiagGaussianActor(nn.Module):
    """
    Diagonal Gaussian Actor for SAC with Tanh squashing.
    
    Outputs actions bounded to [-1, 1] using Tanh transform.
    Supports state-dependent standard deviation.
    
    Args:
        obs_dim: Dimension of observation features
        action_dim: Dimension of action space
        action_horizon: Length of action sequence (for chunked actions)
        hidden_dims: Hidden layer dimensions for feature extractor
        log_std_range: (min, max) range for log standard deviation
        state_dependent_std: Whether std depends on state (True) or is learned (False)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_horizon: int = 8,
        hidden_dims: List[int] = [256, 256, 256],
        log_std_range: Tuple[float, float] = (-5.0, 2.0),
        state_dependent_std: bool = True,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.output_dim = action_horizon * action_dim
        self.log_std_min, self.log_std_max = log_std_range
        self.state_dependent_std = state_dependent_std
        
        # Feature extractor
        self.feature_extractor = MLPFeatureExtractor(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=None,  # Use last hidden dim
            activation=nn.Mish,
            layer_norm=True,
        )
        
        feature_dim = hidden_dims[-1]
        
        # Mean output head
        self.mean_head = nn.Linear(feature_dim, self.output_dim)
        
        # Log std head (state-dependent or learned)
        if state_dependent_std:
            self.log_std_head = nn.Linear(feature_dim, self.output_dim)
        else:
            # Learned log_std parameter (not state-dependent)
            self.log_std = nn.Parameter(torch.zeros(self.output_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize output heads with small weights."""
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        
        if self.state_dependent_std:
            nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
            nn.init.zeros_(self.log_std_head.bias)
    
    def forward(
        self, 
        obs_cond: torch.Tensor,
    ) -> SquashedNormal:
        """
        Forward pass to get action distribution.
        
        Args:
            obs_cond: (B, obs_dim) observation features
            
        Returns:
            dist: SquashedNormal distribution for action chunked sequence
        """
        # Extract features
        features = self.feature_extractor(obs_cond)  # (B, feature_dim)
        
        # Compute mean
        mean = self.mean_head(features)  # (B, output_dim)
        
        # Compute log_std
        if self.state_dependent_std:
            log_std = self.log_std_head(features)
        else:
            log_std = self.log_std.expand_as(mean)
        
        # Clamp log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return SquashedNormal(mean, std)
    
    def get_action(
        self,
        obs_cond: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample action from the policy.
        
        Args:
            obs_cond: (B, obs_dim) observation features
            deterministic: If True, return mean action
            
        Returns:
            action: (B, action_horizon, action_dim) sampled/mean action
            log_prob: (B,) log probability (None if deterministic)
        """
        dist = self.forward(obs_cond)
        
        if deterministic:
            # Use mean (tanh of Gaussian mean)
            action_flat = dist.mean  # (B, output_dim)
            log_prob = None
        else:
            # Sample with reparameterization
            action_flat, log_prob = dist.sample_with_log_prob()  # (B, output_dim), (B,)
        
        # Reshape to action sequence
        B = obs_cond.shape[0]
        action = action_flat.reshape(B, self.action_horizon, self.action_dim)
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.
        
        Args:
            obs_cond: (B, obs_dim) observation features
            actions: (B, action_horizon, action_dim) actions to evaluate
            
        Returns:
            log_prob: (B,) log probability of actions
            entropy: (B,) entropy of the distribution
        """
        dist = self.forward(obs_cond)
        
        # Flatten actions for distribution
        B = actions.shape[0]
        action_flat = actions.reshape(B, -1)  # (B, output_dim)
        
        log_prob = dist.log_prob(action_flat)  # (B,)
        entropy = dist.entropy()  # (B,)
        
        return log_prob, entropy


class LearnableTemperature(nn.Module):
    """
    Learnable temperature parameter for SAC entropy regularization.
    
    α is optimized to maintain a target entropy level.
    Uses log(α) parameterization for numerical stability.
    
    Args:
        init_temperature: Initial temperature value (default: 1.0)
    """
    
    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(np.log(init_temperature)))
    
    @property
    def alpha(self) -> torch.Tensor:
        """Return the temperature value."""
        return self.log_alpha.exp()
    
    def forward(self) -> torch.Tensor:
        """Return the temperature value."""
        return self.alpha
