"""
Actor Networks for Policy Learning.

- DiagGaussianActor: Diagonal Gaussian policy with Tanh squashing (for SAC)
- LearnableTemperature: Learnable temperature for entropy regularization
- SquashedNormal: Tanh-squashed Normal distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import math
from typing import List, Tuple, Optional


# =============================================================================
# Probability Distributions
# =============================================================================

class SquashedNormal:
    """Tanh-squashed Normal distribution for bounded action spaces [-1, 1].
    
    This is the standard distribution used in SAC for continuous control.
    The distribution is: a = tanh(z), where z ~ N(mu, sigma)
    
    Args:
        loc: Mean of the underlying Normal distribution [B, D]
        scale: Std of the underlying Normal distribution [B, D]
    """
    
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc = loc
        self.scale = scale
        self._base_dist = Normal(loc, scale)
    
    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the distribution (after tanh transform)."""
        return torch.tanh(self.loc)
    
    @property
    def mode(self) -> torch.Tensor:
        """Return the mode of the distribution (same as mean for Gaussian)."""
        return self.mean
    
    def sample_with_log_prob(
        self, 
        sample_shape: torch.Size = torch.Size()
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the distribution and compute log probability.
        
        Returns:
            samples: Sampled actions [sample_shape, B, D]
            log_prob: Log probability of samples [sample_shape, B]
        """
        # Reparameterized sample from base distribution
        z = self._base_dist.rsample(sample_shape)
        
        # Apply tanh transform
        action = torch.tanh(z)
        
        # Compute log_prob with Jacobian correction
        log_prob_z = self._base_dist.log_prob(z)
        
        # Jacobian correction: 2 * (log(2) - z - softplus(-2*z))
        log_abs_det_jacobian = 2 * (math.log(2) - z - F.softplus(-2 * z))
        
        # Sum over action dimensions
        log_prob = log_prob_z.sum(dim=-1) - log_abs_det_jacobian.sum(dim=-1)
        
        return action, log_prob
    
    def rsample_with_log_prob(
        self, 
        sample_shape: torch.Size = torch.Size()
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for sample_with_log_prob (always reparameterized)."""
        return self.sample_with_log_prob(sample_shape)
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability of an action."""
        eps = 1e-6
        value = torch.clamp(value, -1 + eps, 1 - eps)
        
        # Inverse tanh to get z
        z = 0.5 * (torch.log1p(value) - torch.log1p(-value))  # arctanh
        
        # Log prob of z under base distribution
        log_prob_z = self._base_dist.log_prob(z)
        
        # Jacobian correction
        log_abs_det_jacobian = 2 * (math.log(2) - z - F.softplus(-2 * z))
        
        # Sum over action dimensions
        return log_prob_z.sum(dim=-1) - log_abs_det_jacobian.sum(dim=-1)
    
    def entropy(self) -> torch.Tensor:
        """Approximate entropy of the squashed distribution."""
        return self._base_dist.entropy().sum(dim=-1)


# =============================================================================
# Actor Networks
# =============================================================================

class DiagGaussianActor(nn.Module):
    """Diagonal Gaussian Actor for SAC with Tanh squashing.
    
    Outputs actions bounded to [-1, 1] using Tanh transform.
    Supports state-dependent standard deviation.
    
    Args:
        obs_dim: Dimension of observation features
        action_dim: Dimension of action space
        action_horizon: Length of action sequence (for chunked actions)
        hidden_dims: Hidden layer dimensions for feature extractor
        log_std_range: (min, max) range for log standard deviation
        state_dependent_std: Whether std depends on state
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
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
            ])
            in_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        
        feature_dim = hidden_dims[-1]
        
        # Mean output head
        self.mean_head = nn.Linear(feature_dim, self.output_dim)
        
        # Log std head
        if state_dependent_std:
            self.log_std_head = nn.Linear(feature_dim, self.output_dim)
        else:
            self.log_std = nn.Parameter(torch.zeros(self.output_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize output heads with small weights."""
        for module in self.feature_extractor.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        
        if self.state_dependent_std:
            nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
            nn.init.zeros_(self.log_std_head.bias)
    
    def forward(self, obs_cond: torch.Tensor) -> SquashedNormal:
        """
        Forward pass to get action distribution.
        
        Args:
            obs_cond: (B, obs_dim) observation features
            
        Returns:
            dist: SquashedNormal distribution
        """
        features = self.feature_extractor(obs_cond)
        mean = self.mean_head(features)
        
        if self.state_dependent_std:
            log_std = self.log_std_head(features)
        else:
            log_std = self.log_std.expand_as(mean)
        
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
            action_flat = dist.mean
            log_prob = None
        else:
            action_flat, log_prob = dist.sample_with_log_prob()
        
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
        
        B = actions.shape[0]
        action_flat = actions.reshape(B, -1)
        
        log_prob = dist.log_prob(action_flat)
        entropy = dist.entropy()
        
        return log_prob, entropy


class LearnableTemperature(nn.Module):
    """Learnable temperature parameter for SAC entropy regularization.
    
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
