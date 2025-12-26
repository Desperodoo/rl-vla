"""
Probability distributions for SAC policy.

Implements SquashedNormal (TanhNormal) distribution for bounded action spaces.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform
from typing import Optional, Tuple
import math


class SquashedNormal(TransformedDistribution):
    """
    Tanh-squashed Normal distribution for bounded action spaces [-1, 1].
    
    This is the standard distribution used in SAC for continuous control.
    The distribution is: a = tanh(z), where z ~ N(mu, sigma)
    
    Key features:
    - Outputs are bounded to [-1, 1]
    - log_prob is computed with proper Jacobian correction
    - Supports reparameterized sampling for gradient flow
    
    Args:
        loc: Mean of the underlying Normal distribution [B, D]
        scale: Std of the underlying Normal distribution [B, D]
        
    Reference:
        Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
        Reinforcement Learning with a Stochastic Actor", ICML 2018
    """
    
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc = loc
        self.scale = scale
        self._base_dist = Normal(loc, scale)
        
        # TanhTransform squashes to [-1, 1]
        transforms = [TanhTransform(cache_size=1)]
        super().__init__(self._base_dist, transforms)
    
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
        
        More numerically stable than calling sample() then log_prob() separately.
        
        Args:
            sample_shape: Shape of samples to draw
            
        Returns:
            samples: Sampled actions [sample_shape, B, D]
            log_prob: Log probability of samples [sample_shape, B]
        """
        # Reparameterized sample from base distribution
        z = self._base_dist.rsample(sample_shape)
        
        # Apply tanh transform
        action = torch.tanh(z)
        
        # Compute log_prob with Jacobian correction
        # log p(a) = log p(z) - log |det(da/dz)|
        # For tanh: log |det(da/dz)| = sum(log(1 - tanh(z)^2))
        log_prob_z = self._base_dist.log_prob(z)
        
        # Jacobian correction: log(1 - tanh(z)^2) = log(1 - a^2)
        # Use numerically stable version: 2 * (log(2) - z - softplus(-2*z))
        log_abs_det_jacobian = 2 * (math.log(2) - z - nn.functional.softplus(-2 * z))
        
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
        """
        Compute log probability of an action.
        
        Note: Due to numerical issues at boundaries, clamp the value slightly.
        
        Args:
            value: Action values in [-1, 1] [B, D]
            
        Returns:
            log_prob: Log probability [B]
        """
        # Clamp to avoid numerical issues at boundaries
        eps = 1e-6
        value = torch.clamp(value, -1 + eps, 1 - eps)
        
        # Inverse tanh to get z
        z = 0.5 * (torch.log1p(value) - torch.log1p(-value))  # arctanh
        
        # Log prob of z under base distribution
        log_prob_z = self._base_dist.log_prob(z)
        
        # Jacobian correction
        log_abs_det_jacobian = 2 * (math.log(2) - z - nn.functional.softplus(-2 * z))
        
        # Sum over action dimensions
        return log_prob_z.sum(dim=-1) - log_abs_det_jacobian.sum(dim=-1)
    
    def entropy(self) -> torch.Tensor:
        """
        Approximate entropy of the squashed distribution.
        
        Exact entropy is intractable, so we use the entropy of the base
        Gaussian plus expected log-Jacobian term.
        
        Returns:
            entropy: Approximate entropy [B]
        """
        # Base Gaussian entropy
        base_entropy = self._base_dist.entropy().sum(dim=-1)
        
        # Expected log |det Jacobian| - approximate with samples
        # For efficiency, just return base entropy (common approximation)
        return base_entropy


class TanhBijector:
    """
    Utility class for tanh transform operations.
    
    Provides stable implementations of:
    - atanh (inverse tanh)
    - log_prob correction for tanh-squashed distributions
    """
    
    @staticmethod
    def atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Stable inverse hyperbolic tangent."""
        x = torch.clamp(x, -1 + eps, 1 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))
    
    @staticmethod
    def log_prob_correction(action: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Compute the log probability correction for tanh squashing.
        
        log |det J| = sum_i log(1 - tanh^2(z_i)) = sum_i log(1 - a_i^2)
        
        Args:
            action: Squashed actions in [-1, 1]
            
        Returns:
            correction: Log Jacobian determinant to subtract from log_prob [B]
        """
        # log(1 - a^2) can be unstable for |a| close to 1
        # Use: log(1 - a^2) = log((1-a)(1+a)) = log(1-a) + log(1+a)
        action = torch.clamp(action, -1 + eps, 1 - eps)
        return torch.log1p(-action.pow(2) + eps).sum(dim=-1)
