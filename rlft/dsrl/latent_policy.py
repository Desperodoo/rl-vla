"""
Latent Gaussian Policy for DSRL

Implements a diagonal Gaussian policy in latent (noise) space.
The policy steers the initial noise of ShortCut Flow sampling.

Key design choices:
1. TanhNormal distribution: π_w(w | obs) naturally bounds output to [-magnitude, +magnitude]
   - Matches official DSRL implementation which uses action_magnitude with TanhNormal
   - No need for explicit clipping as tanh naturally constrains range
2. Support both full latent (pred_horizon, action_dim) and act_horizon-only steering
3. Reparameterization trick for gradient flow
4. Proper log_prob correction for tanh squashing

References:
- Official DSRL π₀: uses TanhMultivariateNormalDiag with action_magnitude
- SAC paper: Haarnoja et al., https://arxiv.org/abs/1812.05905
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Literal


class LatentGaussianPolicy(nn.Module):
    """TanhNormal Policy for latent/noise steering.
    
    Given observation conditioning, outputs a TanhNormal distribution over
    the initial noise (latent) used in ShortCut Flow sampling. Uses tanh
    squashing to naturally bound the output range to [-action_magnitude, +action_magnitude].
    
    This matches the official DSRL π₀ implementation which uses:
    - TanhMultivariateNormalDiag distribution
    - action_magnitude parameter to control output range
    
    Args:
        obs_dim: Dimension of observation conditioning (flattened)
        pred_horizon: Full prediction horizon of flow policy
        action_dim: Action dimension
        hidden_dims: Hidden layer dimensions for MLP
        log_std_range: (min, max) range for log standard deviation
        steer_mode: Which part of latent to steer
            - "full": steer entire (pred_horizon, action_dim) latent
            - "act_horizon": only steer first act_horizon steps
        act_horizon: Action execution horizon (used when steer_mode="act_horizon")
        state_dependent_std: Whether std depends on state
        action_magnitude: Output range scaling factor. Output is in [-magnitude, +magnitude].
            - From official DSRL: 1.0 for Libero, 2.0 for Aloha, 2.5 for real robot
            - Default 1.5 works well for ManiSkill
    """
    
    def __init__(
        self,
        obs_dim: int,
        pred_horizon: int,
        action_dim: int,
        hidden_dims: list = [256, 256, 256],
        log_std_range: Tuple[float, float] = (-5.0, 2.0),
        steer_mode: Literal["full", "act_horizon"] = "full",
        act_horizon: int = 8,
        state_dependent_std: bool = True,
        action_magnitude: float = 1.5,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.steer_mode = steer_mode
        self.act_horizon = act_horizon
        self.state_dependent_std = state_dependent_std
        self.log_std_min, self.log_std_max = log_std_range
        self.action_magnitude = action_magnitude
        
        # Compute output dimension based on steering mode
        if steer_mode == "full":
            self.steer_horizon = pred_horizon
        else:  # "act_horizon"
            self.steer_horizon = act_horizon
        
        self.latent_dim = self.steer_horizon * action_dim
        
        # Feature extractor MLP
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
            ])
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        
        self.feature_dim = hidden_dims[-1]
        
        # Mean output head
        self.mean_head = nn.Linear(self.feature_dim, self.latent_dim)
        
        # Log std head (state-dependent or learned)
        if state_dependent_std:
            self.log_std_head = nn.Linear(self.feature_dim, self.latent_dim)
        else:
            self.log_std = nn.Parameter(torch.zeros(self.latent_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Smaller weights for output heads
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        
        if self.state_dependent_std:
            nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
            nn.init.zeros_(self.log_std_head.bias)
    
    def forward(
        self,
        obs_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and log_std of latent distribution.
        
        Args:
            obs_cond: (B, obs_dim) flattened observation conditioning
            
        Returns:
            mean: (B, steer_horizon, action_dim) mean of latent Gaussian
            log_std: (B, steer_horizon, action_dim) log std of latent Gaussian
        """
        B = obs_cond.shape[0]
        
        # Extract features
        features = self.feature_net(obs_cond)
        
        # Compute mean
        mean_flat = self.mean_head(features)  # (B, latent_dim)
        mean = mean_flat.view(B, self.steer_horizon, self.action_dim)
        
        # Compute log_std
        if self.state_dependent_std:
            log_std_flat = self.log_std_head(features)
        else:
            log_std_flat = self.log_std.unsqueeze(0).expand(B, -1)
        
        # Clamp log_std for numerical stability
        log_std_flat = torch.clamp(log_std_flat, self.log_std_min, self.log_std_max)
        log_std = log_std_flat.view(B, self.steer_horizon, self.action_dim)
        
        return mean, log_std
    
    def sample(
        self,
        obs_cond: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample latent from TanhNormal distribution using reparameterization.
        
        Uses tanh squashing to naturally bound output to [-action_magnitude, +action_magnitude].
        This matches official DSRL implementation with TanhMultivariateNormalDiag.
        
        Args:
            obs_cond: (B, obs_dim) observation conditioning
            deterministic: If True, return tanh(mean) instead of sampling
            
        Returns:
            w: (B, pred_horizon, action_dim) sampled latent (squashed)
                - For steer_mode="full": entire latent is from policy
                - For steer_mode="act_horizon": first act_horizon from policy,
                  remaining from prior N(0, I) also squashed
            log_prob: (B,) log probability of the sampled latent (with tanh correction)
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        mean, log_std = self.forward(obs_cond)
        std = log_std.exp()
        
        if deterministic:
            # Use mean without noise, apply tanh squashing
            w_pre_tanh = mean
            w_steer = torch.tanh(w_pre_tanh) * self.action_magnitude
            log_prob = torch.zeros(B, device=device)
        else:
            # Reparameterization: sample from N(mean, std), then squash with tanh
            eps = torch.randn_like(mean)
            w_pre_tanh = mean + std * eps
            w_steer = torch.tanh(w_pre_tanh) * self.action_magnitude
            
            # Log probability with tanh correction (SAC-style)
            # log π(w|s) = log N(w_pre; μ, σ) - sum(log(1 - tanh²(w_pre)))
            # The action_magnitude scaling adds: -log(action_magnitude) per dimension
            
            # Gaussian log prob (pre-tanh)
            gaussian_log_prob = -0.5 * (
                ((w_pre_tanh - mean) / std).pow(2) + 
                2 * log_std + 
                np.log(2 * np.pi)
            )
            
            # Tanh correction: -log(1 - tanh²(x)) = -log(sech²(x))
            # Numerically stable version
            log_det_jacobian = 2 * (np.log(2) - w_pre_tanh - F.softplus(-2 * w_pre_tanh))
            
            # Sum over all dimensions
            log_prob = (gaussian_log_prob - log_det_jacobian).sum(dim=(1, 2))
            
            # Correction for action_magnitude scaling: -log(magnitude) per dimension
            log_prob = log_prob - self.latent_dim * np.log(self.action_magnitude)
        
        # Construct full latent
        if self.steer_mode == "full":
            w = w_steer
        else:
            # Pad remaining steps with prior samples (also squashed)
            remaining_horizon = self.pred_horizon - self.act_horizon
            w_prior_pre = torch.randn(B, remaining_horizon, self.action_dim, device=device)
            w_prior = torch.tanh(w_prior_pre) * self.action_magnitude
            w = torch.cat([w_steer, w_prior], dim=1)
        
        return w, log_prob
    
    def log_prob(
        self,
        obs_cond: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of a given latent under TanhNormal policy.
        
        NOTE: This requires inverse tanh (atanh) to recover pre-tanh values.
        
        Args:
            obs_cond: (B, obs_dim) observation conditioning
            w: (B, pred_horizon, action_dim) latent to evaluate (squashed)
            
        Returns:
            log_prob: (B,) log probability
        """
        mean, log_std = self.forward(obs_cond)
        std = log_std.exp()
        
        # Extract the steered portion of latent
        w_steer = w[:, :self.steer_horizon, :]
        
        # Inverse tanh to recover pre-tanh values
        # atanh(x) = 0.5 * log((1+x)/(1-x))
        w_scaled = w_steer / self.action_magnitude
        # Clamp to avoid numerical issues at boundaries
        w_scaled = torch.clamp(w_scaled, -0.999999, 0.999999)
        w_pre_tanh = torch.atanh(w_scaled)
        
        # Gaussian log prob (pre-tanh)
        gaussian_log_prob = -0.5 * (
            ((w_pre_tanh - mean) / std).pow(2) + 
            2 * log_std + 
            np.log(2 * np.pi)
        )
        
        # Tanh correction
        log_det_jacobian = 2 * (np.log(2) - w_pre_tanh - F.softplus(-2 * w_pre_tanh))
        
        # Sum over all dimensions
        log_prob = (gaussian_log_prob - log_det_jacobian).sum(dim=(1, 2))
        
        # Correction for action_magnitude scaling
        log_prob = log_prob - self.latent_dim * np.log(self.action_magnitude)
        
        return log_prob
    
    def kl_to_prior(
        self,
        obs_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute approximate KL divergence from TanhNormal policy to squashed prior.
        
        NOTE: Exact KL for TanhNormal is intractable. This computes KL in the 
        pre-tanh space which provides a useful regularization signal.
        
        KL(N(μ, σ²) || N(0, I)) = 0.5 * (μ² + σ² - 1 - log σ²)
        
        Args:
            obs_cond: (B, obs_dim) observation conditioning
            
        Returns:
            kl: (B,) KL divergence per sample (approximate)
        """
        mean, log_std = self.forward(obs_cond)
        var = (2 * log_std).exp()
        
        # KL for each dimension in pre-tanh space, then sum
        kl_per_dim = 0.5 * (mean.pow(2) + var - 1 - 2 * log_std)
        kl = kl_per_dim.sum(dim=(1, 2))
        
        return kl
    
    def entropy(
        self,
        obs_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute approximate entropy of the TanhNormal distribution.
        
        NOTE: Exact entropy for TanhNormal is intractable. This computes entropy
        in pre-tanh space as an approximation. For SAC, using -log_prob is preferred.
        
        H(N(μ, σ²)) ≈ 0.5 * (1 + log(2π) + log σ²) in pre-tanh space
        
        Args:
            obs_cond: (B, obs_dim) observation conditioning
            
        Returns:
            entropy: (B,) entropy per sample (approximate)
        """
        _, log_std = self.forward(obs_cond)
        
        # Entropy for each dimension in pre-tanh space, then sum
        entropy_per_dim = 0.5 * (1 + np.log(2 * np.pi) + 2 * log_std)
        entropy = entropy_per_dim.sum(dim=(1, 2))
        
        return entropy
