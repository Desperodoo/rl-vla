"""
DSRL Agent: Dual-Stage Reinforcement Learning with Latent Steering

Combines frozen ShortCut Flow policy with trainable latent steering policy.
The latent policy steers the initial noise of flow sampling to improve actions.

Two-stage training:
- Stage 1 (Offline): Advantage-Weighted MLE to warm-start latent policy
- Stage 2 (Online): Macro-step PPO to fine-tune latent policy

Key principle: Flow/Shortcut model stays FROZEN, RL only in latent space.

References:
- ShortCut Flow: diffusion_policy/algorithms/shortcut_flow.py
- AW-ShortCut Flow: diffusion_policy/algorithms/aw_shortcut_flow.py
- ToDo.md: Two-stage DSRL design specification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union, Literal
import copy
import math

# Import base components from diffusion_policy
import sys
from pathlib import Path

# Add diffusion_policy to path for imports
_dp_path = Path(__file__).parent.parent / "diffusion_policy"
if str(_dp_path) not in sys.path:
    sys.path.insert(0, str(_dp_path))

# Add dsrl to path for imports (support both package and standalone usage)
_dsrl_path = Path(__file__).parent
if str(_dsrl_path) not in sys.path:
    sys.path.insert(0, str(_dsrl_path))

from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.algorithms.networks import soft_update, DoubleQNetwork
from diffusion_policy.rlpd.networks import EnsembleQNetwork

# Use absolute imports to support both package and standalone usage
from latent_policy import LatentGaussianPolicy
from value_network import ValueNetwork


class DSRLAgent(nn.Module):
    """DSRL Agent with frozen flow policy and trainable latent steering.
    
    The agent combines:
    1. Frozen ShortCut Flow velocity network (pretrained)
    2. Frozen Ensemble Q-network (pretrained from offline RL)
    3. Trainable latent steering policy π_w(w | obs_cond)
    4. Trainable value network V(s) for online PPO (Stage 2)
    
    Training stages:
    - Stage 1: Offline AW-MLE to warm-start latent policy using Q-values
    - Stage 2: Online PPO to fine-tune latent policy in real environment
    
    Args:
        velocity_net: Pretrained ShortCutVelocityUNet1D (will be frozen)
        q_network: Pretrained EnsembleQNetwork (will be frozen)
        latent_policy: LatentGaussianPolicy for noise steering
        value_network: ValueNetwork for online PPO (optional, Stage 2)
        action_dim: Dimension of action space
        obs_dim: Dimension of flattened observation features
        obs_horizon: Number of observation frames
        pred_horizon: Length of action sequence to predict
        act_horizon: Length of action sequence for Q-learning and execution
        num_inference_steps: Number of flow ODE integration steps
        # Stage 1 (Offline AW-MLE) parameters
        num_candidates: Number of latent candidates M for AW-MLE
        kappa: UCB coefficient for Q aggregation (μ - κσ)
        tau: Soft baseline temperature for advantage
        beta_latent: Temperature for advantage weighting
        advantage_clip: Clip range for advantage
        kl_coef: KL-to-prior regularization coefficient
        # Stage 2 (Online PPO) parameters
        ppo_clip: PPO clip ratio
        entropy_coef: Entropy bonus coefficient
        kl_prior_coef: KL-to-prior coefficient for online
        # Exploration parameters
        prior_mix_ratio: Probability of using prior instead of policy
        prior_mix_decay: Decay rate for prior_mix_ratio per update
        prior_mix_min: Minimum prior_mix_ratio
        device: Device to run on
    """
    
    def __init__(
        self,
        velocity_net: ShortCutVelocityUNet1D,
        q_network: EnsembleQNetwork,
        latent_policy: LatentGaussianPolicy,
        value_network: Optional[ValueNetwork] = None,
        action_dim: int = 7,
        obs_dim: int = 512,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        act_horizon: int = 8,
        num_inference_steps: int = 8,
        # Stage 1 parameters
        num_candidates: int = 16,
        kappa: float = 1.0,
        use_ucb: bool = True,
        tau: float = 5.0,
        beta_latent: float = 1.0,
        advantage_clip: float = 20.0,
        kl_coef: float = 1e-3,
        # Stage 2 parameters
        ppo_clip: float = 0.2,
        entropy_coef: float = 1e-3,
        kl_prior_coef: float = 1e-3,
        # Exploration parameters
        prior_mix_ratio: float = 0.3,
        prior_mix_decay: float = 0.995,
        prior_mix_min: float = 0.05,
        device: str = "cuda",
    ):
        super().__init__()
        
        # Frozen components (pretrained)
        self.velocity_net = velocity_net
        self.q_network = q_network
        
        # Freeze velocity net and Q-network
        for param in self.velocity_net.parameters():
            param.requires_grad = False
        for param in self.q_network.parameters():
            param.requires_grad = False
        
        # EMA velocity net for sampling
        self.velocity_net_ema = copy.deepcopy(self.velocity_net)
        for param in self.velocity_net_ema.parameters():
            param.requires_grad = False
        
        # Trainable components
        self.latent_policy = latent_policy
        self.value_network = value_network
        
        # Old policy for PPO ratio computation
        self.latent_policy_old = copy.deepcopy(latent_policy)
        for param in self.latent_policy_old.parameters():
            param.requires_grad = False
        
        # Dimensions
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.num_inference_steps = num_inference_steps
        self.device = device
        
        # Stage 1 parameters
        self.num_candidates = num_candidates
        self.kappa = kappa
        self.use_ucb = use_ucb
        self.tau = tau
        self.beta_latent = beta_latent
        self.advantage_clip = advantage_clip
        self.kl_coef = kl_coef
        
        # Stage 2 parameters
        self.ppo_clip = ppo_clip
        self.entropy_coef = entropy_coef
        self.kl_prior_coef = kl_prior_coef
        
        # Latent policy steering mode (synced from LatentGaussianPolicy)
        self.steer_mode = latent_policy.steer_mode if latent_policy else "full"
        self.steer_horizon = latent_policy.steer_horizon if latent_policy else pred_horizon
        
        # Exploration parameters
        self.prior_mix_ratio = prior_mix_ratio
        self.prior_mix_decay = prior_mix_decay
        self.prior_mix_min = prior_mix_min
        self._current_prior_mix = prior_mix_ratio
        
        # Detect Q-network type for compatibility
        self._is_double_q = isinstance(q_network, DoubleQNetwork)
    
    def _get_q_values(self, actions: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        """Get Q-values in unified format (num_qs, B, 1).
        
        Handles both DoubleQNetwork and EnsembleQNetwork formats.
        
        Args:
            actions: (B, act_horizon, action_dim) action sequence
            obs_cond: (B, obs_dim) observation features
            
        Returns:
            q_values: (num_qs, B, 1) stacked Q-values
        """
        if self._is_double_q:
            q1, q2 = self.q_network(actions, obs_cond)
            return torch.stack([q1, q2], dim=0)  # (2, B, 1)
        else:
            return self.q_network(actions, obs_cond)  # (num_qs, B, 1)
    
    # ==================== Core Sampling Interface ====================
    
    def sample_actions_from_latent(
        self,
        obs_cond: torch.Tensor,
        w: torch.Tensor,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """
        Sample actions by integrating flow ODE from given latent.
        
        This is the KEY interface: instead of starting from random noise,
        we start from the given latent w (which is steered by latent policy).
        
        Args:
            obs_cond: (B, obs_dim) flattened observation conditioning
            w: (B, pred_horizon, action_dim) initial latent (steered noise)
            use_ema: Whether to use EMA velocity network
            
        Returns:
            actions: (B, pred_horizon, action_dim) generated action sequence
        """
        net = self.velocity_net_ema if use_ema else self.velocity_net
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        # Start from given latent instead of randn
        x = w.clone()
        
        # Uniform steps for flow ODE integration
        dt = 1.0 / self.num_inference_steps
        d = torch.full((B,), dt, device=device)
        
        with torch.no_grad():
            for i in range(self.num_inference_steps):
                t = torch.full((B,), i * dt, device=device)
                v = net(x, t, d, obs_cond)
                x = x + dt * v
        
        return torch.clamp(x, -1.0, 1.0)
    
    @torch.no_grad()
    def get_action(
        self,
        obs_features: torch.Tensor,
        use_latent_policy: bool = True,
        deterministic: bool = False,
        use_prior_mixing: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Sample action sequence for execution.
        
        Args:
            obs_features: (B, obs_horizon, obs_dim) or (B, obs_dim) observation
            use_latent_policy: If True, use latent policy for steering
            deterministic: If True, use policy mean instead of sampling
            use_prior_mixing: If True, mix with prior based on prior_mix_ratio
            
        Returns:
            actions: (B, pred_horizon, action_dim) action sequence
            latent_w: (B, pred_horizon, action_dim) sampled latent (if use_latent_policy)
            log_prob: (B,) log probability of latent (if use_latent_policy)
        """
        # Flatten obs if needed
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        else:
            obs_cond = obs_features
        
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        if use_latent_policy:
            # Sample latent from policy
            w, log_prob = self.latent_policy.sample(obs_cond, deterministic=deterministic)
            
            # Mixed exploration: with probability η, use prior instead
            if use_prior_mixing and not deterministic and self._current_prior_mix > 0:
                mask = torch.rand(B, device=device) < self._current_prior_mix
                w_prior = torch.randn_like(w)
                w = torch.where(mask.view(-1, 1, 1), w_prior, w)
                
                # CRITICAL: Recompute log_prob for the actual latent used
                # This is essential for correct PPO ratio computation.
                # old_log_prob must match the actual w stored in buffer,
                # otherwise ratio = exp(new_log_prob - old_log_prob) will explode.
                log_prob = self.latent_policy.log_prob(obs_cond, w)
            
            # Generate actions from latent
            actions = self.sample_actions_from_latent(obs_cond, w, use_ema=True)
            
            return actions, w, log_prob
        else:
            # Standard sampling with random noise (baseline)
            w = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
            actions = self.sample_actions_from_latent(obs_cond, w, use_ema=True)
            
            return actions, None, None
    
    # ==================== Stage 1: Offline AW-MLE ====================
    
    def compute_offline_loss(
        self,
        obs_cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Stage 1 offline loss: Advantage-Weighted MLE for latent policy.
        
        This trains the latent policy to prefer latents that lead to high-Q actions,
        without directly maximizing Q (which could cause OOD issues).
        
        Process:
        1. Sample M latent candidates from prior
        2. Generate actions for each candidate
        3. Score with Q-network using UCB aggregation (μ - κσ)
        4. Compute advantage with soft baseline
        5. Train latent policy with weighted MLE
        
        Args:
            obs_cond: (B, obs_dim) flattened observation conditioning
            
        Returns:
            Dict with loss components
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        M = self.num_candidates
        
        # ===== 1. Sample M latent candidates from prior =====
        # Key fix: When steer_mode="act_horizon", only the steered portion should vary.
        # The tail (non-steered) portion should be fixed per observation to avoid
        # credit assignment pollution.
        if self.steer_mode == "act_horizon":
            # Sample M candidates only for steered head portion
            # Shape: (B, M, steer_horizon, action_dim)
            w_head_candidates = torch.randn(B, M, self.steer_horizon, self.action_dim, device=device)
            
            # Sample ONE fixed tail per observation (not per candidate)
            # Shape: (B, 1, remaining_horizon, action_dim) -> broadcast to (B, M, ...)
            remaining_horizon = self.pred_horizon - self.steer_horizon
            w_tail_fixed = torch.randn(B, 1, remaining_horizon, self.action_dim, device=device)
            w_tail_fixed = w_tail_fixed.expand(-1, M, -1, -1)  # (B, M, remaining_horizon, action_dim)
            
            # Concatenate: each candidate has same tail, different head
            w_candidates = torch.cat([w_head_candidates, w_tail_fixed], dim=2)  # (B, M, pred_horizon, action_dim)
        else:
            # Full steering mode: vary entire latent
            # Shape: (B, M, pred_horizon, action_dim)
            w_candidates = torch.randn(B, M, self.pred_horizon, self.action_dim, device=device)
        
        # ===== 2. Generate actions for all candidates =====
        # Flatten to (B*M, ...) for efficient forward pass
        obs_cond_expand = obs_cond.unsqueeze(1).expand(-1, M, -1)  # (B, M, obs_dim)
        obs_cond_flat = obs_cond_expand.reshape(B * M, -1)  # (B*M, obs_dim)
        w_flat = w_candidates.reshape(B * M, self.pred_horizon, self.action_dim)
        
        # Generate actions (no grad, using EMA net)
        with torch.no_grad():
            actions_full = self.sample_actions_from_latent(obs_cond_flat, w_flat, use_ema=True)
            actions_for_q = actions_full[:, :self.act_horizon, :]  # (B*M, act_horizon, action_dim)
        
        # ===== 3. Score with Q-network using UCB aggregation =====
        with torch.no_grad():
            # Get all Q-values (supports both DoubleQNetwork and EnsembleQNetwork)
            q_all = self._get_q_values(actions_for_q, obs_cond_flat)  # (num_qs, B*M, 1)
            
            # Aggregation method: UCB (μ - κσ) or simple mean
            q_mean = q_all.mean(dim=0).squeeze(-1)  # (B*M,)
            
            if self.use_ucb:
                # UCB aggregation: μ - κσ (pessimistic)
                # Note: Use unbiased=False for std() to avoid high variance with DoubleQ (only 2 heads)
                # With unbiased=True (default), std of 2 samples uses N-1=1 in denominator, causing instability
                q_std = q_all.std(dim=0, unbiased=False).squeeze(-1)  # (B*M,)
                q_scores_flat = q_mean - self.kappa * q_std  # (B*M,)
            else:
                # Simple mean aggregation (no pessimism)
                q_scores_flat = q_mean
            
            # Reshape back to (B, M)
            q_scores = q_scores_flat.view(B, M)
        
        # ===== 4. Compute advantage with soft baseline =====
        with torch.no_grad():
            # Soft baseline: τ * log(mean(exp(q/τ)))
            baseline = self.tau * torch.logsumexp(q_scores / self.tau, dim=1, keepdim=True) - self.tau * math.log(M)
            
            # Advantage
            advantage = q_scores - baseline  # (B, M)
            
            # Clip advantage for stability
            advantage = torch.clamp(advantage, -self.advantage_clip, self.advantage_clip)
            
            # Softmax weights: ω_i = softmax(A_i / β)
            weights = F.softmax(advantage / self.beta_latent, dim=1)  # (B, M)
        
        # ===== 5. Compute weighted MLE loss =====
        # Get log probability of each candidate under current policy
        w_candidates_flat = w_candidates.reshape(B * M, self.pred_horizon, self.action_dim)
        obs_cond_expand_flat = obs_cond_flat
        
        # Need to compute log_prob for each candidate
        log_probs = self.latent_policy.log_prob(obs_cond_expand_flat, w_candidates_flat)  # (B*M,)
        log_probs = log_probs.view(B, M)  # (B, M)
        
        # Weighted negative log-likelihood
        nll_loss = -(weights * log_probs).sum(dim=1).mean()
        
        # ===== 6. KL regularization to prior =====
        kl_loss = self.latent_policy.kl_to_prior(obs_cond).mean()
        
        # Total loss
        total_loss = nll_loss + self.kl_coef * kl_loss
        
        # ===== 7. Compute diagnostic metrics =====
        # Effective sample size: eff_num = 1 / sum(ω²)
        # Ideal value is M (uniform weights), low value indicates weight concentration
        eff_num = 1.0 / (weights.pow(2).sum(dim=1) + 1e-8)  # (B,)
        eff_num_mean = eff_num.mean()
        
        # Correlation between advantage and log_prob: corr(A, log_prob)
        # Positive correlation indicates policy is learning to generate high-advantage latents
        # Negative or zero indicates policy is being pulled back to prior
        A_flat = advantage.reshape(-1)  # (B*M,)
        lp_flat = log_probs.reshape(-1)  # (B*M,)
        A_centered = A_flat - A_flat.mean()
        lp_centered = lp_flat - lp_flat.mean()
        cov_A_lp = (A_centered * lp_centered).mean()
        A_std_val = A_flat.std() + 1e-8
        lp_std_val = lp_flat.std() + 1e-8
        corr_A_logprob = cov_A_lp / (A_std_val * lp_std_val)
        
        # Monitor policy std (log_std) to detect collapse or divergence
        with torch.no_grad():
            _, log_std = self.latent_policy.forward(obs_cond)  # (B, steer_horizon, action_dim)
            log_std_mean = log_std.mean()
            log_std_std = log_std.std()
        
        return {
            "loss": total_loss,
            "nll_loss": nll_loss,
            "kl_loss": kl_loss,
            "q_mean": q_scores.mean(),
            "q_std": q_scores.std(),
            "advantage_mean": advantage.mean(),
            "advantage_std": advantage.std(),
            "weight_max": weights.max(),
            "weight_entropy": -(weights * (weights + 1e-8).log()).sum(dim=1).mean(),
            # New diagnostic metrics
            "eff_num": eff_num_mean,
            "corr_A_logprob": corr_A_logprob,
            "log_std_mean": log_std_mean,
            "log_std_std": log_std_std,
        }
    
    # ==================== Stage 2: Online PPO ====================
    
    def compute_ppo_loss(
        self,
        obs_cond: torch.Tensor,
        latent_w: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantage: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Stage 2 online PPO loss for latent policy.
        
        Args:
            obs_cond: (B, obs_dim) observation conditioning
            latent_w: (B, pred_horizon, action_dim) sampled latents
            old_log_prob: (B,) log probabilities from old policy
            advantage: (B,) GAE advantage estimates
            returns: (B,) GAE returns for value loss
            
        Returns:
            Dict with loss components
        """
        B = obs_cond.shape[0]
        
        # ===== Policy Loss (PPO Clip) =====
        # New log probability
        new_log_prob = self.latent_policy.log_prob(obs_cond, latent_w)
        
        # Ratio
        ratio = torch.exp(new_log_prob - old_log_prob)
        
        # ===== Compute approx_kl (critical diagnostic) =====
        # approx_kl = E[(ratio - 1) - log(ratio)] ≈ KL(old || new)
        # Also known as "KL(3)" in some implementations
        with torch.no_grad():
            log_ratio = new_log_prob - old_log_prob
            approx_kl = ((ratio - 1) - log_ratio).mean()
            # Alternative: approx_kl_simple = (old_log_prob - new_log_prob).mean()
        
        # Record raw advantage stats BEFORE normalization
        adv_mean = advantage.mean()
        adv_std = advantage.std()
        adv_min = advantage.min()
        adv_max = advantage.max()
        
        # Normalize advantage
        advantage_norm = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        # Clipped objective
        obj1 = ratio * advantage_norm
        obj2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantage_norm
        policy_loss = -torch.min(obj1, obj2).mean()
        
        # ===== Compute clip_frac (critical diagnostic) =====
        with torch.no_grad():
            clip_frac = ((ratio < 1 - self.ppo_clip) | (ratio > 1 + self.ppo_clip)).float().mean()
        
        # ===== Entropy Bonus =====
        entropy = self.latent_policy.entropy(obs_cond).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # ===== KL to Prior Regularization =====
        kl_prior = self.latent_policy.kl_to_prior(obs_cond).mean()
        kl_loss = self.kl_prior_coef * kl_prior
        
        # ===== Value Loss (if value network exists) =====
        value_loss = torch.tensor(0.0, device=obs_cond.device)
        if self.value_network is not None:
            value_pred = self.value_network(obs_cond).squeeze(-1)
            value_loss = F.mse_loss(value_pred, returns)
        
        # Total loss
        total_loss = policy_loss + entropy_loss + kl_loss + value_loss
        
        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "kl_prior": kl_prior,
            # Critical PPO diagnostics
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "adv_min": adv_min,
            "adv_max": adv_max,
            # Ratio statistics
            "ratio_mean": ratio.mean(),
            "ratio_std": ratio.std(),
        }
    
    def sync_old_policy(self):
        """Sync old policy with current policy (call before PPO update)."""
        self.latent_policy_old.load_state_dict(self.latent_policy.state_dict())
    
    def decay_prior_mix_ratio(self):
        """Decay prior mixing ratio (call after each update epoch)."""
        self._current_prior_mix = max(
            self.prior_mix_min,
            self._current_prior_mix * self.prior_mix_decay
        )
    
    def get_prior_mix_ratio(self) -> float:
        """Get current prior mixing ratio."""
        return self._current_prior_mix
    
    def set_prior_mix_ratio(self, value: float):
        """Set prior mixing ratio."""
        self._current_prior_mix = max(self.prior_mix_min, value)
    
    # ==================== Value Estimation ====================
    
    @torch.no_grad()
    def get_value(
        self,
        obs_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get value estimate for observations.
        
        Args:
            obs_features: (B, obs_horizon, obs_dim) or (B, obs_dim) observation
            
        Returns:
            value: (B, 1) value estimates
        """
        if self.value_network is None:
            raise ValueError("Value network not initialized. Required for Stage 2.")
        
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        else:
            obs_cond = obs_features
        
        return self.value_network(obs_cond)
    
    # ==================== Q-Value Utilities ====================
    
    @torch.no_grad()
    def get_q_value(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
        aggregation: Literal["min", "mean", "ucb"] = "ucb",
    ) -> torch.Tensor:
        """
        Get Q-value estimate for state-action pairs.
        
        Args:
            obs_cond: (B, obs_dim) observation conditioning
            actions: (B, act_horizon, action_dim) action sequence
            aggregation: How to aggregate ensemble Q-values
                - "min": min over ensemble (conservative)
                - "mean": mean over ensemble
                - "ucb": μ - κσ (UCB-style)
            
        Returns:
            q_value: (B,) Q-value estimates
        """
        q_all = self._get_q_values(actions, obs_cond)  # (num_qs, B, 1)
        
        if aggregation == "min":
            return q_all.min(dim=0).values.squeeze(-1)
        elif aggregation == "mean":
            return q_all.mean(dim=0).squeeze(-1)
        elif aggregation == "ucb":
            q_mean = q_all.mean(dim=0).squeeze(-1)
            q_std = q_all.std(dim=0).squeeze(-1)
            return q_mean - self.kappa * q_std
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    # ==================== Checkpoint Utils ====================
    
    def get_trainable_parameters(self) -> list:
        """Get list of trainable parameters (latent policy + value network)."""
        params = list(self.latent_policy.parameters())
        if self.value_network is not None:
            params += list(self.value_network.parameters())
        return params
    
    def save_checkpoint(self, path: str):
        """Save trainable components to checkpoint."""
        state = {
            "latent_policy": self.latent_policy.state_dict(),
            "prior_mix_ratio": self._current_prior_mix,
        }
        if self.value_network is not None:
            state["value_network"] = self.value_network.state_dict()
        torch.save(state, path)
    
    def load_checkpoint(self, path: str, load_value: bool = True):
        """Load trainable components from checkpoint."""
        state = torch.load(path, map_location=self.device)
        self.latent_policy.load_state_dict(state["latent_policy"])
        self._current_prior_mix = state.get("prior_mix_ratio", self.prior_mix_ratio)
        
        if load_value and self.value_network is not None and "value_network" in state:
            self.value_network.load_state_dict(state["value_network"])
