"""
DSRL SAC Agent: Off-Policy Dual-Stage Reinforcement Learning with Latent Steering

Combines frozen ShortCut Flow policy with trainable latent steering policy,
using SAC for off-policy learning in latent space.

Two-stage training:
- Stage 1 (Offline): Train Q^W via distillation from Q^A, then train π_W
  - Option A (awmle): Advantage-Weighted MLE using Q^W scores
  - Option B (sac): SAC-style policy gradient using Q^W
- Stage 2 (Online): SAC on latent space with high UTD ratio

Key principle: Flow/Shortcut model stays FROZEN, RL only in latent space.

Aligned with original DSRL-NA:
- Q^W distillation is mandatory (not optional)
- Policy update uses Q^W(s, w) not Q^A(s, a)
- No explicit action→noise inversion (removed NA module)
- No KL-to-prior regularization (SAC entropy regularization instead)

References:
- DSRL-NA: https://flow-fine-tuning.github.io/
- SAC: https://arxiv.org/abs/1812.05905
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union, Literal
import copy
import math
import sys
from pathlib import Path

# Add diffusion_policy to path for imports
_dp_path = Path(__file__).parent.parent.parent / "diffusion_policy"
if str(_dp_path) not in sys.path:
    sys.path.insert(0, str(_dp_path))

# Add dsrl to path for shared components
_dsrl_path = Path(__file__).parent.parent.parent / "dsrl"
if str(_dsrl_path) not in sys.path:
    sys.path.insert(0, str(_dsrl_path))

from diffusion_policy.algorithms.shortcut_flow import ShortCutVelocityUNet1D
from diffusion_policy.algorithms.networks import DoubleQNetwork
from diffusion_policy.rlpd.networks import EnsembleQNetwork

# Import shared components from on-policy DSRL
from latent_policy import LatentGaussianPolicy

# Import off-policy specific components
from ..models.latent_q_network import DoubleLatentQNetwork, Temperature
# NOTE: NoiseAliasingModule removed - original DSRL-NA doesn't need explicit action→noise inversion


class DSRLSACAgent(nn.Module):
    """DSRL Agent with SAC-based latent steering (off-policy).
    
    The agent combines:
    1. Frozen ShortCut Flow velocity network (pretrained)
    2. Frozen Q-network Q^A for distillation target (pretrained)
    3. Trainable latent steering policy π_w(w | obs_cond)
    4. Trainable latent Q-networks Q^W(s, w) for policy update
    5. Learnable temperature for entropy regularization
    
    Training stages:
    - Stage 1 (Offline): Q^W distillation + policy update (AWMLE or SAC)
    - Stage 2 (Online): SAC with latent Q-networks Q^W(s, w)
    
    Aligned with original DSRL-NA:
    - Q^W distillation is mandatory (not optional)
    - Policy update uses Q^W(s, w) not Q^A(s, a)
    - No explicit action→noise inversion
    - No KL-to-prior regularization (SAC entropy instead)
    
    Args:
        velocity_net: Pretrained ShortCutVelocityUNet1D (will be frozen)
        q_network: Pretrained Q-network for distillation target (will be frozen)
        latent_policy: LatentGaussianPolicy for noise steering
        latent_q_network: DoubleLatentQNetwork for Q^W (REQUIRED)
        action_dim: Dimension of action space
        obs_dim: Dimension of flattened observation features
        obs_horizon: Number of observation frames
        pred_horizon: Length of action sequence to predict
        act_horizon: Length of action sequence for Q-learning and execution
        num_inference_steps: Number of flow ODE integration steps
        # Stage 1 (Offline) parameters
        num_candidates: Number of latent candidates M for AWMLE
        tau: Soft baseline temperature for advantage
        beta_latent: Temperature for advantage weighting
        advantage_clip: Clip range for advantage
        qw_distill_coef: Coefficient for Q^W distillation loss
        # TanhNormal action magnitude (replaces noise clipping)
        action_magnitude: Output range for TanhNormal policy [-magnitude, +magnitude]
            - From official DSRL: 1.0 for Libero, 2.0 for Aloha, 2.5 for real robot
            - Default 1.5 works well for ManiSkill
        # Stage 2 (Online SAC) parameters
        gamma: Discount factor
        tau_target: Target network soft update rate
        init_temperature: Initial SAC temperature
        learnable_temp: Whether temperature is learnable
        target_entropy: Target entropy for auto-tuning (None = auto)
        backup_entropy: Whether to include entropy in TD target
        device: Device to run on
    """
    
    def __init__(
        self,
        velocity_net: ShortCutVelocityUNet1D,
        q_network: Union[DoubleQNetwork, EnsembleQNetwork],
        latent_policy: LatentGaussianPolicy,
        latent_q_network: DoubleLatentQNetwork,  # Now REQUIRED
        action_dim: int = 7,
        obs_dim: int = 512,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        act_horizon: int = 8,
        num_inference_steps: int = 8,
        # Stage 1 parameters
        num_candidates: int = 16,
        tau: float = 5.0,
        beta_latent: float = 1.0,
        advantage_clip: float = 20.0,
        qw_distill_coef: float = 1.0,
        # TanhNormal action magnitude (replaces noise clipping)
        action_magnitude: float = 1.5,
        # Stage 2 SAC parameters
        gamma: float = 0.99,
        tau_target: float = 0.005,
        init_temperature: float = 0.1,
        learnable_temp: bool = True,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        
        # Frozen components (pretrained)
        self.velocity_net = velocity_net
        self.q_network = q_network  # For Q^W distillation (Q^A target)
        
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
        self.latent_q_network = latent_q_network  # Q^W network (REQUIRED)
        
        # Temperature for SAC
        self.temperature = Temperature(
            initial_temperature=init_temperature,
            learnable=learnable_temp,
        )
        
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
        self.tau = tau
        self.beta_latent = beta_latent
        self.advantage_clip = advantage_clip
        self.qw_distill_coef = qw_distill_coef
        self.action_magnitude = action_magnitude  # For TanhNormal distribution
        
        # Stage 2 SAC parameters
        self.gamma = gamma
        self.tau_target = tau_target
        self.backup_entropy = backup_entropy
        
        # Latent policy steering mode (synced from LatentGaussianPolicy)
        self.steer_mode = latent_policy.steer_mode if latent_policy else "full"
        self.steer_horizon = latent_policy.steer_horizon if latent_policy else pred_horizon
        
        # Auto-compute target entropy if not provided
        if target_entropy is None:
            # latent_dim = self.steer_horizon * action_dim
            self.target_entropy = -0.5 * action_dim
        else:
            self.target_entropy = target_entropy
        
        # Detect Q-network type for compatibility
        self._is_double_q = isinstance(q_network, DoubleQNetwork)
    
    def _get_q_values(self, actions: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        """Get Q-values from action-space Q-network in unified format (num_qs, B, 1).
        
        Used in Stage 1 for action scoring.
        
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Sample action sequence for execution.
        
        Args:
            obs_features: (B, obs_horizon, obs_dim) or (B, obs_dim) observation
            use_latent_policy: If True, use latent policy for steering
            deterministic: If True, use policy mean instead of sampling
            
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
            # Sample latent from policy (TanhNormal already bounds to [-action_magnitude, +action_magnitude])
            w, log_prob = self.latent_policy.sample(obs_cond, deterministic=deterministic)
            
            # Generate actions from latent
            actions = self.sample_actions_from_latent(obs_cond, w, use_ema=True)
            
            return actions, w, log_prob
        else:
            # Standard sampling with random noise (squashed via tanh)
            w = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
            w = torch.tanh(w) * self.action_magnitude
            actions = self.sample_actions_from_latent(obs_cond, w, use_ema=True)
            
            return actions, None, None
    
    # ==================== Stage 1: Offline Training with Q^W ====================
    
    def compute_offline_loss(
        self,
        obs_cond: torch.Tensor,
        algo: Literal["awmle", "sac"] = "awmle",
        update_actor: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Stage 1 offline loss with Q^W distillation and policy update.
        
        This implements the original DSRL-NA approach:
        1. Q^W distillation: Train Q^W to match Q^A via MSE
        2. Policy update using Q^W (not Q^A)
        
        Supports two training phases:
        - Q^W warmup (update_actor=False): Only train Q^W
        - Full training (update_actor=True): Train both Q^W and π_W
        
        Supports two policy update algorithms:
        - "awmle": Advantage-Weighted MLE using Q^W scores
        - "sac": SAC-style policy gradient using Q^W
        
        Args:
            obs_cond: (B, obs_dim) flattened observation conditioning
            algo: Policy update algorithm ("awmle" or "sac")
            update_actor: Whether to update actor (False = Q^W warmup only)
            
        Returns:
            Dict with loss components and metrics
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        M = self.num_candidates
        
        # ===== 1. Sample latent candidates from prior (squashed via tanh) =====
        if self.steer_mode == "act_horizon":
            w_head_pre = torch.randn(B, M, self.steer_horizon, self.action_dim, device=device)
            w_head = torch.tanh(w_head_pre) * self.action_magnitude
            remaining = self.pred_horizon - self.steer_horizon
            w_tail_pre = torch.randn(B, 1, remaining, self.action_dim, device=device).expand(-1, M, -1, -1)
            w_tail = torch.tanh(w_tail_pre) * self.action_magnitude
            w_candidates = torch.cat([w_head, w_tail], dim=2)
        else:
            w_candidates_pre = torch.randn(B, M, self.pred_horizon, self.action_dim, device=device)
            w_candidates = torch.tanh(w_candidates_pre) * self.action_magnitude
        
        # ===== 2. Generate actions for Q^A distillation target =====
        obs_cond_expand = obs_cond.unsqueeze(1).expand(-1, M, -1)
        obs_cond_flat = obs_cond_expand.reshape(B * M, -1)
        w_flat = w_candidates.reshape(B * M, self.pred_horizon, self.action_dim)
        
        with torch.no_grad():
            actions_full = self.sample_actions_from_latent(obs_cond_flat, w_flat, use_ema=True)
            actions_for_q = actions_full[:, :self.act_horizon, :]
        
        # ===== 3. Get Q^A targets for distillation =====
        with torch.no_grad():
            q_all = self._get_q_values(actions_for_q, obs_cond_flat)
            qa_scores_flat = q_all.mean(dim=0).squeeze(-1)  # (B*M,)
            qa_scores = qa_scores_flat.view(B, M)  # For metrics
        
        # ===== 4. Q^W distillation loss (ALWAYS computed) =====
        q1_w, q2_w = self.latent_q_network(obs_cond_flat, w_flat)
        qw_pred = (q1_w + q2_w) / 2  # (B*M, 1)
        qa_target = qa_scores_flat.unsqueeze(-1).detach()
        qw_distill_loss = F.mse_loss(qw_pred, qa_target)
        
        # ===== 5. Compute policy loss using Q^W (not Q^A!) =====
        actor_loss = torch.tensor(0.0, device=device)
        entropy = torch.tensor(0.0, device=device)
        eff_num = torch.tensor(1.0, device=device)
        
        if update_actor:
            if algo == "awmle":
                actor_loss, eff_num = self._compute_awmle_actor_loss(
                    obs_cond, obs_cond_flat, w_candidates, w_flat, B, M
                )
            else:  # sac
                actor_loss, entropy = self._compute_sac_actor_loss(obs_cond)
        
        # ===== 6. Total loss =====
        total_loss = self.qw_distill_coef * qw_distill_loss
        if update_actor:
            total_loss = total_loss + actor_loss
        
        # ===== 7. Metrics =====
        with torch.no_grad():
            qw_scores = qw_pred.view(B, M).detach()
            _, log_std = self.latent_policy.forward(obs_cond)
            log_std_mean = log_std.mean()
        
        return {
            "loss": total_loss,
            "qw_distill_loss": qw_distill_loss,
            "actor_loss": actor_loss,
            "qa_mean": qa_scores.mean(),
            "qa_std": qa_scores.std(),
            "qw_mean": qw_scores.mean(),
            "qw_std": qw_scores.std(),
            "eff_num": eff_num,
            "entropy": entropy,
            "log_std_mean": log_std_mean,
        }
    
    def _compute_awmle_actor_loss(
        self,
        obs_cond: torch.Tensor,
        obs_cond_flat: torch.Tensor,
        w_candidates: torch.Tensor,
        w_flat: torch.Tensor,
        B: int,
        M: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute AWMLE actor loss using Q^W scores (not Q^A!)."""
        device = obs_cond.device
        
        # Get Q^W scores for advantage computation
        with torch.no_grad():
            q1_w, q2_w = self.latent_q_network(obs_cond_flat, w_flat)
            qw_scores = torch.min(q1_w, q2_w).squeeze(-1).view(B, M)  # Use min for stability
            
            # Compute advantage with soft baseline
            baseline = self.tau * torch.logsumexp(qw_scores / self.tau, dim=1, keepdim=True) - self.tau * math.log(M)
            advantage = qw_scores - baseline
            advantage = torch.clamp(advantage, -self.advantage_clip, self.advantage_clip)
            weights = F.softmax(advantage / self.beta_latent, dim=1)
        
        # Weighted MLE loss
        log_probs = self.latent_policy.log_prob(obs_cond_flat, w_flat)
        log_probs = log_probs.view(B, M)
        nll_loss = -(weights * log_probs).sum(dim=1).mean()
        
        # Effective number of samples
        eff_num = 1.0 / (weights.pow(2).sum(dim=1) + 1e-8)
        
        return nll_loss, eff_num.mean()
    
    def _compute_sac_actor_loss(
        self,
        obs_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute SAC-style actor loss using Q^W.
        
        Uses TARGET Q-network instead of online Q to prevent Q-value extrapolation
        and reduce actor exploitation of overestimated Q-values.
        """
        # Sample from policy with reparameterization (TanhNormal already bounded)
        w, log_prob = self.latent_policy.sample(obs_cond, deterministic=False)
        
        # Use TARGET Q^W to prevent exploitation of overestimated online Q-values
        # This reduces Q-value extrapolation in high-dimensional latent space
        q1_t, q2_t = self.latent_q_network.forward_target(obs_cond, w)
        q_min = torch.min(q1_t, q2_t)
        
        # SAC actor loss: α * log π(w|s) - Q^W_target(s, w)
        alpha = self.temperature.alpha.detach()
        actor_loss = (alpha * log_prob.unsqueeze(-1) - q_min).mean()
        
        entropy = -log_prob.mean().detach()
        
        return actor_loss, entropy
    
    # ==================== Stage 2: Online SAC ====================
    
    def compute_critic_loss(
        self,
        obs_cond: torch.Tensor,
        latent_w: torch.Tensor,
        reward: torch.Tensor,
        discount_factor: torch.Tensor,
        done: torch.Tensor,
        next_obs_cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SAC critic loss for latent Q-networks.
        
        TD target: y = R + γ^τ * (1 - done) * (min Q_target(s', w') - α log π(w'|s'))
        
        Args:
            obs_cond: (B, obs_dim) observation conditioning
            latent_w: (B, pred_horizon, action_dim) sampled latents
            reward: (B,) cumulative chunk rewards
            discount_factor: (B,) γ^τ discount factors
            done: (B,) done flags
            next_obs_cond: (B, obs_dim) next observation conditioning
            
        Returns:
            Dict with critic loss and metrics
        """
        if self.latent_q_network is None:
            raise ValueError("Latent Q-network not initialized. Required for Stage 2.")
        
        # Sample next latent from policy
        with torch.no_grad():
            next_w, next_log_prob = self.latent_policy.sample(next_obs_cond, deterministic=False)
            
            # Target Q-values
            q1_target, q2_target = self.latent_q_network.forward_target(next_obs_cond, next_w)
            q_target_min = torch.min(q1_target, q2_target)
            
            # Entropy bonus in TD target
            alpha = self.temperature.alpha.detach()
            if self.backup_entropy:
                v_target = q_target_min - alpha * next_log_prob.unsqueeze(-1)
            else:
                v_target = q_target_min
            
            # SMDP TD target
            td_target = reward.unsqueeze(-1) + discount_factor.unsqueeze(-1) * (1 - done.unsqueeze(-1)) * v_target
        
        # Current Q-values
        q1, q2 = self.latent_q_network(obs_cond, latent_w)
        
        # MSE loss for both Q-networks
        q1_loss = F.mse_loss(q1, td_target)
        q2_loss = F.mse_loss(q2, td_target)
        critic_loss = q1_loss + q2_loss
        
        # Metrics
        with torch.no_grad():
            td_error = ((q1 + q2) / 2 - td_target).abs().mean()
            q_mean = (q1.mean() + q2.mean()) / 2
        
        return {
            "loss": critic_loss,
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "td_error": td_error,
            "q_mean": q_mean,
            "td_target_mean": td_target.mean(),
        }
    
    def compute_actor_loss(
        self,
        obs_cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SAC actor loss for latent policy.
        
        Loss: E[α log π(w|s) - min Q_target(s, w)]
        
        Uses TARGET Q-network instead of online Q to prevent Q-value extrapolation
        and reduce actor exploitation of overestimated Q-values in high-dimensional
        latent space.
        
        Args:
            obs_cond: (B, obs_dim) observation conditioning
            
        Returns:
            Dict with actor loss and metrics
        """
        if self.latent_q_network is None:
            raise ValueError("Latent Q-network not initialized. Required for Stage 2.")
        
        # Sample latent with reparameterization (TanhNormal already bounded)
        w, log_prob = self.latent_policy.sample(obs_cond, deterministic=False)
        
        # Use TARGET Q-values to prevent exploitation of overestimated online Q-values
        # This is critical for stability in high-dimensional latent space
        q1, q2 = self.latent_q_network.forward_target(obs_cond, w)
        q_min = torch.min(q1, q2)
        
        # SAC actor loss
        alpha = self.temperature.alpha.detach()
        actor_loss = (alpha * log_prob.unsqueeze(-1) - q_min).mean()
        
        # Metrics
        with torch.no_grad():
            # Sampled entropy (Monte Carlo estimate): -E[log π(w|s)]
            sample_entropy = -log_prob.mean()
            
            # Analytic entropy (pre-tanh Gaussian approximation)
            gaussian_entropy = self.latent_policy.entropy(obs_cond).mean()
            
            # KL divergence to prior (regularization indicator)
            kl_to_prior = self.latent_policy.kl_to_prior(obs_cond).mean()
            
            # Policy distribution statistics
            mean, log_std = self.latent_policy.forward(obs_cond)
            log_std_mean = log_std.mean()
            log_std_std = log_std.std()
            std_mean = log_std.exp().mean()
            
            # Mean (policy center) statistics
            policy_mean_abs = mean.abs().mean()
            policy_mean_std = mean.std()
            
            # Sampled latent statistics  
            latent_mean = w.mean()
            latent_std = w.std()
            latent_abs_mean = w.abs().mean()
            latent_max = w.abs().max()
        
        return {
            "loss": actor_loss,
            # Entropy metrics
            "entropy": sample_entropy,  # For backward compatibility (sample-based)
            "gaussian_entropy": gaussian_entropy,  # Analytic entropy (pre-tanh)
            # Regularization
            "kl_to_prior": kl_to_prior,
            # Q-value
            "q_policy": q_min.mean(),
            "log_prob_mean": log_prob.mean(),
            # Policy distribution stats
            "log_std_mean": log_std_mean,
            "log_std_std": log_std_std,
            "std_mean": std_mean,
            "policy_mean_abs": policy_mean_abs,
            "policy_mean_std": policy_mean_std,
            # Sampled latent stats
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "latent_abs_mean": latent_abs_mean,
            "latent_max": latent_max,
        }
    
    def compute_temperature_loss(
        self,
        obs_cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute temperature loss for auto-tuning.
        
        Loss: -α * (log π(w|s) + target_entropy)
        
        Args:
            obs_cond: (B, obs_dim) observation conditioning
            
        Returns:
            Dict with temperature loss and metrics
        """
        if not self.temperature.learnable:
            return {"loss": torch.tensor(0.0, device=obs_cond.device), "alpha": self.temperature.alpha}
        
        with torch.no_grad():
            _, log_prob = self.latent_policy.sample(obs_cond, deterministic=False)
        
        alpha = self.temperature.alpha
        temp_loss = (-alpha * (log_prob + self.target_entropy)).mean()
        
        return {
            "loss": temp_loss,
            "alpha": alpha.detach(),
        }
    
    def update_target(self):
        """Soft update target networks."""
        if self.latent_q_network is not None:
            self.latent_q_network.soft_update_target()
    
    # ==================== Diagnostics ====================
    
    @torch.no_grad()
    def diagnose_q_sensitivity(
        self,
        obs_cond: torch.Tensor,
        num_samples: int = 64,
    ) -> Dict[str, float]:
        """
        Diagnose Q-network sensitivity to latent w.
        
        For each observation, samples N latents and evaluates Q(s, w).
        If Q has low variance over w, the critic hasn't learned to distinguish
        good vs bad latents, and actor gradients will be weak.
        
        Args:
            obs_cond: (B, obs_dim) observation conditioning
            num_samples: Number of latent samples per observation
            
        Returns:
            Dict with sensitivity metrics:
            - prior_q_std: Std of Q over prior samples (per obs, averaged)
            - prior_q_range: Max - Min of Q over prior samples
            - prior_q_top1_vs_median: Top-1 Q minus median Q
            - policy_q_std: Same metrics for policy samples
            - policy_q_range, policy_q_top1_vs_median
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        N = num_samples
        
        # Expand obs for batched Q evaluation: (B, obs_dim) -> (B*N, obs_dim)
        obs_expand = obs_cond.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        
        # ===== Sample from Prior (N(0,I) squashed with tanh) =====
        prior_w_pre = torch.randn(B * N, self.pred_horizon, self.action_dim, device=device)
        prior_w = torch.tanh(prior_w_pre) * self.action_magnitude
        
        # Get Q values for prior samples
        q1_prior, q2_prior = self.latent_q_network(obs_expand, prior_w)
        q_prior = torch.min(q1_prior, q2_prior).squeeze(-1)  # (B*N,)
        q_prior = q_prior.view(B, N)
        
        # Prior statistics
        prior_q_mean = q_prior.mean(dim=1)  # (B,)
        prior_q_std = q_prior.std(dim=1)  # (B,)
        prior_q_max = q_prior.max(dim=1).values
        prior_q_min = q_prior.min(dim=1).values
        prior_q_range = prior_q_max - prior_q_min
        prior_q_median = q_prior.median(dim=1).values
        prior_q_top1_vs_median = prior_q_max - prior_q_median
        prior_q_top1_vs_mean = prior_q_max - prior_q_mean
        
        # ===== Sample from Policy =====
        # Need to sample N times per obs
        policy_w_list = []
        for _ in range(N):
            w, _ = self.latent_policy.sample(obs_cond, deterministic=False)
            policy_w_list.append(w)
        policy_w = torch.stack(policy_w_list, dim=1)  # (B, N, pred_horizon, action_dim)
        policy_w_flat = policy_w.view(B * N, self.pred_horizon, self.action_dim)
        
        # Get Q values for policy samples
        q1_policy, q2_policy = self.latent_q_network(obs_expand, policy_w_flat)
        q_policy = torch.min(q1_policy, q2_policy).squeeze(-1)  # (B*N,)
        q_policy = q_policy.view(B, N)
        
        # Policy statistics
        policy_q_mean = q_policy.mean(dim=1)
        policy_q_std = q_policy.std(dim=1)
        policy_q_max = q_policy.max(dim=1).values
        policy_q_min = q_policy.min(dim=1).values
        policy_q_range = policy_q_max - policy_q_min
        policy_q_median = q_policy.median(dim=1).values
        policy_q_top1_vs_median = policy_q_max - policy_q_median
        policy_q_top1_vs_mean = policy_q_max - policy_q_mean
        
        return {
            # Prior sampling metrics
            "prior_q_mean": prior_q_mean.mean().item(),
            "prior_q_std": prior_q_std.mean().item(),
            "prior_q_range": prior_q_range.mean().item(),
            "prior_q_top1_vs_median": prior_q_top1_vs_median.mean().item(),
            "prior_q_top1_vs_mean": prior_q_top1_vs_mean.mean().item(),
            # Policy sampling metrics
            "policy_q_mean": policy_q_mean.mean().item(),
            "policy_q_std": policy_q_std.mean().item(),
            "policy_q_range": policy_q_range.mean().item(),
            "policy_q_top1_vs_median": policy_q_top1_vs_median.mean().item(),
            "policy_q_top1_vs_mean": policy_q_top1_vs_mean.mean().item(),
            # Absolute Q values for reference
            "prior_q_abs_mean": prior_q_mean.abs().mean().item(),
            "policy_q_abs_mean": policy_q_mean.abs().mean().item(),
        }
    
    # ==================== Checkpoint Utils ====================
    
    def get_actor_parameters(self) -> list:
        """Get latent policy parameters."""
        return list(self.latent_policy.parameters())
    
    def get_critic_parameters(self) -> list:
        """Get latent Q-network parameters."""
        if self.latent_q_network is not None:
            return self.latent_q_network.get_parameters()
        return []
    
    def get_temperature_parameters(self) -> list:
        """Get temperature parameters."""
        if self.temperature.learnable:
            return [self.temperature.log_alpha]
        return []
    
    def save_checkpoint(self, path: str):
        """Save trainable components to checkpoint."""
        state = {
            "latent_policy": self.latent_policy.state_dict(),
            "temperature": self.temperature.state_dict(),
        }
        if self.latent_q_network is not None:
            state["latent_q_network"] = self.latent_q_network.state_dict()
        torch.save(state, path)
    
    def load_checkpoint(self, path: str, load_critic: bool = True):
        """Load trainable components from checkpoint."""
        state = torch.load(path, map_location=self.device)
        self.latent_policy.load_state_dict(state["latent_policy"])
        
        if "temperature" in state:
            self.temperature.load_state_dict(state["temperature"])
        
        if load_critic and self.latent_q_network is not None and "latent_q_network" in state:
            self.latent_q_network.load_state_dict(state["latent_q_network"])
