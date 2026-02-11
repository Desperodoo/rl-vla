"""
Offline SAC Agent with Action Chunking.

Adapts Soft Actor-Critic for offline RL training (no environment interaction).
Uses the same DiagGaussianActor + EnsembleQNetwork architecture as online SAC (RLPD),
but with a unified compute_loss() interface compatible with the offline training pipeline.

Key differences from online SAC:
- compute_loss() handles all three losses (actor, critic, temperature) in a single call
- Gradient separation handled internally via requires_grad toggling
  (works correctly with a single optimizer)
- Matches offline RL interface (actions_for_q, cumulative_reward, SMDP formulation)
- get_action() returns action chunks matching flow-based agent convention

Why Offline SAC?
- Serves as a strong baseline for offline RL (comparable to SAC-N / RLPD offline-only)
- EnsembleQ with subsample+min provides implicit conservatism (no CQL penalty needed)
- Gaussian policy is simpler/faster than flow-based policies
- Can be compared against flow-based offline RL (AWCP, CPQL, AW-SC)

Architecture:
- Actor: DiagGaussianActor (MLP with tanh squashing)
- Critic: EnsembleQNetwork (N Q-networks, subsample M for min-Q)
- Temperature: LearnableTemperature (auto-tuned entropy coefficient)

References:
- SAC: Haarnoja et al., "Soft Actor-Critic", ICML 2018
- SAC-N: An et al., "Uncertainty-Based Offline RL with Diversified Q-Ensemble", NeurIPS 2021
- RLPD: Ball et al., "Efficient Online RL with Offline Data", ICML 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, Optional, Tuple

from rlft.networks import EnsembleQNetwork, DiagGaussianActor, LearnableTemperature, soft_update


class OfflineSACAgent(nn.Module):
    """Offline SAC Agent with Action Chunking for batch offline RL training.
    
    Combines DiagGaussianActor (tanh-squashed Gaussian) with EnsembleQNetwork
    and learnable temperature. Designed for single-optimizer training where
    gradient separation between actor/critic/temperature is handled internally.
    
    The compute_loss() method:
    1. Computes critic loss (TD error) — gradients flow to critic only
    2. Temporarily disables critic gradients, computes actor loss 
       (max entropy RL objective) — gradients flow to actor only via 
       reparameterization through the frozen critic
    3. Computes temperature loss — gradients flow to temperature only
    
    This approach correctly implements SAC's separate optimization objectives
    within a single backward() call.
    
    Args:
        obs_dim: Dimension of observation features (after visual encoding)
        action_dim: Dimension of action space
        obs_horizon: Number of observation frames
        pred_horizon: Prediction horizon (for evaluation output shape)
        act_horizon: Action horizon (for Q-network and actor output)
        hidden_dims: Hidden layer dimensions for actor and critic
        num_qs: Number of Q-networks in ensemble
        num_min_qs: Number of Q-networks for subsample + min
        gamma: Discount factor
        tau: Soft update coefficient for target network
        init_temperature: Initial entropy temperature
        target_entropy: Target entropy (default: -action_dim * act_horizon)
        backup_entropy: Whether to subtract entropy in Q-target
        reward_scale: Scale factor for rewards
        q_target_clip: Clip range for Q-target values
        actor_q_mode: Q aggregation for actor loss: 'min' (SB3 default) or 'mean'
        action_bounds: (min, max) bounds for action clamping
        device: Device to run on
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 8,
        act_horizon: int = 8,
        hidden_dims: list = [256, 256, 256],
        num_qs: int = 10,
        num_min_qs: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_temperature: float = 1.0,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = False,
        reward_scale: float = 1.0,
        q_target_clip: float = 100.0,
        actor_q_mode: str = "min",
        action_bounds: Optional[tuple] = None,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.gamma = gamma
        self.tau = tau
        self.backup_entropy = backup_entropy
        self.reward_scale = reward_scale
        self.q_target_clip = q_target_clip
        self.actor_q_mode = actor_q_mode
        self.action_bounds = action_bounds
        self.device = device
        
        if target_entropy is None:
            self.target_entropy = -float(action_dim * act_horizon)
        else:
            self.target_entropy = target_entropy
        
        # Actor: Gaussian policy with tanh squashing
        # Uses act_horizon as output length (matches Q-network input)
        self.actor = DiagGaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_horizon=act_horizon,
            hidden_dims=hidden_dims,
            log_std_range=(-20.0, 2.0),
            state_dependent_std=True,
        )
        
        # Critic: Ensemble Q-networks with subsample + min
        self.critic = EnsembleQNetwork(
            action_dim=action_dim,
            obs_dim=obs_dim,
            action_horizon=act_horizon,
            hidden_dims=hidden_dims,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
        )
        
        # Target critic (no gradients)
        self.critic_target = copy.deepcopy(self.critic)
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # Learnable temperature
        self.temperature = LearnableTemperature(init_temperature)
    
    def _flatten_obs(self, obs_features: torch.Tensor) -> torch.Tensor:
        """Flatten (B, obs_horizon, feat_dim) → (B, obs_horizon*feat_dim)."""
        if obs_features.dim() == 3:
            return obs_features.reshape(obs_features.shape[0], -1)
        return obs_features
    
    def compute_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs_features: torch.Tensor,
        dones: torch.Tensor,
        actions_for_q: Optional[torch.Tensor] = None,
        cumulative_reward: Optional[torch.Tensor] = None,
        chunk_done: Optional[torch.Tensor] = None,
        discount_factor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined actor + critic + temperature loss.
        
        Handles gradient separation internally so the total loss can be used
        with a single optimizer. The trick: temporarily disable critic gradients
        during actor loss computation so no spurious gradients leak to the critic.
        
        Args:
            obs_features: (B, obs_horizon, feat) or (B, feat) observation features
            actions: (B, pred_horizon, action_dim) full action sequence
            rewards: (B,) or (B, 1) step rewards
            next_obs_features: (B, obs_horizon, feat) next observation features
            dones: (B,) or (B, 1) done flags
            actions_for_q: (B, act_horizon, action_dim) sliced actions for Q-input
            cumulative_reward: (B,) SMDP cumulative reward (optional)
            chunk_done: (B,) SMDP chunk done flag (optional)
            discount_factor: (B,) SMDP discount factor (optional)
        
        Returns:
            Dict with 'loss' (total), 'actor_loss', 'critic_loss', 
            'temperature_loss', and diagnostic metrics
        """
        if actions_for_q is None:
            actions_for_q = actions[:, :self.act_horizon, :]
        
        obs_cond = self._flatten_obs(obs_features)
        next_obs_cond = self._flatten_obs(next_obs_features)
        
        # =====================================================================
        # Step 1: Critic loss (gradients to critic only)
        # =====================================================================
        critic_loss, critic_metrics = self._compute_critic_loss(
            obs_cond, actions_for_q, next_obs_cond, rewards, dones,
            cumulative_reward, chunk_done, discount_factor,
        )
        
        # =====================================================================
        # Step 2: Actor loss (gradients to actor only)
        # Temporarily disable critic gradients so reparameterization gradient
        # ∂Q/∂a · ∂a/∂θ_actor flows to actor but NOT to critic weights.
        # =====================================================================
        for p in self.critic.parameters():
            p.requires_grad_(False)
        
        actor_loss, actor_metrics = self._compute_actor_loss(obs_cond)
        
        for p in self.critic.parameters():
            p.requires_grad_(True)
        
        # =====================================================================
        # Step 3: Temperature loss (gradients to temperature only)
        # =====================================================================
        temp_loss, temp_metrics = self._compute_temperature_loss(obs_cond)
        
        # =====================================================================
        # Combine: single backward() distributes gradients correctly
        # =====================================================================
        total_loss = critic_loss + actor_loss + temp_loss
        
        result = {
            "loss": total_loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "temperature_loss": temp_loss,
        }
        result.update(critic_metrics)
        result.update(actor_metrics)
        result.update(temp_metrics)
        
        return result
    
    def _compute_critic_loss(
        self,
        obs_cond: torch.Tensor,
        actions_for_q: torch.Tensor,
        next_obs_cond: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        cumulative_reward: Optional[torch.Tensor] = None,
        chunk_done: Optional[torch.Tensor] = None,
        discount_factor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute critic loss using SMDP Bellman equation with ensemble Q."""
        # Use SMDP formulation if available
        if cumulative_reward is not None:
            r = cumulative_reward
            d = chunk_done if chunk_done is not None else dones
            gamma_tau = discount_factor if discount_factor is not None else torch.full_like(r, self.gamma)
        else:
            r = rewards
            d = dones
            gamma_tau = torch.full_like(r, self.gamma)
        
        if r.dim() == 1: r = r.unsqueeze(-1)
        if d.dim() == 1: d = d.unsqueeze(-1)
        if gamma_tau.dim() == 1: gamma_tau = gamma_tau.unsqueeze(-1)
        
        scaled_rewards = r * self.reward_scale
        
        with torch.no_grad():
            # Sample next actions from current policy
            next_action, next_log_prob = self.actor.get_action(next_obs_cond, deterministic=False)
            
            # Compute target Q using subsample + min from target ensemble
            target_q = self.critic_target.get_min_q(next_action, next_obs_cond, random_subset=True)
            
            if self.backup_entropy:
                alpha = self.temperature.alpha
                target_q = target_q - alpha * next_log_prob.unsqueeze(-1)
            
            td_target = scaled_rewards + (1 - d) * gamma_tau * target_q
            
            if self.q_target_clip is not None:
                td_target = torch.clamp(td_target, -self.q_target_clip, self.q_target_clip)
        
        # Compute ensemble Q-values and MSE loss for each
        q_values = self.critic(actions_for_q, obs_cond)  # (num_qs, B, 1)
        
        critic_loss = 0.0
        for q in q_values:
            critic_loss = critic_loss + F.mse_loss(q, td_target)
        
        metrics = {
            "q_mean": q_values.mean().item(),
            "q_std": q_values.std(dim=0).mean().item(),
            "td_target_mean": td_target.mean().item(),
        }
        
        return critic_loss, metrics
    
    def _compute_actor_loss(
        self,
        obs_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute actor loss: maximize Q - alpha * log_prob."""
        action, log_prob = self.actor.get_action(obs_cond, deterministic=False)
        
        # Q value of sampled actions (gradient flows through action to actor)
        if self.actor_q_mode == "min":
            q_value = self.critic.get_min_q(action, obs_cond, random_subset=False)
        else:
            q_value = self.critic.get_mean_q(action, obs_cond)
        
        alpha = self.temperature.alpha.detach()
        actor_loss = (alpha * log_prob - q_value.squeeze(-1)).mean()
        
        metrics = {
            "actor_entropy": -log_prob.mean().item(),
            "actor_q": q_value.mean().item(),
        }
        
        return actor_loss, metrics
    
    def _compute_temperature_loss(
        self,
        obs_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute temperature loss for entropy-constrained optimization.
        
        Optimizes log(alpha) instead of alpha directly, matching SB3 convention.
        This is more stable as discussed in:
        https://github.com/rail-berkeley/softlearning/issues/37
        """
        with torch.no_grad():
            _, log_prob = self.actor.get_action(obs_cond, deterministic=False)
        
        # Optimize log_alpha (not alpha) for stability — matches SB3
        log_alpha = self.temperature.log_alpha
        temp_loss = -(log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        metrics = {
            "temperature": self.temperature.alpha.item(),
            "entropy": -log_prob.mean().item(),
            "target_entropy": self.target_entropy,
        }
        
        return temp_loss, metrics
    
    def update_target(self):
        """Soft update target critic network."""
        soft_update(self.critic_target, self.critic, self.tau)
    
    @torch.no_grad()
    def get_action(self, obs_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get action for evaluation.
        
        Returns action chunk compatible with AgentWrapper slicing convention.
        If pred_horizon > act_horizon, pads with zeros to match flow-based agents.
        
        Args:
            obs_features: (B, obs_dim) or (B, obs_horizon, feat_dim) features
            
        Returns:
            actions: (B, max(pred_horizon, act_horizon), action_dim)
        """
        self.actor.eval()
        
        obs_cond = self._flatten_obs(obs_features)
        action, _ = self.actor.get_action(obs_cond, deterministic=True)
        # action: (B, act_horizon, action_dim)
        
        # Pad to pred_horizon if needed (for AgentWrapper compatibility)
        if self.pred_horizon > self.act_horizon:
            B = action.shape[0]
            pad = torch.zeros(
                B, self.pred_horizon - self.act_horizon, self.action_dim,
                device=action.device,
            )
            action = torch.cat([action, pad], dim=1)
        
        if self.action_bounds is not None:
            action = torch.clamp(action, self.action_bounds[0], self.action_bounds[1])
        
        self.actor.train()
        return action
