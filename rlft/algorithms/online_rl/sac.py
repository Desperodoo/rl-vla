"""
SAC Agent with Action Chunking for RLPD.

Implements Soft Actor-Critic with:
- Ensemble Q-networks (num_qs > 2 support)
- Action chunking (SMDP formulation)
- Learnable temperature parameter
- Online + offline mixed training (RLPD)

Reference:
- SAC: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
  Reinforcement Learning with a Stochastic Actor", ICML 2018
- RLPD: Ball et al., "Efficient Online Reinforcement Learning with Offline Data", 
  ICML 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, Optional, Tuple

from rlft.networks import EnsembleQNetwork, DiagGaussianActor, LearnableTemperature, soft_update


class SACAgent(nn.Module):
    """
    Soft Actor-Critic Agent with Action Chunking.
    
    Extends standard SAC to support:
    - Action chunk output (action_horizon steps)
    - Ensemble Q-networks with subsample + min
    - SMDP Bellman equation for chunk-level learning
    - RGB observation support via external visual encoder
    
    Args:
        obs_dim: Dimension of observation features (after visual encoding if any)
        action_dim: Dimension of action space
        action_horizon: Length of action chunks (SMDP horizon)
        hidden_dims: Hidden layer dimensions for actor and critic
        num_qs: Number of Q-networks in ensemble (default: 10)
        num_min_qs: Number of Q-networks for subsample + min (default: 2)
        gamma: Discount factor (default: 0.99)
        tau: Soft update coefficient (default: 0.005)
        init_temperature: Initial entropy temperature (default: 1.0)
        target_entropy: Target entropy for temperature tuning
        backup_entropy: Whether to use entropy backup in Q-target (default: True)
        reward_scale: Scale factor for rewards (default: 1.0)
        device: Device to run on (default: "cuda")
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_horizon: int = 8,
        hidden_dims: list = [256, 256, 256],
        num_qs: int = 10,
        num_min_qs: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_temperature: float = 1.0,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = False,
        reward_scale: float = 1.0,
        action_bounds: Optional[tuple] = None,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.gamma = gamma
        self.tau = tau
        self.backup_entropy = backup_entropy
        self.reward_scale = reward_scale
        self.action_bounds = action_bounds
        self.device = device
        
        if target_entropy is None:
            self.target_entropy = -float(action_dim * action_horizon)
        else:
            self.target_entropy = target_entropy
        
        self.actor = DiagGaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            hidden_dims=hidden_dims,
            log_std_range=(-5.0, 2.0),
            state_dependent_std=True,
        )
        
        self.critic = EnsembleQNetwork(
            action_dim=action_dim,
            obs_dim=obs_dim,
            action_horizon=action_horizon,
            hidden_dims=hidden_dims,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
        )
        
        self.critic_target = copy.deepcopy(self.critic)
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        self.temperature = LearnableTemperature(init_temperature)
    
    def get_action(
        self,
        obs_features: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample action chunk from the policy."""
        return self.actor.get_action(obs_features, deterministic=deterministic)
    
    @torch.no_grad()
    def select_action(
        self,
        obs_features: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Select action for environment interaction (no gradient)."""
        squeeze = False
        if obs_features.dim() == 1:
            obs_features = obs_features.unsqueeze(0)
            squeeze = True
        
        action, _ = self.get_action(obs_features, deterministic=deterministic)
        
        # Note: DiagGaussianActor uses tanh squashing, output is already in [-1, 1]
        # Additional clamp if action_bounds is explicitly set
        if self.action_bounds is not None:
            action = torch.clamp(action, self.action_bounds[0], self.action_bounds[1])
        
        if squeeze:
            action = action.squeeze(0)
        
        return action
    
    def compute_actor_loss(
        self,
        obs_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute actor loss (policy gradient with entropy regularization)."""
        action, log_prob = self.get_action(obs_features, deterministic=False)
        
        q_value = self.critic.get_mean_q(action, obs_features)
        
        alpha = self.temperature.alpha.detach()
        actor_loss = (alpha * log_prob - q_value.squeeze(-1)).mean()
        
        metrics = {
            "actor_loss": actor_loss.item(),
            "actor_entropy": -log_prob.mean().item(),
            "actor_q": q_value.mean().item(),
        }
        
        return actor_loss, metrics
    
    def compute_critic_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
        next_obs_features: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        cumulative_reward: Optional[torch.Tensor] = None,
        chunk_done: Optional[torch.Tensor] = None,
        discount_factor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute critic loss (SMDP Bellman error for ensemble)."""
        if cumulative_reward is not None:
            r = cumulative_reward
            d = chunk_done if chunk_done is not None else dones
            gamma_tau = discount_factor if discount_factor is not None else torch.full_like(r, self.gamma)
        else:
            r = rewards
            d = dones
            gamma_tau = torch.full_like(r, self.gamma)
        
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        if d.dim() == 1:
            d = d.unsqueeze(-1)
        if gamma_tau.dim() == 1:
            gamma_tau = gamma_tau.unsqueeze(-1)
        
        scaled_rewards = r * self.reward_scale
        
        with torch.no_grad():
            next_action, next_log_prob = self.get_action(next_obs_features, deterministic=False)
            target_q = self.critic_target.get_min_q(next_action, next_obs_features, random_subset=True)
            
            if self.backup_entropy:
                alpha = self.temperature.alpha
                target_q = target_q - alpha * next_log_prob.unsqueeze(-1)
            
            td_target = scaled_rewards + (1 - d) * gamma_tau * target_q
        
        q_values = self.critic(actions, obs_features)
        
        critic_loss = 0.0
        for q in q_values:
            critic_loss = critic_loss + F.mse_loss(q, td_target)
        
        metrics = {
            "critic_loss": critic_loss.item(),
            "q_mean": q_values.mean().item(),
            "q_std": q_values.std().item(),
            "td_target_mean": td_target.mean().item(),
        }
        
        return critic_loss, metrics
    
    def compute_temperature_loss(
        self,
        obs_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute temperature loss for entropy-constrained optimization."""
        with torch.no_grad():
            _, log_prob = self.get_action(obs_features, deterministic=False)
        
        alpha = self.temperature.alpha
        temp_loss = -alpha * (log_prob + self.target_entropy).mean()
        
        metrics = {
            "temperature_loss": temp_loss.item(),
            "temperature": alpha.item(),
            "entropy": -log_prob.mean().item(),
            "target_entropy": self.target_entropy,
        }
        
        return temp_loss, metrics
    
    def compute_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
        next_obs_features: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        cumulative_reward: Optional[torch.Tensor] = None,
        chunk_done: Optional[torch.Tensor] = None,
        discount_factor: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses for training."""
        critic_loss, critic_metrics = self.compute_critic_loss(
            obs_features, actions, next_obs_features, rewards, dones,
            cumulative_reward, chunk_done, discount_factor
        )
        
        actor_loss, actor_metrics = self.compute_actor_loss(obs_features)
        
        temp_loss, temp_metrics = self.compute_temperature_loss(obs_features)
        
        result = {
            "loss": actor_loss + critic_loss + temp_loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "temperature_loss": temp_loss,
        }
        result.update(critic_metrics)
        result.update(actor_metrics)
        result.update(temp_metrics)
        
        return result
    
    def update_target(self):
        """Soft update target critic network."""
        soft_update(self.critic_target, self.critic, self.tau)
