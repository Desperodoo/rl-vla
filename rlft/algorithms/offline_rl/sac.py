"""
Offline SAC Agent with Action Chunking and Multiple Regularization Methods.

Adapts Soft Actor-Critic for offline RL training (no environment interaction).
Uses the same DiagGaussianActor + EnsembleQNetwork architecture as online SAC (RLPD),
with configurable actor/critic regularization to handle offline distribution shift.

Supported Actor Regularization (actor_loss_type):
  - "sac":   Pure SAC — reparameterization Q-maximization (no data constraint)
  - "td3bc": TD3+BC — reparameterization + MSE BC to data (Q-normalized)
  - "awr":   Advantage Weighted Regression — weight log π(a_data|s) by exp(A/β)
  - "iql":   IQL-style — expectile V-function + advantage-weighted actor

Supported Critic Regularization (cql_alpha):
  - cql_alpha = 0:  Standard Bellman (ensemble min-Q provides implicit conservatism)
  - cql_alpha > 0:  CQL penalty — penalize Q for policy actions vs data actions

IQL Mode (actor_loss_type="iql"):
  - Adds a ValueNetwork V(s) trained with expectile regression
  - Critic uses V-target instead of Q-target (avoids evaluating Q on OOD actions)
  - Actor is advantage-weighted: L = -exp(A/β) * log π(a_data|s)

Architecture:
- Actor: DiagGaussianActor (MLP with tanh squashing)
- Critic: EnsembleQNetwork (N Q-networks, subsample M for min-Q)
- Temperature: LearnableTemperature (auto-tuned entropy coefficient)
- Value: ValueNetwork (MLP, IQL mode only)

References:
- SAC: Haarnoja et al., "Soft Actor-Critic", ICML 2018
- SAC-N: An et al., "Uncertainty-Based Offline RL with Diversified Q-Ensemble", NeurIPS 2021
- TD3+BC: Fujimoto & Gu, "A Minimalist Approach to Offline RL", NeurIPS 2021
- CQL: Kumar et al., "Conservative Q-Learning for Offline RL", NeurIPS 2020
- IQL: Kostrikov et al., "Offline RL with Implicit Q-Learning", ICLR 2022
- AWR: Peng et al., "Advantage-Weighted Regression", arXiv 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, Optional, Tuple

from rlft.networks import EnsembleQNetwork, DiagGaussianActor, LearnableTemperature, soft_update


class ValueNetwork(nn.Module):
    """State Value Network V(s) for IQL.
    
    Simple MLP that maps observations to a scalar value estimate.
    Uses the same architecture conventions as actor/critic (LayerNorm + Mish).
    """
    
    def __init__(self, obs_dim: int, hidden_dims: list = [256, 256, 256]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.LayerNorm(h), nn.Mish()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        final = self.net[-1]
        nn.init.orthogonal_(final.weight, gain=0.01)
        nn.init.zeros_(final.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns V(s): (B, 1)."""
        return self.net(obs)


class OfflineSACAgent(nn.Module):
    """Offline SAC Agent with Action Chunking and configurable regularization.
    
    Supports multiple offline RL regularization strategies via two orthogonal axes:
    
    Actor regularization (actor_loss_type):
      "sac"   — Pure reparameterization Q-maximization (prone to OOD exploit)
      "td3bc" — TD3+BC: Q-maximization + MSE BC loss (default, recommended)
      "awr"   — Advantage Weighted Regression (no ∂Q/∂a gradient)
      "iql"   — IQL: advantage-weighted with learned V-function
    
    Critic regularization (cql_alpha):
      0.0     — Standard Bellman (ensemble min-Q is the only conservatism)
      >0.0    — CQL penalty on critic (penalizes high Q for policy actions)
    
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
        actor_q_mode: Q aggregation for actor loss: 'min' or 'mean'
        actor_loss_type: Actor regularization method
        actor_bc_weight: BC weight for td3bc (lam = 1/(1+w))
        cql_alpha: CQL conservative penalty weight (0 = disabled)
        awr_temperature: β for AWR/IQL advantage weighting
        iql_expectile: τ for IQL expectile regression (0.5-1.0)
        action_bounds: (min, max) bounds for action clamping
        device: Device to run on
    """
    
    VALID_ACTOR_LOSS_TYPES = ("sac", "td3bc", "awr", "iql")
    
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
        # --- Regularization parameters ---
        actor_loss_type: str = "td3bc",
        actor_bc_weight: float = 2.0,
        cql_alpha: float = 0.0,
        awr_temperature: float = 1.0,
        iql_expectile: float = 0.7,
        action_bounds: Optional[tuple] = None,
        device: str = "cuda",
    ):
        super().__init__()
        
        assert actor_loss_type in self.VALID_ACTOR_LOSS_TYPES, \
            f"actor_loss_type must be one of {self.VALID_ACTOR_LOSS_TYPES}, got '{actor_loss_type}'"
        
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
        self.actor_loss_type = actor_loss_type
        self.actor_bc_weight = actor_bc_weight
        self.cql_alpha = cql_alpha
        self.awr_temperature = awr_temperature
        self.iql_expectile = iql_expectile
        self.action_bounds = action_bounds
        self.device = device
        
        if target_entropy is None:
            self.target_entropy = -float(action_dim * act_horizon)
        else:
            self.target_entropy = target_entropy
        
        # Actor: Gaussian policy with tanh squashing
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
        
        # IQL: V-network and its target (only created when needed)
        if actor_loss_type == "iql":
            self.value_net = ValueNetwork(obs_dim, hidden_dims)
            self.value_target = copy.deepcopy(self.value_net)
            for param in self.value_target.parameters():
                param.requires_grad = False
    
    @property
    def uses_reparameterization(self) -> bool:
        """Whether actor loss uses reparameterization gradient (∂Q/∂a → ∂a/∂θ).
        
        sac/td3bc: Yes — need to freeze critic during actor backward.
        awr/iql: No — actor gradient comes from weighted log-prob, not Q.
        """
        return self.actor_loss_type in ("sac", "td3bc")
    
    @property
    def has_value_network(self) -> bool:
        """Whether agent has a V-network (IQL mode only)."""
        return self.actor_loss_type == "iql"
    
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
        """Compute combined loss. Primarily used for single-optimizer mode.
        
        For the training loop with separate optimizers, use the individual
        _compute_*_loss methods directly instead.
        """
        if actions_for_q is None:
            actions_for_q = actions[:, :self.act_horizon, :]
        
        obs_cond = self._flatten_obs(obs_features)
        next_obs_cond = self._flatten_obs(next_obs_features)
        
        # Value loss (IQL only)
        value_loss = torch.tensor(0.0, device=obs_cond.device)
        value_metrics = {}
        if self.has_value_network:
            value_loss, value_metrics = self._compute_value_loss(obs_cond, actions_for_q)
        
        # Critic loss
        critic_loss, critic_metrics = self._compute_critic_loss(
            obs_cond, actions_for_q, next_obs_cond, rewards, dones,
            cumulative_reward, chunk_done, discount_factor,
        )
        
        # Actor loss
        if self.uses_reparameterization:
            for p in self.critic.parameters():
                p.requires_grad_(False)
        
        actor_loss, actor_metrics = self._compute_actor_loss(obs_cond, data_actions=actions_for_q)
        
        if self.uses_reparameterization:
            for p in self.critic.parameters():
                p.requires_grad_(True)
        
        # Temperature loss
        temp_loss, temp_metrics = self._compute_temperature_loss(obs_cond)
        
        total_loss = value_loss + critic_loss + actor_loss + temp_loss
        
        result = {
            "loss": total_loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "temperature_loss": temp_loss,
            "value_loss": value_loss,
        }
        result.update(value_metrics)
        result.update(critic_metrics)
        result.update(actor_metrics)
        result.update(temp_metrics)
        
        return result
    
    # =========================================================================
    # Critic Loss
    # =========================================================================
    
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
        """Compute critic loss using SMDP Bellman equation with ensemble Q.
        
        Supports two modes:
        - Standard: Q-target from next Q-values (SAC/TD3+BC/AWR)
        - IQL: Q-target from V-target (avoids OOD next-action evaluation)
        
        Optionally adds CQL penalty when cql_alpha > 0.
        """
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
            if self.actor_loss_type == "iql":
                # IQL: use V-target instead of Q-target
                # This is the key insight — no need to evaluate Q on next actions,
                # completely avoiding the OOD action problem.
                next_v = self.value_target(next_obs_cond)
                td_target = scaled_rewards + (1 - d) * gamma_tau * next_v
            else:
                # Standard: sample next actions from current policy
                next_action, next_log_prob = self.actor.get_action(next_obs_cond, deterministic=False)
                target_q = self.critic_target.get_min_q(next_action, next_obs_cond, random_subset=True)
                
                if self.backup_entropy:
                    alpha = self.temperature.alpha
                    target_q = target_q - alpha * next_log_prob.unsqueeze(-1)
                
                td_target = scaled_rewards + (1 - d) * gamma_tau * target_q
            
            if self.q_target_clip is not None:
                td_target = torch.clamp(td_target, -self.q_target_clip, self.q_target_clip)
        
        # Ensemble Q-values and MSE loss (×0.5 matches SB3 convention)
        q_values = self.critic(actions_for_q, obs_cond)  # (num_qs, B, 1)
        
        critic_loss = 0.0
        for q in q_values:
            critic_loss = critic_loss + F.mse_loss(q, td_target)
        critic_loss = 0.5 * critic_loss
        
        metrics = {
            "q_mean": q_values.mean().item(),
            "q_std": q_values.std(dim=0).mean().item(),
            "td_target_mean": td_target.mean().item(),
        }
        
        # CQL conservative penalty (optional, orthogonal to actor regularization)
        if self.cql_alpha > 0.0:
            cql_penalty, cql_metrics = self._compute_cql_penalty(obs_cond, actions_for_q)
            critic_loss = critic_loss + self.cql_alpha * cql_penalty
            metrics.update(cql_metrics)
        
        return critic_loss, metrics
    
    def _compute_cql_penalty(
        self,
        obs_cond: torch.Tensor,
        data_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """CQL conservative penalty: E_π[Q(s,a)] - E_D[Q(s,a)].
        
        Penalizes the critic for assigning high Q-values to policy-sampled actions
        relative to data actions. This makes Q conservative for OOD actions.
        
        Uses CQL(ρ) variant with policy + random actions for the logsumexp.
        """
        B = obs_cond.shape[0]
        
        # Q for data actions (should NOT be penalized)
        q_data = self.critic(data_actions, obs_cond)  # (num_qs, B, 1)
        
        # Q for policy-sampled actions (SHOULD be penalized)
        with torch.no_grad():
            policy_actions, policy_log_prob = self.actor.get_action(obs_cond, deterministic=False)
        q_policy = self.critic(policy_actions, obs_cond)  # (num_qs, B, 1)
        
        # Q for random actions (uniform [-1, 1])
        random_actions = torch.rand_like(data_actions) * 2 - 1
        q_random = self.critic(random_actions, obs_cond)  # (num_qs, B, 1)
        
        # CQL penalty: logsumexp(Q_policy, Q_random) - Q_data
        # Average over ensemble, then over batch
        q_cat = torch.cat([q_policy, q_random], dim=2)  # (num_qs, B, 2)
        cql_logsumexp = torch.logsumexp(q_cat, dim=2, keepdim=True)  # (num_qs, B, 1)
        
        cql_penalty = (cql_logsumexp - q_data).mean()
        
        metrics = {
            "cql_penalty": cql_penalty.item(),
            "cql_q_policy": q_policy.mean().item(),
            "cql_q_random": q_random.mean().item(),
            "cql_q_data": q_data.mean().item(),
        }
        
        return cql_penalty, metrics
    
    # =========================================================================
    # Value Loss (IQL only)
    # =========================================================================
    
    def _compute_value_loss(
        self,
        obs_cond: torch.Tensor,
        data_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """IQL expectile regression for V-network.
        
        L_V = E_D[ |τ - 1(Q_target(s,a) - V(s) < 0)| * (Q_target(s,a) - V(s))² ]
        
        τ > 0.5 makes V an optimistic estimate, effectively performing
        in-sample policy improvement without querying OOD actions.
        """
        assert self.has_value_network, "value_loss requires IQL mode"
        
        with torch.no_grad():
            # Use target Q on data actions (no OOD evaluation)
            q_target = self.critic_target.get_min_q(data_actions, obs_cond, random_subset=False)
            # q_target: (B, 1)
        
        v = self.value_net(obs_cond)  # (B, 1)
        diff = q_target - v
        
        # Asymmetric loss: τ weight for positive diff, (1-τ) for negative
        weight = torch.where(diff > 0, self.iql_expectile, 1.0 - self.iql_expectile)
        value_loss = (weight * diff.pow(2)).mean()
        
        metrics = {
            "v_mean": v.mean().item(),
            "v_loss": value_loss.item(),
            "iql_advantage_mean": diff.mean().item(),
        }
        
        return value_loss, metrics
    
    # =========================================================================
    # Actor Loss (multiple modes)
    # =========================================================================
    
    def _compute_actor_loss(
        self,
        obs_cond: torch.Tensor,
        data_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute actor loss based on actor_loss_type.
        
        Dispatches to the appropriate actor loss computation method.
        """
        if self.actor_loss_type == "sac":
            return self._actor_loss_sac(obs_cond)
        elif self.actor_loss_type == "td3bc":
            return self._actor_loss_td3bc(obs_cond, data_actions)
        elif self.actor_loss_type == "awr":
            return self._actor_loss_awr(obs_cond, data_actions)
        elif self.actor_loss_type == "iql":
            return self._actor_loss_iql(obs_cond, data_actions)
        else:
            raise ValueError(f"Unknown actor_loss_type: {self.actor_loss_type}")
    
    def _actor_loss_sac(
        self,
        obs_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Pure SAC: maximize Q via reparameterization + entropy bonus."""
        action, log_prob = self.actor.get_action(obs_cond, deterministic=False)
        
        if self.actor_q_mode == "min":
            q_value = self.critic.get_min_q(action, obs_cond, random_subset=False)
        else:
            q_value = self.critic.get_mean_q(action, obs_cond)
        
        alpha = self.temperature.alpha.detach()
        q_flat = q_value.squeeze(-1)
        
        actor_loss = (alpha * log_prob - q_flat).mean()
        
        metrics = {
            "actor_entropy": -log_prob.mean().item(),
            "actor_q": q_flat.mean().item(),
            "bc_loss": 0.0,
            "actor_lam": 1.0,
        }
        return actor_loss, metrics
    
    def _actor_loss_td3bc(
        self,
        obs_cond: torch.Tensor,
        data_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """TD3+BC: reparameterization Q-maximization + MSE BC regularization.
        
        actor_loss = -lam * Q_normalized + alpha * log_prob + (1-lam) * MSE(a, a_data)
        where lam = 1 / (1 + bc_weight) and Q_normalized = Q / max(mean|Q|, 1)
        """
        action, log_prob = self.actor.get_action(obs_cond, deterministic=False)
        
        if self.actor_q_mode == "min":
            q_value = self.critic.get_min_q(action, obs_cond, random_subset=False)
        else:
            q_value = self.critic.get_mean_q(action, obs_cond)
        
        alpha = self.temperature.alpha.detach()
        q_flat = q_value.squeeze(-1)
        
        if self.actor_bc_weight > 0.0 and data_actions is not None:
            lam = 1.0 / (1.0 + self.actor_bc_weight)
            q_normalizer = torch.clamp(q_flat.abs().mean().detach(), min=1.0)
            q_normalized = q_flat / q_normalizer
            
            action_flat = action.reshape(action.shape[0], -1)
            data_flat = data_actions.reshape(data_actions.shape[0], -1)
            bc_loss = F.mse_loss(action_flat, data_flat)
            
            rl_loss = (alpha * log_prob - lam * q_normalized).mean()
            actor_loss = rl_loss + (1.0 - lam) * bc_loss
        else:
            actor_loss = (alpha * log_prob - q_flat).mean()
            bc_loss = torch.tensor(0.0)
            lam = 1.0
        
        metrics = {
            "actor_entropy": -log_prob.mean().item(),
            "actor_q": q_flat.mean().item(),
            "bc_loss": bc_loss.item() if isinstance(bc_loss, torch.Tensor) else bc_loss,
            "actor_lam": lam,
        }
        return actor_loss, metrics
    
    def _actor_loss_awr(
        self,
        obs_cond: torch.Tensor,
        data_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """AWR: Advantage Weighted Regression.
        
        No reparameterization gradient — instead, weight the log-likelihood
        of data actions by exponentiated advantage:
          L = -E_D[ exp(A(s,a)/β) * log π(a|s) ]
        where A(s,a) = Q(s,a) - V(s), and V(s) ≈ mean_ensemble Q(s, a_data).
        
        This avoids querying Q on OOD policy actions entirely.
        """
        assert data_actions is not None, "AWR requires data_actions"
        
        with torch.no_grad():
            # Q-value of data actions
            q_all = self.critic(data_actions, obs_cond)  # (num_qs, B, 1)
            q_min = q_all.min(dim=0).values.squeeze(-1)  # (B,)
            
            # Baseline V ≈ ensemble mean Q on data actions
            v_baseline = q_all.mean(dim=0).squeeze(-1)  # (B,)
            
            # Advantage = Q_min - V_baseline (conservative Q, average baseline)
            advantage = q_min - v_baseline
            
            # Normalize advantage for stability
            adv_std = advantage.std().clamp(min=1e-6)
            advantage_normalized = (advantage - advantage.mean()) / adv_std
            
            # Exponentiated weights, clipped for stability
            weights = torch.exp(advantage_normalized / self.awr_temperature).clamp(max=100.0)
        
        # Log probability of data actions under current policy
        log_prob, _ = self.actor.evaluate_actions(obs_cond, data_actions)  # (B,)
        
        # Weighted negative log-likelihood
        actor_loss = -(weights * log_prob).mean()
        
        # Optional entropy bonus
        alpha = self.temperature.alpha.detach()
        _, sample_log_prob = self.actor.get_action(obs_cond, deterministic=False)
        actor_loss = actor_loss + alpha * sample_log_prob.mean()
        
        metrics = {
            "actor_entropy": -sample_log_prob.mean().item(),
            "actor_q": q_min.mean().item(),
            "bc_loss": 0.0,
            "actor_lam": 0.0,
            "awr_advantage_mean": advantage.mean().item(),
            "awr_weights_mean": weights.mean().item(),
            "awr_weights_max": weights.max().item(),
        }
        return actor_loss, metrics
    
    def _actor_loss_iql(
        self,
        obs_cond: torch.Tensor,
        data_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """IQL actor: advantage-weighted using learned V-function.
        
        L = -E_D[ exp(clip(A/β, -C, C)) * log π(a|s) ]
        where A(s,a) = Q_target(s,a) - V(s) from the IQL value network.
        
        Uses target Q to compute advantages (more stable than online Q).
        """
        assert data_actions is not None, "IQL requires data_actions"
        assert self.has_value_network, "IQL requires value network"
        
        with torch.no_grad():
            q_target = self.critic_target.get_min_q(data_actions, obs_cond, random_subset=False)
            q_flat = q_target.squeeze(-1)  # (B,)
            
            v = self.value_net(obs_cond).squeeze(-1)  # (B,)
            advantage = q_flat - v
            
            # Clip advantage to prevent extreme weights
            advantage_clipped = torch.clamp(advantage / self.awr_temperature, -8.0, 8.0)
            weights = torch.exp(advantage_clipped)
            # Normalize weights to stabilize
            weights = weights / weights.mean().clamp(min=1e-6)
        
        # Log probability of data actions under current policy
        log_prob, _ = self.actor.evaluate_actions(obs_cond, data_actions)  # (B,)
        
        # Weighted negative log-likelihood
        actor_loss = -(weights * log_prob).mean()
        
        metrics = {
            "actor_entropy": 0.0,  # Computed from data actions, not meaningful
            "actor_q": q_flat.mean().item(),
            "bc_loss": 0.0,
            "actor_lam": 0.0,
            "iql_advantage_mean": advantage.mean().item(),
            "iql_weights_mean": weights.mean().item(),
            "iql_weights_max": weights.max().item(),
            "iql_v_mean": v.mean().item(),
        }
        return actor_loss, metrics
    
    # =========================================================================
    # Temperature Loss
    # =========================================================================
    
    def _compute_temperature_loss(
        self,
        obs_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute temperature loss for entropy-constrained optimization.
        
        For AWR/IQL modes, temperature is still useful for the entropy bonus
        but may be less critical than for reparameterization modes.
        """
        with torch.no_grad():
            _, log_prob = self.actor.get_action(obs_cond, deterministic=False)
        
        log_alpha = self.temperature.log_alpha
        temp_loss = -(log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        metrics = {
            "temperature": self.temperature.alpha.item(),
            "entropy": -log_prob.mean().item(),
            "target_entropy": self.target_entropy,
        }
        
        return temp_loss, metrics
    
    # =========================================================================
    # Target updates
    # =========================================================================
    
    def update_target(self):
        """Soft update target networks (critic + value for IQL)."""
        soft_update(self.critic_target, self.critic, self.tau)
        if self.has_value_network:
            soft_update(self.value_target, self.value_net, self.tau)
    
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
