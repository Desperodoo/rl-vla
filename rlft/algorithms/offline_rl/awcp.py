"""
Advantage-Weighted Consistency Policy (AWCP) Agent

Combines Consistency Flow Matching with AWAC-style Q-weighted BC for stable offline RL.
Instead of maximizing Q directly (which causes distribution shift), uses Q to weight BC samples.

Key Algorithm Design:
=====================
AWCP uses Q-values to reweight BC samples rather than directly maximizing Q:
- CPQL: policy_loss = bc_loss + alpha * (-Q(s, π(s)))  [direct Q maximization]
- AWCP: policy_loss = weighted_bc_loss where weights = exp(β * advantage)  [Q-weighted BC]

This approach is more stable in offline settings because:
1. Policy stays close to data distribution (no OOD action generation for Q)
2. Q is only used to distinguish "better vs worse" samples in the dataset
3. No gradient from Q to policy, avoiding Q-value explosion

The total policy loss is:
    policy_loss = bc_weight * Σ w_i * flow_loss_i + consistency_weight * Σ w_i * cons_loss_i
    
where w_i = normalize(exp(β * (Q(s_i, a_i) - baseline)))

Consistency Loss Design (optimized from sweep):
- Full t range [0, 1] for temporal diversity
- Small fixed delta (0.01) for stable teacher target  
- Teacher integrates from t_cons, student from t_plus (endpoint consistency)
- Endpoint loss space (not velocity-space) for better convergence
- 2-step teacher for accurate target prediction

Hyperparameter Recommendations:
- beta: 10.0 (advantage temperature, higher = more aggressive weighting)
- bc_weight: 1.0 (flow matching loss weight)
- consistency_weight: 1.0 (consistency loss weight)
- weight_clip: 100.0 (prevent outlier dominance)
- use_advantage: True (use Q - baseline instead of raw Q)

References:
- AWAC: https://arxiv.org/abs/2006.09359 (Advantage Weighted Actor-Critic)
- IQL: https://arxiv.org/abs/2110.06169 (Implicit Q-Learning)
- CPQL: https://github.com/cccedric/cpql
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import copy

from rlft.networks import VelocityUNet1D, DoubleQNetwork, soft_update


class AWCPAgent(nn.Module):
    """Advantage-Weighted Consistency Policy Agent.
    
    Uses Q-values to weight BC samples instead of directly maximizing Q.
    This is more stable for offline RL with expert demonstrations.
    
    Algorithm Overview:
    ------------------
    Unlike CPQL which has a -Q term in the loss, AWCP only uses Q to compute
    sample weights. This keeps the policy close to the data distribution while
    still benefiting from Q-value guidance.
    
    Weight computation:
        advantage_i = Q(s_i, a_i) - mean(Q)  # baseline subtraction
        weight_i = exp(β * advantage_i)      # exponential weighting
        weight_i = clamp(weight_i, max=weight_clip)  # prevent outliers
        weight_i = weight_i / mean(weight)   # normalize to mean=1
    
    Hyperparameter Recommendations (from sweep):
    - beta: 10.0 (temperature, higher = more aggressive weighting)
    - bc_weight: 1.0 (flow matching loss weight)
    - consistency_weight: 1.0 (self-consistency loss weight)
    - reward_scale: 0.1 (scale rewards for stable Q-learning)
    - weight_clip: 100.0 (max weight to prevent outlier dominance)
    - use_advantage: True (subtract baseline for reduced variance)
    
    Args:
        velocity_net: VelocityUNet1D for velocity prediction (flow policy)
        q_network: DoubleQNetwork for Q-value estimation (critic)
        action_dim: Dimension of action space
        obs_horizon: Number of observation frames (default: 2)
        pred_horizon: Length of action sequence to predict (default: 16)
        act_horizon: Length of action sequence for Q-learning (default: 8)
        num_flow_steps: Number of ODE integration steps (default: 10)
        beta: Temperature for advantage weighting (default: 10.0)
            Higher values make the weighting more aggressive (sharper)
        bc_weight: Weight for flow matching loss (default: 1.0)
        consistency_weight: Weight for consistency loss (default: 1.0)
        gamma: Discount factor (default: 0.99)
        tau: Soft update coefficient (default: 0.005)
        reward_scale: Scale factor for rewards (default: 0.1)
        q_target_clip: Clip range for Q target (default: 100.0)
        ema_decay: Decay rate for EMA velocity network (default: 0.999)
        weight_clip: Maximum weight to prevent outliers (default: 100.0)
        use_advantage: Whether to use advantage (Q - baseline) or raw Q (default: True)
            True: w = exp(β * (Q - mean(Q))) - reduces variance
            False: w = exp(β * Q) - can be unstable with large Q values
        device: Device to run on (default: "cuda")
    """
    
    def __init__(
        self,
        velocity_net: VelocityUNet1D,
        q_network: DoubleQNetwork,
        action_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        act_horizon: int = 8,
        num_flow_steps: int = 20,  # Best from sweep (20 > 10 > 5)
        beta: float = 10.0,  # Best from sweep (aggressive weighting)
        bc_weight: float = 1.0,
        consistency_weight: float = 0.3,  # Best from sweep (conservative config)
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scale: float = 0.1,
        q_target_clip: float = 100.0,
        ema_decay: float = 0.999,
        weight_clip: float = 100.0,
        use_advantage: bool = True,
        action_bounds: Optional[tuple] = None,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.velocity_net = velocity_net
        self.critic = q_network
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.num_flow_steps = num_flow_steps
        self.beta = beta
        self.bc_weight = bc_weight
        self.consistency_weight = consistency_weight
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.q_target_clip = q_target_clip
        self.ema_decay = ema_decay
        self.weight_clip = weight_clip
        self.use_advantage = use_advantage
        self.action_bounds = action_bounds
        self.device = device
        
        # Consistency loss hyperparameters (optimized from sweep: endpoint consistency style)
        # Full t range [0, 1] for temporal diversity
        # Small fixed delta for stable teacher target (not random range)
        # 2-step teacher for accurate target prediction
        self.t_min = 0.0
        self.t_max = 1.0
        self.delta_min = 0.01
        self.delta_max = 0.01  # Fixed delta (not random range)
        self.teacher_steps = 2
        
        # EMA velocity network for consistency loss
        self.velocity_net_ema = copy.deepcopy(self.velocity_net)
        for p in self.velocity_net_ema.parameters():
            p.requires_grad = False
        
        # Target critic
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False
    
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
        """Compute combined policy and critic loss.
        
        Note: Unlike CPQL, the policy loss here does NOT include a direct Q maximization
        term. Instead, Q-values are used to compute sample weights for the BC loss.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, global_cond_dim]
            actions: (B, pred_horizon, action_dim) expert action sequence for BC training
            rewards: (B,) or (B, 1) single-step rewards
            next_obs_features: Next observation features
            dones: (B,) or (B, 1) done flags
            actions_for_q: (B, act_horizon, action_dim) action sequence for Q-learning
            cumulative_reward: (B,) or (B, 1) SMDP cumulative discounted reward
            chunk_done: (B,) or (B, 1) SMDP done flag
            discount_factor: (B,) or (B, 1) SMDP discount
            
        Returns:
            Dict with loss components including weight statistics for monitoring
        """
        if actions_for_q is None:
            actions_for_q = actions
        
        # Flatten obs_features if 3D
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        else:
            obs_cond = obs_features
            
        if next_obs_features.dim() == 3:
            next_obs_cond = next_obs_features.reshape(next_obs_features.shape[0], -1)
        else:
            next_obs_cond = next_obs_features
        
        # Compute policy loss with Q-weighted BC
        policy_dict = self._compute_policy_loss(obs_cond, actions, actions_for_q)
        
        # Compute critic loss
        critic_loss = self._compute_critic_loss(
            obs_cond.detach(), next_obs_cond.detach(), actions_for_q, rewards, dones,
            cumulative_reward=cumulative_reward,
            chunk_done=chunk_done,
            discount_factor=discount_factor,
        )
        
        return {
            "loss": policy_dict["policy_loss"] + critic_loss,
            "actor_loss": policy_dict["policy_loss"],
            "policy_loss": policy_dict["policy_loss"],
            "flow_loss": policy_dict["flow_loss"],
            "consistency_loss": policy_dict["consistency_loss"],
            "q_policy_loss": policy_dict["q_policy_loss"],
            "critic_loss": critic_loss,
            "q_mean": policy_dict["q_mean"],
            "weight_mean": policy_dict["weight_mean"],
            "weight_std": policy_dict["weight_std"],
            "advantage_mean": policy_dict["advantage_mean"],
        }
    
    def _linear_interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation: x_t = (1 - t) * x_0 + t * x_1"""
        t_expand = t.view(-1, 1, 1)
        return (1 - t_expand) * x_0 + t_expand * x_1
    
    def _compute_policy_loss(
        self, 
        obs_cond: torch.Tensor, 
        action_seq: torch.Tensor,
        actions_for_q_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute policy loss: Q-weighted Flow BC + Endpoint Consistency.
        
        Key differences from CPQL:
        - No direct Q maximization term (no gradient from Q to policy)
        - Q-values only used to compute per-sample weights
        - Both flow loss and consistency loss are weighted by Q-derived weights
        
        Weight computation (AWAC-style):
            1. Q_data = min(Q1, Q2)  for conservative estimate
            2. advantage = Q_data - mean(Q_data)  if use_advantage else Q_data
            3. weights = exp(β * advantage)
            4. weights = clamp(weights, max=weight_clip)
            5. weights = weights / mean(weights)  normalize to mean=1
        
        Consistency Loss (endpoint style):
        - Teacher (EMA): Multi-step integration from t_cons to t=1
        - Student: Multi-step integration from t_plus to t=1
        - Loss: ||student_endpoint - teacher_endpoint||^2
        
        Args:
            obs_cond: Observation conditioning [B, global_cond_dim]
            action_seq: Full pred_horizon actions for BC training [B, pred_horizon, action_dim]
            actions_for_q_input: act_horizon actions for Q-learning [B, act_horizon, action_dim]
            
        Returns:
            Dict with policy_loss, flow_loss, consistency_loss, q_policy_loss (always 0),
            q_mean, weight_mean, weight_std, advantage_mean
        """
        B = action_seq.shape[0]
        device = action_seq.device
        
        # ===== Compute AWAC-style weights from Q-values =====
        with torch.no_grad():
            q1_data, q2_data = self.critic(actions_for_q_input, obs_cond)
            q_data = torch.min(q1_data, q2_data)
            
            if self.use_advantage:
                baseline = q_data.mean()
                advantage = q_data - baseline
            else:
                advantage = q_data
            
            weights = torch.clamp(torch.exp(self.beta * advantage), max=self.weight_clip)
            weights = weights / weights.mean()
            weights = weights.squeeze(-1)
        
        x_0 = torch.randn_like(action_seq)
        
        # ===== Q-Weighted Flow Matching Loss =====
        t = torch.rand(B, device=device)
        x_t = self._linear_interpolate(x_0, action_seq, t)
        v_target = action_seq - x_0
        v_pred = self.velocity_net(x_t, t, global_cond=obs_cond)
        
        flow_loss_per_sample = F.mse_loss(v_pred, v_target, reduction="none")
        flow_loss_per_sample = flow_loss_per_sample.mean(dim=(1, 2))
        flow_loss = (weights * flow_loss_per_sample).mean()
        
        # ===== Consistency Loss =====
        t_cons = self.t_min + torch.rand(B, device=device) * (self.t_max - self.t_min)
        delta_t = torch.full_like(t_cons, self.delta_min)
        t_plus = torch.clamp(t_cons + delta_t, max=1.0)
        
        x_t_cons = self._linear_interpolate(x_0, action_seq, t_cons)
        x_t_plus = self._linear_interpolate(x_0, action_seq, t_plus)
        
        # Teacher: multi-step integration from t_cons
        with torch.no_grad():
            x_teacher = x_t_cons.clone()
            current_t = t_cons.clone()
            remaining_time = 1.0 - current_t
            dt_teacher = remaining_time / self.teacher_steps
            
            for _ in range(self.teacher_steps):
                v_teacher = self.velocity_net_ema(x_teacher, current_t, global_cond=obs_cond)
                dt_expand = dt_teacher.view(-1, 1, 1)
                x_teacher = x_teacher + v_teacher * dt_expand
                current_t = current_t + dt_teacher
            
            target_x1 = x_teacher
        
        # Student: predict from t_plus
        x_student = x_t_plus.clone()
        current_t_student = t_plus.clone()
        remaining_time_student = 1.0 - current_t_student
        dt_student = remaining_time_student / self.teacher_steps
        
        for _ in range(self.teacher_steps):
            v_student = self.velocity_net(x_student, current_t_student, global_cond=obs_cond)
            dt_expand = dt_student.view(-1, 1, 1)
            x_student = x_student + v_student * dt_expand
            current_t_student = current_t_student + dt_student
        
        consistency_loss_per_sample = F.mse_loss(x_student, target_x1, reduction="none")
        consistency_loss_per_sample = consistency_loss_per_sample.mean(dim=(1, 2))
        consistency_loss = (weights * consistency_loss_per_sample).mean()
        
        # No direct Q maximization in AWCP
        q_policy_loss = torch.tensor(0.0, device=device)
        
        policy_loss = (
            self.bc_weight * flow_loss +
            self.consistency_weight * consistency_loss
        )
        
        return {
            "policy_loss": policy_loss,
            "flow_loss": flow_loss,
            "consistency_loss": consistency_loss,
            "q_policy_loss": q_policy_loss,
            "q_mean": q_data.mean(),
            "weight_mean": weights.mean(),
            "weight_std": weights.std(),
            "advantage_mean": advantage.mean(),
        }
    
    def _compute_critic_loss(
        self,
        obs_cond: torch.Tensor,
        next_obs_cond: torch.Tensor,
        action_seq: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        cumulative_reward: Optional[torch.Tensor] = None,
        chunk_done: Optional[torch.Tensor] = None,
        discount_factor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute critic loss using SMDP Bellman equation (same as CPQL).
        
        Standard TD target (when SMDP fields not provided):
            y = r * reward_scale + (1 - d) * γ * min Q_target(s', π(s'))
        
        SMDP TD target (for action chunking):
            y = R^(τ) * reward_scale + (1 - d^(τ)) * γ^τ * min Q_target(s_{t+τ}, π(s_{t+τ}))
        
        Next actions are sampled using the EMA policy network for stability.
        
        Args:
            obs_cond: Current observation conditioning [B, global_cond_dim]
            next_obs_cond: Next observation conditioning [B, global_cond_dim]
            action_seq: Actions taken [B, act_horizon, action_dim]
            rewards: Immediate rewards [B] or [B, 1]
            dones: Done flags [B] or [B, 1]
            cumulative_reward: SMDP cumulative reward (optional)
            chunk_done: SMDP done flag (optional)
            discount_factor: SMDP discount γ^τ (optional)
            
        Returns:
            critic_loss: Combined MSE loss for both Q-networks
        """
        if cumulative_reward is not None:
            r = cumulative_reward
            d = chunk_done if chunk_done is not None else dones
            gamma_tau = discount_factor if discount_factor is not None else torch.full_like(r, self.gamma)
        else:
            r = rewards
            d = dones
            gamma_tau = torch.full_like(r if r.dim() == 1 else r.squeeze(-1), self.gamma)
        
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        if d.dim() == 1:
            d = d.unsqueeze(-1)
        if gamma_tau.dim() == 1:
            gamma_tau = gamma_tau.unsqueeze(-1)
        
        scaled_rewards = r * self.reward_scale
        
        with torch.no_grad():
            next_actions_full = self._sample_actions_batch(next_obs_cond, use_ema=True)
            next_actions = next_actions_full[:, :self.act_horizon, :]
            
            target_q1, target_q2 = self.critic_target(next_actions, next_obs_cond)
            target_q = torch.min(target_q1, target_q2)
            target_q = scaled_rewards + (1 - d) * gamma_tau * target_q
            
            if self.q_target_clip is not None:
                target_q = torch.clamp(target_q, -self.q_target_clip, self.q_target_clip)
        
        current_q1, current_q2 = self.critic(action_seq, obs_cond)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        return critic_loss
    
    def _sample_actions_batch(
        self,
        obs_cond: torch.Tensor,
        use_ema: bool = False
    ) -> torch.Tensor:
        """Sample actions using flow ODE integration (Euler method).
        
        Args:
            obs_cond: Observation conditioning [B, global_cond_dim]
            use_ema: Whether to use EMA velocity network (for stable evaluation)
            
        Returns:
            actions: Generated action sequence [B, pred_horizon, action_dim], clamped to [-1, 1]
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        net = self.velocity_net_ema if use_ema else self.velocity_net
        
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        dt = 1.0 / self.num_flow_steps
        
        for i in range(self.num_flow_steps):
            t = torch.full((B,), i * dt, device=device)
            v = net(x, t, global_cond=obs_cond)
            x = x + v * dt
        
        # Only clamp if action_bounds is explicitly set
        if self.action_bounds is not None:
            x = torch.clamp(x, self.action_bounds[0], self.action_bounds[1])
        return x
    
    def update_ema(self):
        """Update EMA velocity network.
        
        Formula: θ_ema = ema_decay * θ_ema + (1 - ema_decay) * θ
        """
        soft_update(self.velocity_net_ema, self.velocity_net, 1 - self.ema_decay)
    
    def update_target(self):
        """Soft update target critic network.
        
        Formula: θ_target = tau * θ + (1 - tau) * θ_target
        """
        soft_update(self.critic_target, self.critic, self.tau)
    
    @torch.no_grad()
    def get_action(
        self,
        obs_features: torch.Tensor,
        use_ema: bool = True,
        **kwargs,  # Accept extra kwargs for API compatibility
    ) -> torch.Tensor:
        """Sample action sequence for evaluation (stochastic from noise).
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, global_cond_dim]
            use_ema: Whether to use EMA network for sampling (recommended for evaluation)
            
        Returns:
            actions: Generated action sequence [B, pred_horizon, action_dim]
        """
        self.velocity_net.eval()
        
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        else:
            obs_cond = obs_features
        
        action_seq = self._sample_actions_batch(obs_cond, use_ema=use_ema)
        
        self.velocity_net.train()
        return action_seq

    @torch.no_grad()
    def get_action_deterministic(
        self,
        obs_features: torch.Tensor,
        use_ema: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Sample action sequence deterministically (starting from zero instead of noise).
        
        Useful for reproducible evaluation. Starts ODE integration from zeros
        instead of random noise, producing deterministic outputs.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, global_cond_dim]
            use_ema: Whether to use EMA network for sampling (recommended)
            
        Returns:
            actions: Generated action sequence [B, pred_horizon, action_dim]
        """
        self.velocity_net.eval()
        
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        else:
            obs_cond = obs_features
        
        B = obs_cond.shape[0]
        device = obs_cond.device
        net = self.velocity_net_ema if use_ema else self.velocity_net
        
        # Start from zeros (deterministic)
        x = torch.zeros(B, self.pred_horizon, self.action_dim, device=device)
        dt = 1.0 / self.num_flow_steps
        
        for i in range(self.num_flow_steps):
            t = torch.full((B,), i * dt, device=device)
            v = net(x, t, global_cond=obs_cond)
            x = x + v * dt
        
        # Only clamp if action_bounds is explicitly set
        if self.action_bounds is not None:
            x = torch.clamp(x, self.action_bounds[0], self.action_bounds[1])
        
        self.velocity_net.train()
        return x
