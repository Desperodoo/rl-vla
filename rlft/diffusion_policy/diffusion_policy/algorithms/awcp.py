"""
Advantage-Weighted Consistency Policy (AWCP) Agent

Combines Consistency Flow Matching with AWAC-style Q-weighted BC for stable offline RL.
Instead of maximizing Q directly (which causes distribution shift), uses Q to weight BC samples.

Key difference from CPQL:
- CPQL: policy_loss = bc_loss + alpha * (-Q(s, π(s)))  [direct Q maximization]
- AWCP: policy_loss = weighted_bc_loss where weights = exp(β * advantage)  [Q-weighted BC]

This approach is more stable in offline settings because:
1. Policy stays close to data distribution (no OOD action generation for Q)
2. Q is only used to distinguish "better vs worse" samples in the dataset
3. No gradient from Q to policy, avoiding Q-value explosion

References:
- AWAC: https://arxiv.org/abs/2006.09359
- IQL: https://arxiv.org/abs/2110.06169
- CPQL: https://github.com/cccedric/cpql
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import copy

from .networks import VelocityUNet1D, DoubleQNetwork, soft_update


class AWCPAgent(nn.Module):
    """Advantage-Weighted Consistency Policy Agent.
    
    Uses Q-values to weight BC samples instead of directly maximizing Q.
    This is more stable for offline RL with expert demonstrations.
    
    Consistency design (optimized from sweep):
    - Full t range [0, 1] for temporal diversity
    - Small fixed delta (0.01) for stable teacher target
    - Teacher integrates from t_cons, student from t_plus (endpoint consistency)
    - Endpoint loss space (not velocity-space) for better convergence
    - 2-3 multi-step teacher for accurate target prediction
    
    Args:
        velocity_net: VelocityUNet1D for velocity prediction (flow policy)
        q_network: DoubleQNetwork for Q-value estimation (critic)
        action_dim: Dimension of action space
        obs_horizon: Number of observation frames (default: 2)
        pred_horizon: Length of action sequence to predict (default: 16)
        act_horizon: Length of action sequence for Q-learning (default: 8)
        num_flow_steps: Number of ODE integration steps (default: 10)
        beta: Temperature for advantage weighting (default: 10.0)
        bc_weight: Weight for flow matching loss (default: 1.0)
        consistency_weight: Weight for consistency loss (default: 1.0)
        gamma: Discount factor (default: 0.99)
        tau: Soft update coefficient (default: 0.005)
        reward_scale: Scale factor for rewards (default: 0.1)
        q_target_clip: Clip range for Q target (default: 100.0)
        ema_decay: Decay rate for EMA velocity network (default: 0.999)
        weight_clip: Maximum weight to prevent outliers (default: 100.0)
        use_advantage: Whether to use advantage (Q - baseline) or raw Q (default: True)
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
        num_flow_steps: int = 10,
        beta: float = 10.0,
        bc_weight: float = 1.0,
        consistency_weight: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scale: float = 0.1,
        q_target_clip: float = 100.0,
        ema_decay: float = 0.999,
        weight_clip: float = 100.0,
        use_advantage: bool = True,
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
        self.device = device
        
        # Consistency loss hyperparameters (optimized from sweep: flow_endpoint style)
        # Use full t range [0, 1] and small fixed delta for stable consistency
        self.t_min = 0.0  # Full t range [0, 1]
        self.t_max = 1.0  # Full t range [0, 1]
        self.delta_min = 0.01  # Small fixed delta for consistency
        self.delta_max = 0.01  # Small fixed delta (avoid large teacher error)
        self.teacher_steps = 2  # Multi-step teacher for accurate targets
        
        # Additional consistency parameters (matching sweep recommendations)
        self.cons_teacher_from = "t_cons"  # Teacher network starts from t_cons
        self.cons_student_point = "t_plus"  # Student consistency at t_plus
        self.cons_loss_space = "endpoint"  # Use endpoint consistency loss (not velocity)
        
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
            Dict with loss components
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
        
        # Compute critic loss (same as CPQL)
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
            "q_policy_loss": policy_dict["q_policy_loss"],  # Will be 0 for AWCP
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
        """Compute policy loss: Q-weighted Flow BC + Consistency.
        
        Key difference from CPQL:
        - No direct Q maximization (no gradient from Q to policy)
        - Instead, use Q to weight BC samples (AWAC-style)
        
        Args:
            obs_cond: Observation conditioning [B, global_cond_dim]
            action_seq: Full pred_horizon actions for BC training [B, pred_horizon, action_dim]
            actions_for_q_input: act_horizon actions for Q-learning [B, act_horizon, action_dim]
        """
        B = action_seq.shape[0]
        device = action_seq.device
        
        # ===== Compute AWAC-style weights from Q-values =====
        with torch.no_grad():
            # Get Q-values for the dataset actions (no gradient needed)
            q1_data, q2_data = self.critic(actions_for_q_input, obs_cond)
            q_data = torch.min(q1_data, q2_data)  # Conservative Q estimate
            
            if self.use_advantage:
                # Compute advantage: A(s,a) = Q(s,a) - V(s) ≈ Q(s,a) - E[Q(s,a')]
                # Use batch mean as baseline (simple but effective)
                baseline = q_data.mean()
                advantage = q_data - baseline
            else:
                # Use raw Q-values (no baseline subtraction)
                advantage = q_data
            
            # AWAC-style exponential weights: w = exp(β * A)
            # Clamp to prevent numerical issues and outlier dominance
            weights = torch.clamp(torch.exp(self.beta * advantage), max=self.weight_clip)
            
            # Normalize weights to have mean 1 (optional, helps stability)
            weights = weights / weights.mean()
            
            # Flatten weights for per-sample weighting [B]
            weights = weights.squeeze(-1)
        
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(action_seq)
        
        # ===== Q-Weighted Flow Matching Loss =====
        t = torch.rand(B, device=device)
        x_t = self._linear_interpolate(x_0, action_seq, t)
        
        # Target velocity: v = x_1 - x_0
        v_target = action_seq - x_0
        
        # Predict velocity
        v_pred = self.velocity_net(x_t, t, global_cond=obs_cond)
        
        # Per-sample MSE loss [B, T, D] -> [B]
        flow_loss_per_sample = F.mse_loss(v_pred, v_target, reduction="none")
        flow_loss_per_sample = flow_loss_per_sample.mean(dim=(1, 2))  # [B]
        
        # Weighted average: higher Q samples get higher weight
        flow_loss = (weights * flow_loss_per_sample).mean()
        
        # ===== Consistency Loss (endpoint matching, not velocity-space) =====
        # Sample t for consistency: use full range [t_min, t_max]
        t_cons = self.t_min + torch.rand(B, device=device) * (self.t_max - self.t_min)
        # Fixed delta (not random range) for stable teacher target
        delta_t = torch.full_like(t_cons, self.delta_min)
        t_plus = torch.clamp(t_cons + delta_t, max=1.0)
        
        x_t_cons = self._linear_interpolate(x_0, action_seq, t_cons)
        x_t_plus = self._linear_interpolate(x_0, action_seq, t_plus)
        
        # Teacher: multi-step integration from t_cons (not t_plus) using EMA network
        # This predicts x(1) starting from x(t_cons)
        with torch.no_grad():
            # Teacher trajectory: x_t_cons -> x_t_plus -> ... -> x(1)
            x_teacher = x_t_cons.clone()
            current_t = t_cons.clone()
            remaining_time = 1.0 - current_t
            dt_teacher = remaining_time / self.teacher_steps
            
            for _ in range(self.teacher_steps):
                v_teacher = self.velocity_net_ema(x_teacher, current_t, global_cond=obs_cond)
                dt_expand = dt_teacher.view(-1, 1, 1)
                x_teacher = x_teacher + v_teacher * dt_expand
                current_t = current_t + dt_teacher
            
            target_x1 = x_teacher  # Teacher's prediction of final state
        
        # Student: predict x(1) from x(t_plus)
        # Consistency: both should reach same final state
        x_student = x_t_plus.clone()
        current_t_student = t_plus.clone()
        remaining_time_student = 1.0 - current_t_student
        dt_student = remaining_time_student / self.teacher_steps
        
        for _ in range(self.teacher_steps):
            v_student = self.velocity_net(x_student, current_t_student, global_cond=obs_cond)
            dt_expand = dt_student.view(-1, 1, 1)
            x_student = x_student + v_student * dt_expand
            current_t_student = current_t_student + dt_student
        
        # Endpoint consistency loss: ||student_endpoint - teacher_endpoint||^2
        consistency_loss_per_sample = F.mse_loss(x_student, target_x1, reduction="none")
        consistency_loss_per_sample = consistency_loss_per_sample.mean(dim=(1, 2))  # [B]
        consistency_loss = (weights * consistency_loss_per_sample).mean()
        
        # ===== No direct Q maximization loss (this is the key difference!) =====
        # In AWCP, we don't have alpha * (-Q) term
        # Instead, Q is only used to weight the BC samples above
        q_policy_loss = torch.tensor(0.0, device=device)
        
        # Total policy loss
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
        """Compute critic loss using SMDP Bellman equation (same as CPQL)."""
        # Use SMDP fields if provided
        if cumulative_reward is not None:
            r = cumulative_reward
            d = chunk_done if chunk_done is not None else dones
            gamma_tau = discount_factor if discount_factor is not None else torch.full_like(r, self.gamma)
        else:
            r = rewards
            d = dones
            gamma_tau = torch.full_like(r if r.dim() == 1 else r.squeeze(-1), self.gamma)
        
        # Ensure proper shape
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        if d.dim() == 1:
            d = d.unsqueeze(-1)
        if gamma_tau.dim() == 1:
            gamma_tau = gamma_tau.unsqueeze(-1)
        
        # Scale rewards
        scaled_rewards = r * self.reward_scale
        
        with torch.no_grad():
            # Sample next actions using EMA policy
            next_actions_full = self._sample_actions_batch(next_obs_cond, use_ema=True)
            next_actions = next_actions_full[:, :self.act_horizon, :]
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_actions, next_obs_cond)
            target_q = torch.min(target_q1, target_q2)
            
            # TD target
            target_q = scaled_rewards + (1 - d) * gamma_tau * target_q
            
            if self.q_target_clip is not None:
                target_q = torch.clamp(target_q, -self.q_target_clip, self.q_target_clip)
        
        # Current Q-values
        current_q1, current_q2 = self.critic(action_seq, obs_cond)
        
        # MSE loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        return critic_loss
    
    def _sample_actions_batch(
        self,
        obs_cond: torch.Tensor,
        use_ema: bool = False
    ) -> torch.Tensor:
        """Sample actions using flow ODE."""
        B = obs_cond.shape[0]
        device = obs_cond.device
        net = self.velocity_net_ema if use_ema else self.velocity_net
        
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        dt = 1.0 / self.num_flow_steps
        
        for i in range(self.num_flow_steps):
            t = torch.full((B,), i * dt, device=device)
            v = net(x, t, global_cond=obs_cond)
            x = x + v * dt
        
        return torch.clamp(x, -1.0, 1.0)
    
    def update_ema(self):
        """Update EMA velocity network."""
        soft_update(self.velocity_net_ema, self.velocity_net, 1 - self.ema_decay)
    
    def update_target(self):
        """Soft update target critic network."""
        soft_update(self.critic_target, self.critic, self.tau)
    
    @torch.no_grad()
    def get_action(
        self,
        obs_features: torch.Tensor,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Sample action sequence for evaluation."""
        self.velocity_net.eval()
        
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        else:
            obs_cond = obs_features
        
        action_seq = self._sample_actions_batch(obs_cond, use_ema=use_ema)
        
        self.velocity_net.train()
        return action_seq
