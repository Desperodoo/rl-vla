"""
Consistency Policy Q-Learning (CPQL) Agent

Combines Consistency Flow Matching with Q-Learning for efficient offline RL.
Uses 1D U-Net architecture aligned with the official diffusion_policy implementation.

Key Algorithm Design:
=====================
CPQL uses three loss terms to train the policy:
1. Flow Matching Loss (BC): Standard CFM loss to match expert actions
2. Consistency Loss: Self-consistency regularization for faster inference
3. Q-Value Maximization: Direct policy gradient through Q-network

The total policy loss is:
    policy_loss = bc_weight * flow_loss + consistency_weight * consistency_loss + alpha * q_loss

Key Implementation Details:
- Q-loss normalization trick: Randomly select Q1 or Q2 and normalize by the other's magnitude
  to prevent Q-value explosion in offline RL
- Configurable gradient chain: "single_step" (fastest), "last_few", or "whole_grad"
- SMDP support: For action chunking with cumulative rewards and discount factors

Consistency Loss Design (from sweep best practices):
- t_min=0.05, t_max=0.95: Avoid boundary instability
- delta_min=0.02, delta_max=0.15: Random delta for temporal diversity
- teacher_steps=2: Multi-step teacher for accurate targets
- velocity-space loss: Better gradient flow than endpoint-space

References:
- CPQL: https://github.com/cccedric/cpql
- Diffusion-QL: https://arxiv.org/abs/2208.06193
- Consistency Flow Matching: consistency_flow.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import copy

from rlft.networks import VelocityUNet1D, DoubleQNetwork, soft_update


class CPQLAgent(nn.Module):
    """Consistency Policy Q-Learning Agent with 1D U-Net architecture.
    
    Combines Consistency Flow Matching with Q-Learning.
    Uses self-consistency loss to enable single-step generation.
    
    Algorithm Overview:
    ------------------
    Unlike AWCP which weights BC samples by Q-values, CPQL directly maximizes Q:
    - CPQL: policy_loss = bc_loss + alpha * (-Q(s, π(s)))  [direct Q maximization]
    - AWCP: policy_loss = weighted_bc_loss where weights = exp(β * advantage)  [Q-weighted BC]
    
    This makes CPQL more aggressive in policy improvement but potentially less stable.
    
    Hyperparameter Recommendations (from sweep):
    - alpha: 0.01 (Q-value weight, reduced for stability in offline RL)
    - bc_weight: 1.0 (flow matching loss weight)
    - consistency_weight: 1.0 (self-consistency loss weight)
    - reward_scale: 0.1 (scale rewards to prevent Q-value explosion)
    - tau: 0.005 (soft update coefficient for target networks)
    - gamma: 0.99 (discount factor)
    - num_flow_steps: 10 (ODE integration steps)
    - q_grad_mode: "single_step" (fastest, works well in practice)
    
    Args:
        velocity_net: VelocityUNet1D for velocity prediction (flow policy)
        q_network: DoubleQNetwork for Q-value estimation (critic)
        action_dim: Dimension of action space
        obs_horizon: Number of observation frames (default: 2)
        pred_horizon: Length of action sequence to predict (default: 16)
        act_horizon: Length of action sequence for Q-learning (default: 8)
        num_flow_steps: Number of ODE integration steps (default: 10)
        alpha: Weight for Q-value term (default: 0.01)
        bc_weight: Weight for flow matching loss (default: 1.0)
        consistency_weight: Weight for consistency loss (default: 1.0)
        gamma: Discount factor (default: 0.99)
        tau: Soft update coefficient (default: 0.005)
        reward_scale: Scale factor for rewards (default: 0.1)
        q_target_clip: Clip range for Q target (default: 100.0)
        ema_decay: Decay rate for EMA velocity network (default: 0.999)
        q_grad_mode: Gradient mode for Q-loss:
            - "single_step": Predict x_1 = x_0 + v(x_0, 0) in one step (fastest)
            - "last_few": Gradient through last q_grad_steps flow steps
            - "whole_grad": Full gradient chain through all flow steps
        q_grad_steps: Number of steps with gradient when q_grad_mode="last_few"
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
        alpha: float = 0.001,  # Best from sweep (smaller is more stable)
        bc_weight: float = 0.5,  # Best from sweep (0.5 > 1.0)
        consistency_weight: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scale: float = 0.1,
        q_target_clip: float = 100.0,
        ema_decay: float = 0.999,
        q_grad_mode: str = "single_step",
        q_grad_steps: int = 1,
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
        self.alpha = alpha
        self.bc_weight = bc_weight
        self.consistency_weight = consistency_weight
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.q_target_clip = q_target_clip
        self.ema_decay = ema_decay
        self.q_grad_mode = q_grad_mode
        self.q_grad_steps = q_grad_steps
        self.action_bounds = action_bounds
        self.device = device
        
        assert q_grad_mode in ["whole_grad", "last_few", "single_step"], \
            f"q_grad_mode must be 'whole_grad', 'last_few', or 'single_step', got {q_grad_mode}"
        
        # Consistency loss hyperparameters (aligned with best practices from sweep)
        # t_min=0.05, t_max=0.95: Avoid boundary instability at t≈0 and t≈1
        # delta_min=0.02, delta_max=0.15: Random delta for diversity
        # teacher_steps=2: Multi-step teacher for accurate targets
        self.t_min = 0.05
        self.t_max = 0.95
        self.delta_min = 0.02
        self.delta_max = 0.15
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
        
        Supports both standard TD learning and SMDP formulation for action chunking.
        For SMDP (action chunking), the Bellman target is:
            y_t = R_t^(τ) + (1 - d_t^(τ)) * γ^τ * min Q(s_{t+τ}, a')
        
        where:
            - R_t^(τ) = Σ_{i=0}^{τ-1} γ^i r_{t+i} (cumulative discounted reward)
            - d_t^(τ) = 1 if episode ends within chunk
            - γ^τ = discount factor over τ steps
            - s_{t+τ} = state after chunk execution
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, global_cond_dim]
            actions: (B, pred_horizon, action_dim) expert action sequence for BC training
            rewards: (B,) or (B, 1) single-step rewards (used if cumulative_reward not provided)
            next_obs_features: Next observation features (s_{t+τ} for SMDP)
            dones: (B,) or (B, 1) done flags (used if chunk_done not provided)
            actions_for_q: (B, act_horizon, action_dim) action sequence for Q-learning (matches reward)
            cumulative_reward: (B,) or (B, 1) SMDP cumulative discounted reward R_t^(τ)
            chunk_done: (B,) or (B, 1) SMDP done flag (1 if episode ends within chunk)
            discount_factor: (B,) or (B, 1) SMDP discount γ^τ
            
        Returns:
            Dict with loss components:
                - loss: Total loss (policy_loss + critic_loss) for logging
                - actor_loss / policy_loss: Policy loss for actor backward
                - flow_loss: Flow matching (BC) loss component
                - consistency_loss: Self-consistency loss component
                - q_policy_loss: Q-value maximization loss component
                - critic_loss: Critic (Q-network) loss for critic backward
                - q_mean: Mean Q-value for logging
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
        
        # Compute policy loss
        policy_dict = self._compute_policy_loss(obs_cond, actions, actions_for_q)
        
        # Compute critic loss (detach obs for critic)
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
        """Compute policy loss: Flow BC + Consistency + Q-value maximization.
        
        The policy loss consists of three components:
        1. Flow Matching Loss: Standard CFM objective ||v_pred - v_target||^2
           where v_target = x_1 - x_0 (straight-line velocity)
        
        2. Consistency Loss: Self-consistency regularization
           - Teacher (EMA): Multi-step integration from t_plus to 1
           - Student: Predict velocity at t_plus to match teacher's endpoint
           - Uses velocity-space loss: ||v_student - (target_x1 - x_0)||^2
        
        3. Q-Value Maximization: Direct policy gradient
           - Generate actions using flow sampling
           - Maximize Q(s, generated_actions)
           - Uses normalization trick: q_loss = -Q1 / |Q2| (or vice versa)
        
        Args:
            obs_cond: Observation conditioning [B, global_cond_dim]
            action_seq: Full pred_horizon actions for BC training [B, pred_horizon, action_dim]
            actions_for_q_input: act_horizon actions for Q-learning [B, act_horizon, action_dim]
            
        Returns:
            Dict with policy_loss, flow_loss, consistency_loss, q_policy_loss, q_mean
        """
        B = action_seq.shape[0]
        device = action_seq.device
        act_horizon = actions_for_q_input.shape[1]
        
        x_0 = torch.randn_like(action_seq)
        
        # ===== Flow Matching Loss =====
        t = torch.rand(B, device=device)
        x_t = self._linear_interpolate(x_0, action_seq, t)
        v_target = action_seq - x_0
        v_pred = self.velocity_net(x_t, t, global_cond=obs_cond)
        flow_loss = F.mse_loss(v_pred, v_target)
        
        # ===== Consistency Loss =====
        t_cons = self.t_min + torch.rand(B, device=device) * (self.t_max - self.t_min)
        delta_t = self.delta_min + torch.rand(B, device=device) * (self.delta_max - self.delta_min)
        t_plus = torch.clamp(t_cons + delta_t, max=self.t_max)
        
        x_t_plus = self._linear_interpolate(x_0, action_seq, t_plus)
        
        with torch.no_grad():
            x_teacher = x_t_plus.clone()
            current_t = t_plus.clone()
            remaining_time = 1.0 - current_t
            dt_teacher = remaining_time / self.teacher_steps
            
            for _ in range(self.teacher_steps):
                v_teacher = self.velocity_net_ema(x_teacher, current_t, global_cond=obs_cond)
                dt_expand = dt_teacher.view(-1, 1, 1)
                x_teacher = x_teacher + v_teacher * dt_expand
                current_t = current_t + dt_teacher
            
            target_x1 = x_teacher
        
        v_cons_target = target_x1 - x_0
        v_cons_pred = self.velocity_net(x_t_plus, t_plus, global_cond=obs_cond)
        consistency_loss = F.mse_loss(v_cons_pred, v_cons_target)
        
        # ===== Q-Value Maximization Loss =====
        if self.q_grad_mode == "single_step":
            generated_actions_full = self._sample_actions_single_step(obs_cond)
        elif self.q_grad_mode == "whole_grad":
            generated_actions_full = self._sample_actions_with_grad(
                obs_cond, grad_steps=self.num_flow_steps
            )
        elif self.q_grad_mode == "last_few":
            generated_actions_full = self._sample_actions_with_grad(
                obs_cond, grad_steps=self.q_grad_steps
            )
        
        # Only clamp if action_bounds is explicitly set
        if self.action_bounds is not None:
            generated_actions_full = torch.clamp(generated_actions_full, self.action_bounds[0], self.action_bounds[1])
        actions_for_q = generated_actions_full[:, :act_horizon, :]
        
        # Disable critic gradients for Q-loss
        for p in self.critic.parameters():
            p.requires_grad = False
        
        q1_value, q2_value = self.critic(actions_for_q, obs_cond)
        
        for p in self.critic.parameters():
            p.requires_grad = True
        
        # Q-loss normalization trick from CPQL
        if torch.rand(1).item() > 0.5:
            q_loss = -q1_value.mean() / q2_value.abs().mean().detach()
        else:
            q_loss = -q2_value.mean() / q1_value.abs().mean().detach()
        
        q_mean = torch.min(q1_value, q2_value).mean()
        
        policy_loss = (
            self.bc_weight * flow_loss +
            self.consistency_weight * consistency_loss +
            self.alpha * q_loss
        )
        
        return {
            "policy_loss": policy_loss,
            "flow_loss": flow_loss,
            "consistency_loss": consistency_loss,
            "q_policy_loss": q_loss,
            "q_mean": q_mean,
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
        """Compute critic loss using SMDP Bellman equation.
        
        Standard TD target (when SMDP fields not provided):
            y = r + (1 - d) * γ * min Q_target(s', a')
        
        SMDP TD target (for action chunking):
            y = R^(τ) + (1 - d^(τ)) * γ^τ * min Q_target(s_{t+τ}, a')
        
        The critic loss is MSE over both Q-networks:
            critic_loss = MSE(Q1, y) + MSE(Q2, y)
        
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
    
    def _sample_actions_with_grad(
        self,
        obs_cond: torch.Tensor,
        grad_steps: int,
    ) -> torch.Tensor:
        """Sample actions with gradient flowing through the last `grad_steps` flow steps.
        
        This implements configurable gradient chain length for Q-loss backpropagation.
        Shorter chains are faster but may have higher variance; longer chains are
        more accurate but slower.
        
        Args:
            obs_cond: Observation conditioning [B, global_cond_dim]
            grad_steps: Number of steps to keep gradients (from the end)
            
        Returns:
            actions: Generated actions with gradient [B, pred_horizon, action_dim]
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        dt = 1.0 / self.num_flow_steps
        
        # Steps without gradient (first num_flow_steps - grad_steps)
        no_grad_steps = self.num_flow_steps - grad_steps
        
        if no_grad_steps > 0:
            with torch.no_grad():
                for i in range(no_grad_steps):
                    t = torch.full((B,), i * dt, device=device)
                    v = self.velocity_net(x, t, global_cond=obs_cond)
                    x = x + v * dt
        
        # Steps with gradient (last grad_steps)
        for i in range(no_grad_steps, self.num_flow_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.velocity_net(x, t, global_cond=obs_cond)
            x = x + v * dt
        
        return x
    
    def _sample_actions_single_step(
        self,
        obs_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step flow sampling for fast Q-loss computation.
        
        Directly predict velocity at t=0 and step to t=1 in one step:
            x_1 = x_0 + v(x_0, 0) * 1.0
        
        This is the fastest method but assumes the velocity field is accurate
        enough for single-step prediction (which consistency training helps with).
        
        Args:
            obs_cond: Observation conditioning [B, global_cond_dim]
            
        Returns:
            actions: Generated actions [B, pred_horizon, action_dim]
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        x_0 = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        t = torch.zeros(B, device=device)
        v = self.velocity_net(x_0, t, global_cond=obs_cond)
        x_1 = x_0 + v * 1.0
        
        return x_1
    
    def update_ema(self):
        """Update EMA velocity network.
        
        Formula: θ_ema = ema_decay * θ_ema + (1 - ema_decay) * θ
        Equivalent to: soft_update(ema, source, tau=1-ema_decay)
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
        """Sample action sequence for evaluation.
        
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
