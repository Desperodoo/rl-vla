"""
Consistency Policy Q-Learning (CPQL) Agent

Combines Consistency Flow Matching with Q-Learning for efficient offline RL.
Uses 1D U-Net architecture aligned with the official diffusion_policy implementation.

References:
- CPQL: https://github.com/cccedric/cpql
- Diffusion-QL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import copy

from .networks import VelocityUNet1D, DoubleQNetwork, soft_update


class CPQLAgent(nn.Module):
    """Consistency Policy Q-Learning Agent with 1D U-Net architecture.
    
    Combines Consistency Flow Matching with Q-Learning.
    Uses self-consistency loss to enable single-step generation.
    
    Hyperparameters inherited from toy_diffusion_rl:
    - alpha: 0.01 (Q-value weight, reduced for stability)
    - bc_weight: 1.0 (flow matching loss weight)
    - consistency_weight: 1.0 (self-consistency loss weight)
    - reward_scale: 0.1
    - tau: 0.005
    - gamma: 0.99
    - num_flow_steps: 10 (ODE integration steps)
    
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
        q_grad_mode: Gradient mode for Q-loss ("whole_grad", "last_few", "single_step")
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
        num_flow_steps: int = 10,
        alpha: float = 0.01,
        bc_weight: float = 1.0,
        consistency_weight: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scale: float = 0.1,
        q_target_clip: float = 100.0,
        ema_decay: float = 0.999,
        q_grad_mode: str = "single_step",
        q_grad_steps: int = 1,
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
        self.device = device
        
        # Validate q_grad_mode
        assert q_grad_mode in ["whole_grad", "last_few", "single_step"], \
            f"q_grad_mode must be 'whole_grad', 'last_few', or 'single_step', got {q_grad_mode}"
        
        # Consistency loss hyperparameters (aligned with best practices)
        self.t_min = 0.05  # Avoid boundary instability at t≈0
        self.t_max = 0.95  # Avoid boundary instability at t≈1
        self.delta_min = 0.02  # Minimum delta for consistency
        self.delta_max = 0.15  # Maximum delta (avoid large teacher error)
        self.teacher_steps = 2  # Multi-step teacher for accurate targets
        
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
        # actions_for_q: act_horizon length actions for Q-learning (matches reward horizon)
        actions_for_q: Optional[torch.Tensor] = None,
        # SMDP chunk-level fields (optional, for proper action chunking)
        cumulative_reward: Optional[torch.Tensor] = None,
        chunk_done: Optional[torch.Tensor] = None,
        discount_factor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined policy and critic loss.
        
        Supports both standard TD learning and SMDP formulation for action chunking.
        
        For SMDP (action chunking), the Bellman target is:
        y_t = R_t^(τ) + (1 - d_t^(τ)) * γ^τ * max Q(s_{t+τ}, a')
        
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
            Dict with loss components
        """
        # Use actions_for_q if provided, otherwise use full actions (backward compatibility)
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
        
        # Compute policy loss (BC uses full pred_horizon, Q uses act_horizon)
        policy_dict = self._compute_policy_loss(obs_cond, actions, actions_for_q)
        
        # Compute critic loss (uses act_horizon actions to match reward)
        # IMPORTANT: Detach obs_cond and next_obs_cond for critic to prevent
        # gradients from flowing back to visual encoder through critic loss
        critic_loss = self._compute_critic_loss(
            obs_cond.detach(), next_obs_cond.detach(), actions_for_q, rewards, dones,
            cumulative_reward=cumulative_reward,
            chunk_done=chunk_done,
            discount_factor=discount_factor,
        )
        
        # Return policy_loss and critic_loss separately (NOT summed)
        # This allows train_offline_rl.py to do separate backward passes
        return {
            "loss": policy_dict["policy_loss"] + critic_loss,  # For logging only
            "actor_loss": policy_dict["policy_loss"],  # Alias for compatibility
            "policy_loss": policy_dict["policy_loss"],  # For actor backward
            "flow_loss": policy_dict["flow_loss"],
            "consistency_loss": policy_dict["consistency_loss"],
            "q_policy_loss": policy_dict["q_policy_loss"],
            "critic_loss": critic_loss,  # For critic backward
            "q_mean": policy_dict["q_mean"],
        }
    
    def _linear_interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation: x_t = (1 - t) * x_0 + t * x_1"""
        t_expand = t.view(-1, 1, 1)  # (B, 1, 1) for (B, T, D)
        return (1 - t_expand) * x_0 + t_expand * x_1
    
    def _compute_policy_loss(
        self, 
        obs_cond: torch.Tensor, 
        action_seq: torch.Tensor,
        actions_for_q_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute policy loss: Flow BC + Consistency + Q-value.
        
        Args:
            obs_cond: Observation conditioning [B, global_cond_dim]
            action_seq: Full pred_horizon actions for BC training [B, pred_horizon, action_dim]
            actions_for_q_input: act_horizon actions for Q-learning [B, act_horizon, action_dim]
        """
        B = action_seq.shape[0]
        device = action_seq.device
        act_horizon = actions_for_q_input.shape[1]
        
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(action_seq)
        
        # ===== Flow Matching Loss (uses full pred_horizon) =====
        t = torch.rand(B, device=device)
        x_t = self._linear_interpolate(x_0, action_seq, t)
        
        # Target velocity: v = x_1 - x_0
        v_target = action_seq - x_0
        
        # Predict velocity
        v_pred = self.velocity_net(x_t, t, global_cond=obs_cond)
        flow_loss = F.mse_loss(v_pred, v_target)
        
        # ===== Consistency Loss (uses full pred_horizon) =====
        # Sample t in restricted range [t_min, t_max] to avoid boundary instability
        t_cons = self.t_min + torch.rand(B, device=device) * (self.t_max - self.t_min)
        
        # Random delta in [delta_min, delta_max], clamped to not exceed t_max
        delta_t = self.delta_min + torch.rand(B, device=device) * (self.delta_max - self.delta_min)
        t_plus = torch.clamp(t_cons + delta_t, max=self.t_max)
        
        # Get x at t_plus (student input point - Consistency Flow style)
        x_t_plus = self._linear_interpolate(x_0, action_seq, t_plus)
        
        # Teacher: multi-step integration from t_plus to 1 using EMA network
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
        
        # Student: predict velocity at t_plus
        # Target velocity in velocity space: v_target = target_x1 - x_0
        v_cons_target = target_x1 - x_0
        v_cons_pred = self.velocity_net(x_t_plus, t_plus, global_cond=obs_cond)
        consistency_loss = F.mse_loss(v_cons_pred, v_cons_target)
        
        # ===== Q-Value Maximization Loss (uses act_horizon) =====
        # Generate actions with configurable gradient chain
        if self.q_grad_mode == "single_step":
            # Single-step approximation: flow from t=0 to t=1 in one step
            generated_actions_full = self._sample_actions_single_step(obs_cond)
        elif self.q_grad_mode == "whole_grad":
            # Full gradient chain through all flow steps
            generated_actions_full = self._sample_actions_with_grad(
                obs_cond, grad_steps=self.num_flow_steps
            )
        elif self.q_grad_mode == "last_few":
            # Gradient through last N steps only
            generated_actions_full = self._sample_actions_with_grad(
                obs_cond, grad_steps=self.q_grad_steps
            )
        
        generated_actions_full = torch.clamp(generated_actions_full, -1.0, 1.0)
        
        # Truncate to act_horizon for Q-value computation
        actions_for_q = generated_actions_full[:, :act_horizon, :]
        
        # IMPORTANT: We need gradients to flow to policy (velocity_net) but NOT to critic
        # The Q-maximization gradient should only update the policy, not the critic
        # We achieve this by temporarily disabling gradients for critic during this forward pass
        for p in self.critic.parameters():
            p.requires_grad = False
        
        # Get Q-values from both Q-networks for normalization trick
        q1_value, q2_value = self.critic(actions_for_q, obs_cond)
        
        for p in self.critic.parameters():
            p.requires_grad = True
        
        # Q-loss normalization trick from original CPQL
        # Randomly select which Q to optimize and normalize by the other's magnitude
        # This prevents Q-value explosion in offline RL
        # Reference: https://github.com/cccedric/cpql/blob/main/agents/ql_cm.py#L157-L161
        if torch.rand(1).item() > 0.5:
            q_loss = -q1_value.mean() / q2_value.abs().mean().detach()
        else:
            q_loss = -q2_value.mean() / q1_value.abs().mean().detach()
        
        # Store q_mean for logging (use min of Q1 and Q2)
        q_mean = torch.min(q1_value, q2_value).mean()
        
        # Total policy loss
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
        # SMDP fields
        cumulative_reward: Optional[torch.Tensor] = None,
        chunk_done: Optional[torch.Tensor] = None,
        discount_factor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute critic loss using SMDP Bellman equation for action chunking.
        
        SMDP Bellman target:
        y_t = R_t^(τ) + (1 - d_t^(τ)) * γ^τ * min Q(s_{t+τ}, a')
        
        where:
        - R_t^(τ) = Σ_{i=0}^{τ-1} γ^i r_{t+i} (cumulative discounted reward)
        - d_t^(τ) = 1 if episode ends within chunk
        - γ^τ = discount factor over τ steps
        - s_{t+τ} = state after chunk execution
        
        Falls back to standard single-step TD if SMDP fields not provided.
        """
        # Use SMDP fields if provided, otherwise fall back to single-step
        if cumulative_reward is not None:
            r = cumulative_reward
            d = chunk_done if chunk_done is not None else dones
            gamma_tau = discount_factor if discount_factor is not None else torch.full_like(r, self.gamma)
        else:
            # Fall back to single-step TD (legacy behavior)
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
            # Sample next actions using EMA policy (full pred_horizon)
            next_actions_full = self._sample_actions_batch(next_obs_cond, use_ema=True)
            # Truncate to act_horizon to match Q-network input
            next_actions = next_actions_full[:, :self.act_horizon, :]
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_actions, next_obs_cond)
            target_q = torch.min(target_q1, target_q2)
            
            # SMDP TD target: R_t^(τ) + (1 - d_t^(τ)) * γ^τ * Q(s_{t+τ}, a')
            target_q = scaled_rewards + (1 - d) * gamma_tau * target_q
            
            # Clip targets
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
    
    def _sample_actions_with_grad(
        self,
        obs_cond: torch.Tensor,
        grad_steps: int,
    ) -> torch.Tensor:
        """Sample actions with gradient flowing through the last `grad_steps` flow steps.
        
        This implements configurable gradient chain length for Q-loss backpropagation.
        
        Args:
            obs_cond: Observation conditioning [B, global_cond_dim]
            grad_steps: Number of steps to keep gradients (from the end)
            
        Returns:
            actions: (B, pred_horizon, action_dim) generated actions with gradient
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        dt = 1.0 / self.num_flow_steps
        
        # Steps without gradient (first N - grad_steps)
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
        
        Directly predict velocity at t=0 and step to t=1 in one step.
        This is equivalent to predicting x_1 = x_0 + v(x_0, 0) * 1.0
        
        Args:
            obs_cond: Observation conditioning [B, global_cond_dim]
            
        Returns:
            actions: (B, pred_horizon, action_dim) generated actions
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        # Start from noise
        x_0 = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        
        # Single step: predict velocity at t=0, step to t=1
        t = torch.zeros(B, device=device)
        v = self.velocity_net(x_0, t, global_cond=obs_cond)
        x_1 = x_0 + v * 1.0  # dt = 1.0 for single step
        
        return x_1
    
    def update_ema(self):
        """Update EMA velocity network.
        
        Formula: θ_ema = ema_decay * θ_ema + (1 - ema_decay) * θ
        Equivalent to: soft_update(ema, source, tau=1-ema_decay)
        """
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
        """Sample action sequence for evaluation.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, global_cond_dim]
            use_ema: Whether to use EMA network for sampling
            
        Returns:
            actions: (B, pred_horizon, action_dim) action sequence
        """
        self.velocity_net.eval()
        
        # Flatten obs_features if 3D
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        else:
            obs_cond = obs_features
        
        action_seq = self._sample_actions_batch(obs_cond, use_ema=use_ema)
        
        self.velocity_net.train()
        return action_seq
