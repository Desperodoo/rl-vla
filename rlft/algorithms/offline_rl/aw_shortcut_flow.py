"""
Advantage-Weighted ShortCut Flow (AW-SCF / AWSC) Agent

Combines ShortCut Flow with AWAC-style Q-weighted BC for stable offline RL.
This is the "bridge" between pure BC (ShortCut Flow) and online RL (ReinFlow).

Key Algorithm Design:
=====================
AW-ShortCut Flow (AWSC) extends AWCP by replacing standard CFM with ShortCut Flow:
- Standard CFM: v(x_t, t) predicts velocity, requires many steps for sampling
- ShortCut Flow: v(x_t, t, d) predicts velocity conditioned on step size d
  Learns to take larger steps when possible, enabling faster inference

Key design principles:
1. Policy stays on demo distribution (no OOD exploration)
2. Q is used to weight BC samples (not to maximize directly)
3. Critic learns within data distribution (bounded estimation error)
4. ShortCut structure is preserved for downstream ReinFlow fine-tuning

ShortCut Flow Training:
- Flow loss: Standard CFM with step-size conditioning
- Shortcut loss: Self-consistency for larger step sizes
  v(x_t, t, 2d) ≈ average of two d-steps from teacher

Three-stage Pipeline (Recommended):
- Stage 1: ShortCut Flow BC pretrain (pure BC, shortcut_flow.py)
- Stage 2: AW-ShortCut Flow offline RL (this module)
- Stage 3: ReinFlow online RL (fine-tuning with rewards)

ShortCut Flow Parameters (from sweep best practices):
- step_size_mode: "fixed" with fixed_step_size=0.0625 (1/16)
  Small fixed step is more stable than random/power2
- target_mode: "velocity" (not endpoint)
  Velocity-space target has better gradient flow
- use_ema_teacher: True
  EMA teacher provides stable shortcut targets
- teacher_steps: 1 (single step preserves locality)
- flow_weight > shortcut_weight (e.g., 1.0 vs 0.3)
- inference_mode: "uniform" (not adaptive)
  Uniform steps avoid solver mismatch

Critic Architecture Options:
- DoubleQNetwork: Standard double Q-learning (num_qs=2)
- EnsembleQNetwork: Ensemble Q-learning (num_qs>=2)
  Enables seamless checkpoint transfer to online RLPD training

References:
- ShortCut Flow: shortcut_flow.py (local ODE solver approximation)
- AWAC: https://arxiv.org/abs/2006.09359
- IQL: https://arxiv.org/abs/2110.06169
- ReinFlow: For downstream online RL fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal, Union
import copy
import math

from rlft.networks import ShortCutVelocityUNet1D, DoubleQNetwork, EnsembleQNetwork, soft_update


class AWShortCutFlowAgent(nn.Module):
    """Advantage-Weighted ShortCut Flow Agent.
    
    Combines ShortCut Flow (local ODE solver) with AWAC-style Q-weighting.
    Uses Q-values to weight BC samples instead of directly maximizing Q.
    
    Algorithm Overview:
    ------------------
    Similar to AWCP but uses ShortCut Flow instead of standard CFM:
    - velocity_net(x_t, t, d) predicts velocity conditioned on step size d
    - Shortcut consistency loss enables larger steps at inference
    - Q-values weight both flow loss and shortcut loss
    
    Supports both DoubleQNetwork (num_qs=2) and EnsembleQNetwork (num_qs>=2)
    for critic architecture. When using EnsembleQNetwork, it enables seamless
    checkpoint transfer to online RLPD training (AWSC Agent).
    
    Hyperparameter Recommendations (from sweep):
    - beta: 10.0 (advantage temperature)
    - bc_weight: 1.0 (flow matching loss weight)
    - shortcut_weight: 0.3 (shortcut consistency loss weight)
      Lower than flow weight for stability
    - self_consistency_k: 0.1 (fraction of batch for consistency)
    - step_size_mode: "fixed" with fixed_step_size=0.0625 (1/16)
    - target_mode: "velocity" (not endpoint)
    - teacher_steps: 1 (single step preserves locality)
    - inference_mode: "uniform" (not adaptive)
    
    Args:
        velocity_net: ShortCutVelocityUNet1D for velocity prediction
            Takes (x_t, t, d, obs_cond) where d is step size
        q_network: DoubleQNetwork or EnsembleQNetwork for Q-value estimation
            EnsembleQNetwork recommended for downstream online RL
        action_dim: Dimension of action space
        obs_horizon: Number of observation frames (default: 2)
        pred_horizon: Length of action sequence to predict (default: 16)
        act_horizon: Length of action sequence for Q-learning (default: 8)
        max_denoising_steps: Maximum denoising steps (default: 8)
        num_inference_steps: Number of inference steps (default: 8)
        beta: Temperature for advantage weighting (default: 10.0)
        bc_weight: Weight for flow matching loss (default: 1.0)
        shortcut_weight: Weight for shortcut consistency loss (default: 0.3)
        self_consistency_k: Fraction of batch for consistency (default: 0.1)
        gamma: Discount factor (default: 0.99)
        tau: Soft update coefficient (default: 0.005)
        reward_scale: Scale factor for rewards (default: 0.1)
        q_target_clip: Clip range for Q target (default: 100.0)
        ema_decay: Decay rate for EMA velocity network (default: 0.999)
        weight_clip: Maximum weight to prevent outliers (default: 100.0)
        step_size_mode: Step size sampling mode:
            - "fixed": Use fixed_step_size (recommended)
            - "uniform": Sample from [min_step_size, max_step_size]
            - "power2": Sample from {1/N, 2/N, 4/N, ...}
        fixed_step_size: Fixed step size (default: 0.0625 = 1/16)
        min_step_size: Minimum step size for uniform mode (default: 0.0625)
        max_step_size: Maximum step size for uniform mode (default: 0.125)
        target_mode: Shortcut target mode:
            - "velocity": Target is average velocity over 2d
            - "endpoint": Target is endpoint after 2d steps
        teacher_steps: Teacher rollout steps (default: 1)
        use_ema_teacher: Use EMA for teacher (default: True)
        t_min: Minimum t for time sampling (default: 0.0)
        t_max: Maximum t for time sampling (default: 1.0)
        inference_mode: Inference step mode:
            - "uniform": Fixed step size 1/num_inference_steps
            - "adaptive": Use power-of-2 adaptive steps
        device: Device to run on (default: "cuda")
    """
    
    def __init__(
        self,
        velocity_net: ShortCutVelocityUNet1D,
        q_network: Union[DoubleQNetwork, EnsembleQNetwork],
        action_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        act_horizon: int = 8,
        max_denoising_steps: int = 8,
        num_inference_steps: int = 8,
        # Offline RL parameters (best from sweep: aggressive config)
        beta: float = 10.0,  # Best from sweep (aggressive)
        bc_weight: float = 1.0,
        shortcut_weight: float = 0.3,  # Best from sweep (consistency_weight)
        self_consistency_k: float = 0.25,  # Best from sweep
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scale: float = 0.1,  # Best from sweep
        q_target_clip: float = 100.0,
        ema_decay: float = 0.999,
        weight_clip: float = 100.0,
        # ShortCut Flow parameters (best from sweep)
        step_size_mode: Literal["power2", "uniform", "fixed"] = "fixed",
        fixed_step_size: float = 0.125,  # Best from sweep (1/8)
        min_step_size: float = 0.0625,
        max_step_size: float = 0.125,
        target_mode: Literal["velocity", "endpoint"] = "velocity",
        teacher_steps: int = 1,
        use_ema_teacher: bool = True,
        t_min: float = 0.0,
        t_max: float = 1.0,
        inference_mode: Literal["adaptive", "uniform"] = "uniform",
        action_bounds: Optional[tuple] = None,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.velocity_net = velocity_net
        self.critic = q_network
        self._use_ensemble_q = isinstance(q_network, EnsembleQNetwork)
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.max_denoising_steps = max_denoising_steps
        self.num_inference_steps = num_inference_steps
        self.device = device
        
        # Offline RL hyperparameters
        self.beta = beta
        self.bc_weight = bc_weight
        self.shortcut_weight = shortcut_weight
        self.self_consistency_k = self_consistency_k
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.q_target_clip = q_target_clip
        self.ema_decay = ema_decay
        self.weight_clip = weight_clip
        
        # ShortCut Flow parameters
        self.step_size_mode = step_size_mode
        self.fixed_step_size = fixed_step_size
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.target_mode = target_mode
        self.teacher_steps = teacher_steps
        self.use_ema_teacher = use_ema_teacher
        self.t_min = t_min
        self.t_max = t_max
        self.inference_mode = inference_mode
        self.action_bounds = action_bounds
        
        self.log_max_steps = int(math.log2(max_denoising_steps))
        
        # EMA velocity network
        self.velocity_net_ema = copy.deepcopy(self.velocity_net)
        for p in self.velocity_net_ema.parameters():
            p.requires_grad = False
        
        # Target critic
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False
    
    def _sample_step_size(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample step sizes d based on step_size_mode.
        
        Modes:
        - power2: d from {1/N, 2/N, 4/N, ...} where N is max_denoising_steps (log-uniform)
        - uniform: d ~ U[min_step_size, max_step_size]
        - fixed: d = fixed_step_size (recommended, most stable)
        
        Args:
            batch_size: Number of samples
            device: Device to create tensor on
            
        Returns:
            d: Step sizes [batch_size]
        """
        if self.step_size_mode == "power2":
            powers = torch.randint(0, self.log_max_steps + 1, (batch_size,), device=device)
            d = (2.0 ** powers.float()) / self.max_denoising_steps
        elif self.step_size_mode == "uniform":
            d = self.min_step_size + torch.rand(batch_size, device=device) * (self.max_step_size - self.min_step_size)
        elif self.step_size_mode == "fixed":
            d = torch.full((batch_size,), self.fixed_step_size, device=device)
        else:
            raise ValueError(f"Unknown step_size_mode: {self.step_size_mode}")
        return d
    
    def _linear_interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation: x_t = (1 - t) * x_0 + t * x_1"""
        t_expand = t.view(-1, 1, 1)
        return (1 - t_expand) * x_0 + t_expand * x_1
    
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
        
        Policy loss = bc_weight * weighted_flow_loss + shortcut_weight * weighted_shortcut_loss
        
        Both losses are weighted by Q-derived sample weights (AWAC-style).
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, global_cond_dim]
            actions: (B, pred_horizon, action_dim) expert action sequence for BC training
            rewards: (B,) or (B, 1) single-step rewards
            next_obs_features: Next observation features
            dones: (B,) or (B, 1) done flags
            actions_for_q: (B, act_horizon, action_dim) action sequence for Q-learning
            cumulative_reward: (B,) SMDP cumulative discounted reward (optional)
            chunk_done: (B,) SMDP done flag (optional)
            discount_factor: (B,) SMDP discount γ^τ (optional)
            
        Returns:
            Dict with:
                - loss: Total loss (for logging)
                - actor_loss / policy_loss: Policy loss for actor backward
                - flow_loss: Flow matching loss component
                - shortcut_loss: Shortcut consistency loss component
                - critic_loss: Critic loss for critic backward
                - q_mean: Mean Q-value
                - weight_mean, weight_std, advantage_mean: Weight statistics
        """
        if actions_for_q is None:
            actions_for_q = actions[:, :self.act_horizon]
        
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        else:
            obs_cond = obs_features
            
        if next_obs_features.dim() == 3:
            next_obs_cond = next_obs_features.reshape(next_obs_features.shape[0], -1)
        else:
            next_obs_cond = next_obs_features
        
        policy_dict = self._compute_policy_loss(obs_cond, actions, actions_for_q)
        
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
            "shortcut_loss": policy_dict["shortcut_loss"],
            "critic_loss": critic_loss,
            "q_mean": policy_dict["q_mean"],
            "weight_mean": policy_dict["weight_mean"],
            "weight_std": policy_dict["weight_std"],
            "advantage_mean": policy_dict["advantage_mean"],
        }
    
    def _compute_policy_loss(
        self, 
        obs_cond: torch.Tensor, 
        action_seq: torch.Tensor,
        actions_for_q: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute policy loss: Q-weighted Flow BC + Q-weighted Shortcut Consistency.
        
        Key differences from AWCP:
        - velocity_net takes additional step_size argument d
        - Shortcut consistency loss instead of endpoint consistency loss
        
        Flow Loss (with step-size conditioning):
            For each sample, predict velocity with step size d:
            flow_loss_i = ||v(x_t, t, d) - (x_1 - x_0)||^2
            flow_loss = Σ w_i * flow_loss_i
        
        Shortcut Consistency Loss:
            Teacher: Two d-steps from x_t to get endpoint
                x_{t+d} = x_t + d * v_ema(x_t, t, d)
                x_{t+2d} = x_{t+d} + d * v_ema(x_{t+d}, t+d, d)
            Student: One 2d-step
                v_pred = v(x_t, t, 2d)
            Target: shortcut_v = (v_1 + v_2) / 2 (for velocity mode)
            shortcut_loss = Σ w_i * ||v_pred - shortcut_v||^2
        
        Args:
            obs_cond: Observation conditioning [B, global_cond_dim]
            action_seq: Full pred_horizon actions for BC [B, pred_horizon, action_dim]
            actions_for_q: act_horizon actions for Q [B, act_horizon, action_dim]
            
        Returns:
            Dict with policy_loss, flow_loss, shortcut_loss, q_mean, weight stats
        """
        B = action_seq.shape[0]
        device = action_seq.device
        
        # ===== Compute AWAC-style weights from Q-values =====
        with torch.no_grad():
            if self._use_ensemble_q:
                q_data = self.critic.get_min_q(actions_for_q, obs_cond, random_subset=True)
            else:
                q1_data, q2_data = self.critic(actions_for_q, obs_cond)
                q_data = torch.min(q1_data, q2_data)
            
            baseline = q_data.mean()
            advantage = q_data - baseline
            
            weights = torch.clamp(torch.exp(self.beta * advantage), max=self.weight_clip)
            weights = weights / weights.mean()
            weights = weights.squeeze(-1)
        
        x_0 = torch.randn_like(action_seq)
        d = self._sample_step_size(B, device)
        
        # ===== Q-Weighted Flow Matching Loss =====
        t_flow = torch.rand(B, device=device)
        x_t = self._linear_interpolate(x_0, action_seq, t_flow)
        v_target = action_seq - x_0
        v_pred = self.velocity_net(x_t, t_flow, d, obs_cond)
        
        flow_loss_per_sample = F.mse_loss(v_pred, v_target, reduction="none")
        flow_loss_per_sample = flow_loss_per_sample.mean(dim=(1, 2))
        flow_loss = (weights * flow_loss_per_sample).mean()
        
        # ===== Q-Weighted Shortcut Consistency Loss =====
        shortcut_loss = torch.tensor(0.0, device=device)
        
        if self.shortcut_weight > 0 and self.self_consistency_k > 0:
            n_consistency = max(1, int(B * self.self_consistency_k))
            idx = torch.randperm(B)[:n_consistency]
            
            x_0_sub = x_0[idx]
            actions_sub = action_seq[idx]
            obs_sub = obs_cond[idx]
            d_sub = d[idx]
            weights_sub = weights[idx]
            
            t_cons = self.t_min + torch.rand(n_consistency, device=device) * (self.t_max - self.t_min)
            
            d_double = 2 * d_sub
            max_t = (1.0 - d_double).clamp(min=self.t_min)
            t_cons = torch.min(t_cons, max_t)
            
            x_t_cons = self._linear_interpolate(x_0_sub, actions_sub, t_cons)
            
            valid_mask = (t_cons + d_double) <= 1.0
            
            if valid_mask.sum() > 0:
                x_t_valid = x_t_cons[valid_mask]
                t_valid = t_cons[valid_mask]
                d_valid = d_sub[valid_mask]
                d_double_valid = d_double[valid_mask]
                obs_valid = obs_sub[valid_mask]
                x_0_valid = x_0_sub[valid_mask]
                actions_valid = actions_sub[valid_mask]
                weights_valid = weights_sub[valid_mask]
                
                shortcut_target = self._compute_shortcut_target(
                    x_t_valid, t_valid, d_valid, obs_valid, x_0_valid, actions_valid
                )
                
                v_pred_2d = self.velocity_net(x_t_valid, t_valid, d_double_valid, obs_valid)
                
                if self.target_mode == "velocity":
                    shortcut_loss_per_sample = F.mse_loss(v_pred_2d, shortcut_target, reduction="none")
                    shortcut_loss_per_sample = shortcut_loss_per_sample.mean(dim=(1, 2))
                else:
                    d_double_expand = d_double_valid.view(-1, 1, 1)
                    pred_endpoint = x_t_valid + d_double_expand * v_pred_2d
                    shortcut_loss_per_sample = F.mse_loss(pred_endpoint, shortcut_target, reduction="none")
                    shortcut_loss_per_sample = shortcut_loss_per_sample.mean(dim=(1, 2))
                
                shortcut_loss = (weights_valid * shortcut_loss_per_sample).mean()
        
        policy_loss = self.bc_weight * flow_loss + self.shortcut_weight * shortcut_loss
        
        return {
            "policy_loss": policy_loss,
            "flow_loss": flow_loss,
            "shortcut_loss": shortcut_loss,
            "q_mean": q_data.mean(),
            "weight_mean": weights.mean(),
            "weight_std": weights.std(),
            "advantage_mean": advantage.mean(),
        }
    
    def _compute_shortcut_target(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        d: torch.Tensor,
        obs_cond: torch.Tensor,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute shortcut target using teacher network.
        
        The shortcut target is what the student should predict for step size 2d,
        derived from the teacher's two d-step predictions.
        
        For teacher_steps=1 (recommended):
            Step 1: v_1 = v_ema(x_t, t, d)
                    x_{t+d} = x_t + d * v_1
            Step 2: v_2 = v_ema(x_{t+d}, t+d, d)
            
            If target_mode="velocity":
                shortcut_target = (v_1 + v_2) / 2
            If target_mode="endpoint":
                shortcut_target = x_t + d * v_1 + d * v_2
        
        For teacher_steps>1:
            Takes multiple d-steps, computes equivalent velocity
        
        Args:
            x_t: Current noisy state at time t [B, T, D]
            t: Current time [B]
            d: Current step size [B]
            obs_cond: Observation conditioning [B, cond_dim]
            x_0: Initial noise (unused but kept for API consistency)
            x_1: Target actions (unused but kept for API consistency)
            
        Returns:
            shortcut_target: Target velocity or endpoint depending on target_mode
        """
        teacher_net = self.velocity_net_ema if self.use_ema_teacher else self.velocity_net
        
        with torch.no_grad():
            d_expand = d.view(-1, 1, 1)
            
            if self.teacher_steps == 1:
                v_1 = teacher_net(x_t, t, d, obs_cond)
                x_t_plus_d = x_t + d_expand * v_1
                
                t_plus_d = t + d
                v_2 = teacher_net(x_t_plus_d, t_plus_d, d, obs_cond)
                
                target_endpoint = x_t + d_expand * v_1 + d_expand * v_2
                shortcut_v = (v_1 + v_2) / 2
            else:
                x = x_t.clone()
                current_t = t.clone()
                
                for step in range(self.teacher_steps):
                    v = teacher_net(x, current_t, d, obs_cond)
                    x = x + d_expand * v
                    current_t = current_t + d
                
                target_endpoint = x
                shortcut_v = (target_endpoint - x_t) / (self.teacher_steps * d_expand)
            
            if self.target_mode == "velocity":
                return shortcut_v
            else:
                return target_endpoint
    
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
        
        Supports both DoubleQNetwork and EnsembleQNetwork.
        
        For DoubleQNetwork:
            critic_loss = MSE(Q1, target) + MSE(Q2, target)
        
        For EnsembleQNetwork:
            critic_loss = MSE(Q_all, target) averaged over ensemble
        
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
            critic_loss: Combined critic loss
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
            
            if self._use_ensemble_q:
                target_q = self.critic_target.get_min_q(next_actions, next_obs_cond, random_subset=True)
            else:
                target_q1, target_q2 = self.critic_target(next_actions, next_obs_cond)
                target_q = torch.min(target_q1, target_q2)
            
            target_q = scaled_rewards + (1 - d) * gamma_tau * target_q
            
            if self.q_target_clip is not None:
                target_q = torch.clamp(target_q, -self.q_target_clip, self.q_target_clip)
        
        if self._use_ensemble_q:
            q_all = self.critic(action_seq, obs_cond)
            critic_loss = F.mse_loss(q_all, target_q.unsqueeze(0).expand_as(q_all))
        else:
            current_q1, current_q2 = self.critic(action_seq, obs_cond)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        return critic_loss
    
    def _sample_actions_batch(
        self,
        obs_cond: torch.Tensor,
        use_ema: bool = False
    ) -> torch.Tensor:
        """Sample actions using ShortCut flow ODE integration.
        
        Uses uniform step sizes (not adaptive) as recommended from sweep.
        
        Args:
            obs_cond: Observation conditioning [B, global_cond_dim]
            use_ema: Whether to use EMA velocity network
            
        Returns:
            actions: Generated action sequence [B, pred_horizon, action_dim], clamped to [-1, 1]
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        net = self.velocity_net_ema if use_ema else self.velocity_net
        
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        
        # Uniform step size (from sweep: uniform > adaptive)
        dt = 1.0 / self.num_inference_steps
        d = torch.full((B,), dt, device=device)
        
        for i in range(self.num_inference_steps):
            t = torch.full((B,), i * dt, device=device)
            v = net(x, t, d, obs_cond)
            x = x + dt * v
        
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
        
        Supports both uniform and adaptive inference modes.
        
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
        
        net = self.velocity_net_ema if use_ema else self.velocity_net
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        
        if self.inference_mode == "adaptive":
            t = torch.zeros(B, device=device)
            
            while t[0] < 1.0:
                remaining = 1.0 - t[0]
                
                d_val = min(remaining.item(), self.max_step_size)
                for power in range(self.log_max_steps, -1, -1):
                    candidate = (2.0 ** power) / self.max_denoising_steps
                    if candidate <= remaining and candidate >= self.min_step_size:
                        d_val = candidate
                        break
                
                d = torch.full((B,), d_val, device=device)
                v = net(x, t, d, obs_cond)
                x = x + d.view(-1, 1, 1) * v
                t = t + d
                
                if d_val < 1e-6:
                    break
        else:
            dt = 1.0 / self.num_inference_steps
            d = torch.full((B,), dt, device=device)
            
            for i in range(self.num_inference_steps):
                t = torch.full((B,), i * dt, device=device)
                v = net(x, t, d, obs_cond)
                x = x + dt * v
        
        # Only clamp if action_bounds is explicitly set
        if self.action_bounds is not None:
            x = torch.clamp(x, self.action_bounds[0], self.action_bounds[1])
        
        self.velocity_net.train()
        return x

    @torch.no_grad()
    def get_action_deterministic(
        self,
        obs_features: torch.Tensor,
        use_ema: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Sample action sequence deterministically (starting from zero).
        
        Unlike get_action() which starts from random noise, this method starts
        from zeros to produce deterministic outputs given the same observation.
        This is useful for:
        - Evaluation with reduced variance
        - Debugging and reproducibility
        - Comparing algorithm behavior
        
        Note: For flow-based models, starting from zeros produces a "mean-like"
        trajectory through the flow field, though it may be OOD for the learned
        velocity field which was trained with Gaussian noise inputs.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, global_cond_dim]
            use_ema: Whether to use EMA network for sampling (recommended)
            
        Returns:
            actions: Generated action sequence [B, pred_horizon, action_dim], clamped to [-1, 1]
        """
        self.velocity_net.eval()
        
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        else:
            obs_cond = obs_features
        
        net = self.velocity_net_ema if use_ema else self.velocity_net
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        # Start from zeros (deterministic)
        x = torch.zeros(B, self.pred_horizon, self.action_dim, device=device)
        
        if self.inference_mode == "adaptive":
            t = torch.zeros(B, device=device)
            
            while t[0] < 1.0:
                remaining = 1.0 - t[0]
                
                d_val = min(remaining.item(), self.max_step_size)
                for power in range(self.log_max_steps, -1, -1):
                    candidate = (2.0 ** power) / self.max_denoising_steps
                    if candidate <= remaining and candidate >= self.min_step_size:
                        d_val = candidate
                        break
                
                d = torch.full((B,), d_val, device=device)
                v = net(x, t, d, obs_cond)
                x = x + d.view(-1, 1, 1) * v
                t = t + d
                
                if d_val < 1e-6:
                    break
        else:
            dt = 1.0 / self.num_inference_steps
            d = torch.full((B,), dt, device=device)
            
            for i in range(self.num_inference_steps):
                t = torch.full((B,), i * dt, device=device)
                v = net(x, t, d, obs_cond)
                x = x + dt * v
        
        # Only clamp if action_bounds is explicitly set
        if self.action_bounds is not None:
            x = torch.clamp(x, self.action_bounds[0], self.action_bounds[1])
        return x
