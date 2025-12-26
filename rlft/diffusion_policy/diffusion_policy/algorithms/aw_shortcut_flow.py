"""
Advantage-Weighted ShortCut Flow (AW-SCF) Agent

Combines ShortCut Flow with AWAC-style Q-weighted BC for stable offline RL.
This is the "bridge" between pure BC (ShortCut Flow) and online RL (ReinFlow).

Key design principles:
1. Policy stays on demo distribution (no OOD exploration)
2. Q is used to weight BC samples (not to maximize directly)
3. Critic learns within data distribution (bounded estimation error)
4. ShortCut structure is preserved for downstream ReinFlow fine-tuning

Three-stage pipeline:
- Stage 1: ShortCut Flow BC pretrain (pure BC)
- Stage 2: AW-ShortCut Flow offline RL (this module) 
- Stage 3: ReinFlow online RL (fine-tuning)

References:
- ShortCut Flow: shortcut_flow.py (local ODE solver approximation)
- AWAC: https://arxiv.org/abs/2006.09359
- IQL: https://arxiv.org/abs/2110.06169
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal, Union
import copy
import math

from .shortcut_flow import ShortCutVelocityUNet1D
from .networks import DoubleQNetwork, EnsembleQNetwork, soft_update


class AWShortCutFlowAgent(nn.Module):
    """Advantage-Weighted ShortCut Flow Agent.
    
    Combines ShortCut Flow (local ODE solver) with AWAC-style Q-weighting.
    Uses Q-values to weight BC samples instead of directly maximizing Q,
    which is more stable for offline RL with expert demonstrations.
    
    Supports both DoubleQNetwork (num_qs=2) and EnsembleQNetwork (num_qs>=2)
    for critic architecture. When using EnsembleQNetwork, it enables seamless
    checkpoint transfer to online RLPD training (AWSC Agent).
    
    Based on sweep results, recommended config for offline RL stage:
    - step_size_mode: "fixed" with fixed_step_size=0.0625 (1/16)
    - target_mode: "velocity" (not endpoint)
    - use_ema_teacher: True
    - flow_weight > shortcut_weight (e.g., 1.0 vs 0.3)
    - inference_mode: "uniform" (not adaptive)
    
    Args:
        velocity_net: ShortCutVelocityUNet1D for velocity prediction
        q_network: DoubleQNetwork or EnsembleQNetwork for Q-value estimation.
            EnsembleQNetwork recommended for downstream online RL fine-tuning.
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
        # ShortCut Flow specific (from sweep best practices)
        step_size_mode: Step size sampling mode (default: "fixed")
        fixed_step_size: Fixed step size (default: 0.0625)
        min_step_size: Minimum step size for uniform mode (default: 0.0625)
        max_step_size: Maximum step size for uniform mode (default: 0.125)
        target_mode: Shortcut target mode (default: "velocity")
        teacher_steps: Teacher rollout steps (default: 1)
        use_ema_teacher: Use EMA for teacher (default: True)
        t_min: Minimum t for time sampling (default: 0.0)
        t_max: Maximum t for time sampling (default: 1.0)
        inference_mode: Inference step mode (default: "uniform")
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
        # Offline RL parameters
        beta: float = 10.0,
        bc_weight: float = 1.0,
        shortcut_weight: float = 0.3,
        self_consistency_k: float = 0.1,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scale: float = 0.1,
        q_target_clip: float = 100.0,
        ema_decay: float = 0.999,
        weight_clip: float = 100.0,
        # ShortCut Flow parameters (from sweep best practices)
        step_size_mode: Literal["power2", "uniform", "fixed"] = "fixed",
        fixed_step_size: float = 0.0625,  # 1/16 (small step for reliable targets)
        min_step_size: float = 0.0625,
        max_step_size: float = 0.125,  # [1/16, 1/8] for uniform mode
        target_mode: Literal["velocity", "endpoint"] = "velocity",
        teacher_steps: int = 1,  # Single step for local approximation
        use_ema_teacher: bool = True,
        t_min: float = 0.0,
        t_max: float = 1.0,
        inference_mode: Literal["adaptive", "uniform"] = "uniform",
        device: str = "cuda",
    ):
        super().__init__()
        
        self.velocity_net = velocity_net
        self.critic = q_network
        # Track critic type for correct Q-value computation
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
        
        # Log2 of max steps for power2 mode
        self.log_max_steps = int(math.log2(max_denoising_steps))
        
        # EMA velocity network for shortcut targets and critic targets
        self.velocity_net_ema = copy.deepcopy(self.velocity_net)
        for p in self.velocity_net_ema.parameters():
            p.requires_grad = False
        
        # Target critic
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False
    
    def _sample_step_size(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample step sizes d based on step_size_mode."""
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
            actions_for_q = actions[:, :self.act_horizon]
        
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
        
        # Compute critic loss (SMDP Bellman)
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
        """Compute policy loss: Q-weighted Flow BC + Shortcut Consistency.
        
        Key difference from CPQL:
        - No direct Q maximization (no gradient from Q to policy)
        - Instead, use Q to weight BC samples (AWAC-style)
        
        Args:
            obs_cond: Observation conditioning [B, global_cond_dim]
            action_seq: Full pred_horizon actions for BC training [B, pred_horizon, action_dim]
            actions_for_q: act_horizon actions for Q-learning [B, act_horizon, action_dim]
        """
        B = action_seq.shape[0]
        device = action_seq.device
        
        # ===== Compute AWAC-style weights from Q-values =====
        with torch.no_grad():
            # Get Q-values for the dataset actions (no gradient needed)
            # Compatible with both DoubleQNetwork and EnsembleQNetwork
            if self._use_ensemble_q:
                # EnsembleQNetwork: use get_min_q for conservative estimate
                q_data = self.critic.get_min_q(actions_for_q, obs_cond, random_subset=True)
            else:
                # DoubleQNetwork: use min(q1, q2)
                q1_data, q2_data = self.critic(actions_for_q, obs_cond)
                q_data = torch.min(q1_data, q2_data)  # Conservative Q estimate
            
            # Compute advantage: A(s,a) = Q(s,a) - V(s) ≈ Q(s,a) - E[Q(s,a')]
            # Use batch mean as baseline (simple but effective)
            baseline = q_data.mean()
            advantage = q_data - baseline
            
            # AWAC-style exponential weights: w = exp(β * A)
            # Clamp to prevent numerical issues and outlier dominance
            weights = torch.clamp(torch.exp(self.beta * advantage), max=self.weight_clip)
            
            # Normalize weights to have mean 1 (helps stability)
            weights = weights / weights.mean()
            
            # Flatten weights for per-sample weighting [B]
            weights = weights.squeeze(-1)
        
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(action_seq)
        
        # Sample step sizes
        d = self._sample_step_size(B, device)
        
        # ===== Q-Weighted Flow Matching Loss =====
        t_flow = torch.rand(B, device=device)
        x_t = self._linear_interpolate(x_0, action_seq, t_flow)
        
        # Target velocity: v = x_1 - x_0
        v_target = action_seq - x_0
        
        # Predict velocity with step size d
        v_pred = self.velocity_net(x_t, t_flow, d, obs_cond)
        
        # Per-sample MSE loss [B, T, D] -> [B]
        flow_loss_per_sample = F.mse_loss(v_pred, v_target, reduction="none")
        flow_loss_per_sample = flow_loss_per_sample.mean(dim=(1, 2))  # [B]
        
        # Weighted average: higher Q samples get higher weight
        flow_loss = (weights * flow_loss_per_sample).mean()
        
        # ===== Q-Weighted Shortcut Consistency Loss =====
        shortcut_loss = torch.tensor(0.0, device=device)
        
        if self.shortcut_weight > 0 and self.self_consistency_k > 0:
            # Only compute for subset of batch (efficiency)
            n_consistency = max(1, int(B * self.self_consistency_k))
            idx = torch.randperm(B)[:n_consistency]
            
            x_0_sub = x_0[idx]
            actions_sub = action_seq[idx]
            obs_sub = obs_cond[idx]
            d_sub = d[idx]
            weights_sub = weights[idx]
            
            # Sample time for consistency (ensure room for 2d)
            t_cons = self.t_min + torch.rand(n_consistency, device=device) * (self.t_max - self.t_min)
            
            # Ensure t + 2d <= 1
            d_double = 2 * d_sub
            max_t = (1.0 - d_double).clamp(min=self.t_min)
            t_cons = torch.min(t_cons, max_t)
            
            # Interpolate
            x_t_cons = self._linear_interpolate(x_0_sub, actions_sub, t_cons)
            
            # Only train where 2d is valid
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
                
                # Compute shortcut target using teacher
                shortcut_target = self._compute_shortcut_target(
                    x_t_valid, t_valid, d_valid, obs_valid, x_0_valid, actions_valid
                )
                
                # Predict with 2d step size
                v_pred_2d = self.velocity_net(x_t_valid, t_valid, d_double_valid, obs_valid)
                
                if self.target_mode == "velocity":
                    # Per-sample loss
                    shortcut_loss_per_sample = F.mse_loss(v_pred_2d, shortcut_target, reduction="none")
                    shortcut_loss_per_sample = shortcut_loss_per_sample.mean(dim=(1, 2))  # [n_valid]
                else:  # endpoint
                    d_double_expand = d_double_valid.view(-1, 1, 1)
                    pred_endpoint = x_t_valid + d_double_expand * v_pred_2d
                    shortcut_loss_per_sample = F.mse_loss(pred_endpoint, shortcut_target, reduction="none")
                    shortcut_loss_per_sample = shortcut_loss_per_sample.mean(dim=(1, 2))
                
                # Weighted average
                shortcut_loss = (weights_valid * shortcut_loss_per_sample).mean()
        
        # Total policy loss
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
        
        For velocity mode: target is the equivalent velocity for 2d step
        For endpoint mode: target is the endpoint after 2d steps
        """
        teacher_net = self.velocity_net_ema if self.use_ema_teacher else self.velocity_net
        
        with torch.no_grad():
            d_expand = d.view(-1, 1, 1)
            
            if self.teacher_steps == 1:
                # Two-step teacher: x_t -> x_{t+d} -> x_{t+2d}
                v_1 = teacher_net(x_t, t, d, obs_cond)
                x_t_plus_d = x_t + d_expand * v_1
                
                t_plus_d = t + d
                v_2 = teacher_net(x_t_plus_d, t_plus_d, d, obs_cond)
                
                target_endpoint = x_t + d_expand * v_1 + d_expand * v_2
                shortcut_v = (v_1 + v_2) / 2  # Average velocity for 2d step
            else:
                # Multi-step rollout
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
        """Compute critic loss using SMDP Bellman equation."""
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
            
            # Compute target Q-values (conservative estimate)
            # Compatible with both DoubleQNetwork and EnsembleQNetwork
            if self._use_ensemble_q:
                target_q = self.critic_target.get_min_q(next_actions, next_obs_cond, random_subset=True)
            else:
                target_q1, target_q2 = self.critic_target(next_actions, next_obs_cond)
                target_q = torch.min(target_q1, target_q2)
            
            # TD target with SMDP discount
            target_q = scaled_rewards + (1 - d) * gamma_tau * target_q
            
            if self.q_target_clip is not None:
                target_q = torch.clamp(target_q, -self.q_target_clip, self.q_target_clip)
        
        # Current Q-values and critic loss
        # Compatible with both DoubleQNetwork and EnsembleQNetwork
        if self._use_ensemble_q:
            # EnsembleQNetwork: compute loss over all Q-networks
            q_all = self.critic(action_seq, obs_cond)  # (num_qs, B, 1)
            # Mean squared error across all Q-networks
            critic_loss = F.mse_loss(q_all, target_q.unsqueeze(0).expand_as(q_all))
        else:
            # DoubleQNetwork: traditional double Q loss
            current_q1, current_q2 = self.critic(action_seq, obs_cond)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        return critic_loss
    
    def _sample_actions_batch(
        self,
        obs_cond: torch.Tensor,
        use_ema: bool = False
    ) -> torch.Tensor:
        """Sample actions using flow ODE integration."""
        B = obs_cond.shape[0]
        device = obs_cond.device
        net = self.velocity_net_ema if use_ema else self.velocity_net
        
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        
        # Use uniform steps for sampling (from sweep: uniform > adaptive)
        dt = 1.0 / self.num_inference_steps
        d = torch.full((B,), dt, device=device)
        
        for i in range(self.num_inference_steps):
            t = torch.full((B,), i * dt, device=device)
            v = net(x, t, d, obs_cond)
            x = x + dt * v
        
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
        """Sample action sequence for evaluation.
        
        Args:
            obs_features: Encoded observation features
            use_ema: Whether to use EMA network (default: True)
            
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
        
        # Start from noise
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        
        # Use inference mode from config
        if self.inference_mode == "adaptive":
            # Adaptive step sizes (power of 2)
            t = torch.zeros(B, device=device)
            
            while t[0] < 1.0:
                remaining = 1.0 - t[0]
                
                # Find largest valid power-of-2 step
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
            # Uniform steps (recommended from sweep)
            dt = 1.0 / self.num_inference_steps
            d = torch.full((B,), dt, device=device)
            
            for i in range(self.num_inference_steps):
                t = torch.full((B,), i * dt, device=device)
                v = net(x, t, d, obs_cond)
                x = x + dt * v
        
        x = torch.clamp(x, -1.0, 1.0)
        
        self.velocity_net.train()
        return x
