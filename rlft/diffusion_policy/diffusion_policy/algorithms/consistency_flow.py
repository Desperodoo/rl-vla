"""
Consistency Flow Policy with self-consistency loss.
Migrated from toy_diffusion_rl to diffusion_policy framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal
import copy

from .networks import VelocityUNet1D, soft_update


class ConsistencyFlowAgent(nn.Module):
    """
    Consistency Flow Matching agent.
    Combines flow matching with consistency loss for faster inference.
    Uses EMA teacher for stable consistency targets.
    """
    
    def __init__(
        self,
        velocity_net: VelocityUNet1D,
        action_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        num_flow_steps: int = 10,
        flow_weight: float = 1.0,
        consistency_weight: float = 1.0,
        ema_decay: float = 0.999,
        consistency_delta: float = 0.01,  # Small time step for consistency
        # Consistency design toggles
        cons_use_flow_t: bool = False,
        cons_full_t_range: bool = False,
        cons_t_min: float = 0.05,
        cons_t_max: float = 0.95,
        cons_t_upper: float = 0.95,
        cons_delta_mode: Literal["random", "fixed"] = "random",
        cons_delta_min: float = 0.02,
        cons_delta_max: float = 0.15,
        cons_delta_fixed: float = 0.01,
        cons_delta_dynamic_max: bool = False,
        cons_delta_cap: float = 0.99,
        teacher_steps: int = 2,
        teacher_from: Literal["t_plus", "t_cons"] = "t_plus",
        student_point: Literal["t_plus", "t_cons"] = "t_plus",
        consistency_loss_space: Literal["velocity", "endpoint"] = "velocity",
        action_bounds: Optional[tuple] = None,  # (min, max) for action clipping, None to disable
        device: str = "cuda",
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.velocity_net_ema = copy.deepcopy(velocity_net)
        
        # Freeze EMA model
        for param in self.velocity_net_ema.parameters():
            param.requires_grad = False
            
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.num_flow_steps = num_flow_steps
        self.flow_weight = flow_weight
        self.consistency_weight = consistency_weight
        self.ema_decay = ema_decay
        self.consistency_delta = consistency_delta  # Not used anymore, kept for API compatibility
        self.device = device
        
        # Consistency loss hyperparameters (aligned with best practices)
        self.cons_use_flow_t = cons_use_flow_t
        self.cons_full_t_range = cons_full_t_range
        self.cons_delta_mode = cons_delta_mode
        self.t_min = cons_t_min  # Avoid boundary instability at t≈0
        self.t_max = cons_t_max  # Avoid boundary instability at t≈1
        self.t_upper = cons_t_upper  # Clamp for t_plus
        self.delta_min = cons_delta_min  # Minimum delta for consistency
        self.delta_max = cons_delta_max  # Maximum delta (avoid large teacher error)
        self.delta_fixed = cons_delta_fixed
        self.delta_dynamic_max = cons_delta_dynamic_max
        self.delta_cap = cons_delta_cap
        self.teacher_steps = teacher_steps  # Teacher rollout steps to 1
        self.teacher_from = teacher_from
        self.student_point = student_point
        self.consistency_loss_space = consistency_loss_space
        self.action_bounds = action_bounds  # None means no clipping
        
    def update_ema(self):
        """Update EMA velocity network.
        
        Formula: θ_ema = ema_decay * θ_ema + (1 - ema_decay) * θ
        Equivalent to: soft_update(ema, source, tau=1-ema_decay)
        """
        soft_update(self.velocity_net_ema, self.velocity_net, 1 - self.ema_decay)
    
    def _get_consistency_targets(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        obs_features: torch.Tensor,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """
        Compute consistency target by integrating from t to 1 using EMA model.
        
        Args:
            x_t: Current state at time t
            t: Current time
            obs_features: Observation conditioning
            num_steps: Number of integration steps to take
        """
        x = x_t.clone()
        current_t = t.clone()
        
        remaining_time = 1.0 - current_t
        dt = remaining_time / num_steps
        
        for _ in range(num_steps):
            v = self.velocity_net_ema(x, current_t, obs_features)
            # Expand dt for broadcasting
            dt_expand = dt.view(-1, 1, 1)
            x = x + v * dt_expand
            current_t = current_t + dt
            
        return x
    
    def compute_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mixed flow matching + consistency loss.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim]
            actions: Expert actions [B, pred_horizon, action_dim]
        """
        batch_size = actions.shape[0]
        
        # Sample noise
        x_0 = torch.randn_like(actions)
        
        # Sample time uniformly
        t_flow = torch.rand(batch_size, device=actions.device)
        
        # Interpolate
        t_flow_expand = t_flow.view(-1, 1, 1)
        x_t_flow = (1 - t_flow_expand) * x_0 + t_flow_expand * actions
        
        # Flow matching target
        v_target = actions - x_0
        
        # Predict velocity
        v_pred = self.velocity_net(x_t_flow, t_flow, obs_features)
        
        # Flow matching loss
        flow_loss = F.mse_loss(v_pred, v_target)
        
        # Consistency loss: predictions from t and t+delta should reach same endpoint
        # Using velocity space loss for gradient stability
        consistency_loss = torch.tensor(0.0, device=actions.device)
        if self.consistency_weight > 0:
            # Choose sampling range
            t_low = 0.0 if self.cons_full_t_range else self.t_min
            t_high = 1.0 if self.cons_full_t_range else self.t_max
            t_upper = self.t_upper if self.t_upper is not None else t_high
            
            # Optionally reuse flow t (clamped) or resample for consistency branch
            if self.cons_use_flow_t:
                t_cons = torch.clamp(t_flow, min=t_low, max=t_high)
                t_cons_expand = t_cons.view(-1, 1, 1)
                x_t_cons = (1 - t_cons_expand) * x_0 + t_cons_expand * actions
            else:
                range_span = max(t_high - t_low, 1e-6)
                t_cons = t_low + torch.rand(batch_size, device=actions.device) * range_span
                t_cons_expand = t_cons.view(-1, 1, 1)
                x_t_cons = (1 - t_cons_expand) * x_0 + t_cons_expand * actions
            
            # Delta strategy: random or fixed, optional dynamic cap
            if self.cons_delta_mode == "fixed":
                delta_t = torch.full_like(t_cons, self.delta_fixed)
            else:
                if self.delta_dynamic_max:
                    delta_high = torch.clamp(self.delta_cap - t_cons, min=0.0)
                else:
                    delta_high = torch.full_like(t_cons, self.delta_max)
                delta_range = torch.clamp(delta_high - self.delta_min, min=0.0)
                delta_t = self.delta_min + torch.rand_like(t_cons) * delta_range
                delta_t = torch.clamp(delta_t, min=self.delta_min)
            
            t_plus = torch.clamp(t_cons + delta_t, max=t_upper)
            t_plus_expand = t_plus.view(-1, 1, 1)
            x_t_plus = (1 - t_plus_expand) * x_0 + t_plus_expand * actions
            
            # Teacher rollout start: either from t_cons or t_plus
            if self.teacher_from == "t_cons":
                teacher_x = x_t_cons
                teacher_t = t_cons
            else:
                teacher_x = x_t_plus
                teacher_t = t_plus
            
            with torch.no_grad():
                target_x1 = self._get_consistency_targets(
                    teacher_x, teacher_t, obs_features, num_steps=self.teacher_steps
                )
            
            # Student evaluation point
            if self.student_point == "t_cons":
                student_x = x_t_cons
                student_t = t_cons
            else:
                student_x = x_t_plus
                student_t = t_plus
            
            v_pred_student = self.velocity_net(student_x, student_t, obs_features)
            
            if self.consistency_loss_space == "velocity":
                v_target_cons = target_x1 - x_0
                consistency_loss = F.mse_loss(v_pred_student, v_target_cons)
            else:
                remaining = (1.0 - student_t).view(-1, 1, 1)
                pred_x1 = student_x + remaining * v_pred_student
                consistency_loss = F.mse_loss(pred_x1, target_x1)
        
        total_loss = (
            self.flow_weight * flow_loss + 
            self.consistency_weight * consistency_loss
        )
        
        return {
            "loss": total_loss,
            "flow_loss": flow_loss,
            "consistency_loss": consistency_loss,
        }
    
    @torch.no_grad()
    def get_action(
        self,
        obs_features: torch.Tensor,
        num_steps: Optional[int] = None,
        integration_method: Literal["euler", "rk4"] = "euler",
    ) -> torch.Tensor:
        """
        Generate action using flow ODE integration.
        Consistency training allows fewer steps at inference.
        
        Args:
            obs_features: Encoded observation [B, obs_horizon, obs_dim]
            num_steps: Override default flow steps (can use fewer due to consistency)
            integration_method: ODE solver to use
            
        Returns:
            actions: Generated action sequence [B, pred_horizon, action_dim]
        """
        self.velocity_net.eval()
        batch_size = obs_features.shape[0]
        
        # Can use fewer steps due to consistency training
        steps = num_steps if num_steps is not None else self.num_flow_steps
        
        # Start from noise
        x = torch.randn(
            batch_size, self.pred_horizon, self.action_dim,
            device=obs_features.device
        )
        
        dt = 1.0 / steps
        
        for i in range(steps):
            t = torch.full((batch_size,), i * dt, device=obs_features.device)
            
            if integration_method == "euler":
                v = self.velocity_net(x, t, obs_features)
                x = x + v * dt
            else:  # RK4
                t_mid = t + 0.5 * dt
                t_end = t + dt
                
                k1 = self.velocity_net(x, t, obs_features)
                k2 = self.velocity_net(x + 0.5 * dt * k1, t_mid, obs_features)
                k3 = self.velocity_net(x + 0.5 * dt * k2, t_mid, obs_features)
                k4 = self.velocity_net(x + dt * k3, t_end, obs_features)
                
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Clamp to action bounds (optional)
        if self.action_bounds is not None:
            x = torch.clamp(x, self.action_bounds[0], self.action_bounds[1])
        
        self.velocity_net.train()
        return x
