"""
Reflected Flow Policy for bounded action spaces.

Uses boundary reflection to handle action limits [-1, 1].
This is particularly useful for robotics where joint limits are strict.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Literal
import copy


class ReflectedFlowAgent(nn.Module):
    """
    Reflected Flow Matching agent for bounded action spaces.
    Uses boundary reflection to handle action limits [-1, 1].
    
    Args:
        velocity_net: Velocity network (VelocityUNet1D)
        action_dim: Dimension of action space
        obs_horizon: Number of observation frames
        pred_horizon: Number of actions to predict
        num_flow_steps: ODE integration steps
        action_low: Lower bound for actions
        action_high: Upper bound for actions
        reflection_mode: "hard" (reflect trajectory) or "soft" (regularization only)
        boundary_reg_weight: Weight for boundary regularization loss
        device: Device to run on
    """
    
    def __init__(
        self,
        velocity_net: nn.Module,
        action_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        num_flow_steps: int = 20,  # Best from sweep (20 > 10 > 5)
        action_low: float = -1.0,
        action_high: float = 1.0,
        reflection_mode: Literal["hard", "soft"] = "soft",  # Best from sweep
        boundary_reg_weight: float = 0.01,
        device: str = "cuda",
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.num_flow_steps = num_flow_steps
        self.action_low = action_low
        self.action_high = action_high
        self.reflection_mode = reflection_mode
        self.boundary_reg_weight = boundary_reg_weight
        self.device = device
        
    def _reflect_trajectory(
        self, 
        x: torch.Tensor, 
        low: float = -1.0, 
        high: float = 1.0
    ) -> torch.Tensor:
        """
        Reflect trajectory to keep it within bounds.
        Uses modular reflection to handle multiple boundary crossings.
        """
        range_size = high - low
        
        # Shift to [0, range_size]
        x_shifted = x - low
        
        # Number of full reflections
        n_reflections = torch.floor(x_shifted / range_size).long()
        
        # Position within current range
        x_mod = x_shifted - n_reflections.float() * range_size
        
        # Reflect odd number of crossings
        should_reflect = (n_reflections % 2) == 1
        x_reflected = torch.where(
            should_reflect,
            range_size - x_mod,
            x_mod
        )
        
        # Shift back
        return x_reflected + low
    
    def _compute_reflected_target(
        self,
        x_0: torch.Tensor,  # noise
        x_1: torch.Tensor,  # data
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reflected flow matching target.
        
        Returns:
            x_t: interpolated point
            v_target: target velocity (considering reflection)
        """
        # Standard linear interpolation
        t_expand = t.view(-1, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        
        if self.reflection_mode == "hard":
            # Hard reflection: reflect x_t and compute adjusted velocity
            x_t_reflected = self._reflect_trajectory(x_t)
            
            # Velocity target: use standard CFM target (x_1 - x_0) for numerical stability
            # The reflection is only applied to x_t to keep trajectories in bounds
            # Dividing by remaining_time causes numerical explosion when t → 1
            # 
            # Alternative interpretation: the velocity should point from x_0 to x_1
            # regardless of where x_t lands after reflection. The reflection handles
            # the boundary constraint, while the velocity field learns the flow direction.
            v_target = x_1 - x_0
            
            return x_t_reflected, v_target
        else:
            # Soft reflection: use smooth boundary penalty
            v_target = x_1 - x_0
            return x_t, v_target
    
    def compute_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reflected flow matching loss.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, cond_dim]
            actions: Expert actions [B, pred_horizon, action_dim]
        """
        batch_size = actions.shape[0]
        
        # Sample noise
        x_0 = torch.randn_like(actions)
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=actions.device)
        
        # Compute reflected interpolation and target
        x_t, v_target = self._compute_reflected_target(x_0, actions, t)
        
        # Predict velocity
        v_pred = self.velocity_net(x_t, t, obs_features)
        
        # Flow matching loss
        flow_loss = F.mse_loss(v_pred, v_target)
        
        # Boundary regularization (soft penalty for being near boundaries)
        boundary_loss = torch.tensor(0.0, device=actions.device)
        if self.boundary_reg_weight > 0:
            # Penalize predictions that would go out of bounds
            margin = 0.1
            low_violation = F.relu(self.action_low + margin - x_t)
            high_violation = F.relu(x_t - self.action_high + margin)
            boundary_loss = (low_violation.pow(2) + high_violation.pow(2)).mean()
        
        total_loss = flow_loss + self.boundary_reg_weight * boundary_loss
        
        return {
            "loss": total_loss,
            "flow_loss": flow_loss,
            "boundary_loss": boundary_loss,
        }
    
    @torch.no_grad()
    def get_action(
        self,
        obs_features: torch.Tensor,
        integration_method: Literal["euler", "rk4"] = "euler",
        **kwargs,  # Accept extra kwargs for API compatibility
    ) -> torch.Tensor:
        """
        Generate action using reflected flow ODE integration.
        
        Args:
            obs_features: Encoded observation [B, obs_horizon, obs_dim] or [B, cond_dim]
            integration_method: ODE solver to use
            
        Returns:
            actions: Generated action sequence [B, pred_horizon, action_dim]
        """
        self.velocity_net.eval()
        batch_size = obs_features.shape[0]
        
        # Start from noise
        x = torch.randn(
            batch_size, self.pred_horizon, self.action_dim,
            device=obs_features.device
        )
        
        dt = 1.0 / self.num_flow_steps
        
        for i in range(self.num_flow_steps):
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
            
            # Apply reflection after each step
            x = self._reflect_trajectory(x, self.action_low, self.action_high)
        
        self.velocity_net.train()
        return x

    @torch.no_grad()
    def get_action_deterministic(
        self,
        obs_features: torch.Tensor,
        integration_method: Literal["euler", "rk4"] = "euler",
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate action deterministically using reflected flow ODE integration.
        Starts from zero instead of random noise for reproducible results.
        
        Args:
            obs_features: Encoded observation [B, obs_horizon, obs_dim] or [B, cond_dim]
            integration_method: ODE solver to use
            
        Returns:
            actions: Generated action sequence [B, pred_horizon, action_dim]
        """
        self.velocity_net.eval()
        batch_size = obs_features.shape[0]
        
        # Start from zeros (deterministic)
        x = torch.zeros(
            batch_size, self.pred_horizon, self.action_dim,
            device=obs_features.device
        )
        
        dt = 1.0 / self.num_flow_steps
        
        for i in range(self.num_flow_steps):
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
            
            # Apply reflection after each step
            x = self._reflect_trajectory(x, self.action_low, self.action_high)
        
        self.velocity_net.train()
        return x
