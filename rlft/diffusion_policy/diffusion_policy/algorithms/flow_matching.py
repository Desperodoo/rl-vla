"""
Flow Matching Policy Agent

Standard Conditional Flow Matching (CFM) policy for imitation learning.
Uses 1D U-Net architecture aligned with the official diffusion_policy implementation.

References:
- Flow Matching for Generative Modeling (Lipman et al., 2022)
- Optimal Transport Flow Matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from .networks import VelocityUNet1D


class FlowMatchingAgent(nn.Module):
    """Flow Matching Agent with 1D U-Net architecture.
    
    Uses CFM objective to learn a velocity field that transports
    Gaussian noise to the action distribution.
    
    Args:
        velocity_net: VelocityUNet1D for velocity prediction
        action_dim: Dimension of action space
        obs_horizon: Number of observation frames (default: 2)
        pred_horizon: Length of action sequence to predict (default: 16)
        num_flow_steps: Number of ODE integration steps (default: 10)
        device: Device to run on (default: "cuda")
    """
    
    def __init__(
        self,
        velocity_net: VelocityUNet1D,
        action_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        num_flow_steps: int = 10,
        action_bounds: Optional[tuple] = None,  # (min, max) for action clipping, None to disable
        device: str = "cuda",
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.num_flow_steps = num_flow_steps
        self.action_bounds = action_bounds
        self.device = device
    
    def compute_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute CFM training loss.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, global_cond_dim]
            actions: (B, pred_horizon, action_dim) expert action sequence
            
        Returns:
            Dict with loss and loss components
        """
        B = actions.shape[0]
        device = actions.device
        
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(actions)
        
        # Sample random timesteps t ~ U(0, 1)
        t = torch.rand(B, device=device)
        
        # Linear interpolation: x_t = (1 - t) * x_0 + t * x_1
        t_expand = t.view(-1, 1, 1)  # (B, 1, 1) for broadcasting
        x_t = (1 - t_expand) * x_0 + t_expand * actions
        
        # Target velocity: v = x_1 - x_0
        target_velocity = actions - x_0
        
        # Predict velocity
        predicted_velocity = self.velocity_net(x_t, t, global_cond=obs_features)
        
        # MSE loss
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
        return {"loss": loss, "flow_loss": loss}
    
    @torch.no_grad()
    def get_action(
        self,
        obs_features: torch.Tensor,
        use_rk4: bool = False,
    ) -> torch.Tensor:
        """Sample action sequence by integrating the learned ODE.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, global_cond_dim]
            use_rk4: Use RK4 integrator instead of Euler
            
        Returns:
            actions: (B, pred_horizon, action_dim) action sequence
        """
        self.velocity_net.eval()
        batch_size = obs_features.shape[0]
        device = obs_features.device
        
        # Initialize from Gaussian noise
        x = torch.randn(batch_size, self.pred_horizon, self.action_dim, device=device)
        
        # Integrate ODE
        if use_rk4:
            x = self._rk4_integrate(x, obs_features)
        else:
            x = self._euler_integrate(x, obs_features)
        
        # Clamp to valid action range (optional)
        if self.action_bounds is not None:
            x = torch.clamp(x, self.action_bounds[0], self.action_bounds[1])
        
        self.velocity_net.train()
        return x
    
    def _euler_integrate(self, x: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        """Euler integration of the velocity field."""
        B = x.shape[0]
        device = x.device
        dt = 1.0 / self.num_flow_steps
        
        for i in range(self.num_flow_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.velocity_net(x, t, global_cond=obs_cond)
            x = x + v * dt
        
        return x
    
    def _rk4_integrate(self, x: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
        """RK4 integration of the velocity field."""
        B = x.shape[0]
        device = x.device
        dt = 1.0 / self.num_flow_steps
        
        for i in range(self.num_flow_steps):
            t = i * dt
            t_tensor = torch.full((B,), t, device=device)
            
            k1 = self.velocity_net(x, t_tensor, global_cond=obs_cond)
            
            t_mid = torch.full((B,), t + dt/2, device=device)
            k2 = self.velocity_net(x + k1 * dt/2, t_mid, global_cond=obs_cond)
            k3 = self.velocity_net(x + k2 * dt/2, t_mid, global_cond=obs_cond)
            
            t_end = torch.full((B,), t + dt, device=device)
            k4 = self.velocity_net(x + k3 * dt, t_end, global_cond=obs_cond)
            
            x = x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        
        return x
