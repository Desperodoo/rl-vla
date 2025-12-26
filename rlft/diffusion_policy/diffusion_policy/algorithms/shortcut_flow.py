"""
ShortCut Flow Policy with adaptive step size.
Migrated from toy_diffusion_rl (ReinFlow) to diffusion_policy framework.

Configurable design axes (for ablation studies):
1. Time sampling: t_min, t_max, t_sampling_mode
2. Step size sampling: step_size_mode, min_step_size, max_step_size
3. Target computation: target_mode (velocity/endpoint), teacher_steps, use_ema_teacher
4. Training: flow_weight, shortcut_weight, self_consistency_k
5. Inference: inference_mode (adaptive/uniform), num_inference_steps
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal, Tuple
import copy
import math

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from .networks import soft_update


class ShortCutVelocityUNet1D(nn.Module):
    """
    Velocity network with step size conditioning for ShortCut Flow.
    Extends ConditionalUnet1D to accept both time t and step size d.
    """
    
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.unet = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        
        # Additional embedding for step size d
        self.step_size_embed = nn.Sequential(
            nn.Linear(1, diffusion_step_embed_dim),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
        )
        
        # Combine t and d embeddings
        self.combine_embed = nn.Sequential(
            nn.Linear(diffusion_step_embed_dim * 2, diffusion_step_embed_dim),
            nn.Mish(),
        )
        
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        step_size: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with time and step size conditioning.
        
        Args:
            sample: [B, T, input_dim] action sequence
            timestep: [B] diffusion timestep (0-1)
            step_size: [B] step size d (0-1)
            global_cond: [B, cond_dim] or [B, obs_horizon, cond_dim] conditioning features
        """
        # Get step size embedding (currently unused, but available for future extensions)
        d_embed = self.step_size_embed(step_size.unsqueeze(-1))
        
        # Flatten obs for global conditioning
        if global_cond is not None and global_cond.dim() == 3:
            global_cond = global_cond.reshape(global_cond.shape[0], -1)
        
        # Use timestep as integer (scaled by 100 for embedding, matching VelocityUNet1D)
        timestep_int = (timestep * 100).long()
        
        # ConditionalUnet1D expects input as (B, T, input_dim) and returns same shape
        output = self.unet(sample, timestep_int, global_cond=global_cond)
        
        return output


class ShortCutFlowAgent(nn.Module):
    """
    ShortCut Flow Matching agent with adaptive step sizes.
    Learns to take larger steps when possible, enabling faster inference.
    Based on ReinFlow implementation.
    
    Configurable hyperparameters for ablation:
    - Time sampling: t_min, t_max, t_sampling_mode (uniform/truncated)
    - Step size: step_size_mode (power2/uniform/fixed), min/max step sizes
    - Target: target_mode (velocity/endpoint), teacher_steps, use_ema_teacher
    - Inference: inference_mode (adaptive/uniform), num_inference_steps
    
    Based on sweep results, recommended config:
    - step_size_mode: "fixed" with fixed_step_size=0.0625 (1/16)
    - target_mode: "velocity" (not endpoint)
    - use_ema_teacher: True
    - teacher_steps: 1 (single step preserves locality)
    - flow_weight > shortcut_weight (e.g., 1.0 vs 0.3)
    - inference_mode: "uniform" (avoids solver mismatch)
    """
    
    def __init__(
        self,
        velocity_net: ShortCutVelocityUNet1D,
        action_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        max_denoising_steps: int = 8,
        self_consistency_k: float = 0.25,  # Fraction of batch for consistency
        flow_weight: float = 1.0,
        shortcut_weight: float = 0.3,  # Sweep: flow-heavy is best
        ema_decay: float = 0.999,
        # Time sampling hyperparameters
        t_min: float = 0.0,
        t_max: float = 1.0,
        t_sampling_mode: Literal["uniform", "truncated"] = "uniform",
        # Step size hyperparameters (sweep: fixed small step is best)
        step_size_mode: Literal["power2", "uniform", "fixed"] = "fixed",
        min_step_size: float = 0.0625,  # 1/16 by default
        max_step_size: float = 0.5,     # 1/2 by default
        fixed_step_size: float = 0.0625, # 1/16 (sweep best)
        # Target computation hyperparameters
        target_mode: Literal["velocity", "endpoint"] = "velocity",
        teacher_steps: int = 1,  # Sweep: single step preserves locality
        use_ema_teacher: bool = True,
        # Inference hyperparameters (sweep: uniform avoids solver mismatch)
        inference_mode: Literal["adaptive", "uniform"] = "uniform",
        num_inference_steps: int = 8,
        device: str = "cuda",
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.max_denoising_steps = max_denoising_steps
        self.self_consistency_k = self_consistency_k
        self.flow_weight = flow_weight
        self.shortcut_weight = shortcut_weight
        self.ema_decay = ema_decay
        self.device = device
        
        # Time sampling config
        self.t_min = t_min
        self.t_max = t_max
        self.t_sampling_mode = t_sampling_mode
        
        # Step size config
        self.step_size_mode = step_size_mode
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.fixed_step_size = fixed_step_size
        
        # Target computation config
        self.target_mode = target_mode
        self.teacher_steps = teacher_steps
        self.use_ema_teacher = use_ema_teacher
        
        # Inference config
        self.inference_mode = inference_mode
        self.num_inference_steps = num_inference_steps
        
        # EMA velocity network for stable shortcut targets
        self.velocity_net_ema = copy.deepcopy(velocity_net)
        for param in self.velocity_net_ema.parameters():
            param.requires_grad = False
        
        # Log2 of max steps for step size sampling (power2 mode)
        self.log_max_steps = int(math.log2(max_denoising_steps))
        
    def _sample_step_size(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample step sizes d based on step_size_mode.
        
        Modes:
        - power2: d from {1/N, 2/N, 4/N, ...} where N is max_steps (log-uniform)
        - uniform: d ~ U[min_step_size, max_step_size]
        - fixed: d = fixed_step_size
        """
        if self.step_size_mode == "power2":
            # Sample power of 2: 0, 1, 2, ..., log_max_steps
            powers = torch.randint(0, self.log_max_steps + 1, (batch_size,), device=device)
            # d = 2^power / max_steps
            d = (2.0 ** powers.float()) / self.max_denoising_steps
        elif self.step_size_mode == "uniform":
            # Uniform sampling in [min_step_size, max_step_size]
            d = self.min_step_size + torch.rand(batch_size, device=device) * (self.max_step_size - self.min_step_size)
        elif self.step_size_mode == "fixed":
            # Fixed step size
            d = torch.full((batch_size,), self.fixed_step_size, device=device)
        else:
            raise ValueError(f"Unknown step_size_mode: {self.step_size_mode}")
        return d
    
    def _sample_time(self, batch_size: int, device: torch.device, d: torch.Tensor) -> torch.Tensor:
        """
        Sample time t based on t_sampling_mode.
        
        Modes:
        - uniform: t ~ U[t_min, t_max], then clamp to leave room for 2d
        - truncated: directly sample in valid range [t_min, 1 - 2d]
        """
        if self.t_sampling_mode == "uniform":
            # Standard uniform sampling, then clamp
            t = self.t_min + torch.rand(batch_size, device=device) * (self.t_max - self.t_min)
            # Ensure t + 2d <= 1 for shortcut training
            max_t = 1.0 - 2 * d
            t = torch.clamp(t * max_t.clamp(min=0.01) / max(self.t_max, 0.01), min=0.0, max=0.99)
        elif self.t_sampling_mode == "truncated":
            # Directly sample in valid range
            max_t = (1.0 - 2 * d).clamp(min=self.t_min)
            t = self.t_min + torch.rand(batch_size, device=device) * (max_t - self.t_min).clamp(min=0.01)
        else:
            raise ValueError(f"Unknown t_sampling_mode: {self.t_sampling_mode}")
        return t
    
    def _compute_shortcut_target(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        d: torch.Tensor,
        obs_features: torch.Tensor,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute shortcut target based on target_mode.
        
        Modes:
        - velocity: Target velocity that achieves 2d displacement
        - endpoint: Target endpoint x_1 reached after 2d steps
        
        Args:
            x_t: Current noisy state at time t
            t: Current time
            d: Current step size
            obs_features: Observation conditioning
            x_0: Initial noise (for endpoint mode)
            x_1: Target actions (for endpoint mode)
        """
        # Select teacher network
        teacher_net = self.velocity_net_ema if self.use_ema_teacher else self.velocity_net
        
        with torch.no_grad():
            if self.teacher_steps == 1:
                # Single-step target: just predict with d and use as 2d target
                v_1 = teacher_net(x_t, t, d, obs_features)
                d_expand = d.view(-1, 1, 1)
                x_t_plus_d = x_t + d_expand * v_1
                
                t_plus_d = t + d
                v_2 = teacher_net(x_t_plus_d, t_plus_d, d, obs_features)
                
                # Combined velocity for step size 2d
                shortcut_v = v_1 + d_expand * v_2
                target_endpoint = x_t + 2 * d_expand * ((v_1 + v_2) / 2)
            else:
                # Multi-step rollout to get target
                x = x_t.clone()
                current_t = t.clone()
                d_expand = d.view(-1, 1, 1)
                
                # Take teacher_steps small steps, each of size d
                for step in range(self.teacher_steps):
                    if step < self.teacher_steps:
                        v = teacher_net(x, current_t, d, obs_features)
                        x = x + d_expand * v
                        current_t = current_t + d
                
                target_endpoint = x
                # Compute equivalent velocity for 2d step
                shortcut_v = (target_endpoint - x_t) / (self.teacher_steps * d_expand)
            
            if self.target_mode == "velocity":
                return shortcut_v
            else:  # endpoint
                return target_endpoint
    
    def compute_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ShortCut Flow loss: standard flow + self-consistency.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim]
            actions: Expert actions [B, pred_horizon, action_dim]
        """
        batch_size = actions.shape[0]
        device = actions.device
        
        # Sample noise
        x_0 = torch.randn_like(actions)
        
        # Sample step sizes
        d = self._sample_step_size(batch_size, device)
        
        # Sample time using configurable method
        t = self._sample_time(batch_size, device, d)
        
        # Interpolate
        t_expand = t.view(-1, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * actions
        
        # Standard flow matching target (for d -> 0)
        v_target = actions - x_0
        
        # Predict velocity with step size d
        v_pred = self.velocity_net(x_t, t, d, obs_features)
        
        # Flow matching loss (for small d, v should match standard CFM)
        flow_loss = F.mse_loss(v_pred, v_target)
        
        # Self-consistency loss for larger step sizes
        shortcut_loss = torch.tensor(0.0, device=device)
        
        if self.shortcut_weight > 0 and self.self_consistency_k > 0:
            # Only compute for subset of batch (efficiency)
            n_consistency = max(1, int(batch_size * self.self_consistency_k))
            idx = torch.randperm(batch_size)[:n_consistency]
            
            x_t_sub = x_t[idx]
            t_sub = t[idx]
            d_sub = d[idx]
            obs_sub = obs_features[idx]
            x_0_sub = x_0[idx]
            actions_sub = actions[idx]
            
            # Double the step size for shortcut
            d_double = 2 * d_sub
            
            # Only train where 2d is valid (not exceeding 1)
            valid_mask = (t_sub + d_double) <= 1.0
            
            if valid_mask.sum() > 0:
                x_t_valid = x_t_sub[valid_mask]
                t_valid = t_sub[valid_mask]
                d_valid = d_sub[valid_mask]
                d_double_valid = d_double[valid_mask]
                obs_valid = obs_sub[valid_mask]
                x_0_valid = x_0_sub[valid_mask]
                actions_valid = actions_sub[valid_mask]
                
                # Shortcut target from teacher rollout
                shortcut_target = self._compute_shortcut_target(
                    x_t_valid, t_valid, d_valid, obs_valid, x_0_valid, actions_valid
                )
                
                # Predict with 2d step size
                v_pred_2d = self.velocity_net(
                    x_t_valid, t_valid, d_double_valid, obs_valid
                )
                
                if self.target_mode == "velocity":
                    shortcut_loss = F.mse_loss(v_pred_2d, shortcut_target)
                else:  # endpoint
                    # Compute predicted endpoint from student
                    d_double_expand = d_double_valid.view(-1, 1, 1)
                    pred_endpoint = x_t_valid + d_double_expand * v_pred_2d
                    shortcut_loss = F.mse_loss(pred_endpoint, shortcut_target)
        
        total_loss = self.flow_weight * flow_loss + self.shortcut_weight * shortcut_loss
        
        return {
            "loss": total_loss,
            "flow_loss": flow_loss,
            "shortcut_loss": shortcut_loss,
        }
    
    def update_ema(self):
        """Update EMA velocity network.
        
        Formula: θ_ema = ema_decay * θ_ema + (1 - ema_decay) * θ
        Equivalent to: soft_update(ema, source, tau=1-ema_decay)
        """
        soft_update(self.velocity_net_ema, self.velocity_net, 1 - self.ema_decay)
    
    @torch.no_grad()
    def get_action(
        self,
        obs_features: torch.Tensor,
        num_steps: Optional[int] = None,
        adaptive: Optional[bool] = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """
        Generate action using ShortCut flow with adaptive step sizes.
        
        Args:
            obs_features: Encoded observation [B, obs_horizon, obs_dim]
            num_steps: Override number of steps (default: num_inference_steps)
            adaptive: Override inference_mode (default: uses self.inference_mode)
            use_ema: Whether to use EMA network for sampling (default: True)
            
        Returns:
            actions: Generated action sequence [B, pred_horizon, action_dim]
        """
        self.velocity_net.eval()
        net = self.velocity_net_ema if use_ema else self.velocity_net
        batch_size = obs_features.shape[0]
        device = obs_features.device
        
        # Determine inference mode
        use_adaptive = adaptive if adaptive is not None else (self.inference_mode == "adaptive")
        steps = num_steps if num_steps is not None else self.num_inference_steps
        
        # Start from noise
        x = torch.randn(
            batch_size, self.pred_horizon, self.action_dim,
            device=device
        )
        
        if use_adaptive:
            # Use adaptive step sizes (powers of 2)
            # Start with largest possible step size
            t = torch.zeros(batch_size, device=device)
            
            while t[0] < 1.0:
                # Find largest valid step size
                remaining = 1.0 - t[0]
                
                # Use power-of-2 step sizes
                d_val = min(remaining.item(), self.max_step_size)
                for power in range(self.log_max_steps, -1, -1):
                    candidate = (2.0 ** power) / self.max_denoising_steps
                    if candidate <= remaining and candidate >= self.min_step_size:
                        d_val = candidate
                        break
                
                d = torch.full((batch_size,), d_val, device=device)
                
                v = net(x, t, d, obs_features)
                x = x + d.view(-1, 1, 1) * v
                t = t + d
                
                # Safety check
                if d_val < 1e-6:
                    break
        else:
            # Uniform steps
            dt = 1.0 / steps
            d = torch.full((batch_size,), dt, device=device)
            
            for i in range(steps):
                t = torch.full((batch_size,), i * dt, device=device)
                v = net(x, t, d, obs_features)
                x = x + dt * v
        
        # Clamp to action bounds
        x = torch.clamp(x, -1.0, 1.0)
        
        self.velocity_net.train()
        return x
