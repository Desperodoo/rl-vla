"""
Velocity Networks for Flow Matching and ShortCut Flow.

- VelocityUNet1D: Standard velocity field prediction network
- ShortCutVelocityUNet1D: Velocity network with step size conditioning
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .unet import ConditionalUnet1D


class VelocityUNet1D(nn.Module):
    """1D U-Net for velocity field prediction (Flow Matching).
    
    Similar to ConditionalUnet1D but predicts velocity instead of noise.
    Uses FiLM conditioning with observation features.
    
    Args:
        input_dim: Action dimension
        global_cond_dim: Dimension of global conditioning (obs features)
        diffusion_step_embed_dim: Dimension of timestep embedding
        down_dims: Channel dimensions for each downsampling stage
        n_groups: Number of groups for GroupNorm
    """
    
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 64,
        down_dims: List[int] = [64, 128, 256],
        n_groups: int = 8,
    ):
        super().__init__()
        # Reuse ConditionalUnet1D architecture for velocity prediction
        self.unet = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            n_groups=n_groups,
        )
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sample: (B, pred_horizon, action_dim) noisy action sequence
            timestep: (B,) timestep values in [0, 1] for flow matching
            global_cond: (B, global_cond_dim) or (B, obs_horizon, cond_dim) observation features
            
        Returns:
            velocity: (B, pred_horizon, action_dim) predicted velocity field
        """
        # Flatten global_cond if it's 3D
        if global_cond.dim() == 3:
            global_cond = global_cond.reshape(global_cond.shape[0], -1)
        
        # Convert continuous timestep [0, 1] to integer for U-Net embedding
        timestep_int = (timestep * 100).long()
        return self.unet(sample, timestep_int, global_cond=global_cond)


class ShortCutVelocityUNet1D(nn.Module):
    """Velocity network with step size conditioning for ShortCut Flow.
    
    Extends ConditionalUnet1D to accept both time t and step size d.
    The step size embedding is available for future extensions.
    
    Args:
        input_dim: Action dimension
        global_cond_dim: Dimension of global conditioning
        diffusion_step_embed_dim: Dimension of timestep embedding
        down_dims: Channel dimensions for U-Net levels
        kernel_size: Convolution kernel size
        n_groups: Number of groups for GroupNorm
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
        
        # Combine t and d embeddings (for future use)
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
            sample: (B, T, input_dim) action sequence
            timestep: (B,) diffusion timestep (0-1)
            step_size: (B,) step size d (0-1)
            global_cond: (B, cond_dim) or (B, obs_horizon, cond_dim) conditioning
            
        Returns:
            velocity: (B, T, input_dim) predicted velocity
        """
        # Get step size embedding
        # TODO: d_embed is computed but NOT actually passed to the UNet.
        # ShortCut conditioning v(x_t, t, d) degrades to v(x_t, t) (regular Flow Matching).
        # To fix: need to modify UNet to accept combined (t, d) embedding.
        # WARNING: Existing pretrained checkpoints were trained with this bug,
        # so fixing this requires retraining the offline model.
        d_embed = self.step_size_embed(step_size.unsqueeze(-1))
        
        # Flatten obs for global conditioning
        if global_cond is not None and global_cond.dim() == 3:
            global_cond = global_cond.reshape(global_cond.shape[0], -1)
        
        # Use timestep as integer (scaled by 100 for embedding)
        timestep_int = (timestep * 100).long()
        
        output = self.unet(sample, timestep_int, global_cond=global_cond)
        
        return output


class GripperHead(nn.Module):
    """Gripper classification head for discrete gripper control.
    
    Takes encoded observation features and predicts open/close 
    for each timestep in the prediction horizon.
    
    Args:
        obs_dim: Dimension of observation features
        pred_horizon: Action prediction horizon
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        obs_dim: int,
        pred_horizon: int = 16,
        hidden_dim: int = 256,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, pred_horizon * 2),  # 2 classes per timestep
            )
        else:
            # Legacy architecture (ReLU, no LayerNorm — matches older checkpoints)
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, pred_horizon * 2),
            )
    
    def forward(self, obs_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_features: (B, obs_dim) encoded observation features
            
        Returns:
            logits: (B, pred_horizon, 2) gripper logits
        """
        B = obs_features.shape[0]
        out = self.net(obs_features)
        return out.view(B, self.pred_horizon, 2)
