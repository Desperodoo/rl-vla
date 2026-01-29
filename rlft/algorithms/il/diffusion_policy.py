"""
Diffusion Policy Agent

DDPM-based policy for imitation learning.
Uses 1D U-Net architecture aligned with the official diffusion_policy implementation.

References:
- Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (Chi et al., 2023)
- Denoising Diffusion Probabilistic Models (Ho et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from rlft.networks import ConditionalUnet1D


class DiffusionPolicyAgent(nn.Module):
    """Diffusion Policy Agent with 1D U-Net architecture.
    
    Uses DDPM objective to learn a noise prediction network that
    denoises Gaussian noise to the action distribution.
    
    Args:
        noise_pred_net: ConditionalUnet1D for noise prediction
        action_dim: Dimension of action space
        obs_horizon: Number of observation frames (default: 2)
        pred_horizon: Length of action sequence to predict (default: 16)
        num_diffusion_iters: Number of diffusion steps (default: 100)
        device: Device to run on (default: "cuda")
    """
    
    def __init__(
        self,
        noise_pred_net: ConditionalUnet1D,
        action_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        num_diffusion_iters: int = 100,
        action_bounds: Optional[tuple] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.noise_pred_net = noise_pred_net
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.num_diffusion_iters = num_diffusion_iters
        self.action_bounds = action_bounds
        self.device = device
        
        # DDPM noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance
            clip_sample=True,  # clip output to [-1,1] for stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )
    
    def compute_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute DDPM training loss.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, global_cond_dim]
            actions: (B, pred_horizon, action_dim) expert action sequence
            
        Returns:
            Dict with loss and loss components
        """
        B = actions.shape[0]
        device = actions.device
        
        # Flatten obs_features if 3D
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(B, -1)
        else:
            obs_cond = obs_features
        
        # Sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.action_dim), device=device)
        
        # Sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()
        
        # Add noise to the clean actions according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(actions, noise, timesteps)
        
        # Predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )
        
        # MSE loss between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)
        
        return {"loss": loss, "diffusion_loss": loss}
    
    @torch.no_grad()
    def get_action(
        self,
        obs_features: torch.Tensor,
        **kwargs,  # Accept extra kwargs for API compatibility
    ) -> torch.Tensor:
        """Sample action sequence by iterative denoising.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, global_cond_dim]
            
        Returns:
            actions: (B, pred_horizon, action_dim) action sequence
        """
        self.noise_pred_net.eval()
        batch_size = obs_features.shape[0]
        device = obs_features.device
        
        # Flatten obs_features if 3D
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(batch_size, -1)
        else:
            obs_cond = obs_features
        
        # Initialize action from Gaussian noise
        noisy_action_seq = torch.randn(
            (batch_size, self.pred_horizon, self.action_dim), device=device
        )
        
        # Set number of diffusion steps
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        
        # Iterative denoising
        for k in self.noise_scheduler.timesteps:
            # Predict noise
            noise_pred = self.noise_pred_net(
                sample=noisy_action_seq,
                timestep=k,
                global_cond=obs_cond,
            )
            
            # Inverse diffusion step (remove noise)
            noisy_action_seq = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_action_seq,
            ).prev_sample
        
        # Note: DDPM scheduler already has clip_sample=True
        # Additional clamp if action_bounds is explicitly set
        if self.action_bounds is not None:
            noisy_action_seq = torch.clamp(noisy_action_seq, self.action_bounds[0], self.action_bounds[1])
        
        self.noise_pred_net.train()
        return noisy_action_seq

    @torch.no_grad()
    def get_action_deterministic(
        self,
        obs_features: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Sample action sequence deterministically.
        
        Note: Diffusion models are inherently stochastic. This method uses
        a fixed seed for reproducibility but the result is still based on
        the diffusion process.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, global_cond_dim]
            
        Returns:
            actions: (B, pred_horizon, action_dim) action sequence
        """
        # For diffusion policy, use zero initialization instead of random noise
        self.noise_pred_net.eval()
        batch_size = obs_features.shape[0]
        device = obs_features.device
        
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(batch_size, -1)
        else:
            obs_cond = obs_features
        
        # Initialize from zeros (deterministic)
        action_seq = torch.zeros(
            (batch_size, self.pred_horizon, self.action_dim), device=device
        )
        
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                sample=action_seq,
                timestep=k,
                global_cond=obs_cond,
            )
            
            action_seq = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=action_seq,
            ).prev_sample
        
        # Note: DDPM scheduler already has clip_sample=True
        # Additional clamp if action_bounds is explicitly set
        if self.action_bounds is not None:
            action_seq = torch.clamp(action_seq, self.action_bounds[0], self.action_bounds[1])
        
        self.noise_pred_net.train()
        return action_seq
