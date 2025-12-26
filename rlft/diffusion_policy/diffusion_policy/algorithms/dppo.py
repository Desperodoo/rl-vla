"""
DPPO - Diffusion Policy Policy Optimization

DDPM-based diffusion policy with partial chain fine-tuning for online RL.
The key idea is to freeze the early denoising steps and only fine-tune
the last K steps, improving stability and sample efficiency.

For offline RL, this agent can be used for BC pretraining via compute_loss().

References:
- DPPO: https://diffusion-ppo.github.io/
- Official code: https://github.com/irom-princeton/dppo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import copy

from ..conditional_unet1d import ConditionalUnet1D
from .networks import soft_update


class ValueNetwork(nn.Module):
    """Value network for PPO.
    
    Simple MLP that takes observation features and outputs scalar value.
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: list = [256, 256],
    ):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Smaller weights for output layer
        final_layer = self.net[-1]
        nn.init.orthogonal_(final_layer.weight, gain=0.01)
    
    def forward(self, obs_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_features: (B, obs_dim) observation features
            
        Returns:
            value: (B, 1) state value
        """
        return self.net(obs_features)


class DPPOAgent(nn.Module):
    """DPPO Agent for diffusion policy fine-tuning.
    
    Core features:
    1. Partial chain fine-tuning: only last K denoising steps are trainable
    2. noise_pred_net: frozen base network for early steps
    3. noise_pred_net_ft: trainable network for final steps
    4. ValueNetwork for PPO advantage estimation
    
    For offline RL, use compute_loss() for BC pretraining.
    For online RL, use collect_rollout() + update() for PPO fine-tuning.
    
    Args:
        noise_pred_net: ConditionalUnet1D for noise prediction
        obs_dim: Dimension of flattened observation features
        act_dim: Action dimension
        pred_horizon: Prediction horizon (action chunk length)
        obs_horizon: Observation horizon
        act_horizon: Action execution horizon
        num_diffusion_iters: Total DDPM denoising steps
        ft_denoising_steps: Number of steps to fine-tune (from the end)
        ema_decay: EMA decay rate for target network
        gamma: Discount factor for RL
        gae_lambda: GAE lambda
        clip_ratio: PPO clip ratio
        entropy_coef: Entropy coefficient
        value_coef: Value loss coefficient
        use_ema: Whether to use EMA for inference
    """
    
    def __init__(
        self,
        noise_pred_net: ConditionalUnet1D,
        obs_dim: int,
        act_dim: int,
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        act_horizon: int = 8,
        num_diffusion_iters: int = 100,
        ft_denoising_steps: int = 5,
        ema_decay: float = 0.999,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        use_ema: bool = True,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.num_diffusion_iters = num_diffusion_iters
        self.ft_denoising_steps = min(ft_denoising_steps, num_diffusion_iters)
        self.ema_decay = ema_decay
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.use_ema = use_ema
        
        # Frozen base noise predictor (for early denoising steps)
        self.noise_pred_net = noise_pred_net
        for param in self.noise_pred_net.parameters():
            param.requires_grad = False
        
        # Trainable noise predictor (for last ft_denoising_steps)
        self.noise_pred_net_ft = copy.deepcopy(noise_pred_net)
        for param in self.noise_pred_net_ft.parameters():
            param.requires_grad = True
        
        # EMA of trainable network for stable inference
        if use_ema:
            self.noise_pred_net_ft_ema = copy.deepcopy(noise_pred_net)
            for param in self.noise_pred_net_ft_ema.parameters():
                param.requires_grad = False
        else:
            self.noise_pred_net_ft_ema = None
        
        # Value network for PPO
        self.value_net = ValueNetwork(obs_dim=obs_dim)
        
        # DDPM scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        
        # Learnable log_std for stochastic policy
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Old policy for PPO ratio computation
        self.noise_pred_net_ft_old = copy.deepcopy(noise_pred_net)
        for param in self.noise_pred_net_ft_old.parameters():
            param.requires_grad = False
    
    def _get_noise_predictor(self, timestep: int, use_ema: bool = False) -> nn.Module:
        """Get appropriate noise predictor for given timestep.
        
        Uses fine-tuned network for last ft_denoising_steps, frozen for earlier.
        """
        if timestep < self.ft_denoising_steps:
            if use_ema and self.noise_pred_net_ft_ema is not None:
                return self.noise_pred_net_ft_ema
            return self.noise_pred_net_ft
        return self.noise_pred_net
    
    def update_ema(self):
        """Update EMA of fine-tuned network."""
        if self.noise_pred_net_ft_ema is not None:
            soft_update(self.noise_pred_net_ft_ema, self.noise_pred_net_ft, 1 - self.ema_decay)
    
    def sync_old_policy(self):
        """Sync old policy with current for PPO ratio."""
        self.noise_pred_net_ft_old.load_state_dict(self.noise_pred_net_ft.state_dict())
    
    def compute_loss(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute BC loss for pretraining.
        
        Trains BOTH frozen and fine-tuned networks during pretraining.
        After pretraining, call freeze_base_network() to freeze the base.
        
        Args:
            obs_cond: (B, obs_horizon * obs_dim) flattened observation features
            actions: (B, pred_horizon, act_dim) ground truth action sequence
            
        Returns:
            Dictionary with 'loss' and 'bc_loss' keys
        """
        B = actions.shape[0]
        device = actions.device
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.num_diffusion_iters, (B,), device=device
        ).long()
        
        # Add noise to actions
        noise = torch.randn_like(actions)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        
        # Predict noise with fine-tuned network (which is trainable)
        # During pretraining, we train the network that will be fine-tuned
        noise_pred = self.noise_pred_net_ft(noisy_actions, timesteps, global_cond=obs_cond)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return {
            "loss": loss,
            "bc_loss": loss,
        }
    
    def freeze_base_network(self):
        """Freeze base network and copy weights to frozen network.
        
        Call this after BC pretraining to set up partial chain fine-tuning.
        """
        # Copy trained weights to frozen network
        self.noise_pred_net.load_state_dict(self.noise_pred_net_ft.state_dict())
        for param in self.noise_pred_net.parameters():
            param.requires_grad = False
        
        # Also update EMA
        if self.noise_pred_net_ft_ema is not None:
            self.noise_pred_net_ft_ema.load_state_dict(self.noise_pred_net_ft.state_dict())
        
        # Sync old policy
        self.sync_old_policy()
    
    @torch.no_grad()
    def get_action(
        self,
        obs_cond: torch.Tensor,
        deterministic: bool = False,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Sample action using partial chain DDPM denoising.
        
        Uses frozen network for early steps, fine-tuned for final steps.
        
        Args:
            obs_cond: (B, obs_horizon * obs_dim) observation features
            deterministic: If True, use deterministic denoising
            use_ema: Whether to use EMA network for fine-tuned steps
            
        Returns:
            actions: (B, pred_horizon, act_dim) sampled action sequence
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        # Start from Gaussian noise
        noisy_action = torch.randn(
            (B, self.pred_horizon, self.act_dim), device=device
        )
        
        # Set up scheduler
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        
        # Denoising loop
        for k, t in enumerate(self.noise_scheduler.timesteps):
            timestep = t.item()
            
            # Get appropriate network
            net = self._get_noise_predictor(timestep, use_ema=use_ema)
            
            # Predict noise
            noise_pred = net(
                noisy_action,
                torch.full((B,), t, device=device, dtype=torch.long),
                global_cond=obs_cond,
            )
            
            # Denoise step
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_action,
            ).prev_sample
            
            # Add exploration noise for fine-tuned steps (if not deterministic)
            if not deterministic and timestep < self.ft_denoising_steps and timestep > 0:
                std = torch.exp(self.log_std).clamp(min=0.01, max=0.5)
                noise_scale = std * 0.1 * (timestep / self.ft_denoising_steps)
                noisy_action = noisy_action + noise_scale * torch.randn_like(noisy_action)
        
        return noisy_action
    
    def compute_value(self, obs_cond: torch.Tensor) -> torch.Tensor:
        """Compute state value.
        
        Args:
            obs_cond: (B, obs_dim) observation features
            
        Returns:
            value: (B, 1) state value
        """
        return self.value_net(obs_cond)
    
    def compute_action_log_prob(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
        use_old_policy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probability of actions under diffusion policy.
        
        Uses local Gaussian approximation around predicted clean action.
        Only computes for fine-tuned denoising steps (PPO relevant).
        
        Args:
            obs_cond: (B, obs_dim) observation features
            actions: (B, pred_horizon, act_dim) action sequence
            use_old_policy: Whether to use old policy for PPO ratio
            
        Returns:
            log_prob: (B,) log probability
            entropy: (B,) policy entropy
        """
        B = actions.shape[0]
        device = actions.device
        
        total_log_prob = torch.zeros(B, device=device)
        total_entropy = torch.zeros(B, device=device)
        
        std = torch.exp(self.log_std).clamp(min=0.01, max=1.0)
        
        net = self.noise_pred_net_ft_old if use_old_policy else self.noise_pred_net_ft
        
        # Only compute for fine-tuned steps
        for t in range(self.ft_denoising_steps - 1, -1, -1):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Get noisy action at timestep t
            noise = torch.randn_like(actions)
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, t_batch)
            
            # Predict noise
            with torch.set_grad_enabled(not use_old_policy):
                noise_pred = net(noisy_actions, t_batch, global_cond=obs_cond)
            
            # Compute predicted clean action (simplified DDPM formula)
            alpha_bar = self.noise_scheduler.alphas_cumprod[t].to(device)
            pred_a0 = (noisy_actions - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            pred_a0 = torch.clamp(pred_a0, -1.0, 1.0)
            
            # Local Gaussian approximation
            if t == 0:
                dist = torch.distributions.Normal(pred_a0, std.view(1, 1, -1))
                log_prob = dist.log_prob(actions).sum(dim=[1, 2])
                entropy = dist.entropy().sum(dim=[1, 2])
            else:
                scale = std * np.sqrt(t / self.num_diffusion_iters)
                dist = torch.distributions.Normal(pred_a0, scale.view(1, 1, -1))
                log_prob = dist.log_prob(actions).sum(dim=[1, 2]) * 0.1
                entropy = dist.entropy().sum(dim=[1, 2])
            
            total_log_prob = total_log_prob + log_prob
            total_entropy = total_entropy + entropy
        
        # Average entropy
        total_entropy = total_entropy / max(1, self.ft_denoising_steps)
        
        return total_log_prob, total_entropy
    
    def compute_ppo_loss(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO loss for online RL fine-tuning.
        
        Args:
            obs_cond: (B, obs_dim) observation features
            actions: (B, pred_horizon, act_dim) action sequence
            old_log_probs: (B,) log probs from old policy
            advantages: (B,) GAE advantages
            returns: (B,) discounted returns
            
        Returns:
            Dictionary with 'loss', 'policy_loss', 'value_loss', 'entropy'
        """
        # Compute new log probs and entropy
        new_log_probs, entropy = self.compute_action_log_prob(
            obs_cond, actions, use_old_policy=False
        )
        
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        values = self.value_net(obs_cond).squeeze(-1)
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy.mean(),
        }
    
    def save(self, path: str):
        """Save checkpoint."""
        checkpoint = {
            "noise_pred_net": self.noise_pred_net.state_dict(),
            "noise_pred_net_ft": self.noise_pred_net_ft.state_dict(),
            "value_net": self.value_net.state_dict(),
            "log_std": self.log_std,
            "config": {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "pred_horizon": self.pred_horizon,
                "obs_horizon": self.obs_horizon,
                "act_horizon": self.act_horizon,
                "num_diffusion_iters": self.num_diffusion_iters,
                "ft_denoising_steps": self.ft_denoising_steps,
            },
        }
        if self.noise_pred_net_ft_ema is not None:
            checkpoint["noise_pred_net_ft_ema"] = self.noise_pred_net_ft_ema.state_dict()
        torch.save(checkpoint, path)
    
    def load(self, path: str, device: str = "cpu"):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        self.noise_pred_net.load_state_dict(checkpoint["noise_pred_net"])
        self.noise_pred_net_ft.load_state_dict(checkpoint["noise_pred_net_ft"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.log_std.data = checkpoint["log_std"]
        if "noise_pred_net_ft_ema" in checkpoint and self.noise_pred_net_ft_ema is not None:
            self.noise_pred_net_ft_ema.load_state_dict(checkpoint["noise_pred_net_ft_ema"])
        self.sync_old_policy()
    
    def load_pretrained(self, checkpoint_path: str, device: str = "cpu"):
        """Load pretrained diffusion policy weights.
        
        Loads weights into both frozen and fine-tuned networks.
        Use this before calling freeze_base_network().
        
        Args:
            checkpoint_path: Path to pretrained checkpoint
            device: Device to load to
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Try different key names
        if "noise_pred_net" in checkpoint:
            weights = checkpoint["noise_pred_net"]
        elif "model" in checkpoint:
            weights = checkpoint["model"]
        else:
            weights = checkpoint
        
        # Load into both networks
        self.noise_pred_net.load_state_dict(weights, strict=False)
        self.noise_pred_net_ft.load_state_dict(weights, strict=False)
        
        if self.noise_pred_net_ft_ema is not None:
            self.noise_pred_net_ft_ema.load_state_dict(weights, strict=False)
        
        self.sync_old_policy()
