"""
Diffusion Double Q Learning Agent

Combines Diffusion Policy with Double Q-Learning for offline RL.
Uses 1D U-Net architecture aligned with the official diffusion_policy implementation.

References:
- Diffusion-QL: https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL
- IDQL: https://github.com/philippe-eecs/IDQL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal
import copy
import numpy as np

from .networks import DoubleQNetwork, soft_update
from ..conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class DiffusionDoubleQAgent(nn.Module):
    """Diffusion Double Q Agent with 1D U-Net architecture.
    
    Combines diffusion-based actor with twin Q-networks for offline RL.
    
    Hyperparameters inherited from toy_diffusion_rl:
    - alpha: 0.01 (Q-value weight, reduced for stability with dense rewards)
    - bc_weight: 1.0
    - reward_scale: 0.1
    - tau: 0.005
    - gamma: 0.99
    
    Q-Gradient Modes (q_grad_mode):
    - "whole_grad": Gradient flows through ALL diffusion steps (original Diffusion-QL)
    - "last_few": Gradient flows through last N steps (q_grad_steps controls N)
    - "single_step": Single-step denoising approximation from t=1 to t=0 (fast but approximate)
    
    Args:
        noise_pred_net: ConditionalUnet1D for noise prediction (actor)
        q_network: DoubleQNetwork for Q-value estimation (critic)
        action_dim: Dimension of action space
        obs_horizon: Number of observation frames (default: 2)
        pred_horizon: Length of action sequence to predict (default: 16)
        act_horizon: Length of action sequence for Q-learning (default: 8)
        num_diffusion_iters: Number of diffusion steps (default: 100)
        alpha: Weight for Q-value term (default: 0.01, reduced for dense reward stability)
        bc_weight: Weight for BC loss (default: 1.0)
        gamma: Discount factor (default: 0.99)
        tau: Soft update coefficient (default: 0.005)
        reward_scale: Scale factor for rewards (default: 0.1)
        q_target_clip: Clip range for Q target (default: 100.0)
        q_grad_mode: How to compute Q-gradient ("whole_grad", "last_few", "single_step")
        q_grad_steps: Number of steps with gradient in "last_few" mode (default: 5)
        device: Device to run on (default: "cuda")
    """
    
    def __init__(
        self,
        noise_pred_net: ConditionalUnet1D,
        q_network: DoubleQNetwork,
        action_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        act_horizon: int = 8,
        num_diffusion_iters: int = 100,
        alpha: float = 0.01,
        bc_weight: float = 1.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scale: float = 0.1,
        q_target_clip: float = 100.0,
        q_grad_mode: Literal["whole_grad", "last_few", "single_step"] = "last_few",
        q_grad_steps: int = 5,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.noise_pred_net = noise_pred_net
        self.critic = q_network
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.num_diffusion_iters = num_diffusion_iters
        self.alpha = alpha
        self.bc_weight = bc_weight
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.q_target_clip = q_target_clip
        self.q_grad_mode = q_grad_mode
        self.q_grad_steps = q_grad_steps
        self.device = device
        
        # Target critic
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        # Noise scheduler (aligned with official implementation)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
    
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
        """Compute combined actor and critic loss.
        
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
        
        # Compute actor loss (BC uses full pred_horizon actions)
        actor_dict = self._compute_actor_loss(obs_cond, actions, actions_for_q)
        
        # Compute critic loss (uses act_horizon actions to match reward)
        # IMPORTANT: Detach obs_cond and next_obs_cond for critic to prevent
        # gradients from flowing back to visual encoder through critic loss
        critic_loss = self._compute_critic_loss(
            obs_cond.detach(), next_obs_cond.detach(), actions_for_q, rewards, dones,
            cumulative_reward=cumulative_reward,
            chunk_done=chunk_done,
            discount_factor=discount_factor,
        )
        
        # Return actor_loss and critic_loss separately (NOT summed)
        # This allows train_offline_rl.py to do separate backward passes
        return {
            "loss": actor_dict["actor_loss"] + critic_loss,  # For logging only
            "actor_loss": actor_dict["actor_loss"],  # For actor backward
            "bc_loss": actor_dict["bc_loss"],
            "q_policy_loss": actor_dict["q_loss"],
            "critic_loss": critic_loss,  # For critic backward
            "q_mean": actor_dict["q_mean"],
        }
    
    def _compute_actor_loss(
        self, 
        obs_cond: torch.Tensor, 
        action_seq: torch.Tensor,
        actions_for_q: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute actor loss: BC loss + Q-value maximization.
        
        Args:
            obs_cond: Observation conditioning [B, global_cond_dim]
            action_seq: Full pred_horizon actions for BC training [B, pred_horizon, action_dim]
            actions_for_q: act_horizon actions for Q-learning [B, act_horizon, action_dim]
        """
        B = action_seq.shape[0]
        device = action_seq.device
        act_horizon = actions_for_q.shape[1]
        
        # BC Loss: Diffusion training objective (uses full pred_horizon)
        noise = torch.randn_like(action_seq)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)
        bc_loss = F.mse_loss(noise_pred, noise)
        
        # Q-value maximization: sample actions with gradient
        # Disable gradients for critic during Q-maximization (only update actor)
        for p in self.critic.parameters():
            p.requires_grad = False
        
        # Sample actions based on q_grad_mode
        if self.q_grad_mode == "whole_grad":
            # Full gradient through entire diffusion chain (original Diffusion-QL)
            sampled_actions = self._sample_actions_with_grad(obs_cond, act_horizon, grad_steps=self.num_diffusion_iters)
        elif self.q_grad_mode == "last_few":
            # Gradient only through last N steps
            sampled_actions = self._sample_actions_with_grad(obs_cond, act_horizon, grad_steps=self.q_grad_steps)
        else:  # "single_step"
            # Single-step denoising approximation (fast but approximate)
            sampled_actions = self._sample_actions_single_step(obs_cond, act_horizon)
        
        q_value = self.critic.q1_forward(sampled_actions, obs_cond)
        
        for p in self.critic.parameters():
            p.requires_grad = True
        
        q_loss = -q_value.mean()
        
        # Total actor loss
        actor_loss = self.bc_weight * bc_loss + self.alpha * q_loss
        
        return {
            "actor_loss": actor_loss,
            "bc_loss": bc_loss,
            "q_loss": q_loss,
            "q_mean": q_value.mean(),
        }
    
    def _sample_actions_with_grad(
        self, 
        obs_cond: torch.Tensor, 
        act_horizon: int,
        grad_steps: int,
    ) -> torch.Tensor:
        """Sample actions with gradient flowing through the diffusion chain.
        
        This implements the original Diffusion-QL approach where gradients
        flow through the entire (or partial) diffusion sampling process.
        
        Args:
            obs_cond: Observation conditioning [B, global_cond_dim]
            act_horizon: Number of action steps to return
            grad_steps: Number of final steps to enable gradient (from T-1 to T-grad_steps)
        
        Returns:
            sampled_actions: [B, act_horizon, action_dim] with gradients
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        # Start from pure noise
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        
        # Determine which steps should have gradient
        # grad_steps = 5 means last 5 steps (t=4,3,2,1,0) have gradient
        grad_threshold = grad_steps  # Steps below this threshold get gradient
        
        # Full DDPM sampling loop
        for i in reversed(range(self.num_diffusion_iters)):
            t_batch = torch.full((B,), i, device=device, dtype=torch.long)
            
            if i < grad_threshold:
                # With gradient
                x = self._p_sample_with_grad(x, t_batch, obs_cond)
            else:
                # Without gradient
                with torch.no_grad():
                    x = self._p_sample_with_grad(x, t_batch, obs_cond)
        
        # Clamp and return act_horizon subset
        x = torch.clamp(x, -1.0, 1.0)
        return x[:, :act_horizon, :]
    
    def _p_sample_with_grad(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        obs_cond: torch.Tensor
    ) -> torch.Tensor:
        """Single DDPM p_sample step, preserving gradients.
        
        Implements: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_theta(x_t, t)) + sigma_t * z
        
        Args:
            x: Current noisy sample [B, pred_horizon, action_dim]
            t: Current timestep [B]
            obs_cond: Observation conditioning [B, global_cond_dim]
            
        Returns:
            x_prev: Denoised sample at t-1
        """
        B = x.shape[0]
        device = x.device
        
        # Predict noise
        noise_pred = self.noise_pred_net(x, t, global_cond=obs_cond)
        
        # Get scheduler parameters for this timestep
        # Note: we manually implement the step to preserve gradients
        t_idx = t[0].item()  # Assumes all t are the same
        
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[t_idx]
        alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[t_idx - 1] if t_idx > 0 else torch.tensor(1.0)
        beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        
        # Compute x_0 prediction from noise prediction
        # x_0 = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
        
        x_0_pred = (x - sqrt_one_minus_alpha_prod_t * noise_pred) / sqrt_alpha_prod_t
        
        # Clip prediction (important for stability)
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # Compute posterior mean
        # mu_t = (sqrt(alpha_bar_{t-1}) * beta_t / (1-alpha_bar_t)) * x_0 
        #      + (sqrt(alpha_t) * (1-alpha_bar_{t-1}) / (1-alpha_bar_t)) * x_t
        sqrt_alpha_prod_t_prev = torch.sqrt(alpha_prod_t_prev)
        alpha_t = alpha_prod_t / alpha_prod_t_prev if t_idx > 0 else alpha_prod_t
        
        coef1 = sqrt_alpha_prod_t_prev * beta_t / (1 - alpha_prod_t)
        coef2 = torch.sqrt(alpha_t) * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
        
        posterior_mean = coef1 * x_0_pred + coef2 * x
        
        # Add noise (except for t=0)
        if t_idx > 0:
            posterior_variance = beta_t * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
            noise = torch.randn_like(x)
            x_prev = posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            x_prev = posterior_mean
        
        return x_prev
    
    def _sample_actions_single_step(
        self, 
        obs_cond: torch.Tensor, 
        act_horizon: int
    ) -> torch.Tensor:
        """Single-step denoising approximation from t=1 to t=0.
        
        This is a fast approximation that only requires one network forward pass.
        Less accurate than full chain but much faster.
        
        Args:
            obs_cond: Observation conditioning [B, global_cond_dim]
            act_horizon: Number of action steps to return
            
        Returns:
            reconstructed_actions: [B, act_horizon, action_dim] with gradients
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        t_one = torch.ones(B, device=device, dtype=torch.long)
        noise_for_q = torch.randn((B, act_horizon, self.action_dim), device=device)
        
        # Pad to pred_horizon for network input
        pad_len = self.pred_horizon - act_horizon
        if pad_len > 0:
            noise_padded = torch.cat([noise_for_q, noise_for_q[:, -1:, :].repeat(1, pad_len, 1)], dim=1)
        else:
            noise_padded = noise_for_q
        
        # Single-step denoising with gradient
        pred_noise_full = self.noise_pred_net(noise_padded, t_one, global_cond=obs_cond)
        pred_noise = pred_noise_full[:, :act_horizon, :]
        
        # Reconstruct action from noise prediction (differentiable!)
        alpha_bar = self.noise_scheduler.alphas_cumprod[1].view(1, 1, 1)  # t=1
        reconstructed_actions = (noise_for_q - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
        reconstructed_actions = torch.clamp(reconstructed_actions, -1.0, 1.0)
        
        return reconstructed_actions
    
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
            # Sample next actions using current policy (full pred_horizon)
            next_actions_full = self._sample_actions_batch(next_obs_cond, num_steps=5)
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
    
    def _sample_actions_batch(self, obs_cond: torch.Tensor, num_steps: int = 5) -> torch.Tensor:
        """Sample actions using DDPM sampling with reduced steps."""
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        
        # Use subset of timesteps for faster sampling
        step_indices = list(range(0, self.num_diffusion_iters, self.num_diffusion_iters // num_steps))[::-1]
        
        for t in step_indices:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.noise_pred_net(x, t_batch, global_cond=obs_cond)
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample
        
        return torch.clamp(x, -1.0, 1.0)
    
    def update_target(self):
        """Soft update target critic network."""
        soft_update(self.critic_target, self.critic, self.tau)
    
    @torch.no_grad()
    def get_action(
        self,
        obs_features: torch.Tensor,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample action sequence for evaluation.
        
        Args:
            obs_features: Encoded observation features [B, obs_horizon, obs_dim] or [B, global_cond_dim]
            num_inference_steps: Number of diffusion steps (default: all steps)
            
        Returns:
            actions: (B, pred_horizon, action_dim) action sequence
        """
        self.noise_pred_net.eval()
        
        # Flatten obs_features if 3D
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        else:
            obs_cond = obs_features
        
        B = obs_cond.shape[0]
        device = obs_cond.device
        
        # Full DDPM sampling
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        
        if num_inference_steps is not None:
            step_indices = list(range(0, self.num_diffusion_iters, self.num_diffusion_iters // num_inference_steps))[::-1]
        else:
            step_indices = self.noise_scheduler.timesteps
        
        for k in step_indices:
            noise_pred = self.noise_pred_net(
                sample=x,
                timestep=k,
                global_cond=obs_cond,
            )
            x = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=x,
            ).prev_sample
        
        x = torch.clamp(x, -1.0, 1.0)
        
        self.noise_pred_net.train()
        return x
