"""
ReinFlow - Reinforcement Learning Fine-tuning for Flow Matching Policies

Key features:
1. Flow matching policy with learnable exploration noise
2. PPO-style policy gradient for online RL
3. Advantage estimation via value network
4. Compatible with ShortCut Flow / AW-ShortCut Flow checkpoints

Based on ReinFlow paper: https://github.com/ReinFlow/ReinFlow

Three-stage pipeline:
- Stage 1: ShortCut Flow BC pretrain (pure BC)
- Stage 2: AW-ShortCut Flow offline RL (Q-weighted BC)
- Stage 3: ReinFlow online RL (this module, PPO fine-tuning)

Architecture:
- ShortCutVelocityUNet1D: Base velocity network with step_size conditioning
- ExploreNoiseNet: Per-dimension exploration noise prediction
- NoisyVelocityUNet1D: Wrapper combining base + exploration
- ValueNetwork: Critic for PPO (V(s_0) predicts return)

Denoising MDP design (from ReinFlow paper):
- State: s_k = (x_k, obs) at denoising step k
- Action: predicted velocity v_k
- Reward: sparse, only final step gets R = A(s, a_final)
- Critic: V(s_0) predicts expected return from executing the action chunk

NOTE: RolloutBufferPPO and compute_smdp_rewards are now in diffusion_policy.rlpd.replay_buffer
      Import from there instead of defining here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Literal
import copy
import math

from .shortcut_flow import ShortCutVelocityUNet1D
from .networks import soft_update
from .dppo import ValueNetwork


class ExploreNoiseNet(nn.Module):
    """Network that predicts exploration noise scale.
    
    Takes timestep embedding and observation embedding,
    outputs per-dimension noise scale.
    
    Args:
        time_embed_dim: Timestep embedding dimension
        obs_embed_dim: Observation embedding dimension  
        hidden_dim: Hidden layer dimension
        action_dim: Action dimension (output size)
        min_noise_std: Minimum noise std (added to output)
        max_noise_std: Maximum noise std (clamped)
    """
    
    def __init__(
        self,
        time_embed_dim: int = 64,
        obs_embed_dim: int = 128,
        hidden_dim: int = 128,
        action_dim: int = 1,
        min_noise_std: float = 0.01,
        max_noise_std: float = 0.3,
    ):
        super().__init__()
        
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        
        self.net = nn.Sequential(
            nn.Linear(time_embed_dim + obs_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus(),  # Ensure positive output
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Small init for output
        self.net[-2].weight.data *= 0.01
    
    def forward(
        self,
        time_emb: torch.Tensor,
        obs_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            time_emb: (B, time_embed_dim) timestep embedding
            obs_emb: (B, obs_embed_dim) observation embedding
            
        Returns:
            noise_std: (B, action_dim) noise scale per dimension
        """
        x = torch.cat([time_emb, obs_emb], dim=-1)
        raw_std = self.net(x)
        # Clamp to [min_noise_std, max_noise_std]
        return torch.clamp(raw_std + self.min_noise_std, max=self.max_noise_std)


class NoisyVelocityUNet1D(nn.Module):
    """Velocity network with learnable exploration noise.
    
    Wraps a base ShortCutVelocityUNet1D and adds a noise prediction network
    for stochastic policy exploration during RL fine-tuning.
    
    Compatible with ShortCut Flow architecture (step_size conditioning).
    
    Args:
        base_velocity_net: ShortCutVelocityUNet1D for velocity prediction
        obs_dim: Observation dimension (flattened)
        action_dim: Action dimension
        time_embed_dim: Timestep embedding dimension
        obs_embed_dim: Observation embedding dimension
        min_noise_std: Minimum exploration noise std
        max_noise_std: Maximum exploration noise std
    """
    
    def __init__(
        self,
        base_velocity_net: ShortCutVelocityUNet1D,
        obs_dim: int,
        action_dim: int,
        time_embed_dim: int = 64,
        obs_embed_dim: int = 128,
        min_noise_std: float = 0.01,
        max_noise_std: float = 0.3,
    ):
        super().__init__()
        
        self.base_net = base_velocity_net
        self.action_dim = action_dim
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Observation embedding
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, obs_embed_dim),
            nn.LayerNorm(obs_embed_dim),
            nn.Mish(),
            nn.Linear(obs_embed_dim, obs_embed_dim),
        )
        
        # Exploration noise network
        self.explore_noise_net = ExploreNoiseNet(
            time_embed_dim=time_embed_dim,
            obs_embed_dim=obs_embed_dim,
            action_dim=action_dim,
            min_noise_std=min_noise_std,
            max_noise_std=max_noise_std,
        )
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        step_size: torch.Tensor,
        global_cond: torch.Tensor,
        sample_noise: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            sample: (B, pred_horizon, action_dim) noisy action sequence
            timestep: (B,) timestep in [0, 1]
            step_size: (B,) step size d for ShortCut Flow
            global_cond: (B, obs_dim) observation features
            sample_noise: Whether to sample exploration noise
            
        Returns:
            velocity: (B, pred_horizon, action_dim) predicted velocity
            noise: (B, pred_horizon, action_dim) sampled noise (or None)
            noise_std: (B, action_dim) noise scale (clamped for numerical stability)
        """
        # Get base velocity prediction (with step_size for ShortCut Flow)
        velocity = self.base_net(sample, timestep, step_size, global_cond)
        
        # Compute embeddings for noise prediction
        t_emb = self.time_embed(timestep.unsqueeze(-1))
        obs_emb = self.obs_embed(global_cond)
        
        # Predict noise scale
        noise_std_raw = self.explore_noise_net(t_emb, obs_emb)
        
        # Clamp noise_std for numerical stability
        # Following original ReinFlow: min_sampling_denoising_std and max_logprob_denoising_std
        noise_std = torch.clamp(noise_std_raw, min=self.min_noise_std, max=self.max_noise_std)
        
        # Sample noise if requested
        if sample_noise:
            # noise_std: (B, action_dim) -> expand to (B, pred_horizon, action_dim)
            noise_std_expanded = noise_std.unsqueeze(1).expand(-1, sample.shape[1], -1)
            noise = torch.randn_like(sample) * noise_std_expanded
        else:
            noise = None
        
        return velocity, noise, noise_std
    
    def update_noise_bounds(self, min_std: float, max_std: float):
        """Update noise bounds for scheduling."""
        self.min_noise_std = min_std
        self.max_noise_std = max_std
        self.explore_noise_net.min_noise_std = min_std
        self.explore_noise_net.max_noise_std = max_std


class ReinFlowAgent(nn.Module):
    """ReinFlow Agent for online RL fine-tuning of flow matching policies.
    
    Core features:
    1. ShortCutVelocityUNet1D for flow matching velocity prediction
    2. ExploreNoiseNet for learnable exploration during RL
    3. Fixed num_inference_steps mode (no adaptive stepping)
    4. PPO-style updates with denoising MDP formulation
    5. Critic warmup support for stable training
    6. Noise scheduling interface for sweep
    
    Compatible with AW-ShortCut Flow checkpoints via load_from_aw_shortcut_flow().
    
    Denoising MDP (from ReinFlow paper):
    - State: s_k = (x_k, obs) at denoising step k
    - Action: predicted velocity v_k  
    - Reward: sparse, only final step gets R = A(s, a_final)
    - Critic: V(s_0) predicts expected return from executing the action chunk
    
    Args:
        velocity_net: ShortCutVelocityUNet1D for velocity field prediction
        obs_dim: Dimension of flattened observation features
        act_dim: Action dimension
        pred_horizon: Prediction horizon (action chunk length)
        obs_horizon: Observation horizon
        act_horizon: Action execution horizon (full chunk for SMDP)
        num_inference_steps: Fixed number of flow integration steps
        ema_decay: EMA decay rate
        use_ema: Whether to use EMA for inference
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_ratio: PPO clip ratio
        entropy_coef: Entropy coefficient
        value_coef: Value loss coefficient
        # Noise scheduling parameters
        min_noise_std: Minimum exploration noise std
        max_noise_std: Maximum exploration noise std
        noise_decay_type: Type of noise decay ("constant", "linear", "exponential")
        noise_decay_steps: Steps over which to decay noise
        # Critic warmup
        critic_warmup_steps: Number of steps to train critic only
    """
    
    def __init__(
        self,
        velocity_net: ShortCutVelocityUNet1D,
        obs_dim: int,
        act_dim: int,
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        act_horizon: int = 8,
        num_inference_steps: int = 8,
        ema_decay: float = 0.999,
        use_ema: bool = True,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        # Noise scheduling
        min_noise_std: float = 0.01,
        max_noise_std: float = 0.3,
        noise_decay_type: Literal["constant", "linear", "exponential"] = "constant",
        noise_decay_steps: int = 100000,
        # Critic warmup
        critic_warmup_steps: int = 0,
        # === NEW: Critic stability improvements (borrowed from AWCP) ===
        reward_scale: float = 1.0,  # Scale rewards to stabilize critic training
        value_target_tau: float = 0.005,  # Soft update rate for target value network
        use_target_value_net: bool = True,  # Whether to use target value network
        value_target_clip: float = 100.0,  # Clip value targets
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.num_inference_steps = num_inference_steps
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Noise scheduling
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        self.initial_max_noise_std = max_noise_std
        self.noise_decay_type = noise_decay_type
        self.noise_decay_steps = noise_decay_steps
        
        # Critic warmup
        self.critic_warmup_steps = critic_warmup_steps
        self._current_step = 0
        
        # === NEW: Critic stability parameters ===
        self.reward_scale = reward_scale
        self.value_target_tau = value_target_tau
        self.use_target_value_net = use_target_value_net
        self.value_target_clip = value_target_clip
        
        # Fixed step size for inference (uniform steps)
        self.fixed_step_size = 1.0 / num_inference_steps
        
        # Wrap velocity net with noisy version
        self.noisy_velocity_net = NoisyVelocityUNet1D(
            base_velocity_net=velocity_net,
            obs_dim=obs_dim,
            action_dim=act_dim,
            min_noise_std=min_noise_std,
            max_noise_std=max_noise_std,
        )
        
        # EMA of base velocity network for stable inference
        if use_ema:
            self.velocity_net_ema = copy.deepcopy(velocity_net)
            for param in self.velocity_net_ema.parameters():
                param.requires_grad = False
        else:
            self.velocity_net_ema = None
        
        # Value network for PPO (predicts V(s_0) = expected return)
        self.value_net = ValueNetwork(obs_dim=obs_dim)
        
        # === NEW: Target value network for stable critic training (borrowed from AWCP) ===
        if use_target_value_net:
            self.value_net_target = copy.deepcopy(self.value_net)
            for param in self.value_net_target.parameters():
                param.requires_grad = False
        else:
            self.value_net_target = None
    
    def get_current_noise_std(self) -> Tuple[float, float]:
        """Get current noise bounds based on decay schedule."""
        if self.noise_decay_type == "constant":
            return self.min_noise_std, self.max_noise_std
        
        progress = min(1.0, self._current_step / max(1, self.noise_decay_steps))
        
        if self.noise_decay_type == "linear":
            current_max = self.initial_max_noise_std * (1.0 - progress) + self.min_noise_std * progress
        elif self.noise_decay_type == "exponential":
            decay_factor = math.exp(-3 * progress)  # Decay to ~5% at end
            current_max = self.min_noise_std + (self.initial_max_noise_std - self.min_noise_std) * decay_factor
        else:
            current_max = self.max_noise_std
        
        return self.min_noise_std, max(self.min_noise_std, current_max)
    
    def update_noise_schedule(self, step: Optional[int] = None):
        """Update noise bounds based on training progress."""
        if step is not None:
            self._current_step = step
        else:
            self._current_step += 1
        
        min_std, max_std = self.get_current_noise_std()
        self.noisy_velocity_net.update_noise_bounds(min_std, max_std)
    
    def update_ema(self):
        """Update EMA of velocity network."""
        if self.velocity_net_ema is not None:
            soft_update(
                self.velocity_net_ema,
                self.noisy_velocity_net.base_net,
                1 - self.ema_decay
            )
    
    def update_target_value_net(self):
        """Soft update target value network (borrowed from AWCP's target critic update)."""
        if self.value_net_target is not None:
            soft_update(
                self.value_net_target,
                self.value_net,
                self.value_target_tau
            )
    
    def compute_loss(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute flow matching BC loss (for pretraining/warmup).
        
        Uses standard conditional flow matching objective with fixed step size.
        
        Args:
            obs_cond: (B, obs_horizon * obs_dim) observation features
            actions: (B, pred_horizon, act_dim) ground truth action sequence
            
        Returns:
            Dictionary with 'loss' and 'bc_loss' keys
        """
        B = actions.shape[0]
        device = actions.device
        
        # Sample random timesteps t ~ U[0, 1]
        t = torch.rand(B, device=device)
        
        # Fixed step size for ShortCut Flow compatibility
        d = torch.full((B,), self.fixed_step_size, device=device)
        
        # Sample initial noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(actions)
        
        # Interpolate: x_t = (1 - t) * x_0 + t * x_1
        t_expand = t.view(B, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * actions
        
        # Target velocity: v = x_1 - x_0
        target_v = actions - x_0
        
        # Predict velocity (no noise during BC training)
        pred_v, _, _ = self.noisy_velocity_net(x_t, t, d, obs_cond, sample_noise=False)
        
        # MSE loss
        loss = F.mse_loss(pred_v, target_v)
        
        return {
            "loss": loss,
            "bc_loss": loss,
        }
    
    @torch.no_grad()
    def get_action(
        self,
        obs_cond: torch.Tensor,
        deterministic: bool = True,
        use_ema: bool = True,
        return_chains: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample action using fixed-step flow integration.
        
        For offline BC training evaluation: use deterministic=True (default)
        For online RL exploration: use deterministic=False to inject learned noise
        
        Following original ReinFlow implementation, stores complete x_chain trajectory
        for accurate log_prob computation in PPO updates.
        
        Args:
            obs_cond: (B, obs_horizon * obs_dim) observation features
            deterministic: If True, don't add exploration noise (default for eval)
            use_ema: Whether to use EMA network for base velocity
            return_chains: Whether to return intermediate states for log_prob
            
        Returns:
            actions: (B, pred_horizon, act_dim) sampled action sequence
            x_chain: (B, K+1, pred_horizon, act_dim) complete denoising trajectory
                     if return_chains=True, else None
                     x_chain[:, 0] is initial noise x_0
                     x_chain[:, K] is final action (before clamp)
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        K = self.num_inference_steps
        
        # Start from Gaussian noise
        x = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        
        # Fixed step size and dt
        dt = self.fixed_step_size
        d = torch.full((B,), dt, device=device)
        
        # Store complete x_chain for log_prob computation: [B, K+1, H, A]
        if return_chains:
            x_chain = torch.zeros((B, K + 1, self.pred_horizon, self.act_dim), device=device)
            x_chain[:, 0] = x.clone()  # Store initial noise x_0
        else:
            x_chain = None
        
        # Choose velocity network
        if use_ema and self.velocity_net_ema is not None and deterministic:
            # Use EMA for deterministic inference
            for i in range(K):
                t = torch.full((B,), i * dt, device=device)
                velocity = self.velocity_net_ema(x, t, d, obs_cond)
                x = x + velocity * dt
                if return_chains:
                    x_chain[:, i + 1] = x.clone()
        else:
            # Use noisy velocity net
            for i in range(K):
                t = torch.full((B,), i * dt, device=device)
                
                if deterministic:
                    velocity, _, noise_std = self.noisy_velocity_net(
                        x, t, d, obs_cond, sample_noise=False
                    )
                    x = x + velocity * dt
                else:
                    velocity, noise, noise_std = self.noisy_velocity_net(
                        x, t, d, obs_cond, sample_noise=True
                    )
                    x = x + velocity * dt
                    # Add exploration noise (following original ReinFlow: no sqrt(dt) factor)
                    # Original: dist = Normal(xt, std); xt = dist.sample()
                    if noise is not None:
                        x = x + noise  # noise already has std = noise_std
                
                if return_chains:
                    x_chain[:, i + 1] = x.clone()
        
        actions = torch.clamp(x, -1.0, 1.0)
        return actions, x_chain
    
    def compute_value(self, obs_cond: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Compute state value V(s_0).
        
        In ReinFlow's denoising MDP, V(s_0) predicts the expected return
        from executing the generated action chunk in the environment.
        
        Args:
            obs_cond: (B, obs_dim) observation features
            use_target: If True and target value network exists, use target network
                       for more stable bootstrap value estimation (recommended for GAE)
        """
        if use_target and self.use_target_value_net and self.value_net_target is not None:
            return self.value_net_target(obs_cond)
        return self.value_net(obs_cond)
    
    def compute_action_log_prob(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
        x_chain: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probability of actions using Markov chain transition probabilities.
        
        Following original ReinFlow implementation:
        log p(trajectory | policy) = sum_{k=0}^{K-1} log p(x_{k+1} | x_k, s, policy)
        
        Note: We do NOT include log p(x_0) because:
        1. x_0 ~ N(0, I) is policy-independent (same for old and new policy)
        2. In PPO ratio computation, log p(x_0) cancels out
        3. Including it would make log_prob values unnecessarily large
        
        where:
        - p(x_{k+1} | x_k, s) = N(x_k + v_k * dt, sigma_k)
        
        Args:
            obs_cond: (B, obs_dim) observation features
            actions: (B, pred_horizon, act_dim) action sequence (not used if x_chain provided)
            x_chain: (B, K+1, pred_horizon, act_dim) complete denoising trajectory from get_action
                     x_chain[:, 0] is initial noise x_0, x_chain[:, K] is final action
            
        Returns:
            log_prob: (B,) log probability of the trajectory (transitions only)
            entropy: (B,) policy entropy estimate
        """
        if x_chain is None:
            raise ValueError("x_chain is required for accurate log_prob computation. "
                           "Call get_action with return_chains=True.")
        
        B = x_chain.shape[0]
        K = self.num_inference_steps
        device = x_chain.device
        
        dt = self.fixed_step_size
        d = torch.full((B,), dt, device=device)
        
        # === Compute sum of transition log probabilities ===
        # For each step k: log p(x_{k+1} | x_k, s) = log N(x_{k+1}; mean_k, std_k)
        # where mean_k = x_k + v_k * dt, std_k = noise_std_k (following original ReinFlow)
        #
        # NOTE: We do NOT include log p(x_0) because it's policy-independent
        # and would cancel out in the PPO ratio anyway.
        
        total_log_prob_trans = torch.zeros(B, device=device)
        total_entropy = torch.zeros(B, device=device)
        
        for k in range(K):
            # Detach x_chain states since they're fixed from rollout
            # Only the network parameters should have gradients
            x_k = x_chain[:, k].detach()        # (B, H, A) - current state  
            x_next = x_chain[:, k + 1].detach()  # (B, H, A) - next state
            
            # Compute timestep
            t = torch.full((B,), k * dt, device=device)
            
            # Get velocity and noise_std from policy network
            velocity, _, noise_std = self.noisy_velocity_net(x_k, t, d, obs_cond, sample_noise=False)
            # noise_std: (B, A)
            
            # Mean of transition: x_k + velocity * dt
            mean_next = x_k + velocity * dt  # (B, H, A)
            
            # Std of transition: noise_std directly (following original ReinFlow)
            # Original: dist = Normal(xt, std); logprob = dist.log_prob(xt).sum()
            trans_std = noise_std  # (B, A) - no sqrt(dt) factor!
            
            # Expand std to match action shape: (B, A) -> (B, H, A)
            trans_std_expanded = trans_std.unsqueeze(1).expand(-1, self.pred_horizon, -1)
            
            # Compute log p(x_{k+1} | x_k) = log N(x_{k+1}; mean_next, trans_std)
            # = -0.5 * ((x_next - mean_next) / trans_std)^2 - log(trans_std) - 0.5 * log(2*pi)
            diff = x_next - mean_next
            
            # Normalized squared distance
            normalized_diff = diff / trans_std_expanded
            
            # For numerical stability, clip the normalized difference
            normalized_diff = torch.clamp(normalized_diff, -10.0, 10.0)
            
            log_prob_trans = -0.5 * normalized_diff ** 2 \
                             - torch.log(trans_std_expanded) \
                             - 0.5 * np.log(2 * np.pi)
            
            # Sum over dimensions (H, A) to get per-sample log_prob
            log_prob_trans = log_prob_trans.sum(dim=[1, 2])  # (B,)
            
            total_log_prob_trans = total_log_prob_trans + log_prob_trans
            
            # Accumulate entropy: H = 0.5 * (1 + log(2*pi*sigma^2)) per dim
            entropy_per_dim = 0.5 * (1 + np.log(2 * np.pi)) + torch.log(trans_std_expanded)
            total_entropy = total_entropy + entropy_per_dim.sum(dim=[1, 2])
        
        # === Final log probability ===
        # Only transition probabilities, not including log p(x_0)
        log_prob = total_log_prob_trans
        
        # No clamping here - let the values flow through for proper gradient computation
        # The ratio will be clamped in compute_ppo_loss anyway
        
        # Average entropy per step
        entropy = total_entropy / K
        
        return log_prob, entropy
    
    def compute_ppo_loss(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        x_chain: Optional[torch.Tensor] = None,
        old_values: Optional[torch.Tensor] = None,
        clip_value: bool = False,
        value_clip_range: float = 10.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO loss for online RL.
        
        Following original ReinFlow implementation (ppoflow.py).
        
        Args:
            obs_cond: (B, obs_dim) observation features
            actions: (B, pred_horizon, act_dim) action sequence
            old_log_probs: (B,) log probs from old policy
            advantages: (B,) GAE advantages
            returns: (B,) discounted returns (should be pre-scaled by reward_scale)
            x_chain: (B, K+1, pred_horizon, act_dim) complete denoising trajectory
            old_values: (B,) old value predictions (for value clipping)
            clip_value: Whether to clip value loss
            value_clip_range: Range for value clipping
            
        Returns:
            Dictionary with loss components
        """
        # Compute value predictions
        values = self.value_net(obs_cond).squeeze(-1)
        
        # === Value loss (following original ReinFlow: MSE with 0.5 coefficient) ===
        if clip_value and old_values is not None:
            # Clipped value loss (PPO2 style)
            # Original: v_loss = 0.5 * torch.max((newvalues - returns) ** 2, (v_clipped - returns) ** 2).mean()
            v_clipped = old_values + torch.clamp(
                values - old_values, -value_clip_range, value_clip_range
            )
            v_loss_unclipped = (values - returns) ** 2
            v_loss_clipped = (v_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            # Original: v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
            value_loss = 0.5 * ((values - returns) ** 2).mean()
        
        # Compute new log probs
        new_log_probs, entropy = self.compute_action_log_prob(
            obs_cond, actions, x_chain=x_chain
        )
        
        # === Log prob handling ===
        # NOTE: Original ReinFlow uses normalized log_probs (divided by K or by dimensions)
        # with clamp range [-5, 2]. Our implementation uses joint log_prob (sum over K steps),
        # which results in much larger values (~400-500 for K=8, H=16, A=7).
        # 
        # We have two options:
        # 1. Normalize log_probs by dividing by (K * H * A) - but this changes gradient scale
        # 2. Don't clamp, since the ratio is what matters for PPO, not absolute log_prob values
        #
        # We choose option 2: no clamping, because:
        # - The ratio exp(new_log_prob - old_log_prob) is numerically stable if the difference is small
        # - Clamping would destroy gradients when values are at the boundary
        # - The PPO clip already handles extreme ratios
        
        # PPO clipped objective (no log_prob clamping)
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        
        # === Compute approximate KL divergence ===
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
                
        # Policy loss with clipping (original doesn't clamp ratio before this)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy.mean(),
            "ratio_mean": ratio.mean(),
            "ratio_std": ratio.std(),
            "ratio_min": ratio.min(),
            "ratio_max": ratio.max(),
            "log_ratio_mean": log_ratio.mean(),
            "log_ratio_std": log_ratio.std(),
            "new_log_probs_mean": new_log_probs.mean(),
            "new_log_probs_std": new_log_probs.std(),
            "old_log_probs_mean": old_log_probs.mean(),
            "old_log_probs_std": old_log_probs.std(),
            "returns_mean": returns.mean(),
            "returns_std": returns.std(),
            "values_mean": values.mean(),
            "values_std": values.std(),
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "in_warmup": False,
        }
    
    def save(self, path: str):
        """Save checkpoint."""
        checkpoint = {
            "noisy_velocity_net": self.noisy_velocity_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "current_step": self._current_step,
            "config": {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "pred_horizon": self.pred_horizon,
                "obs_horizon": self.obs_horizon,
                "act_horizon": self.act_horizon,
                "num_inference_steps": self.num_inference_steps,
                "ema_decay": self.ema_decay,
                "min_noise_std": self.min_noise_std,
                "max_noise_std": self.max_noise_std,
                "initial_max_noise_std": self.initial_max_noise_std,
                "noise_decay_type": self.noise_decay_type,
                "noise_decay_steps": self.noise_decay_steps,
                "critic_warmup_steps": self.critic_warmup_steps,
                # === NEW: Save critic stability parameters ===
                "reward_scale": self.reward_scale,
                "value_target_tau": self.value_target_tau,
                "use_target_value_net": self.use_target_value_net,
                "value_target_clip": self.value_target_clip,
            },
        }
        if self.velocity_net_ema is not None:
            checkpoint["velocity_net_ema"] = self.velocity_net_ema.state_dict()
        # === NEW: Save target value network ===
        if self.value_net_target is not None:
            checkpoint["value_net_target"] = self.value_net_target.state_dict()
        torch.save(checkpoint, path)
    
    def load(self, path: str, device: str = "cpu"):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        self.noisy_velocity_net.load_state_dict(checkpoint["noisy_velocity_net"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        if "velocity_net_ema" in checkpoint and self.velocity_net_ema is not None:
            self.velocity_net_ema.load_state_dict(checkpoint["velocity_net_ema"])
        # === NEW: Load target value network ===
        if "value_net_target" in checkpoint and self.value_net_target is not None:
            self.value_net_target.load_state_dict(checkpoint["value_net_target"])
        if "current_step" in checkpoint:
            self._current_step = checkpoint["current_step"]
    
    def load_from_aw_shortcut_flow(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        load_critic: bool = False,
    ):
        """Load pretrained weights from AW-ShortCut Flow checkpoint.
        
        Extracts velocity_net weights from AWShortCutFlowAgent checkpoint
        and loads them into the base velocity network.
        
        AW-ShortCut Flow checkpoint format:
        - checkpoint['agent'] contains the full agent state dict with keys like 'velocity_net.xxx'
        - checkpoint['ema_agent'] contains the EMA agent state dict
        
        Args:
            checkpoint_path: Path to AW-ShortCut Flow checkpoint
            device: Device to load to
            load_critic: Whether to attempt loading critic weights (usually False,
                        since we use ValueNetwork instead of DoubleQNetwork)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # AW-ShortCut Flow checkpoint format: agent/ema_agent with velocity_net.xxx keys
        if "agent" in checkpoint or "ema_agent" in checkpoint:
            # Extract velocity_net weights from agent state dict
            agent_state = checkpoint.get("ema_agent", checkpoint.get("agent", {}))
            
            # Filter velocity_net keys and remove prefix
            velocity_weights = {}
            for k, v in agent_state.items():
                if k.startswith("velocity_net."):
                    # Remove "velocity_net." prefix since we're loading into base_net
                    new_key = k[len("velocity_net."):]
                    velocity_weights[new_key] = v
            
            if not velocity_weights:
                raise ValueError(f"No velocity_net weights found in checkpoint: {checkpoint_path}")
                
        # Fallback: direct velocity_net state dict
        elif "velocity_net" in checkpoint:
            velocity_weights = checkpoint["velocity_net"]
        elif "velocity_net_ema" in checkpoint:
            velocity_weights = checkpoint["velocity_net_ema"]
        elif "model" in checkpoint:
            velocity_weights = checkpoint["model"]
        else:
            # Try loading directly (might be just the state dict)
            velocity_weights = checkpoint
        
        # Load into base network
        missing_keys, unexpected_keys = self.noisy_velocity_net.base_net.load_state_dict(
            velocity_weights, strict=False
        )
        if missing_keys:
            print(f"Warning: Missing keys when loading velocity_net: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading velocity_net: {unexpected_keys[:5]}...")
        
        # Also load into EMA network
        if self.velocity_net_ema is not None:
            self.velocity_net_ema.load_state_dict(velocity_weights, strict=False)
        
        print(f"Loaded velocity network from {checkpoint_path}")
        
        # Note: We don't load critic because AW-ShortCut Flow uses DoubleQNetwork
        # while ReinFlow uses ValueNetwork. The value network should be initialized
        # fresh and trained with critic warmup.
        if load_critic:
            print("Warning: Critic loading not supported (architecture mismatch)")
    
    def load_pretrained(self, checkpoint_path: str, device: str = "cpu"):
        """Load pretrained flow matching weights.
        
        Tries AW-ShortCut Flow format first (checkpoint with 'agent'/'ema_agent' keys),
        then falls back to generic loading for other checkpoint formats.
        
        Args:
            checkpoint_path: Path to pretrained checkpoint
            device: Device to load to
            
        Raises:
            RuntimeError: If checkpoint format doesn't match model architecture
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check checkpoint format and dispatch to appropriate loader
        if "agent" in checkpoint or "ema_agent" in checkpoint:
            # AW-ShortCut Flow format: agent/ema_agent with velocity_net.xxx keys
            self.load_from_aw_shortcut_flow(checkpoint_path, device)
        elif "velocity_net" in checkpoint:
            # Direct velocity_net state dict
            weights = checkpoint["velocity_net"]
            self.noisy_velocity_net.base_net.load_state_dict(weights)
            if self.velocity_net_ema is not None:
                self.velocity_net_ema.load_state_dict(weights)
            print(f"Loaded velocity network from {checkpoint_path}")
        elif "model" in checkpoint:
            # Generic model key
            weights = checkpoint["model"]
            self.noisy_velocity_net.base_net.load_state_dict(weights)
            if self.velocity_net_ema is not None:
                self.velocity_net_ema.load_state_dict(weights)
            print(f"Loaded velocity network from {checkpoint_path}")
        else:
            # Try loading directly (might be just the state dict)
            self.noisy_velocity_net.base_net.load_state_dict(checkpoint)
            if self.velocity_net_ema is not None:
                self.velocity_net_ema.load_state_dict(checkpoint)
            print(f"Loaded velocity network from {checkpoint_path}")
