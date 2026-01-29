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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Literal
import copy
import math

from rlft.networks import ShortCutVelocityUNet1D, soft_update


class ExploreNoiseNet(nn.Module):
    """Network that predicts exploration noise scale.
    
    Takes timestep embedding and observation embedding,
    outputs per-dimension noise scale.
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
            nn.Softplus(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.net[-2].weight.data *= 0.01
    
    def forward(
        self,
        time_emb: torch.Tensor,
        obs_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([time_emb, obs_emb], dim=-1)
        raw_std = self.net(x)
        return torch.clamp(raw_std + self.min_noise_std, max=self.max_noise_std)


class NoisyVelocityUNet1D(nn.Module):
    """Velocity network with learnable exploration noise.
    
    Wraps a base ShortCutVelocityUNet1D and adds a noise prediction network
    for stochastic policy exploration during RL fine-tuning.
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
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, obs_embed_dim),
            nn.LayerNorm(obs_embed_dim),
            nn.Mish(),
            nn.Linear(obs_embed_dim, obs_embed_dim),
        )
        
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
        velocity = self.base_net(sample, timestep, step_size, global_cond)
        
        t_emb = self.time_embed(timestep.unsqueeze(-1))
        obs_emb = self.obs_embed(global_cond)
        
        noise_std_raw = self.explore_noise_net(t_emb, obs_emb)
        noise_std = torch.clamp(noise_std_raw, min=self.min_noise_std, max=self.max_noise_std)
        
        if sample_noise:
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


class ValueNetwork(nn.Module):
    """Value network for PPO critic."""
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: list = [256, 256],
    ):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Mish())
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
        self.net[-1].weight.data *= 0.01
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


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
        min_noise_std: Minimum exploration noise std
        max_noise_std: Maximum exploration noise std
        noise_decay_type: Type of noise decay
        noise_decay_steps: Steps over which to decay noise
        critic_warmup_steps: Number of steps to train critic only
        reward_scale: Scale rewards for critic stability
        value_target_tau: Soft update rate for target value network
        use_target_value_net: Whether to use target value network
        value_target_clip: Clip value targets
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
        min_noise_std: float = 0.01,
        max_noise_std: float = 0.3,
        noise_decay_type: Literal["constant", "linear", "exponential"] = "constant",
        noise_decay_steps: int = 100000,
        critic_warmup_steps: int = 0,
        reward_scale: float = 1.0,
        value_target_tau: float = 0.005,
        use_target_value_net: bool = True,
        value_target_clip: float = 100.0,
        action_bounds: Optional[tuple] = None,
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
        
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        self.initial_max_noise_std = max_noise_std
        self.noise_decay_type = noise_decay_type
        self.noise_decay_steps = noise_decay_steps
        
        self.critic_warmup_steps = critic_warmup_steps
        self._current_step = 0
        
        self.reward_scale = reward_scale
        self.value_target_tau = value_target_tau
        self.use_target_value_net = use_target_value_net
        self.value_target_clip = value_target_clip
        self.action_bounds = action_bounds
        
        self.fixed_step_size = 1.0 / num_inference_steps
        
        self.noisy_velocity_net = NoisyVelocityUNet1D(
            base_velocity_net=velocity_net,
            obs_dim=obs_dim,
            action_dim=act_dim,
            min_noise_std=min_noise_std,
            max_noise_std=max_noise_std,
        )
        
        if use_ema:
            self.velocity_net_ema = copy.deepcopy(velocity_net)
            for param in self.velocity_net_ema.parameters():
                param.requires_grad = False
        else:
            self.velocity_net_ema = None
        
        self.value_net = ValueNetwork(obs_dim=obs_dim)
        
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
            decay_factor = math.exp(-3 * progress)
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
        """Soft update target value network."""
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
        """Compute flow matching BC loss (for pretraining/warmup)."""
        B = actions.shape[0]
        device = actions.device
        
        t = torch.rand(B, device=device)
        d = torch.full((B,), self.fixed_step_size, device=device)
        
        x_0 = torch.randn_like(actions)
        t_expand = t.view(B, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * actions
        
        target_v = actions - x_0
        pred_v, _, _ = self.noisy_velocity_net(x_t, t, d, obs_cond, sample_noise=False)
        
        loss = F.mse_loss(pred_v, target_v)
        
        return {"loss": loss, "bc_loss": loss}
    
    @torch.no_grad()
    def get_action(
        self,
        obs_cond: torch.Tensor,
        deterministic: bool = True,
        use_ema: bool = True,
        return_chains: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample action using fixed-step flow integration."""
        B = obs_cond.shape[0]
        device = obs_cond.device
        K = self.num_inference_steps
        
        x = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        
        dt = self.fixed_step_size
        d = torch.full((B,), dt, device=device)
        
        if return_chains:
            x_chain = torch.zeros((B, K + 1, self.pred_horizon, self.act_dim), device=device)
            x_chain[:, 0] = x.clone()
        else:
            x_chain = None
        
        if use_ema and self.velocity_net_ema is not None and deterministic:
            for i in range(K):
                t = torch.full((B,), i * dt, device=device)
                velocity = self.velocity_net_ema(x, t, d, obs_cond)
                x = x + velocity * dt
                if return_chains:
                    x_chain[:, i + 1] = x.clone()
        else:
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
                    if noise is not None:
                        x = x + noise
                
                if return_chains:
                    x_chain[:, i + 1] = x.clone()
        
        if self.action_bounds is not None:
            x = torch.clamp(x, self.action_bounds[0], self.action_bounds[1])
        return x, x_chain
    
    def compute_value(self, obs_cond: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Compute state value V(s_0)."""
        if use_target and self.use_target_value_net and self.value_net_target is not None:
            return self.value_net_target(obs_cond)
        return self.value_net(obs_cond)
    
    def compute_action_log_prob(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
        x_chain: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probability of actions using Markov chain transition probabilities."""
        if x_chain is None:
            raise ValueError("x_chain is required for accurate log_prob computation.")
        
        B = x_chain.shape[0]
        K = self.num_inference_steps
        device = x_chain.device
        
        dt = self.fixed_step_size
        d = torch.full((B,), dt, device=device)
        
        total_log_prob_trans = torch.zeros(B, device=device)
        total_entropy = torch.zeros(B, device=device)
        
        for k in range(K):
            x_k = x_chain[:, k].detach()
            x_next = x_chain[:, k + 1].detach()
            
            t = torch.full((B,), k * dt, device=device)
            velocity, _, noise_std = self.noisy_velocity_net(x_k, t, d, obs_cond, sample_noise=False)
            
            mean_next = x_k + velocity * dt
            trans_std = noise_std
            trans_std_expanded = trans_std.unsqueeze(1).expand(-1, self.pred_horizon, -1)
            
            diff = x_next - mean_next
            normalized_diff = diff / trans_std_expanded
            normalized_diff = torch.clamp(normalized_diff, -10.0, 10.0)
            
            log_prob_trans = -0.5 * normalized_diff ** 2 \
                             - torch.log(trans_std_expanded) \
                             - 0.5 * np.log(2 * np.pi)
            
            log_prob_trans = log_prob_trans.sum(dim=[1, 2])
            total_log_prob_trans = total_log_prob_trans + log_prob_trans
            
            entropy_per_dim = 0.5 * (1 + np.log(2 * np.pi)) + torch.log(trans_std_expanded)
            total_entropy = total_entropy + entropy_per_dim.sum(dim=[1, 2])
        
        log_prob = total_log_prob_trans
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
        """Compute PPO loss for online RL."""
        values = self.value_net(obs_cond).squeeze(-1)
        
        if clip_value and old_values is not None:
            v_clipped = old_values + torch.clamp(
                values - old_values, -value_clip_range, value_clip_range
            )
            v_loss_unclipped = (values - returns) ** 2
            v_loss_clipped = (v_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            value_loss = 0.5 * ((values - returns) ** 2).mean()
        
        new_log_probs, entropy = self.compute_action_log_prob(
            obs_cond, actions, x_chain=x_chain
        )
        
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
        
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        entropy_loss = -entropy.mean()
        
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy.mean(),
            "ratio_mean": ratio.mean(),
            "ratio_std": ratio.std(),
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
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
            },
        }
        if self.velocity_net_ema is not None:
            checkpoint["velocity_net_ema"] = self.velocity_net_ema.state_dict()
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
        """Load pretrained weights from AW-ShortCut Flow checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if "agent" in checkpoint or "ema_agent" in checkpoint:
            agent_state = checkpoint.get("ema_agent", checkpoint.get("agent", {}))
            
            velocity_weights = {}
            for k, v in agent_state.items():
                if k.startswith("velocity_net."):
                    new_key = k[len("velocity_net."):]
                    velocity_weights[new_key] = v
            
            if not velocity_weights:
                raise ValueError(f"No velocity_net weights found in checkpoint: {checkpoint_path}")
                
        elif "velocity_net" in checkpoint:
            velocity_weights = checkpoint["velocity_net"]
        elif "velocity_net_ema" in checkpoint:
            velocity_weights = checkpoint["velocity_net_ema"]
        elif "model" in checkpoint:
            velocity_weights = checkpoint["model"]
        else:
            velocity_weights = checkpoint
        
        missing_keys, unexpected_keys = self.noisy_velocity_net.base_net.load_state_dict(
            velocity_weights, strict=False
        )
        if missing_keys:
            print(f"Warning: Missing keys when loading velocity_net: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading velocity_net: {unexpected_keys[:5]}...")
        
        if self.velocity_net_ema is not None:
            self.velocity_net_ema.load_state_dict(velocity_weights, strict=False)
        
        print(f"Loaded velocity network from {checkpoint_path}")
