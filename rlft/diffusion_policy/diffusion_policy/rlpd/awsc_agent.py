"""
Advantage-Weighted ShortCut Flow (AWSC) Agent for RLPD Online RL.

Adapts AW-ShortCut Flow from diffusion_policy.algorithms to RLPD framework:
- Ensemble Q-networks (num_qs > 2 support)
- Online + Offline mixed training
- Pretrained velocity network loading
- RGB observation support

Three-stage pipeline:
- Stage 1: ShortCut Flow BC pretrain (pure BC)
- Stage 2: AW-ShortCut Flow offline RL (Q-weighted BC)
- Stage 3: AWSC Online RL (this module, RLPD-style fine-tuning)

References:
- AW-ShortCut Flow: diffusion_policy/algorithms/aw_shortcut_flow.py
- RLPD: Ball et al., "Efficient Online Reinforcement Learning with Offline Data", ICML 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from typing import Dict, Optional, Tuple, Literal

from .networks import EnsembleQNetwork, soft_update


class ShortCutVelocityUNet1D(nn.Module):
    """
    Velocity network with step size conditioning for ShortCut Flow.
    Extends ConditionalUnet1D to accept both time t and step size d.
    
    Copied from diffusion_policy.algorithms.shortcut_flow for independence.
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
        
        # Import here to avoid circular dependency
        from diffusion_policy.conditional_unet1d import ConditionalUnet1D
        
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
        
        # Use timestep as integer (scaled by 100 for embedding)
        timestep_int = (timestep * 100).long()
        
        output = self.unet(sample, timestep_int, global_cond=global_cond)
        
        return output


class AWSCAgent(nn.Module):
    """
    Advantage-Weighted ShortCut Flow Agent for RLPD Online RL.
    
    Key differences from AWShortCutFlowAgent:
    - EnsembleQNetwork instead of DoubleQNetwork
    - Support for pretrained velocity network loading
    - Online exploration with noise injection
    - RLPD-compatible interface
    - Policy-Critic data separation for avoiding failed sample pollution
    
    Args:
        velocity_net: ShortCutVelocityUNet1D for velocity prediction
        q_network: EnsembleQNetwork for Q-value estimation (optional, created internally if None)
        obs_dim: Dimension of observation features
        action_dim: Dimension of action space
        obs_horizon: Number of observation frames (default: 2)
        pred_horizon: Length of action sequence to predict (default: 16)
        act_horizon: Length of action sequence for Q-learning (default: 8)
        num_qs: Number of Q-networks in ensemble (default: 10)
        num_min_qs: Number of Q-networks for subsample + min (default: 2)
        max_denoising_steps: Maximum denoising steps (default: 8)
        num_inference_steps: Number of inference steps (default: 8)
        beta: Temperature for advantage weighting (default: 10.0)
        bc_weight: Weight for flow matching loss (default: 1.0)
        shortcut_weight: Weight for shortcut consistency loss (default: 0.3)
        self_consistency_k: Fraction of batch for consistency (default: 0.1)
        gamma: Discount factor (default: 0.99)
        tau: Soft update coefficient (default: 0.005)
        reward_scale: Scale factor for rewards (default: 1.0)
        q_target_clip: Clip range for Q target (default: 100.0)
        ema_decay: Decay rate for EMA velocity network (default: 0.999)
        weight_clip: Maximum weight to prevent outliers (default: 100.0)
        exploration_noise_std: Std of exploration noise (default: 0.1)
        step_size_mode: Step size sampling mode (default: "fixed")
        fixed_step_size: Fixed step size (default: 0.0625)
        target_mode: Shortcut target mode (default: "velocity")
        teacher_steps: Teacher rollout steps (default: 1)
        use_ema_teacher: Use EMA for teacher (default: True)
        inference_mode: Inference step mode (default: "uniform")
        filter_policy_data: Whether to filter policy training data by advantage (default: False)
        advantage_threshold: Minimum advantage for online samples to be used in policy training (default: 0.0)
        device: Device to run on (default: "cuda")
    """
    
    def __init__(
        self,
        velocity_net: ShortCutVelocityUNet1D,
        q_network: Optional[EnsembleQNetwork] = None,
        obs_dim: int = 256,
        action_dim: int = 7,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        act_horizon: int = 8,
        # Ensemble Q parameters
        num_qs: int = 10,
        num_min_qs: int = 2,
        q_hidden_dims: list = [256, 256, 256],
        # ShortCut Flow parameters
        max_denoising_steps: int = 8,
        num_inference_steps: int = 8,
        # AWAC parameters
        beta: float = 10.0,
        bc_weight: float = 1.0,
        shortcut_weight: float = 0.3,
        self_consistency_k: float = 0.1,
        # RL parameters
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scale: float = 1.0,
        q_target_clip: float = 100.0,
        ema_decay: float = 0.999,
        weight_clip: float = 100.0,
        # Exploration parameters
        exploration_noise_std: float = 0.1,
        # ShortCut Flow specific
        step_size_mode: Literal["power2", "uniform", "fixed"] = "fixed",
        fixed_step_size: float = 0.0625,  # 1/16
        min_step_size: float = 0.0625,
        max_step_size: float = 0.125,
        target_mode: Literal["velocity", "endpoint"] = "velocity",
        teacher_steps: int = 1,
        use_ema_teacher: bool = True,
        t_min: float = 0.0,
        t_max: float = 1.0,
        inference_mode: Literal["adaptive", "uniform"] = "uniform",
        # Policy-Critic data separation
        filter_policy_data: bool = False,
        advantage_threshold: float = 0.0,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.velocity_net = velocity_net
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.max_denoising_steps = max_denoising_steps
        self.num_inference_steps = num_inference_steps
        self.device = device
        
        # AWAC hyperparameters
        self.beta = beta
        self.bc_weight = bc_weight
        self.shortcut_weight = shortcut_weight
        self.self_consistency_k = self_consistency_k
        
        # RL hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.q_target_clip = q_target_clip
        self.ema_decay = ema_decay
        self.weight_clip = weight_clip
        self.exploration_noise_std = exploration_noise_std
        
        # ShortCut Flow parameters
        self.step_size_mode = step_size_mode
        self.fixed_step_size = fixed_step_size
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.target_mode = target_mode
        self.teacher_steps = teacher_steps
        self.use_ema_teacher = use_ema_teacher
        self.t_min = t_min
        self.t_max = t_max
        self.inference_mode = inference_mode
        
        # Policy-Critic data separation parameters
        self.filter_policy_data = filter_policy_data
        self.advantage_threshold = advantage_threshold
        
        # Log2 of max steps for power2 mode
        self.log_max_steps = int(math.log2(max_denoising_steps))
        
        # Create or use provided Q-network (Ensemble)
        if q_network is not None:
            self.critic = q_network
        else:
            self.critic = EnsembleQNetwork(
                action_dim=action_dim,
                obs_dim=obs_dim,
                action_horizon=act_horizon,
                hidden_dims=q_hidden_dims,
                num_qs=num_qs,
                num_min_qs=num_min_qs,
            )
        
        # EMA velocity network for shortcut targets
        self.velocity_net_ema = copy.deepcopy(self.velocity_net)
        for p in self.velocity_net_ema.parameters():
            p.requires_grad = False
        
        # Target critic (frozen copy)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False
    
    def load_pretrained(
        self,
        checkpoint_path: str,
        load_critic: bool = False,
        strict: bool = False,
        use_ema: bool = True,
    ):
        """
        Load pretrained velocity network from ShortCut Flow BC or AW-ShortCut Flow checkpoint.
        
        This method supports loading from both offline RL (AWShortCutFlowAgent) and 
        pure BC (ShortCutFlowAgent) checkpoints. When both offline and online training
        use EnsembleQNetwork (recommended), critic weights can be directly transferred.
        
        Args:
            checkpoint_path: Path to checkpoint file (.pt)
            load_critic: Whether to load critic weights (if available).
                Recommended when continuing from AW-ShortCut offline RL checkpoint.
            strict: Whether to enforce strict state dict matching
            use_ema: Whether to load from ema_agent (recommended for best performance).
                The EMA model is typically better for evaluation/inference.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Prefer ema_agent if available (EMA weights perform better)
        if use_ema and "ema_agent" in checkpoint:
            agent_state = checkpoint["ema_agent"]
            print(f"Loading from ema_agent (EMA weights)")
        elif "agent" in checkpoint:
            agent_state = checkpoint["agent"]
            if use_ema:
                print(f"Warning: ema_agent not found, loading from agent")
        elif "velocity_net" in checkpoint:
            agent_state = checkpoint
        else:
            agent_state = checkpoint
        
        # Load velocity network
        velocity_state = {}
        for key, value in agent_state.items():
            if key.startswith("velocity_net."):
                velocity_state[key.replace("velocity_net.", "")] = value
        
        if velocity_state:
            self.velocity_net.load_state_dict(velocity_state, strict=strict)
            print(f"Loaded velocity network from {checkpoint_path}")
        
        # Also update EMA network
        self.velocity_net_ema.load_state_dict(self.velocity_net.state_dict())
        
        # Optionally load critic
        if load_critic:
            critic_state = {}
            for key, value in agent_state.items():
                if key.startswith("critic."):
                    critic_state[key.replace("critic.", "")] = value
            
            if critic_state:
                # Check critic architecture compatibility
                has_q_nets = any("q_nets" in k for k in critic_state.keys())
                has_q1_q2 = any("q1_net" in k or "q2_net" in k for k in critic_state.keys())
                
                is_ensemble = hasattr(self.critic, 'q_nets')
                
                if has_q_nets and is_ensemble:
                    # Both use EnsembleQNetwork - direct load
                    try:
                        self.critic.load_state_dict(critic_state, strict=strict)
                        self.critic_target.load_state_dict(self.critic.state_dict())
                        print(f"Loaded EnsembleQNetwork critic from {checkpoint_path}")
                    except Exception as e:
                        print(f"Warning: Could not load EnsembleQNetwork critic: {e}")
                elif has_q1_q2 and not is_ensemble:
                    # Both use DoubleQNetwork - direct load
                    try:
                        self.critic.load_state_dict(critic_state, strict=strict)
                        self.critic_target.load_state_dict(self.critic.state_dict())
                        print(f"Loaded DoubleQNetwork critic from {checkpoint_path}")
                    except Exception as e:
                        print(f"Warning: Could not load DoubleQNetwork critic: {e}")
                elif has_q1_q2 and is_ensemble:
                    # Checkpoint has DoubleQNetwork, we have EnsembleQNetwork
                    # This is a mismatch - warn user
                    print(f"Warning: Checkpoint has DoubleQNetwork but current agent uses EnsembleQNetwork.")
                    print(f"Critic weights not loaded. Consider using --use_ensemble_q for offline training.")
                elif has_q_nets and not is_ensemble:
                    # Checkpoint has EnsembleQNetwork, we have DoubleQNetwork
                    print(f"Warning: Checkpoint has EnsembleQNetwork but current agent uses DoubleQNetwork.")
                    print(f"Critic weights not loaded.")
                else:
                    # Try generic load with strict=False
                    try:
                        self.critic.load_state_dict(critic_state, strict=False)
                        self.critic_target.load_state_dict(self.critic.state_dict())
                        print(f"Loaded critic from {checkpoint_path} (partial match)")
                    except Exception as e:
                        print(f"Warning: Could not load critic weights: {e}")
    
    def _sample_step_size(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample step sizes d based on step_size_mode."""
        if self.step_size_mode == "power2":
            powers = torch.randint(0, self.log_max_steps + 1, (batch_size,), device=device)
            d = (2.0 ** powers.float()) / self.max_denoising_steps
        elif self.step_size_mode == "uniform":
            d = self.min_step_size + torch.rand(batch_size, device=device) * (self.max_step_size - self.min_step_size)
        elif self.step_size_mode == "fixed":
            d = torch.full((batch_size,), self.fixed_step_size, device=device)
        else:
            raise ValueError(f"Unknown step_size_mode: {self.step_size_mode}")
        return d
    
    def _linear_interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation: x_t = (1 - t) * x_0 + t * x_1"""
        t_expand = t.view(-1, 1, 1)
        return (1 - t_expand) * x_0 + t_expand * x_1
    
    def get_action(
        self,
        obs_features: torch.Tensor,
        deterministic: bool = False,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """
        Sample action sequence from the policy for evaluation.
        
        IMPORTANT: Returns full pred_horizon sequence for compatibility with AgentWrapper,
        which handles the action chunk slicing ([:, obs_horizon-1:obs_horizon-1+act_horizon]).
        
        Args:
            obs_features: (B, obs_dim) encoded observation features
            deterministic: Whether to disable exploration noise (ignored, flow has no noise)
            use_ema: Whether to use EMA network
            
        Returns:
            actions: (B, pred_horizon, action_dim) full action sequence
        """
        # Sample full action sequence (pred_horizon length)
        actions_full = self._sample_actions_batch(obs_features, use_ema=use_ema)
        
        return actions_full
    
    @torch.no_grad()
    def select_action(
        self,
        obs_features: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Select action for environment interaction (no gradient).
        
        This is used during online training for env interaction.
        Returns act_horizon actions with optional exploration noise.
        
        Args:
            obs_features: (B, obs_dim) or (obs_dim,) observation features
            deterministic: Whether to disable exploration noise
            
        Returns:
            action: (B, act_horizon, action_dim) or (act_horizon, action_dim)
        """
        squeeze = False
        if obs_features.dim() == 1:
            obs_features = obs_features.unsqueeze(0)
            squeeze = True
        
        # Sample full action sequence
        actions_full = self._sample_actions_batch(obs_features, use_ema=True)
        
        # Extract act_horizon chunk for execution
        actions = actions_full[:, :self.act_horizon, :]
        
        # Add exploration noise if not deterministic
        if not deterministic and self.exploration_noise_std > 0:
            noise = torch.randn_like(actions) * self.exploration_noise_std
            actions = torch.clamp(actions + noise, -1.0, 1.0)
        
        if squeeze:
            actions = actions.squeeze(0)
        
        return actions
    
    def _sample_actions_batch(
        self,
        obs_cond: torch.Tensor,
        use_ema: bool = True
    ) -> torch.Tensor:
        """Sample actions using flow ODE integration."""
        B = obs_cond.shape[0]
        device = obs_cond.device
        net = self.velocity_net_ema if use_ema else self.velocity_net
        
        # Ensure obs_cond is flattened
        if obs_cond.dim() == 3:
            obs_cond = obs_cond.reshape(B, -1)
        
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        
        # Use uniform steps for sampling
        dt = 1.0 / self.num_inference_steps
        d = torch.full((B,), dt, device=device)
        
        for i in range(self.num_inference_steps):
            t = torch.full((B,), i * dt, device=device)
            v = net(x, t, d, obs_cond)
            x = x + dt * v
        
        return torch.clamp(x, -1.0, 1.0)
    
    def compute_actor_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
        actions_for_q: Optional[torch.Tensor] = None,
        is_demo: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute actor loss: Q-weighted Flow BC + Shortcut Consistency.
        
        Supports Policy-Critic data separation:
        - If filter_policy_data=True and is_demo is provided:
          - Keep all demo samples (is_demo=True)
          - Filter online samples by advantage threshold
          - If no valid samples remain, use only demo samples
        
        Args:
            obs_features: (B, obs_dim) observation features
            actions: (B, pred_horizon, action_dim) expert actions for BC
            actions_for_q: (B, act_horizon, action_dim) actions for Q-weighting
            is_demo: (B,) bool tensor indicating if sample is from offline demos
            
        Returns:
            actor_loss: Scalar loss
            metrics: Dictionary of metrics
        """
        if actions_for_q is None:
            actions_for_q = actions[:, :self.act_horizon]
        
        # Flatten obs if needed
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        else:
            obs_cond = obs_features
        
        B = actions.shape[0]
        device = actions.device
        
        # ===== Compute AWAC-style weights from Ensemble Q-values =====
        with torch.no_grad():
            # Use subsample + min for conservative Q estimate
            q_data = self.critic.get_min_q(actions_for_q, obs_cond, random_subset=True)
            
            # Compute advantage: A(s,a) = Q(s,a) - baseline
            baseline = q_data.mean()
            advantage = q_data - baseline
            
            # AWAC weights: w = exp(β * A), clamped and normalized
            weights = torch.clamp(torch.exp(self.beta * advantage), max=self.weight_clip)
            weights = weights / weights.mean()
            weights = weights.squeeze(-1)  # [B]
            advantage_flat = advantage.squeeze(-1)  # [B]
        
        # ===== Policy-Critic Data Separation =====
        # Filter data for policy training: keep demos + high-advantage online samples
        n_demo = 0
        n_online_kept = 0
        n_online_filtered = 0
        
        if self.filter_policy_data and is_demo is not None:
            # Identify demo and online samples
            demo_mask = is_demo.bool()
            online_mask = ~demo_mask
            n_demo = demo_mask.sum().item()
            
            # Filter online samples by advantage threshold
            online_high_adv = online_mask & (advantage_flat > self.advantage_threshold)
            n_online_kept = online_high_adv.sum().item()
            n_online_filtered = online_mask.sum().item() - n_online_kept
            
            # Combine: demo + high-advantage online
            policy_mask = demo_mask | online_high_adv
            
            # If no valid samples, fall back to demos only
            if policy_mask.sum() == 0:
                policy_mask = demo_mask
                # If still no demos, use all data (shouldn't happen in practice)
                if policy_mask.sum() == 0:
                    policy_mask = torch.ones(B, dtype=torch.bool, device=device)
            
            # Apply filter
            policy_indices = policy_mask.nonzero(as_tuple=True)[0]
            obs_cond_policy = obs_cond[policy_indices]
            actions_policy = actions[policy_indices]
            weights_policy = weights[policy_indices]
            
            # Re-normalize weights for filtered data
            if weights_policy.sum() > 0:
                weights_policy = weights_policy / weights_policy.mean()
            
            B_policy = len(policy_indices)
        else:
            # No filtering - use all data
            obs_cond_policy = obs_cond
            actions_policy = actions
            weights_policy = weights
            B_policy = B
        
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(actions_policy)
        
        # Sample step sizes
        d = self._sample_step_size(B_policy, device)
        
        # ===== Q-Weighted Flow Matching Loss =====
        t_flow = torch.rand(B_policy, device=device)
        x_t = self._linear_interpolate(x_0, actions_policy, t_flow)
        
        # Target velocity: v = x_1 - x_0
        v_target = actions_policy - x_0
        
        # Predict velocity
        v_pred = self.velocity_net(x_t, t_flow, d, obs_cond_policy)
        
        # Per-sample MSE loss [B, T, D] -> [B]
        flow_loss_per_sample = F.mse_loss(v_pred, v_target, reduction="none")
        flow_loss_per_sample = flow_loss_per_sample.mean(dim=(1, 2))
        
        # Weighted average
        flow_loss = (weights_policy * flow_loss_per_sample).mean()
        
        # ===== Q-Weighted Shortcut Consistency Loss =====
        shortcut_loss = torch.tensor(0.0, device=device)
        
        if self.shortcut_weight > 0 and self.self_consistency_k > 0:
            n_consistency = max(1, int(B_policy * self.self_consistency_k))
            idx = torch.randperm(B_policy)[:n_consistency]
            
            x_0_sub = x_0[idx]
            actions_sub = actions_policy[idx]
            obs_sub = obs_cond_policy[idx]
            d_sub = d[idx]
            weights_sub = weights_policy[idx]
            
            # Sample time for consistency
            t_cons = self.t_min + torch.rand(n_consistency, device=device) * (self.t_max - self.t_min)
            
            # Ensure t + 2d <= 1
            d_double = 2 * d_sub
            max_t = (1.0 - d_double).clamp(min=self.t_min)
            t_cons = torch.min(t_cons, max_t)
            
            x_t_cons = self._linear_interpolate(x_0_sub, actions_sub, t_cons)
            
            valid_mask = (t_cons + d_double) <= 1.0
            
            if valid_mask.sum() > 0:
                x_t_valid = x_t_cons[valid_mask]
                t_valid = t_cons[valid_mask]
                d_valid = d_sub[valid_mask]
                d_double_valid = d_double[valid_mask]
                obs_valid = obs_sub[valid_mask]
                x_0_valid = x_0_sub[valid_mask]
                actions_valid = actions_sub[valid_mask]
                weights_valid = weights_sub[valid_mask]
                
                # Compute shortcut target
                shortcut_target = self._compute_shortcut_target(
                    x_t_valid, t_valid, d_valid, obs_valid, x_0_valid, actions_valid
                )
                
                # Predict with 2d step size
                v_pred_2d = self.velocity_net(x_t_valid, t_valid, d_double_valid, obs_valid)
                
                if self.target_mode == "velocity":
                    shortcut_loss_per_sample = F.mse_loss(v_pred_2d, shortcut_target, reduction="none")
                    shortcut_loss_per_sample = shortcut_loss_per_sample.mean(dim=(1, 2))
                else:
                    d_double_expand = d_double_valid.view(-1, 1, 1)
                    pred_endpoint = x_t_valid + d_double_expand * v_pred_2d
                    shortcut_loss_per_sample = F.mse_loss(pred_endpoint, shortcut_target, reduction="none")
                    shortcut_loss_per_sample = shortcut_loss_per_sample.mean(dim=(1, 2))
                
                shortcut_loss = (weights_valid * shortcut_loss_per_sample).mean()
        
        # Total actor loss
        actor_loss = self.bc_weight * flow_loss + self.shortcut_weight * shortcut_loss
        
        metrics = {
            "actor_loss": actor_loss.item(),
            "flow_loss": flow_loss.item(),
            "shortcut_loss": shortcut_loss.item() if isinstance(shortcut_loss, torch.Tensor) else shortcut_loss,
            "q_mean": q_data.mean().item(),
            "weight_mean": weights.mean().item(),
            "weight_std": weights.std().item(),
            "advantage_mean": advantage.mean().item(),
            # Data separation metrics
            "policy_batch_size": B_policy,
            "n_demo_samples": n_demo,
            "n_online_kept": n_online_kept,
            "n_online_filtered": n_online_filtered,
        }
        
        return actor_loss, metrics
    
    def _compute_shortcut_target(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        d: torch.Tensor,
        obs_cond: torch.Tensor,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute shortcut target using teacher network."""
        teacher_net = self.velocity_net_ema if self.use_ema_teacher else self.velocity_net
        
        with torch.no_grad():
            d_expand = d.view(-1, 1, 1)
            
            if self.teacher_steps == 1:
                # Two-step teacher
                v_1 = teacher_net(x_t, t, d, obs_cond)
                x_t_plus_d = x_t + d_expand * v_1
                
                t_plus_d = t + d
                v_2 = teacher_net(x_t_plus_d, t_plus_d, d, obs_cond)
                
                target_endpoint = x_t + d_expand * v_1 + d_expand * v_2
                shortcut_v = (v_1 + v_2) / 2
            else:
                # Multi-step rollout
                x = x_t.clone()
                current_t = t.clone()
                
                for step in range(self.teacher_steps):
                    v = teacher_net(x, current_t, d, obs_cond)
                    x = x + d_expand * v
                    current_t = current_t + d
                
                target_endpoint = x
                shortcut_v = (target_endpoint - x_t) / (self.teacher_steps * d_expand)
            
            if self.target_mode == "velocity":
                return shortcut_v
            else:
                return target_endpoint
    
    def compute_critic_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
        next_obs_features: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        cumulative_reward: Optional[torch.Tensor] = None,
        chunk_done: Optional[torch.Tensor] = None,
        discount_factor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute critic loss using SMDP Bellman equation with Ensemble Q.
        
        Args:
            obs_features: (B, obs_dim) current observations
            actions: (B, act_horizon, action_dim) action chunks
            next_obs_features: (B, obs_dim) next observations
            rewards: (B,) single-step rewards
            dones: (B,) done flags
            cumulative_reward: (B,) SMDP cumulative rewards
            chunk_done: (B,) SMDP done flags
            discount_factor: (B,) SMDP discount factors
            
        Returns:
            critic_loss: Scalar loss
            metrics: Dictionary of metrics
        """
        # Flatten obs if needed
        if obs_features.dim() == 3:
            obs_cond = obs_features.reshape(obs_features.shape[0], -1)
        else:
            obs_cond = obs_features
            
        if next_obs_features.dim() == 3:
            next_obs_cond = next_obs_features.reshape(next_obs_features.shape[0], -1)
        else:
            next_obs_cond = next_obs_features
        
        # Use SMDP fields if provided
        if cumulative_reward is not None:
            r = cumulative_reward
            d = chunk_done if chunk_done is not None else dones
            gamma_tau = discount_factor if discount_factor is not None else torch.full_like(r, self.gamma)
        else:
            r = rewards
            d = dones
            gamma_tau = torch.full_like(r, self.gamma)
        
        # Ensure proper shapes
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        if d.dim() == 1:
            d = d.unsqueeze(-1)
        if gamma_tau.dim() == 1:
            gamma_tau = gamma_tau.unsqueeze(-1)
        
        # Scale rewards
        scaled_rewards = r * self.reward_scale
        
        with torch.no_grad():
            # Sample next actions using EMA policy
            next_actions_full = self._sample_actions_batch(next_obs_cond, use_ema=True)
            next_actions = next_actions_full[:, :self.act_horizon, :]
            
            # Compute target Q-values (subsample + min)
            target_q = self.critic_target.get_min_q(next_actions, next_obs_cond, random_subset=True)
            
            # TD target with SMDP discount
            target_q = scaled_rewards + (1 - d) * gamma_tau * target_q
            
            if self.q_target_clip is not None:
                target_q = torch.clamp(target_q, -self.q_target_clip, self.q_target_clip)
        
        # Current Q-values from all ensemble members
        q_values = self.critic(actions, obs_cond)  # (num_qs, B, 1)
        
        # MSE loss for each Q-network
        critic_loss = 0.0
        for q in q_values:
            critic_loss = critic_loss + F.mse_loss(q, target_q)
        
        metrics = {
            "critic_loss": critic_loss.item(),
            "q_mean": q_values.mean().item(),
            "q_std": q_values.std().item(),
            "td_target_mean": target_q.mean().item(),
        }
        
        return critic_loss, metrics
    
    def compute_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
        next_obs_features: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        actions_for_q: Optional[torch.Tensor] = None,
        cumulative_reward: Optional[torch.Tensor] = None,
        chunk_done: Optional[torch.Tensor] = None,
        discount_factor: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for training.
        
        Note: In practice, actor and critic should be updated with different optimizers.
        
        Returns:
            Dictionary of losses and metrics
        """
        if actions_for_q is None:
            actions_for_q = actions[:, :self.act_horizon]
        
        # Compute losses
        actor_loss, actor_metrics = self.compute_actor_loss(
            obs_features, actions, actions_for_q
        )
        
        critic_loss, critic_metrics = self.compute_critic_loss(
            obs_features, actions_for_q, next_obs_features, rewards, dones,
            cumulative_reward, chunk_done, discount_factor
        )
        
        # Combine
        result = {
            "loss": actor_loss + critic_loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
        }
        result.update(actor_metrics)
        result.update(critic_metrics)
        
        return result
    
    def update_ema(self):
        """Update EMA velocity network."""
        soft_update(self.velocity_net_ema, self.velocity_net, 1 - self.ema_decay)
    
    def update_target(self):
        """Soft update target critic network."""
        soft_update(self.critic_target, self.critic, self.tau)
