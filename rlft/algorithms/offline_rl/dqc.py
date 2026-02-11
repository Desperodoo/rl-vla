"""
Decoupled Q-Chunking (DQC) Agent

Implements the DQC algorithm (Li, Park, Levine 2025, arXiv:2512.10926) for
offline RL with action chunking. DQC decouples the critic's chunk length from
the policy's chunk length, enabling better temporal credit assignment.

Key Architecture (4 networks):
- Chunk Critic Q^h(s, a_{1:h}): Trained with Bellman equation, uses V-target.
  Horizon h = backup_horizon (can be larger than policy chunk).
- Action Critic Q^{h_a}(s, a_{1:h_a}): Distilled from chunk critic via
  expectile regression. Horizon h_a = act_horizon. 
- Value V(s): IQL expectile regression from target action critic.
- Actor π(a_{1:h_a}|s): Flow matching (pure BC), uses best-of-N at inference.

Key Design Principles:
1. All Q-networks use sigmoid parameterization [0,1] + BCE loss for stability.
2. Actor is pure BC (flow matching) — no Q gradient to policy.
3. Best-of-N inference: sample N actions, score with action critic, pick best.
4. V-target avoids OOD action evaluation in Bellman bootstrap (IQL-style).
5. Dual critic architecture decouples backup horizon from policy horizon.

Training Order (4 steps, separate optimizers):
  Step 1: Value network — expectile regression from target action critic
  Step 2: Chunk critic  — Bellman with V-target bootstrap + BCE loss
  Step 3: Action critic — expectile regression distillation from chunk critic
  Step 4: Actor         — pure flow matching BC loss

References:
- DQC: Li, Park, Levine. "Decoupled Q-Chunking for Offline RL"
  arXiv:2512.10926. Code: github.com/ColinQiyangLi/dqc
- IQL: Kostrikov et al. "Offline RL with Implicit Q-Learning" ICLR 2022
- AWCP: Advantage-Weighted Consistency Policy (local codebase, similar structure)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Optional, Dict, Tuple

from rlft.networks import VelocityUNet1D, SigmoidQNetwork, soft_update


class ValueNetwork(nn.Module):
    """State Value Network V(s) for IQL-style value estimation.
    
    Simple MLP that maps observations to a scalar value estimate.
    Identical to the one in sac.py but included here for self-containedness.
    """
    
    def __init__(self, obs_dim: int, hidden_dims: list = [256, 256, 256]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.LayerNorm(h), nn.Mish()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        final = self.net[-1]
        nn.init.orthogonal_(final.weight, gain=0.01)
        nn.init.zeros_(final.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns V(s): (B, 1)."""
        return self.net(obs)


class DQCAgent(nn.Module):
    """Decoupled Q-Chunking Agent.
    
    Combines flow-based actor (pure BC) with dual sigmoid Q-critics and
    IQL-style value network. Uses best-of-N inference for policy improvement.
    
    Network Architecture:
    --------------------
    1. velocity_net: VelocityUNet1D — flow actor (pred_horizon actions)
    2. chunk_critic: SigmoidQNetwork — Q^h(s, a_{1:h}), h=backup_horizon
    3. action_critic: SigmoidQNetwork — Q^{h_a}(s, a_{1:h_a}), h_a=act_horizon
    4. value_net: ValueNetwork — V(s), IQL expectile from action critic
    
    + Target networks (frozen, soft-updated):
    - velocity_net_ema: EMA of actor (for chunk critic target actions)
    - chunk_critic_target: soft update target
    - action_critic_target: soft update target for value regression
    - value_target: soft update target for Bellman bootstrap
    
    Loss Functions:
    ---------------
    Value:  L_V = E[|κ_b - 1(Q^{h_a}_target - V < 0)| * (Q^{h_a}_target - V)^2]
    Chunk:  L_Q^h = BCE(Q^h_logit, sigmoid_target) where target from V-bootstrap
    Action: L_Q^{h_a} = E[|κ_d - 1(Q^h - Q^{h_a} < 0)| * (Q^h - Q^{h_a})^2]
    Actor:  L_π = E[||v_θ(x_t, t) - (x_1 - x_0)||^2] (pure flow matching)
    
    Args:
        velocity_net: VelocityUNet1D for flow-based policy
        action_dim: Dimension of action space
        obs_dim: Dimension of flattened observation features
        obs_horizon: Number of observation frames (default: 2)
        pred_horizon: Actor prediction horizon (default: 16)
        act_horizon: Action critic horizon h_a (default: 8)
        backup_horizon: Chunk critic horizon h (default: 16)
            Set to pred_horizon by default to avoid dataset changes.
            DQC paper uses h=25 but that requires longer action sequences.
        num_flow_steps: ODE steps for flow sampling (default: 20)
        q_hidden_dims: Hidden dims for Q-networks and V-network
        num_chunk_qs: Number of Q-nets in chunk critic ensemble (default: 2)
        num_action_qs: Number of Q-nets in action critic ensemble (default: 2)
        kappa_b: Expectile for V-network (backup expectile, default: 0.9)
            Higher → more optimistic V → stronger offline improvement
        kappa_d: Expectile for action critic distillation (default: 0.8)
            Higher → action critic tracks upper quantiles of chunk critic
        gamma: Discount factor
        tau: Soft update coefficient for target networks
        ema_decay: EMA decay for actor velocity network
        reward_scale: Scale factor for rewards in Bellman target
        q_target_clip: Clip range for Bellman target (before sigmoid)
        best_of_n: Number of candidates for best-of-N inference
        action_bounds: Optional action clamping bounds
        device: Device to run on
    """
    
    def __init__(
        self,
        velocity_net: VelocityUNet1D,
        action_dim: int,
        obs_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        act_horizon: int = 8,
        backup_horizon: int = 16,
        num_flow_steps: int = 20,
        q_hidden_dims: list = [256, 256, 256],
        num_chunk_qs: int = 2,
        num_action_qs: int = 2,
        kappa_b: float = 0.9,
        kappa_d: float = 0.8,
        gamma: float = 0.99,
        tau: float = 0.005,
        ema_decay: float = 0.999,
        reward_scale: float = 1.0,
        q_target_clip: float = 100.0,
        best_of_n: int = 32,
        action_bounds: Optional[tuple] = None,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.velocity_net = velocity_net
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.backup_horizon = backup_horizon
        self.num_flow_steps = num_flow_steps
        self.kappa_b = kappa_b
        self.kappa_d = kappa_d
        self.gamma = gamma
        self.tau = tau
        self.ema_decay = ema_decay
        self.reward_scale = reward_scale
        self.q_target_clip = q_target_clip
        self.best_of_n = best_of_n
        self.action_bounds = action_bounds
        self.device = device
        
        # --- Chunk Critic: Q^h(s, a_{1:h}) ---
        self.chunk_critic = SigmoidQNetwork(
            action_dim=action_dim,
            obs_dim=obs_dim,
            action_horizon=backup_horizon,
            hidden_dims=q_hidden_dims,
            num_qs=num_chunk_qs,
        )
        
        # --- Action Critic: Q^{h_a}(s, a_{1:h_a}) ---
        self.action_critic = SigmoidQNetwork(
            action_dim=action_dim,
            obs_dim=obs_dim,
            action_horizon=act_horizon,
            hidden_dims=q_hidden_dims,
            num_qs=num_action_qs,
        )
        
        # --- Value Network: V(s) ---
        self.value_net = ValueNetwork(obs_dim, q_hidden_dims)
        
        # --- Target networks (frozen) ---
        self.velocity_net_ema = copy.deepcopy(self.velocity_net)
        for p in self.velocity_net_ema.parameters():
            p.requires_grad = False
        
        self.chunk_critic_target = copy.deepcopy(self.chunk_critic)
        for p in self.chunk_critic_target.parameters():
            p.requires_grad = False
        
        self.action_critic_target = copy.deepcopy(self.action_critic)
        for p in self.action_critic_target.parameters():
            p.requires_grad = False
        
        self.value_target = copy.deepcopy(self.value_net)
        for p in self.value_target.parameters():
            p.requires_grad = False
    
    def _flatten_obs(self, obs_features: torch.Tensor) -> torch.Tensor:
        """Flatten (B, obs_horizon, feat_dim) → (B, obs_horizon*feat_dim)."""
        if obs_features.dim() == 3:
            return obs_features.reshape(obs_features.shape[0], -1)
        return obs_features
    
    # =========================================================================
    # Step 1: Value Network Loss (IQL expectile regression)
    # =========================================================================
    
    def compute_value_loss(
        self,
        obs_cond: torch.Tensor,
        actions_for_q: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """IQL expectile regression: V(s) ← Q^{h_a}_target(s, a_data).
        
        L_V = E[|κ_b - 1(Q_target - V < 0)| * (Q_target - V)^2]
        
        Uses target action critic (not chunk critic) for the regression target.
        This is the key mechanism that avoids OOD action evaluation.
        
        Args:
            obs_cond: Flattened observation features (B, obs_dim)
            actions_for_q: Data actions (B, act_horizon, action_dim) for Q evaluation
        """
        with torch.no_grad():
            q_target = self.action_critic_target.get_min_q(actions_for_q, obs_cond)  # (B, 1)
        
        v = self.value_net(obs_cond)  # (B, 1)
        diff = q_target - v
        
        # Asymmetric loss: κ_b for positive diff, (1 - κ_b) for negative
        weight = torch.where(diff > 0, self.kappa_b, 1.0 - self.kappa_b)
        value_loss = (weight * diff.pow(2)).mean()
        
        metrics = {
            "v_mean": v.mean().item(),
            "v_loss": value_loss.item(),
            "v_target_q_mean": q_target.mean().item(),
            "v_advantage_mean": diff.mean().item(),
        }
        
        return value_loss, metrics
    
    # =========================================================================
    # Step 2: Chunk Critic Loss (Bellman + V-target + BCE)
    # =========================================================================
    
    def compute_chunk_critic_loss(
        self,
        obs_cond: torch.Tensor,
        next_obs_cond: torch.Tensor,
        actions: torch.Tensor,
        cumulative_reward: torch.Tensor,
        chunk_done: torch.Tensor,
        discount_factor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Chunk critic Bellman update with V-target bootstrap and BCE loss.
        
        Target: y = reward_scale * R^(τ) + (1 - d^(τ)) * γ^τ * V_target(s')
        Loss:   BCE(Q^h_logit, sigmoid(y))     — for each Q in ensemble
        
        Uses V-target for bootstrap (IQL-style, no OOD action evaluation).
        Sigmoid targets mean the critic predicts "goodness probability".
        
        Args:
            obs_cond: Current observation (B, obs_dim) 
            next_obs_cond: Next observation (B, obs_dim)
            actions: Full action sequence (B, pred_horizon, action_dim),
                     truncated to backup_horizon for chunk critic
            cumulative_reward: SMDP cumulative reward (B,) or (B, 1)
            chunk_done: SMDP done flag (B,) or (B, 1)
            discount_factor: SMDP discount γ^τ (B,) or (B, 1)
        """
        r = cumulative_reward
        d = chunk_done
        gamma_tau = discount_factor
        
        if r.dim() == 1: r = r.unsqueeze(-1)
        if d.dim() == 1: d = d.unsqueeze(-1)
        if gamma_tau.dim() == 1: gamma_tau = gamma_tau.unsqueeze(-1)
        
        scaled_rewards = r * self.reward_scale
        
        # Truncate actions to backup_horizon for chunk critic
        actions_for_chunk = actions[:, :self.backup_horizon, :]
        
        with torch.no_grad():
            # V-target bootstrap (IQL-style, avoids OOD actions)
            next_v = self.value_target(next_obs_cond)  # (B, 1)
            td_target = scaled_rewards + (1 - d) * gamma_tau * next_v
            
            if self.q_target_clip is not None:
                td_target = torch.clamp(td_target, -self.q_target_clip, self.q_target_clip)
            
            # Sigmoid target for BCE loss
            sigmoid_target = torch.sigmoid(td_target)  # (B, 1)
        
        # BCE loss on logits (numerically stable)
        q_logits = self.chunk_critic.forward_logits(actions_for_chunk, obs_cond)  # (num_qs, B, 1)
        
        chunk_loss = 0.0
        for i in range(q_logits.shape[0]):
            chunk_loss = chunk_loss + F.binary_cross_entropy_with_logits(
                q_logits[i], sigmoid_target
            )
        
        # Monitoring metrics
        with torch.no_grad():
            q_values = torch.sigmoid(q_logits)
        
        metrics = {
            "chunk_q_mean": q_values.mean().item(),
            "chunk_q_std": q_values.std(dim=0).mean().item(),
            "chunk_td_target_mean": td_target.mean().item(),
            "chunk_critic_loss": chunk_loss.item(),
        }
        
        return chunk_loss, metrics
    
    # =========================================================================
    # Step 3: Action Critic Loss (Distillation from Chunk Critic)
    # =========================================================================
    
    def compute_action_critic_loss(
        self,
        obs_cond: torch.Tensor,
        actions_for_q: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Distill action critic from chunk critic via expectile regression.
        
        Target: Q^h(s, a_{1:h}) from chunk critic (current, not target)
        Loss:   E[|κ_d - 1(target - Q^{h_a} < 0)| * (target - Q^{h_a})^2]
        
        The chunk critic evaluates the full backup_horizon action sequence,
        and the action critic is trained to approximate this with only act_horizon.
        Expectile regression (κ_d > 0.5) helps the action critic track the
        upper quantiles of the chunk critic's predictions.
        
        Args:
            obs_cond: Observation features (B, obs_dim)
            actions_for_q: Actions for action critic (B, act_horizon, action_dim)
            actions: Full action sequence (B, pred_horizon, action_dim)
        """
        actions_for_chunk = actions[:, :self.backup_horizon, :]
        
        with torch.no_grad():
            # Chunk critic target (using current chunk critic, not target network,
            # following DQC paper: distillation uses the latest Q^h)
            chunk_q = self.chunk_critic.get_min_q(actions_for_chunk, obs_cond)  # (B, 1)
        
        # Action critic predictions
        action_q_all = self.action_critic.forward(actions_for_q, obs_cond)  # (num_qs, B, 1)
        
        # Expectile regression for each Q network
        action_loss = 0.0
        for i in range(action_q_all.shape[0]):
            diff = chunk_q - action_q_all[i]
            weight = torch.where(diff > 0, self.kappa_d, 1.0 - self.kappa_d)
            action_loss = action_loss + (weight * diff.pow(2)).mean()
        
        metrics = {
            "action_q_mean": action_q_all.mean().item(),
            "action_q_std": action_q_all.std(dim=0).mean().item(),
            "chunk_q_distill_target": chunk_q.mean().item(),
            "action_critic_loss": action_loss.item(),
        }
        
        return action_loss, metrics
    
    # =========================================================================
    # Step 4: Actor Loss (Pure Flow Matching BC)
    # =========================================================================
    
    def compute_actor_loss(
        self,
        obs_cond: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Pure flow matching BC loss (no Q gradient to policy).
        
        L_π = E_{t~U(0,1)} [||v_θ(x_t, t, obs) - (x_1 - x_0)||^2]
        
        where x_t = (1-t)*x_0 + t*x_1, x_0 ~ N(0,I), x_1 = data actions.
        
        Actor quality improvement comes entirely from best-of-N at inference,
        where the action critic scores candidates.
        
        Args:
            obs_cond: Observation features (B, obs_dim)
            actions: Expert action sequence (B, pred_horizon, action_dim)
        """
        B = actions.shape[0]
        device = actions.device
        
        x_0 = torch.randn_like(actions)  # (B, pred_horizon, action_dim)
        x_1 = actions
        
        t = torch.rand(B, device=device)
        t_expand = t.view(-1, 1, 1)
        
        # Linear interpolation
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        
        # Target velocity
        v_target = x_1 - x_0
        
        # Predicted velocity
        v_pred = self.velocity_net(x_t, t, global_cond=obs_cond)
        
        # MSE flow matching loss
        actor_loss = F.mse_loss(v_pred, v_target)
        
        metrics = {
            "flow_loss": actor_loss.item(),
            "actor_loss": actor_loss.item(),
        }
        
        return actor_loss, metrics
    
    # =========================================================================
    # Combined Loss (for single-optimizer compatibility, optional)
    # =========================================================================
    
    def compute_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs_features: torch.Tensor,
        dones: torch.Tensor,
        actions_for_q: Optional[torch.Tensor] = None,
        cumulative_reward: Optional[torch.Tensor] = None,
        chunk_done: Optional[torch.Tensor] = None,
        discount_factor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses combined (for logging/debugging, not primary training).
        
        The primary training loop uses the individual compute_*_loss methods
        with separate optimizers. This method is provided for API compatibility.
        """
        if actions_for_q is None:
            actions_for_q = actions[:, :self.act_horizon, :]
        
        obs_cond = self._flatten_obs(obs_features)
        next_obs_cond = self._flatten_obs(next_obs_features)
        
        # Value loss
        value_loss, value_metrics = self.compute_value_loss(obs_cond, actions_for_q)
        
        # Chunk critic loss
        chunk_loss, chunk_metrics = self.compute_chunk_critic_loss(
            obs_cond, next_obs_cond, actions,
            cumulative_reward, chunk_done, discount_factor,
        )
        
        # Action critic loss
        action_loss, action_metrics = self.compute_action_critic_loss(
            obs_cond, actions_for_q, actions,
        )
        
        # Actor loss
        actor_loss, actor_metrics = self.compute_actor_loss(obs_cond, actions)
        
        total_loss = value_loss + chunk_loss + action_loss + actor_loss
        
        result = {
            "loss": total_loss,
            "actor_loss": actor_loss,
            "value_loss": value_loss,
            "chunk_critic_loss": chunk_loss,
            "action_critic_loss": action_loss,
            "flow_loss": actor_metrics["flow_loss"],
        }
        result.update(value_metrics)
        result.update(chunk_metrics)
        result.update(action_metrics)
        
        return result
    
    # =========================================================================
    # Target Updates
    # =========================================================================
    
    def update_targets(self):
        """Soft update all target networks."""
        soft_update(self.chunk_critic_target, self.chunk_critic, self.tau)
        soft_update(self.action_critic_target, self.action_critic, self.tau)
        soft_update(self.value_target, self.value_net, self.tau)
        soft_update(self.velocity_net_ema, self.velocity_net, 1 - self.ema_decay)
    
    # =========================================================================
    # Sampling & Inference
    # =========================================================================
    
    def _sample_actions_batch(
        self,
        obs_cond: torch.Tensor,
        use_ema: bool = False,
    ) -> torch.Tensor:
        """Sample actions using flow ODE integration (Euler method).
        
        Args:
            obs_cond: Observation conditioning (B, obs_dim)
            use_ema: Whether to use EMA velocity network
            
        Returns:
            actions: (B, pred_horizon, action_dim)
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        net = self.velocity_net_ema if use_ema else self.velocity_net
        
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        dt = 1.0 / self.num_flow_steps
        
        for i in range(self.num_flow_steps):
            t = torch.full((B,), i * dt, device=device)
            v = net(x, t, global_cond=obs_cond)
            x = x + v * dt
        
        if self.action_bounds is not None:
            x = torch.clamp(x, self.action_bounds[0], self.action_bounds[1])
        return x
    
    @torch.no_grad()
    def get_action(
        self,
        obs_features: torch.Tensor,
        use_ema: bool = True,
        best_of_n: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample action with best-of-N selection using action critic.
        
        1. Generate N action candidates from flow actor
        2. Score each candidate with action critic Q^{h_a}
        3. Return the highest-scoring candidate
        
        Args:
            obs_features: Observation features (B, ...) — typically B=1 for eval
            use_ema: Whether to use EMA velocity network
            best_of_n: Override default best_of_n (set to 1 for pure BC)
            
        Returns:
            actions: (B, pred_horizon, action_dim) best action sequence
        """
        self.velocity_net.eval()
        
        obs_cond = self._flatten_obs(obs_features)
        n = best_of_n if best_of_n is not None else self.best_of_n
        B = obs_cond.shape[0]
        
        if n <= 1:
            # Pure BC (no selection)
            action = self._sample_actions_batch(obs_cond, use_ema=use_ema)
            self.velocity_net.train()
            return action
        
        # Best-of-N: generate N candidates per batch element
        # Expand obs_cond: (B, obs_dim) → (B*N, obs_dim)
        obs_expanded = obs_cond.unsqueeze(1).expand(B, n, -1).reshape(B * n, -1)
        
        # Sample B*N actions
        all_actions = self._sample_actions_batch(obs_expanded, use_ema=use_ema)
        # (B*N, pred_horizon, action_dim)
        
        # Score with action critic (using act_horizon portion)
        actions_for_q = all_actions[:, :self.act_horizon, :]  # (B*N, act_horizon, action_dim)
        q_scores = self.action_critic.get_min_q(actions_for_q, obs_expanded)  # (B*N, 1)
        
        # Reshape to (B, N) and pick argmax
        q_scores = q_scores.reshape(B, n)
        best_idx = q_scores.argmax(dim=1)  # (B,)
        
        # Gather best actions
        all_actions = all_actions.reshape(B, n, self.pred_horizon, self.action_dim)
        best_actions = all_actions[torch.arange(B, device=obs_cond.device), best_idx]
        # (B, pred_horizon, action_dim)
        
        self.velocity_net.train()
        return best_actions
    
    @torch.no_grad()
    def get_action_deterministic(
        self,
        obs_features: torch.Tensor,
        use_ema: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Deterministic action: start ODE from zeros, no best-of-N.
        
        Args:
            obs_features: Observation features (B, ...)
            use_ema: Whether to use EMA velocity network
            
        Returns:
            actions: (B, pred_horizon, action_dim)
        """
        self.velocity_net.eval()
        
        obs_cond = self._flatten_obs(obs_features)
        B = obs_cond.shape[0]
        device = obs_cond.device
        net = self.velocity_net_ema if use_ema else self.velocity_net
        
        x = torch.zeros(B, self.pred_horizon, self.action_dim, device=device)
        dt = 1.0 / self.num_flow_steps
        
        for i in range(self.num_flow_steps):
            t = torch.full((B,), i * dt, device=device)
            v = net(x, t, global_cond=obs_cond)
            x = x + v * dt
        
        if self.action_bounds is not None:
            x = torch.clamp(x, self.action_bounds[0], self.action_bounds[1])
        
        self.velocity_net.train()
        return x
