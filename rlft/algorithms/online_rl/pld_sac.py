"""
PLD-SAC Agent — SAC operating in the residual action space of a frozen base policy.

Implements the Stage 1 (Probe) of the PLD (Probe, Learn, Distill) framework:
    Self-Improving Vision-Language-Action Models with Data Generation via Residual RL
    (Xiao et al., arXiv:2511.00091)

The actor outputs a residual action  a_delta ∈ [-action_scale, +action_scale]^D
via a TanhGaussian distribution.  The combined action is:
    a_bar = clamp(a_base + a_delta, -1, 1)

The environment wrapper (``ManiSkillResidualEnvWrapper``) handles the base policy
inference and action composition internally — the agent only sees the residual
action space.

Key differences from ``DSRLSACAgent``:
    * Operates in **residual action space** (additive offset) rather than noise space.
    * Supports **offline/online mixed replay** for RLPD-style training.
    * Includes **Cal-QL critic pretraining** for stable exploration warm-start.
    * Default action_scale=0.3 (PLD sweep: optimal residual range).
    * Hidden dims = [1024,1024,1024], num_qs = 5 (tuned via PLD sweep).

Reference:
    - PLD: https://arxiv.org/abs/2511.00091
    - RLPD: Ball et al., ICML 2023
    - Cal-QL: Nakamoto et al., NeurIPS 2024
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from rlft.networks.q_networks import soft_update


# =====================================================================
# Scaled SquashedNormal  (tanh(z) * scale)
# =====================================================================

class ScaledSquashedNormal:
    """TanhNormal scaled to ``[-scale, +scale]``."""

    def __init__(self, loc: torch.Tensor, scale_std: torch.Tensor, action_scale: float = 1.0):
        self.loc = loc
        self.scale_std = scale_std
        self.action_scale = action_scale
        self._base = torch.distributions.Normal(loc, scale_std)

    @property
    def mean(self) -> torch.Tensor:
        return torch.tanh(self.loc) * self.action_scale

    def sample_with_log_prob(self):
        z = self._base.rsample()
        action = torch.tanh(z) * self.action_scale

        log_prob_z = self._base.log_prob(z)
        log_abs_det = 2 * (math.log(2) - z - F.softplus(-2 * z))
        log_prob = (log_prob_z - log_abs_det).sum(dim=-1)
        return action, log_prob

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        unscaled = torch.clamp(value / self.action_scale, -1 + eps, 1 - eps)
        z = 0.5 * (torch.log1p(unscaled) - torch.log1p(-unscaled))  # arctanh
        log_prob_z = self._base.log_prob(z)
        log_abs_det = 2 * (math.log(2) - z - F.softplus(-2 * z))
        return (log_prob_z - log_abs_det).sum(dim=-1)


# =====================================================================
# PLD Actor — MLP + TanhGaussian (residual action output)
# =====================================================================

class PLDActor(nn.Module):
    """MLP Gaussian actor for PLD-SAC.

    Architecture: ``[Linear → Tanh] × num_layers`` → mean / log_std heads.
    Output is ``tanh(z) * action_scale``, representing the residual action
    a_delta to be added to the base policy output.

    Args:
        obs_dim: Flat observation dim (after visual encoding + history).
        residual_dim: Total residual action dim = act_steps × action_dim.
        hidden_dims: MLP hidden layer sizes.  Default ``[256]*3`` (PLD paper).
        action_scale: Scale for tanh output (ξ in the paper).
        log_std_init: Initial value for log-std bias.
    """

    def __init__(
        self,
        obs_dim: int,
        residual_dim: int,
        hidden_dims: list = None,
        action_scale: float = 0.3,
        log_std_init: float = -3.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        self.residual_dim = residual_dim
        self.action_scale = action_scale

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.Tanh()])
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        self.mean_head = nn.Linear(in_dim, residual_dim)
        self.log_std_head = nn.Linear(in_dim, residual_dim)

        # Initialize log_std bias so initial std ≈ exp(log_std_init)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, log_std_init)

        self._init_weights()

    def _init_weights(self):
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: torch.Tensor) -> ScaledSquashedNormal:
        feat = self.trunk(obs)
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return ScaledSquashedNormal(mean, log_std.exp(), self.action_scale)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dist = self.forward(obs)
        if deterministic:
            return dist.mean, None
        action, log_prob = dist.sample_with_log_prob()
        return action, log_prob


# =====================================================================
# PLD Critic — Ensemble of MLPs
# =====================================================================

class PLDCritic(nn.Module):
    """Ensemble Q-network for PLD-SAC.

    Input = ``[obs_features, combined_action_bar]`` → scalar Q.
    Note: the combined action a_bar is provided by the environment wrapper.

    Uses Tanh activation and (optionally) LayerNorm, matching the PLD paper
    architecture (3×256, LayerNorm).

    Args:
        obs_dim: Flat observation dim.
        action_input_dim: Total action dim for Q input = act_steps × action_dim.
            This should be the combined action a_bar dimension.
        hidden_dims: MLP hidden layer sizes.
        num_qs: Number of Q-networks in ensemble.
        use_layer_norm: Whether to use LayerNorm.
    """

    def __init__(
        self,
        obs_dim: int,
        action_input_dim: int,
        hidden_dims: list = None,
        num_qs: int = 2,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]

        self.num_qs = num_qs
        input_dim = obs_dim + action_input_dim

        self.q_nets = nn.ModuleList()
        for _ in range(num_qs):
            layers = []
            in_d = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(in_d, h))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(h))
                layers.append(nn.Tanh())
                in_d = h
            layers.append(nn.Linear(in_d, 1))
            self.q_nets.append(nn.Sequential(*layers))

        self._init_weights()

    def _init_weights(self):
        for qnet in self.q_nets:
            for m in qnet.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            # Smaller output init for final layer
            final = qnet[-1]
            if isinstance(final, nn.Linear):
                nn.init.orthogonal_(final.weight, gain=0.01)
                nn.init.zeros_(final.bias)

    def forward(self, action: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Returns (num_qs, B, 1)."""
        x = torch.cat([obs, action], dim=-1)
        return torch.stack([q(x) for q in self.q_nets], dim=0)

    def get_min_q(self, action: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(action, obs).min(dim=0).values

    def get_mean_q(self, action: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(action, obs).mean(dim=0)


# =====================================================================
# PLD-SAC Agent
# =====================================================================

class PLDSACAgent(nn.Module):
    """PLD-SAC: Soft Actor-Critic with residual actions on a frozen base policy.

    The agent holds:
    - ``actor``: PLDActor  (outputs residual action a_delta)
    - ``critic`` / ``critic_target``: PLDCritic
    - ``log_alpha``: learnable temperature

    The frozen base policy is held externally (in the env wrapper), identical
    to how DSRL-SAC places the flow policy in ``ManiSkillFlowEnvWrapper``.

    Args:
        obs_dim: Encoded observation dimension.
        act_steps: Number of action-chunk steps.
        action_dim: Per-step action dimension (e.g. 7).
        action_scale: Residual action bound [-ξ, +ξ] (default 0.5).
        hidden_dims: MLP widths for actor and critic.
        num_qs: Number of Q-networks.
        gamma: Discount factor.
        tau: Soft-update rate.
        init_temperature: Initial entropy temperature.
        target_entropy: Target entropy for auto-tuning.
        log_std_init: Actor initial log-std.
        use_layer_norm: Use LayerNorm in critic.
        device: Device.
    """

    def __init__(
        self,
        obs_dim: int,
        act_steps: int = 8,
        action_dim: int = 7,
        action_scale: float = 0.3,
        hidden_dims: list = None,
        num_qs: int = 5,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_temperature: float = 0.1,
        target_entropy: Optional[float] = None,
        log_std_init: float = -5.0,
        use_layer_norm: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 1024, 1024]

        self.obs_dim = obs_dim
        self.act_steps = act_steps
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.residual_dim = act_steps * action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Target entropy: PLD paper uses target_entropy = -dim(a)
        if target_entropy is None:
            self.target_entropy = -float(self.residual_dim)
        else:
            self.target_entropy = target_entropy

        # ---------- networks ----------
        self.actor = PLDActor(
            obs_dim=obs_dim,
            residual_dim=self.residual_dim,
            hidden_dims=hidden_dims,
            action_scale=action_scale,
            log_std_init=log_std_init,
        )

        # Critic evaluates Q(obs, a_delta) where a_delta is the residual action.
        # This is consistent with DSRL which evaluates Q(obs, noise) — the
        # critic operates in the same space as the actor (residual / noise).
        self.critic = PLDCritic(
            obs_dim=obs_dim,
            action_input_dim=self.residual_dim,
            hidden_dims=hidden_dims,
            num_qs=num_qs,
            use_layer_norm=use_layer_norm,
        )

        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ---------- temperature ----------
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(init_temperature), dtype=torch.float32)
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Select flat residual action vector for environment interaction."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        action, _ = self.actor.get_action(obs, deterministic=deterministic)
        return action  # (B, residual_dim)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute critic (Q-network) loss.

        Note: ``actions`` here are the **residual** actions a_delta (not
        the combined a_bar).  The critic learns Q(s, a_delta), which is
        consistent with the actor outputting a_delta.  This mirrors DSRL
        where the critic evaluates Q(s, noise) in the agent's action space.
        """
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)

        with torch.no_grad():
            # Sample a_delta from actor for next-state bootstrap.
            # Q-target consistently evaluates Q(s', a_delta') in the same
            # residual action space as the actor.
            next_a_delta, next_log_prob = self.actor.get_action(next_obs, deterministic=False)
            target_q = self.critic_target.get_min_q(next_a_delta, next_obs)
            target_q = target_q - self.alpha.detach() * next_log_prob.unsqueeze(-1)
            td_target = rewards + (1 - dones) * self.gamma * target_q

        q_all = self.critic(actions, obs)  # (num_qs, B, 1)
        critic_loss = sum(F.mse_loss(q, td_target) for q in q_all)

        metrics = {
            "critic_loss": critic_loss.item(),
            "q_mean": q_all.mean().item(),
            "q_std": q_all.std().item(),
            "td_target_mean": td_target.mean().item(),
        }
        return critic_loss, metrics

    def compute_actor_loss(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute actor loss (maximize Q with entropy regularization)."""
        a_delta, log_prob = self.actor.get_action(obs, deterministic=False)
        q = self.critic.get_min_q(a_delta, obs)
        actor_loss = (self.alpha.detach() * log_prob - q.squeeze(-1)).mean()

        metrics = {
            "actor_loss": actor_loss.item(),
            "actor_entropy": -log_prob.mean().item(),
            "actor_q": q.mean().item(),
        }
        return actor_loss, metrics

    def compute_temperature_loss(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute temperature loss for entropy-constrained optimization."""
        with torch.no_grad():
            _, log_prob = self.actor.get_action(obs, deterministic=False)
        temp_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()
        metrics = {
            "temperature_loss": temp_loss.item(),
            "temperature": self.alpha.item(),
            "entropy": -log_prob.mean().item(),
        }
        return temp_loss, metrics

    # ------------------------------------------------------------------
    # Target update
    # ------------------------------------------------------------------

    def update_target(self):
        """Soft update target critic network."""
        soft_update(self.critic_target, self.critic, self.tau)

    # ------------------------------------------------------------------
    # Cal-QL Critic Pretraining
    # ------------------------------------------------------------------

    def pretrain_critic_calql(
        self,
        buffer,
        steps: int = 5000,
        batch_size: int = 256,
        lr: float = 3e-4,
        calql_alpha: float = 5.0,
        device: str = "cuda",
    ):
        """Pretrain critic using Cal-QL on offline data.

        Cal-QL (Calibrated Q-Learning) prevents critic overestimation on OOD
        actions while avoiding the excessive underestimation of CQL, by
        lower-bounding the conservative Q with the behavior policy's value.

        Objective:
            min_θ  α_cql * E_{a~π}[max(Q_θ(s,a), V_μ(s))]
                   - α_cql * E_{(s,a)~D}[Q_θ(s,a)]
                   + (1/2) * TD_loss

        Args:
            buffer: PLDReplayBuffer with offline data.
            steps: Number of pretraining gradient steps.
            batch_size: Batch size for pretraining.
            lr: Learning rate for critic optimizer.
            calql_alpha: Conservative coefficient (α_cql).
            device: Torch device.
        """
        print(f"[Cal-QL Pretrain] Starting critic pretraining for {steps} steps ...")
        print(f"  calql_alpha={calql_alpha}, batch_size={batch_size}, lr={lr}")

        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.critic.train()
        self.actor.eval()

        for step in tqdm(range(steps), desc="Cal-QL Pretrain"):
            batch = buffer.sample_offline(batch_size)

            obs = batch["obs"]
            actions = batch["actions"]
            next_obs = batch["next_obs"]
            rewards = batch["rewards"]
            dones = batch["dones"]

            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(-1)
            if dones.dim() == 1:
                dones = dones.unsqueeze(-1)

            # ---- Standard TD target ----
            with torch.no_grad():
                next_a_delta, next_log_prob = self.actor.get_action(next_obs, deterministic=False)
                target_q = self.critic_target.get_min_q(next_a_delta, next_obs)
                target_q = target_q - self.alpha.detach() * next_log_prob.unsqueeze(-1)
                td_target = rewards + (1 - dones) * self.gamma * target_q

            # ---- Q values on dataset actions ----
            q_all_data = self.critic(actions, obs)  # (num_qs, B, 1)

            # ---- TD loss ----
            td_loss = sum(F.mse_loss(q, td_target) for q in q_all_data)

            # ---- Cal-QL conservative penalty ----
            # Sample actions from current policy for OOD penalty
            with torch.no_grad():
                policy_actions, _ = self.actor.get_action(obs, deterministic=False)

            q_policy = self.critic(policy_actions, obs)  # (num_qs, B, 1)
            q_data = self.critic(actions, obs)  # (num_qs, B, 1)

            # V_mu(s) ≈ Q(s, a_data) for behavior policy
            v_mu = q_data.mean(dim=0).detach()  # (B, 1)

            # Cal-QL: max(Q(s, a_policy), V_mu(s)) instead of raw Q(s, a_policy)
            calql_penalty = torch.stack([
                torch.maximum(q_policy[i], v_mu) for i in range(q_policy.shape[0])
            ], dim=0).mean()

            calql_bonus = q_data.mean()

            conservative_loss = calql_alpha * (calql_penalty - calql_bonus)

            # ---- Total loss ----
            loss = td_loss + conservative_loss

            critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            critic_optimizer.step()

            # Update target
            self.update_target()

            if (step + 1) % 1000 == 0:
                print(
                    f"  [Step {step+1}/{steps}] "
                    f"td_loss={td_loss.item():.4f}, "
                    f"conservative_loss={conservative_loss.item():.4f}, "
                    f"q_mean={q_all_data.mean().item():.4f}"
                )

        print("[Cal-QL Pretrain] Done.")
        self.critic.train()
