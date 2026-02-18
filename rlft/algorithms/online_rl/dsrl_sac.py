"""
DSRL-SAC Agent — SAC operating in the noise space of a ShortCut Flow policy.

The actor outputs a noise vector  w ∈ [-action_magnitude, +action_magnitude]^D
via a TanhGaussian distribution (scaled by *action_magnitude*).
The critic evaluates Q(obs, w) in the same noise space.

The environment wrapper (``ManiSkillFlowEnvWrapper``) internally decodes
w → real actions through the frozen flow policy.

Key differences from the standard ``SACAgent``:
    * Output range is ``[-action_magnitude, +action_magnitude]`` not ``[-1,1]``.
    * Uses ``Tanh`` activation (matching DSRL official) instead of ``Mish``.
    * Default hidden dims = ``[2048, 2048, 2048]`` (wider, shallower).
    * ``log_std_init`` defaults to −5 (very small initial exploration noise to
      protect the pretrained policy).
    * ``target_entropy`` defaults to −3.5 (moderate negative entropy for
      balanced exploration/exploitation in noise space).

Reference: https://github.com/ajwagen/dsrl
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

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

        # log_prob with Jacobian correction
        log_prob_z = self._base.log_prob(z)
        log_abs_det = 2 * (math.log(2) - z - F.softplus(-2 * z))
        log_prob = (log_prob_z - log_abs_det).sum(dim=-1)
        # Adjust for action_scale: log|d(scale*tanh)/dz| = log(scale) + log|dtanh/dz|
        # Already accounted since we sample z through tanh then multiply —
        # the factor of log(scale) cancels in the ratio ∇log π, so we keep
        # the standard formula and just track the sign convention.
        return action, log_prob

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        unscaled = torch.clamp(value / self.action_scale, -1 + eps, 1 - eps)
        z = 0.5 * (torch.log1p(unscaled) - torch.log1p(-unscaled))  # arctanh
        log_prob_z = self._base.log_prob(z)
        log_abs_det = 2 * (math.log(2) - z - F.softplus(-2 * z))
        return (log_prob_z - log_abs_det).sum(dim=-1)


# =====================================================================
# DSRL Actor — MLP + TanhGaussian  (matches SB3-SAC MlpPolicy)
# =====================================================================

class DSRLActor(nn.Module):
    """MLP Gaussian actor for DSRL-SAC.

    Architecture mirrors the official DSRL config:
        ``[Linear → Tanh]  ×  num_layers``  →  mean / log_std heads.

    Output is ``tanh(z) * action_magnitude``.

    Args:
        obs_dim: Flat observation dim (after visual encoding + history).
        noise_dim: Total noise dim = act_steps × action_dim.
        hidden_dims: MLP hidden layer sizes.  Default ``[2048]*3``.
        action_magnitude: Scale for tanh output.
        log_std_init: Initial value for the constant part of log-std
            (−5 ≈ std 0.007, protects pretrained policy).
    """

    def __init__(
        self,
        obs_dim: int,
        noise_dim: int,
        hidden_dims: list = None,
        action_magnitude: float = 2.5,
        log_std_init: float = -5.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [2048, 2048, 2048]

        self.noise_dim = noise_dim
        self.action_magnitude = action_magnitude

        # Feature extractor — Tanh activation (matching SB3 SAC)
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.Tanh()])
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        self.mean_head = nn.Linear(in_dim, noise_dim)
        self.log_std_head = nn.Linear(in_dim, noise_dim)

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
        return ScaledSquashedNormal(mean, log_std.exp(), self.action_magnitude)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        dist = self.forward(obs)
        if deterministic:
            return dist.mean, None
        action, log_prob = dist.sample_with_log_prob()
        return action, log_prob


# =====================================================================
# DSRL Critic — Ensemble of MLPs (Tanh activation)
# =====================================================================

class DSRLCritic(nn.Module):
    """Ensemble Q-network for DSRL-SAC.

    Input = ``[flat_noise, obs_features]`` → scalar Q.

    Uses Tanh activation and (optionally) LayerNorm, matching the official
    DSRL ``net_arch=[2048]*3, activation=Tanh`` config.
    """

    def __init__(
        self,
        obs_dim: int,
        noise_dim: int,
        hidden_dims: list = None,
        num_qs: int = 10,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [2048, 2048, 2048]

        self.num_qs = num_qs
        input_dim = noise_dim + obs_dim

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
            final = qnet[-1]
            if isinstance(final, nn.Linear):
                nn.init.orthogonal_(final.weight, gain=0.01)
                nn.init.zeros_(final.bias)

    def forward(self, noise: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Returns (num_qs, B, 1)."""
        x = torch.cat([noise, obs], dim=-1)
        return torch.stack([q(x) for q in self.q_nets], dim=0)

    def get_min_q(self, noise: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(noise, obs).min(dim=0).values

    def get_mean_q(self, noise: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(noise, obs).mean(dim=0)


# =====================================================================
# DSRL-SAC Agent
# =====================================================================

class DSRLSACAgent(nn.Module):
    """DSRL-SAC: Soft Actor-Critic in ShortCut Flow noise space.

    The agent holds:
    - ``actor``: DSRLActor  (outputs noise w)
    - ``critic`` / ``critic_target``: DSRLCritic
    - ``temperature``: learnable α

    The frozen ``base_policy`` is held externally (in the env wrapper).

    Args:
        obs_dim: Encoded observation dimension.
        act_steps: Number of action-chunk steps.
        action_dim: Per-step action dimension.
        action_magnitude: Noise bound [-mag, +mag].
        hidden_dims: MLP widths for actor and critic.
        num_qs: Number of Q-networks.
        gamma: Discount factor.
        tau: Soft-update rate.
        init_temperature: Initial entropy temperature.
        target_entropy: Target entropy for auto-tuning.
            Default −3.5 (moderate negative entropy, balances exploration
            and exploitation in noise space).
        log_std_init: Actor initial log-std.
        use_layer_norm: Use LayerNorm in critic.
        device: Device.
    """

    def __init__(
        self,
        obs_dim: int,
        act_steps: int = 8,
        action_dim: int = 7,
        action_magnitude: float = 2.5,
        hidden_dims: list = None,
        num_qs: int = 10,
        gamma: float = 0.95,
        tau: float = 0.005,
        init_temperature: float = 0.5,
        target_entropy: float = -3.5,
        log_std_init: float = -5.0,
        use_layer_norm: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [2048, 2048, 2048]

        self.obs_dim = obs_dim
        self.act_steps = act_steps
        self.action_dim = action_dim
        self.action_magnitude = action_magnitude
        self.noise_dim = act_steps * action_dim
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy
        self.device = device

        # ---------- networks ----------
        self.actor = DSRLActor(
            obs_dim=obs_dim,
            noise_dim=self.noise_dim,
            hidden_dims=hidden_dims,
            action_magnitude=action_magnitude,
            log_std_init=log_std_init,
        )

        self.critic = DSRLCritic(
            obs_dim=obs_dim,
            noise_dim=self.noise_dim,
            hidden_dims=hidden_dims,
            num_qs=num_qs,
            use_layer_norm=use_layer_norm,
        )

        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ---------- temperature ----------
        self.log_alpha = nn.Parameter(torch.tensor(np.log(init_temperature), dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Select flat noise vector for environment interaction."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        action, _ = self.actor.get_action(obs, deterministic=deterministic)
        return action  # (B, noise_dim)

    # ------------------------------------------------------------------
    # Loss computation  (called by the training loop)
    # ------------------------------------------------------------------

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        noise: torch.Tensor,
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute critic (Q-network) loss."""
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)

        with torch.no_grad():
            next_noise, next_log_prob = self.actor.get_action(next_obs, deterministic=False)
            target_q = self.critic_target.get_min_q(next_noise, next_obs)
            target_q = target_q - self.alpha.detach() * next_log_prob.unsqueeze(-1)
            td_target = rewards + (1 - dones) * self.gamma * target_q

        q_all = self.critic(noise, obs)  # (num_qs, B, 1)
        critic_loss = sum(F.mse_loss(q, td_target) for q in q_all)

        metrics = {
            "critic_loss": critic_loss.item(),
            "q_mean": q_all.mean().item(),
            "td_target_mean": td_target.mean().item(),
        }
        return critic_loss, metrics

    def compute_actor_loss(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        noise, log_prob = self.actor.get_action(obs, deterministic=False)
        q = self.critic.get_min_q(noise, obs)
        actor_loss = (self.alpha.detach() * log_prob - q.squeeze(-1)).mean()

        metrics = {
            "actor_loss": actor_loss.item(),
            "actor_entropy": -log_prob.mean().item(),
        }
        return actor_loss, metrics

    def compute_temperature_loss(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
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
        soft_update(self.critic_target, self.critic, self.tau)
