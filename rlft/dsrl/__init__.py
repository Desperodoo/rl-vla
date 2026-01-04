"""
DSRL: Dual-Stage Reinforcement Learning with Latent Steering

Two-stage RL pipeline that preserves pre-trained ShortCut Flow policy:
- Stage 1 (Offline): Advantage-Weighted MLE to warm-start latent steering policy
- Stage 2 (Online): Macro-step PPO to fine-tune latent steering in real environment

Key principle: Flow/Shortcut model stays frozen, RL only happens in latent space.
"""

from .latent_policy import LatentGaussianPolicy
from .dsrl_agent import DSRLAgent
from .value_network import ValueNetwork
from .macro_rollout_buffer import MacroRolloutBuffer

__all__ = [
    "LatentGaussianPolicy",
    "DSRLAgent", 
    "ValueNetwork",
    "MacroRolloutBuffer",
]
