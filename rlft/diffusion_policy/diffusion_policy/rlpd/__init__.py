"""
RLPD (Reinforcement Learning with Prior Data) Module

PyTorch implementation of RLPD algorithms migrated from JAX/Flax.
Supports online RL fine-tuning with offline data mixing.

Algorithms:
- SACAgent: Soft Actor-Critic with Ensemble Q-networks and action chunking
- AWSCAgent: Advantage-Weighted ShortCut Flow for online RL

Key Features:
- EnsembleQNetwork: Configurable num_qs with subsample + min for conservative Q
- DiagGaussianActor: State-dependent Gaussian with Tanh squashing
- OnlineReplayBuffer: SMDP-aware buffer with offline data mixing
- RGB observation support via PlainConv visual encoder
"""

from .networks import (
    EnsembleQNetwork,
    DiagGaussianActor,
    LearnableTemperature,
    soft_update,
)
from .distributions import SquashedNormal
from .sac_agent import SACAgent
from .awsc_agent import AWSCAgent
from .replay_buffer import (
    BaseRolloutBuffer,
    OnlineReplayBufferRaw,
    SMDPChunkCollector,
    RolloutBufferPPO,
    SuccessReplayBuffer,
)

__all__ = [
    "EnsembleQNetwork",
    "DiagGaussianActor",
    "LearnableTemperature",
    "SquashedNormal",
    "SACAgent",
    "AWSCAgent",
    "BaseRolloutBuffer",
    "OnlineReplayBufferRaw",
    "SMDPChunkCollector",
    "RolloutBufferPPO",
    "SuccessReplayBuffer",
    "soft_update",
]
