"""
RLFT: Reinforcement Learning and Flow-based Training

A unified framework for:
- Offline Imitation Learning (Diffusion Policy, Flow Matching, etc.)
- Offline Reinforcement Learning (CPQL, AWCP, etc.)
- Online Reinforcement Learning (SAC, RLPD, ReinFlow, etc.)

Package Structure:
    rlft.offline/      - Offline training scripts
    rlft.online/       - Online training scripts
    rlft.algorithms/   - Policy algorithms (IL, offline RL, online RL)
    rlft.networks/     - Neural network architectures
    rlft.buffers/      - Replay and rollout buffers
    rlft.datasets/     - Dataset classes for different data formats
    rlft.envs/         - Environment creation and evaluation
    rlft.utils/        - Common utilities
    rlft.critic/       - Critic module (for reward learning)
    rlft.roboreward/   - RoboReward labeling tool
"""

__version__ = "0.1.0"

# Import submodules for convenient access
from . import networks
from . import algorithms
from . import buffers
from . import datasets
from . import envs
from . import utils

__all__ = [
    "networks",
    "algorithms",
    "buffers",
    "datasets",
    "envs",
    "utils",
]
