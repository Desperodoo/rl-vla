"""
RLFT: Reinforcement Learning and Flow-based Training

A unified framework for:
- Offline Imitation Learning (Diffusion Policy, Flow Matching, etc.)
- Offline Reinforcement Learning (CPQL, AWCP, etc.)
- Online Reinforcement Learning (SAC, RLPD, AWSC, etc.)

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
# 使用 try-except 防止在专用环境中因可选依赖缺失或二进制不兼容而崩溃
try:
    from . import networks
    from . import algorithms
    from . import buffers
    from . import datasets
    from . import envs
    from . import utils
except Exception:
    pass  # 专用环境只需要部分子模块，跳过主框架依赖

try:
    from . import vlaw
except Exception:
    pass

__all__ = [
    "networks",
    "algorithms",
    "buffers",
    "datasets",
    "envs",
    "utils",
]
