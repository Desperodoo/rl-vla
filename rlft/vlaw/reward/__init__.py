"""rlft.vlaw.reward — 奖励模型相关模块

包含:
    reward_model       ← reward_model.py      (VLM 二分类奖励 P3.1)
    train_reward_model ← train_reward_model.py (VLM 微调训练 P3.2)

导入路径:
    from rlft.vlaw.reward import VLAWRewardModel

注意: train_reward_model 为训练脚本，不在此自动导入（避免加载重型训练依赖）。
"""

# 从子目录模块导入（新路径的权威来源）
from .reward_model import VLAWRewardConfig, VLAWRewardModel, uniform_sample_frames

__all__ = [
    "VLAWRewardConfig",
    "VLAWRewardModel",
    "uniform_sample_frames",
]
