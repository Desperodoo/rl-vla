"""
rlft.vlaw — VLAW 复现模块

公开接口:
    VLAWRewardModel    — VLM 二分类奖励模型 (P3.1)
    VLAWRewardConfig   — 奖励模型配置
    uniform_sample_frames — 轨迹帧均匀采样工具
"""

from .reward_model import VLAWRewardConfig, VLAWRewardModel, uniform_sample_frames

__all__ = [
    "VLAWRewardConfig",
    "VLAWRewardModel",
    "uniform_sample_frames",
]
