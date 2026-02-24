"""
rlft.vlaw — VLAW 复现模块

公开接口:
    VLAWRewardModel       — VLM 二分类奖励模型 (P3.1)
    VLAWRewardConfig      — 奖励模型配置
    uniform_sample_frames — 轨迹帧均匀采样工具
    VLAWDataCollector     — ManiSkill Rollout 收集器 (P1.1)
    CollectorConfig       — 数据收集配置
    VLAWDataPipeline      — VAE 编码管线 (P1.2)
    PipelineConfig        — VAE 管线配置
    DemoConverter         — ManiSkill Demo 转 VLAW 格式 (P1.3)
    DemoPrepConfig        — Demo 准备配置
"""

from .reward_model import VLAWRewardConfig, VLAWRewardModel, uniform_sample_frames
from .data_collector import CollectorConfig, VLAWDataCollector
from .data_pipeline import PipelineConfig, VLAWDataPipeline
from .demo_prep import DemoPrepConfig, DemoConverter

__all__ = [
    "VLAWRewardConfig",
    "VLAWRewardModel",
    "uniform_sample_frames",
    "CollectorConfig",
    "VLAWDataCollector",
    "PipelineConfig",
    "VLAWDataPipeline",
    "DemoPrepConfig",
    "DemoConverter",
]
