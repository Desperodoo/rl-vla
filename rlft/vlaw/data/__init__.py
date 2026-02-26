"""rlft.vlaw.data — 数据相关模块

包含:
    collector   ← data_collector.py (ManiSkill Rollout 收集器 P1.1)
    pipeline    ← data_pipeline.py  (VAE 编码管线 P1.2)
    demo_prep   ← demo_prep.py      (ManiSkill Demo 转换 P1.3)

导入路径:
    from rlft.vlaw.data import CollectorConfig
"""

# 从子目录模块导入（新路径的权威来源）
from .collector import CollectorConfig, VLAWDataCollector
from .pipeline import PipelineConfig, VLAWDataPipeline, concat_cameras
from .demo_prep import DemoPrepConfig, DemoConverter

__all__ = [
    # collector
    "CollectorConfig",
    "VLAWDataCollector",
    # pipeline
    "PipelineConfig",
    "VLAWDataPipeline",
    "concat_cameras",
    # demo_prep
    "DemoPrepConfig",
    "DemoConverter",
]
