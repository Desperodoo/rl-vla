"""
RoboReward: Robot Manipulation Reward Labeler

基于 RoboReward-8B (Qwen3-VL-8B-Instruct) 的机器人操作奖励标注工具。
用于给 HDF5 格式的机械臂数据集自动打 Reward 标签。

主要组件:
- RoboRewardLabeler: 核心推理类，加载模型并对视频帧进行评分
- DatasetConverter: 数据格式转换，将 HDF5 转为模型输入
- batch_label: 批量标注入口脚本

使用示例:
    from rlft.roboreward import RoboRewardLabeler, DatasetConverter
    
    labeler = RoboRewardLabeler()
    converter = DatasetConverter(sample_frames=8)
    
    frames = converter.load_episode_frames("episode_0001.hdf5")
    reward = labeler.score_episode(frames, "pick up the red cube")
"""

from .labeler import RoboRewardLabeler
from .dataset_converter import DatasetConverter
from .config import RoboRewardConfig

__version__ = "0.1.0"
__all__ = ["RoboRewardLabeler", "DatasetConverter", "RoboRewardConfig"]
