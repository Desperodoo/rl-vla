#!/usr/bin/env python3
"""
CARM 推理模块

包含:
    - InferenceNode: 推理节点（核心控制器）
    - InferenceConfig: 类型化配置
    - ActionProcessor: 动作处理流水线
    - InferenceLogger: 推理日志记录器
    - InferenceRecorder: 推理数据采集
"""

from .inference_logger import InferenceLogger
from .config import InferenceConfig
from .action_processor import ActionProcessor, ActionIndices, SafetyResult

__all__ = [
    'InferenceLogger',
    'InferenceConfig',
    'ActionProcessor',
    'ActionIndices',
    'SafetyResult',
]
