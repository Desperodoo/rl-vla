# -*- coding: utf-8 -*-
"""
多进程推理系统

将安全关键的控制逻辑与 GPU 推理完全分离。

组件:
    - shared_state: 共享内存通信模块
    - control_node: 控制进程 (安全核心)
    - inference_node: 推理进程 (GPU + 摄像头)

用法:
    # 终端 1: 启动控制节点
    python -m inference.control_node --model X5 --interface can0

    # 终端 2: 启动推理节点
    python -m inference.inference_node -c /path/to/checkpoint.pt
"""

from .config import RL_VLA_CONFIG, setup_arx5, setup_rlft, setup_all
from .shared_state import SharedState, ControlFlags, SharedMemoryLayout

__all__ = [
    'RL_VLA_CONFIG',
    'setup_arx5', 
    'setup_rlft',
    'setup_all',
    # 多进程通信
    'SharedState',
    'ControlFlags', 
    'SharedMemoryLayout',
]

__version__ = "0.1.0"
