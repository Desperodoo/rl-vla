#!/usr/bin/env python3
"""
CARM 推理模块

包含:
    - inference_ros: 策略推理主程序
    - InferenceLogger: 推理日志记录器
"""

from .inference_logger import InferenceLogger

__all__ = ['InferenceLogger']
