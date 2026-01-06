#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTC 推理配置
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class RTCConfig:
    """RTC 推理配置"""
    
    # ========== 时序参数 ==========
    # 策略推理
    policy_rate: float = 10.0        # Hz, 策略推理频率
    obs_horizon: int = 2             # 观测帧堆叠数
    act_horizon: int = 8             # 执行的动作帧数 (H)
    pred_horizon: int = 16           # 预测的总动作帧数
    
    # 原始数据频率
    record_freq: float = 30.0        # Hz, 训练数据采集频率
    
    # 关键帧参数 (自动计算)
    @property
    def dt_key(self) -> float:
        """关键帧时间间隔 (秒)"""
        return 1.0 / self.record_freq
    
    @property
    def planning_horizon(self) -> float:
        """规划时域 (秒)"""
        return self.act_horizon * self.dt_key
    
    # ========== 模型参数 ==========
    num_flow_steps: int = 10         # flow/diffusion 步数
    image_size: Tuple[int, int] = (128, 128)  # 模型输入尺寸 (H, W)
    
    # ========== 安全参数 ==========
    # 注意: 主要安全限制在 C++ 伺服端执行
    # 这里只做基本检查
    joint_pos_min: Tuple[float, ...] = (-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0.0)
    joint_pos_max: Tuple[float, ...] = (3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 0.08)
    
    # ========== 共享内存 ==========
    shm_name: str = "rtc_keyframes"
    dof: int = 7                     # 6 关节 + 1 夹爪
    
    # ========== 状态维度 ==========
    action_dim: int = 7              # 动作维度
    state_dim: int = 13              # 状态维度 (6 pos + 6 vel + 1 gripper)


@dataclass
class InferenceConfig(RTCConfig):
    """推理节点配置 (继承 RTC 配置)"""
    
    # ========== 设备 ==========
    device: str = "cuda"
    use_ema: bool = True
    
    # ========== 运行模式 ==========
    headless: bool = False           # 无头模式 (无 GUI)
    dry_run: bool = False            # 模拟模式 (不连接机器人)
    verbose: bool = True
    
    # ========== 现有 SharedState 连接 (用于读取机器人状态) ==========
    # 控制进程的共享内存名称
    control_shm_name: str = "arx5_control"


# 默认配置
DEFAULT_CONFIG = InferenceConfig()
