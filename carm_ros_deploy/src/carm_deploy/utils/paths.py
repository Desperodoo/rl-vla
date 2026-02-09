#!/usr/bin/env python3
"""
CARM 路径配置模块

提供统一的路径获取和环境变量处理。

使用方法:
    from utils.paths import get_rl_vla_root, get_recorded_data_dir, get_inference_logs_dir
    
    root = get_rl_vla_root()
    data_dir = get_recorded_data_dir()
    log_dir = get_inference_logs_dir()
    
环境变量:
    RL_VLA_ROOT: 设置项目根目录路径 (可选，默认自动推断)
"""

import os
from typing import Optional


def get_rl_vla_root() -> str:
    """
    获取 rl-vla 项目根目录
    
    优先级:
        1. 环境变量 RL_VLA_ROOT
        2. 从当前文件路径推断
    
    Returns:
        项目根目录绝对路径
    """
    # 1. 尝试从环境变量获取
    env_root = os.environ.get('RL_VLA_ROOT')
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)
    
    # 2. 从当前文件路径推断
    # 当前文件: carm_ros_deploy/src/carm_deploy/utils/paths.py
    # 根目录: ../../../.. (4 层)
    current_file = os.path.abspath(__file__)
    # carm_deploy/utils/paths.py -> carm_deploy/utils -> carm_deploy -> src -> carm_ros_deploy -> rl-vla
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file)))))
    
    # 验证路径有效性（检查是否存在 rlft 目录）
    if os.path.isdir(os.path.join(root, 'rlft')):
        return root
    
    # 3. 最后尝试 home 目录下的默认路径
    default_root = os.path.expanduser('~/rl-vla')
    if os.path.isdir(default_root):
        return default_root
    
    # 找不到则返回推断的路径
    return root


def get_recorded_data_dir(subdir: Optional[str] = None) -> str:
    """
    获取数据记录目录
    
    Args:
        subdir: 子目录名 (如 'mix', 'random_pos')
    
    Returns:
        数据目录绝对路径
    """
    root = get_rl_vla_root()
    data_dir = os.path.join(root, 'recorded_data')
    
    if subdir:
        data_dir = os.path.join(data_dir, subdir)
    
    return data_dir


def get_inference_logs_dir() -> str:
    """
    获取推理日志目录
    
    Returns:
        日志目录绝对路径
    """
    root = get_rl_vla_root()
    return os.path.join(root, 'inference_logs')


def get_safety_config_path() -> str:
    """
    获取默认安全配置文件路径
    
    Returns:
        安全配置文件绝对路径
    """
    # 当前文件在 carm_deploy/utils 目录下
    carm_deploy_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(carm_deploy_root, 'safety_config.json')


def get_training_runs_dir() -> str:
    """
    获取训练运行目录
    
    Returns:
        训练目录绝对路径
    """
    root = get_rl_vla_root()
    return os.path.join(root, 'rlft', 'diffusion_policy', 'runs')


def ensure_dir(path: str) -> str:
    """
    确保目录存在，不存在则创建
    
    Args:
        path: 目录路径
    
    Returns:
        目录绝对路径
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


# 便捷常量
RL_VLA_ROOT = get_rl_vla_root()
RECORDED_DATA_DIR = get_recorded_data_dir()
INFERENCE_LOGS_DIR = get_inference_logs_dir()
SAFETY_CONFIG_PATH = get_safety_config_path()


if __name__ == '__main__':
    # 测试路径配置
    print("rl-vla 路径配置")
    print("=" * 50)
    print(f"RL_VLA_ROOT:       {get_rl_vla_root()}")
    print(f"RECORDED_DATA_DIR: {get_recorded_data_dir()}")
    print(f"  - mix:           {get_recorded_data_dir('mix')}")
    print(f"INFERENCE_LOGS:    {get_inference_logs_dir()}")
    print(f"SAFETY_CONFIG:     {get_safety_config_path()}")
    print(f"TRAINING_RUNS:     {get_training_runs_dir()}")
