#!/usr/bin/env python3
"""
CARM 安全控制器

提供机械臂推理时的安全检查和限制功能。

功能:
    - 关节限位检查
    - 工作空间边界检查
    - 动作幅度限制
    - 低通滤波平滑
    - 速度限制

使用方法:
    from safety_controller import SafetyController
    safety = SafetyController.from_dataset_stats(data_dir, margin=0.1)
    action_safe, warnings = safety.check_and_clip(action, current_state)
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class JointLimits:
    """关节限位配置"""
    # 从数据集统计得到，扩展 10% margin
    # 默认值基于 recorded_data 的统计
    joint_min: np.ndarray = field(default_factory=lambda: np.array([
        -0.065 - 0.1,  # J1
        1.548 - 0.1,   # J2
        -1.033 - 0.1,  # J3
        -0.257 - 0.1,  # J4
        0.447 - 0.1,   # J5
        -0.190 - 0.1,  # J6
    ]))
    joint_max: np.ndarray = field(default_factory=lambda: np.array([
        0.932 + 0.1,   # J1
        1.970 + 0.1,   # J2
        -0.619 + 0.1,  # J3
        0.203 + 0.1,   # J4
        0.949 + 0.1,   # J5
        0.670 + 0.1,   # J6
    ]))
    gripper_min: float = 0.0
    gripper_max: float = 0.08


@dataclass
class WorkspaceLimits:
    """工作空间边界配置"""
    # 基于实际机械臂工作范围设定
    x_min: float = 0.10
    x_max: float = 0.50
    y_min: float = -0.30
    y_max: float = 0.30
    z_min: float = 0.05
    z_max: float = 0.40


@dataclass
class SafetyParams:
    """安全参数配置"""
    # 动作幅度限制 (每步最大变化量)
    max_joint_delta: float = 0.1  # rad per step
    max_gripper_delta: float = 0.02  # m per step
    max_position_delta: float = 0.02  # m per step
    max_rotation_delta: float = 0.1  # rad per step
    
    # 低通滤波
    filter_alpha: float = 0.3  # 0=全滤波, 1=无滤波
    
    # 速度限制
    max_joint_velocity: float = 1.0  # rad/s
    max_gripper_velocity: float = 0.1  # m/s


class SafetyController:
    """
    安全控制器
    
    提供多层安全检查和动作限制
    """
    
    def __init__(
        self,
        joint_limits: Optional[JointLimits] = None,
        workspace_limits: Optional[WorkspaceLimits] = None,
        safety_params: Optional[SafetyParams] = None,
    ):
        self.joint_limits = joint_limits or JointLimits()
        self.workspace_limits = workspace_limits or WorkspaceLimits()
        self.params = safety_params or SafetyParams()
        
        # 上一步动作 (用于滤波)
        self.prev_action = None
        self.prev_timestamp = None
        
        # 统计
        self.stats = {
            'total_checks': 0,
            'joint_clips': 0,
            'workspace_clips': 0,
            'delta_clips': 0,
            'filter_applied': 0,
        }
    
    @classmethod
    def from_dataset_stats(cls, data_dir: str, margin: float = 0.1) -> 'SafetyController':
        """
        从数据集统计信息创建安全控制器
        
        Args:
            data_dir: 数据集目录
            margin: 限位扩展比例 (0.1 = 10%)
            
        Returns:
            SafetyController 实例
        """
        data_dir = os.path.expanduser(data_dir)
        info_path = os.path.join(data_dir, 'dataset_info.json')
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            joint_min = np.array([float(x) for x in info['joint_min']])
            joint_max = np.array([float(x) for x in info['joint_max']])
            
            # 计算范围并扩展 margin
            joint_range = joint_max - joint_min
            joint_min_expanded = joint_min - margin * joint_range
            joint_max_expanded = joint_max + margin * joint_range
            
            # 获取夹爪范围
            gripper_min = info.get('gripper_range', [0.01, 0.08])[0]
            gripper_max = info.get('gripper_range', [0.01, 0.08])[1]
            gripper_range = gripper_max - gripper_min
            
            joint_limits = JointLimits(
                joint_min=joint_min_expanded[:6],
                joint_max=joint_max_expanded[:6],
                gripper_min=max(0, gripper_min - margin * gripper_range),
                gripper_max=min(0.08, gripper_max + margin * gripper_range),
            )
            
            print(f"Loaded joint limits from {info_path}")
            print(f"  Joint min: {joint_limits.joint_min}")
            print(f"  Joint max: {joint_limits.joint_max}")
            print(f"  Gripper range: [{joint_limits.gripper_min:.4f}, {joint_limits.gripper_max:.4f}]")
            
            return cls(joint_limits=joint_limits)
        else:
            print(f"Warning: {info_path} not found, using default limits")
            return cls()
    
    def check_joint_limits(self, action: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        检查并裁剪关节限位
        
        Args:
            action: 动作向量，支持多种格式:
                    - [j1-j6, gripper]: 7D
                    - [j1-j6, gripper, end_pose(7), gripper]: 15D
                    
        Returns:
            clipped_action: 裁剪后的动作
            warnings: 警告信息列表
        """
        warnings = []
        clipped = action.copy()
        
        # 检查关节
        for i in range(6):
            if action[i] < self.joint_limits.joint_min[i]:
                clipped[i] = self.joint_limits.joint_min[i]
                warnings.append(f"Joint {i+1} below min: {action[i]:.3f} < {self.joint_limits.joint_min[i]:.3f}")
            elif action[i] > self.joint_limits.joint_max[i]:
                clipped[i] = self.joint_limits.joint_max[i]
                warnings.append(f"Joint {i+1} above max: {action[i]:.3f} > {self.joint_limits.joint_max[i]:.3f}")
        
        # 检查夹爪 (index 6)
        if len(action) > 6:
            if action[6] < self.joint_limits.gripper_min:
                clipped[6] = self.joint_limits.gripper_min
                warnings.append(f"Gripper below min: {action[6]:.3f} < {self.joint_limits.gripper_min:.3f}")
            elif action[6] > self.joint_limits.gripper_max:
                clipped[6] = self.joint_limits.gripper_max
                warnings.append(f"Gripper above max: {action[6]:.3f} > {self.joint_limits.gripper_max:.3f}")
        
        if warnings:
            self.stats['joint_clips'] += 1
        
        return clipped, warnings
    
    def check_workspace(self, end_pose: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        检查并裁剪工作空间边界
        
        Args:
            end_pose: 末端位姿 [x, y, z, qx, qy, qz, qw] 或 [x, y, z, qx, qy, qz, qw, gripper]
            
        Returns:
            clipped_pose: 裁剪后的位姿
            warnings: 警告信息列表
        """
        warnings = []
        clipped = end_pose.copy()
        
        # 检查位置
        limits = self.workspace_limits
        if end_pose[0] < limits.x_min:
            clipped[0] = limits.x_min
            warnings.append(f"X below min: {end_pose[0]:.3f} < {limits.x_min:.3f}")
        elif end_pose[0] > limits.x_max:
            clipped[0] = limits.x_max
            warnings.append(f"X above max: {end_pose[0]:.3f} > {limits.x_max:.3f}")
        
        if end_pose[1] < limits.y_min:
            clipped[1] = limits.y_min
            warnings.append(f"Y below min: {end_pose[1]:.3f} < {limits.y_min:.3f}")
        elif end_pose[1] > limits.y_max:
            clipped[1] = limits.y_max
            warnings.append(f"Y above max: {end_pose[1]:.3f} > {limits.y_max:.3f}")
        
        if end_pose[2] < limits.z_min:
            clipped[2] = limits.z_min
            warnings.append(f"Z below min: {end_pose[2]:.3f} < {limits.z_min:.3f}")
        elif end_pose[2] > limits.z_max:
            clipped[2] = limits.z_max
            warnings.append(f"Z above max: {end_pose[2]:.3f} > {limits.z_max:.3f}")
        
        if warnings:
            self.stats['workspace_clips'] += 1
        
        return clipped, warnings
    
    def clip_action_delta(self, action: np.ndarray, current_state: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        限制动作幅度
        
        Args:
            action: 目标动作
            current_state: 当前状态
            
        Returns:
            clipped_action: 限制后的动作
            warnings: 警告信息列表
        """
        warnings = []
        clipped = action.copy()
        
        # 关节幅度限制
        for i in range(min(6, len(action), len(current_state))):
            delta = action[i] - current_state[i]
            if abs(delta) > self.params.max_joint_delta:
                clipped[i] = current_state[i] + np.sign(delta) * self.params.max_joint_delta
                warnings.append(f"Joint {i+1} delta clipped: {delta:.3f} -> {np.sign(delta) * self.params.max_joint_delta:.3f}")
        
        # 夹爪幅度限制
        if len(action) > 6 and len(current_state) > 6:
            delta = action[6] - current_state[6]
            if abs(delta) > self.params.max_gripper_delta:
                clipped[6] = current_state[6] + np.sign(delta) * self.params.max_gripper_delta
                warnings.append(f"Gripper delta clipped: {delta:.3f} -> {np.sign(delta) * self.params.max_gripper_delta:.3f}")
        
        if warnings:
            self.stats['delta_clips'] += 1
        
        return clipped, warnings
    
    def apply_low_pass_filter(self, action: np.ndarray) -> np.ndarray:
        """
        应用低通滤波平滑动作
        
        Args:
            action: 当前动作
            
        Returns:
            filtered_action: 滤波后的动作
        """
        if self.prev_action is None:
            self.prev_action = action.copy()
            return action
        
        alpha = self.params.filter_alpha
        filtered = alpha * action + (1 - alpha) * self.prev_action
        self.prev_action = filtered.copy()
        
        self.stats['filter_applied'] += 1
        return filtered
    
    def check_and_clip(
        self,
        action: np.ndarray,
        current_state: np.ndarray,
        apply_filter: bool = True,
        check_workspace: bool = False,
        end_pose: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        执行完整的安全检查和裁剪
        
        Args:
            action: 动作向量
            current_state: 当前状态
            apply_filter: 是否应用低通滤波
            check_workspace: 是否检查工作空间
            end_pose: 末端位姿 (用于工作空间检查)
            
        Returns:
            safe_action: 安全动作
            all_warnings: 所有警告信息
        """
        self.stats['total_checks'] += 1
        all_warnings = []
        
        # 1. 关节限位检查
        action, warnings = self.check_joint_limits(action)
        all_warnings.extend(warnings)
        
        # 2. 动作幅度限制
        action, warnings = self.clip_action_delta(action, current_state)
        all_warnings.extend(warnings)
        
        # 3. 工作空间检查 (可选)
        if check_workspace and end_pose is not None:
            end_pose, warnings = self.check_workspace(end_pose)
            all_warnings.extend(warnings)
        
        # 4. 低通滤波 (可选)
        if apply_filter:
            action = self.apply_low_pass_filter(action)
        
        return action, all_warnings
    
    def reset(self):
        """重置状态"""
        self.prev_action = None
        self.prev_timestamp = None
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def print_stats(self):
        """打印统计信息"""
        print("\n=== Safety Controller Statistics ===")
        print(f"Total checks:     {self.stats['total_checks']}")
        print(f"Joint clips:      {self.stats['joint_clips']}")
        print(f"Workspace clips:  {self.stats['workspace_clips']}")
        print(f"Delta clips:      {self.stats['delta_clips']}")
        print(f"Filter applied:   {self.stats['filter_applied']}")
        print("====================================\n")
    
    def save_config(self, path: str):
        """保存配置到 JSON"""
        config = {
            'joint_limits': {
                'joint_min': self.joint_limits.joint_min.tolist(),
                'joint_max': self.joint_limits.joint_max.tolist(),
                'gripper_min': self.joint_limits.gripper_min,
                'gripper_max': self.joint_limits.gripper_max,
            },
            'workspace_limits': {
                'x_min': self.workspace_limits.x_min,
                'x_max': self.workspace_limits.x_max,
                'y_min': self.workspace_limits.y_min,
                'y_max': self.workspace_limits.y_max,
                'z_min': self.workspace_limits.z_min,
                'z_max': self.workspace_limits.z_max,
            },
            'safety_params': {
                'max_joint_delta': self.params.max_joint_delta,
                'max_gripper_delta': self.params.max_gripper_delta,
                'max_position_delta': self.params.max_position_delta,
                'max_rotation_delta': self.params.max_rotation_delta,
                'filter_alpha': self.params.filter_alpha,
            },
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Safety config saved to: {path}")
    
    @classmethod
    def from_config(cls, path: str) -> 'SafetyController':
        """从 JSON 配置加载"""
        with open(path, 'r') as f:
            config = json.load(f)
        
        joint_limits = JointLimits(
            joint_min=np.array(config['joint_limits']['joint_min']),
            joint_max=np.array(config['joint_limits']['joint_max']),
            gripper_min=config['joint_limits']['gripper_min'],
            gripper_max=config['joint_limits']['gripper_max'],
        )
        
        workspace_limits = WorkspaceLimits(
            x_min=config['workspace_limits']['x_min'],
            x_max=config['workspace_limits']['x_max'],
            y_min=config['workspace_limits']['y_min'],
            y_max=config['workspace_limits']['y_max'],
            z_min=config['workspace_limits']['z_min'],
            z_max=config['workspace_limits']['z_max'],
        )
        
        safety_params = SafetyParams(
            max_joint_delta=config['safety_params']['max_joint_delta'],
            max_gripper_delta=config['safety_params']['max_gripper_delta'],
            max_position_delta=config['safety_params']['max_position_delta'],
            max_rotation_delta=config['safety_params']['max_rotation_delta'],
            filter_alpha=config['safety_params']['filter_alpha'],
        )
        
        return cls(joint_limits, workspace_limits, safety_params)


# 测试代码
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='~/rl-vla/recorded_data')
    parser.add_argument('--save_config', type=str, default=None)
    args = parser.parse_args()
    
    # 从数据集创建安全控制器
    safety = SafetyController.from_dataset_stats(args.data_dir, margin=0.1)
    
    # 测试
    print("\n=== Testing Safety Controller ===")
    
    # 测试关节限位
    test_action = np.array([0.5, 1.7, -0.8, 0.0, 0.7, 0.3, 0.05])
    current_state = np.array([0.4, 1.6, -0.7, 0.0, 0.6, 0.2, 0.04])
    
    safe_action, warnings = safety.check_and_clip(test_action, current_state)
    print(f"Original action: {test_action}")
    print(f"Safe action:     {safe_action}")
    print(f"Warnings: {warnings}")
    
    # 保存配置
    if args.save_config:
        safety.save_config(args.save_config)
    
    safety.print_stats()
