#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关节空间轨迹插值器

使用三次样条插值实现关节空间的平滑轨迹生成。
参考 umi-arx 的 PoseTrajectoryInterpolator，但针对关节空间进行了优化。

特点:
- 支持 6 自由度关节 + 1 夹爪
- 三次样条插值保证速度连续
- 支持 drive_to_waypoint 和 schedule_waypoint 模式
- 支持最大速度限制
"""

from typing import Union, Optional
import numpy as np
from scipy import interpolate


def joint_distance(start_joints: np.ndarray, end_joints: np.ndarray) -> float:
    """
    计算关节空间距离 (最大单关节角度变化)
    
    Args:
        start_joints: 起始关节角度 [N,]
        end_joints: 目标关节角度 [N,]
    
    Returns:
        最大单关节角度变化 (rad)
    """
    return np.max(np.abs(end_joints - start_joints))


class JointTrajectoryInterpolator:
    """
    关节空间轨迹插值器
    
    存储一系列带时间戳的关节角度，并提供插值查询功能。
    使用三次样条插值保证速度连续性。
    """
    
    def __init__(self, times: np.ndarray, joints: np.ndarray):
        """
        Args:
            times: 时间戳数组 [T,]
            joints: 关节角度数组 [T, 7] (6关节 + 1夹爪)
        """
        assert len(times) >= 1, "至少需要一个时间点"
        assert len(joints) == len(times), "时间和关节数量不匹配"
        
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(joints, np.ndarray):
            joints = np.array(joints)
        
        # 确保 joints 是 2D
        if joints.ndim == 1:
            joints = joints.reshape(1, -1)
        
        self.n_joints = joints.shape[1]  # 通常是 7 (6 关节 + 1 夹爪)
        
        if len(times) == 1:
            # 单点情况，直接返回该点
            self.single_step = True
            self._times = times
            self._joints = joints
            self._interp = None
        else:
            self.single_step = False
            # 确保时间严格递增
            assert np.all(times[1:] > times[:-1]), "时间戳必须严格递增"
            
            self._times = times
            self._joints = joints
            
            # 使用三次样条插值
            # kind='cubic' 需要至少4个点，少于4个点时使用 'linear' 或 'quadratic'
            if len(times) >= 4:
                kind = 'cubic'
            elif len(times) >= 3:
                kind = 'quadratic'
            else:
                kind = 'linear'
            
            self._interp = interpolate.interp1d(
                times, joints, axis=0, 
                kind=kind, 
                bounds_error=False,
                fill_value=(joints[0], joints[-1])  # 边界外使用端点值
            )
    
    @property
    def times(self) -> np.ndarray:
        """获取时间戳数组"""
        return self._times.copy()
    
    @property
    def joints(self) -> np.ndarray:
        """获取关节角度数组"""
        return self._joints.copy()
    
    @property
    def start_time(self) -> float:
        """轨迹起始时间"""
        return float(self._times[0])
    
    @property
    def end_time(self) -> float:
        """轨迹结束时间"""
        return float(self._times[-1])
    
    @property
    def duration(self) -> float:
        """轨迹持续时间"""
        return self.end_time - self.start_time
    
    def __call__(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        插值查询
        
        Args:
            t: 查询时间 (标量或数组)
        
        Returns:
            关节角度 [n_joints,] 或 [len(t), n_joints]
        """
        is_single = isinstance(t, (int, float))
        if is_single:
            t = np.array([t])
        
        if self.single_step:
            result = np.tile(self._joints[0], (len(t), 1))
        else:
            result = self._interp(t)
        
        if is_single:
            return result[0]
        return result
    
    def trim(self, start_t: float, end_t: float) -> "JointTrajectoryInterpolator":
        """
        裁剪轨迹到指定时间范围
        
        Args:
            start_t: 起始时间
            end_t: 结束时间
        
        Returns:
            新的插值器
        """
        assert start_t <= end_t, "起始时间必须小于等于结束时间"
        
        # 找出在范围内的时间点
        times = self._times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        
        # 添加边界点
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # 去重
        all_times = np.unique(all_times)
        
        # 插值获取对应的关节角度
        all_joints = self(all_times)
        
        return JointTrajectoryInterpolator(times=all_times, joints=all_joints)
    
    def drive_to_waypoint(
        self,
        joints: np.ndarray,
        time: float,
        curr_time: float,
        max_speed: float = np.inf,
    ) -> "JointTrajectoryInterpolator":
        """
        从当前位置驱动到目标点 (覆盖后续轨迹)
        
        这是一个简单的"立即执行"模式：从当前时间点开始，
        以最大速度限制的方式移动到目标点。
        
        Args:
            joints: 目标关节角度 [n_joints,]
            time: 期望到达时间
            curr_time: 当前时间
            max_speed: 最大关节速度 (rad/s)
        
        Returns:
            新的插值器
        """
        assert max_speed > 0, "最大速度必须大于0"
        time = max(time, curr_time)
        
        # 获取当前关节位置
        curr_joints = self(curr_time)
        
        # 计算最小持续时间 (基于最大速度)
        joint_dist = joint_distance(curr_joints, joints)
        min_duration = joint_dist / max_speed
        
        # 确定实际持续时间
        duration = time - curr_time
        duration = max(duration, min_duration)
        
        # 计算到达时间
        target_time = curr_time + duration
        
        # 创建新的插值器 (只包含当前点和目标点)
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [target_time])
        joints_arr = np.vstack([trimmed_interp.joints, [joints]])
        
        return JointTrajectoryInterpolator(times=times, joints=joints_arr)
    
    def schedule_waypoint(
        self,
        joints: np.ndarray,
        time: float,
        max_speed: float = np.inf,
        curr_time: Optional[float] = None,
        last_waypoint_time: Optional[float] = None,
    ) -> "JointTrajectoryInterpolator":
        """
        调度一个航点 (保留历史轨迹)
        
        这是一个"排队执行"模式：将新的航点添加到轨迹末尾，
        同时考虑最大速度限制。
        
        Args:
            joints: 目标关节角度 [n_joints,]
            time: 期望到达时间
            max_speed: 最大关节速度 (rad/s)
            curr_time: 当前时间 (用于裁剪历史)
            last_waypoint_time: 上一个航点的时间
        
        Returns:
            新的插值器
        """
        assert max_speed > 0, "最大速度必须大于0"
        
        # 确定裁剪范围
        start_time = self._times[0]
        end_time = self._times[-1]
        
        if curr_time is not None:
            if time <= curr_time:
                # 目标时间已过，不做任何改变
                return self
            
            start_time = max(curr_time, start_time)
            
            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time
        
        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        
        # 裁剪当前轨迹
        trimmed_interp = self.trim(start_time, end_time)
        
        # 计算从末端到新航点的持续时间
        duration = time - end_time
        end_joints = trimmed_interp(end_time)
        joint_dist = joint_distance(end_joints, joints)
        min_duration = joint_dist / max_speed
        duration = max(duration, min_duration)
        
        # 计算实际到达时间
        target_time = end_time + duration
        
        # 添加新航点
        times = np.append(trimmed_interp.times, [target_time])
        joints_arr = np.vstack([trimmed_interp.joints, [joints]])
        
        return JointTrajectoryInterpolator(times=times, joints=joints_arr)
    
    def extend(self, other: "JointTrajectoryInterpolator") -> "JointTrajectoryInterpolator":
        """
        扩展轨迹
        
        将另一个插值器的轨迹追加到当前轨迹末尾。
        
        Args:
            other: 另一个插值器
        
        Returns:
            合并后的新插值器
        """
        # 时间偏移
        time_offset = self.end_time - other.start_time
        
        new_times = np.concatenate([self._times, other._times[1:] + time_offset])
        new_joints = np.vstack([self._joints, other._joints[1:]])
        
        return JointTrajectoryInterpolator(times=new_times, joints=new_joints)


class JointTrajectoryBuffer:
    """
    关节轨迹缓冲区
    
    用于累积多个航点，然后一次性创建插值器。
    适用于 UPDATE_TRAJECTORY 命令模式。
    """
    
    def __init__(self):
        self.times = []
        self.joints = []
    
    def add(self, time: float, joints: np.ndarray):
        """添加一个航点"""
        self.times.append(time)
        self.joints.append(np.array(joints))
    
    def clear(self):
        """清空缓冲区"""
        self.times = []
        self.joints = []
    
    @property
    def size(self) -> int:
        """缓冲区中的航点数量"""
        return len(self.times)
    
    def to_interpolator(self) -> Optional[JointTrajectoryInterpolator]:
        """
        将缓冲区转换为插值器
        
        Returns:
            插值器，如果缓冲区为空则返回 None
        """
        if self.size == 0:
            return None
        
        times = np.array(self.times)
        joints = np.array(self.joints)
        
        return JointTrajectoryInterpolator(times=times, joints=joints)


# ===================== 测试代码 =====================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 测试基本插值
    print("=== 测试基本插值 ===")
    times = np.array([0.0, 1.0, 2.0, 3.0])
    joints = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.5],
        [1.0, 0.6, 0.4, 0.2, 0.1, 0.0, 1.0],
        [1.5, 0.9, 0.6, 0.3, 0.2, 0.0, 0.5],
    ])
    
    interp = JointTrajectoryInterpolator(times, joints)
    
    # 查询
    t_query = np.linspace(0, 3, 100)
    j_query = interp(t_query)
    
    print(f"起始时间: {interp.start_time}")
    print(f"结束时间: {interp.end_time}")
    print(f"持续时间: {interp.duration}")
    print(f"t=1.5 时的关节角: {interp(1.5)}")
    
    # 测试 drive_to_waypoint
    print("\n=== 测试 drive_to_waypoint ===")
    target = np.array([0.8, 0.4, 0.3, 0.2, 0.1, 0.0, 0.8])
    new_interp = interp.drive_to_waypoint(
        joints=target,
        time=4.0,
        curr_time=1.5,
        max_speed=2.0
    )
    print(f"新轨迹持续时间: {new_interp.duration}")
    print(f"t=4.0 时的关节角: {new_interp(4.0)}")
    
    # 测试缓冲区
    print("\n=== 测试轨迹缓冲区 ===")
    buffer = JointTrajectoryBuffer()
    buffer.add(0.0, np.zeros(7))
    buffer.add(1.0, np.ones(7) * 0.5)
    buffer.add(2.0, np.ones(7))
    
    buf_interp = buffer.to_interpolator()
    print(f"缓冲区大小: {buffer.size}")
    print(f"t=1.0 时的关节角: {buf_interp(1.0)}")
    
    print("\n=== 测试完成 ===")
