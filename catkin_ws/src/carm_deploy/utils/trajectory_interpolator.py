#!/usr/bin/env python3
"""
时间插值工具
实现 VecTF 等价功能，用于动作轨迹的时间插值
替代 svar 的 vectf.VecTF 功能
"""

import numpy as np
from collections import deque
import threading


class TrajectoryInterpolator:
    """
    轨迹时间插值器
    存储带时间戳的动作序列，支持按时间查询插值
    """
    
    def __init__(self, max_size=1000):
        """
        初始化插值器
        
        Args:
            max_size: 最大存储数量，超过后自动删除旧数据
        """
        self.max_size = max_size
        self.timestamps = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def append(self, timestamp, action):
        """
        添加带时间戳的动作
        
        Args:
            timestamp: 时间戳（秒）
            action: 动作向量（list 或 numpy array）
        """
        with self.lock:
            # 确保时间戳递增
            if len(self.timestamps) > 0 and timestamp <= self.timestamps[-1]:
                return
            
            self.timestamps.append(timestamp)
            self.actions.append(np.array(action))
    
    def get_once(self, query_time):
        """
        按时间查询动作（不插值，返回最近的有效动作）
        
        Args:
            query_time: 查询时间戳
            
        Returns:
            numpy array: 动作向量，如果没有有效动作返回 None
        """
        with self.lock:
            if len(self.timestamps) == 0:
                return None
            
            # 查找第一个大于等于 query_time 的时间戳
            for i, ts in enumerate(self.timestamps):
                if ts >= query_time:
                    return self.actions[i].copy()
            
            # 如果所有时间戳都小于查询时间，返回 None（已过期）
            return None

    def get_once_with_timestamp(self, query_time):
        """
        按时间查询动作并返回其时间戳

        Args:
            query_time: 查询时间戳

        Returns:
            tuple: (action, timestamp) 如果没有有效动作返回 (None, None)
        """
        with self.lock:
            if len(self.timestamps) == 0:
                return None, None

            for i, ts in enumerate(self.timestamps):
                if ts >= query_time:
                    return self.actions[i].copy(), ts

            return None, None
    
    def get_interpolated(self, query_time):
        """
        按时间查询动作（线性插值）
        
        Args:
            query_time: 查询时间戳
            
        Returns:
            numpy array: 插值后的动作向量，如果无法插值返回 None
        """
        with self.lock:
            if len(self.timestamps) < 2:
                return self.get_once(query_time)
            
            timestamps = list(self.timestamps)
            actions = list(self.actions)
            
            # 查找插值区间
            for i in range(len(timestamps) - 1):
                if timestamps[i] <= query_time <= timestamps[i + 1]:
                    # 线性插值
                    t0, t1 = timestamps[i], timestamps[i + 1]
                    a0, a1 = actions[i], actions[i + 1]
                    
                    alpha = (query_time - t0) / (t1 - t0)
                    return a0 + alpha * (a1 - a0)
            
            # 如果查询时间在范围外
            if query_time < timestamps[0]:
                return actions[0].copy()
            elif query_time > timestamps[-1]:
                return None  # 已过期
            
            return None
    
    def clear(self):
        """清空所有数据"""
        with self.lock:
            self.timestamps.clear()
            self.actions.clear()
    
    def clear_before(self, timestamp):
        """
        清除指定时间戳之前的数据
        
        Args:
            timestamp: 时间戳阈值
        """
        with self.lock:
            while len(self.timestamps) > 0 and self.timestamps[0] < timestamp:
                self.timestamps.popleft()
                self.actions.popleft()
    
    def __len__(self):
        return len(self.timestamps)
    
    @property
    def empty(self):
        return len(self.timestamps) == 0
    
    @property
    def latest_timestamp(self):
        """获取最新时间戳"""
        with self.lock:
            if len(self.timestamps) == 0:
                return None
            return self.timestamps[-1]
    
    @property
    def oldest_timestamp(self):
        """获取最旧时间戳"""
        with self.lock:
            if len(self.timestamps) == 0:
                return None
            return self.timestamps[0]


class ActionChunkManager:
    """
    动作块管理器
    管理多个 TrajectoryInterpolator，支持时间加权融合
    类似于原始代码中的 action_tfs 列表
    """
    
    def __init__(self, temporal_factor_k=0.01):
        """
        初始化动作块管理器
        
        Args:
            temporal_factor_k: 时间加权因子
        """
        self.temporal_factor_k = temporal_factor_k
        self.trajectories = []
        self.lock = threading.Lock()
    
    def add_trajectory(self, trajectory):
        """
        添加新的轨迹
        
        Args:
            trajectory: TrajectoryInterpolator 实例
        """
        with self.lock:
            self.trajectories.append(trajectory)
    
    def get_fused_action(self, query_time):
        """
        获取时间加权融合后的动作
        
        Args:
            query_time: 查询时间戳
            
        Returns:
            numpy array: 融合后的动作，如果没有有效动作返回 None
        """
        with self.lock:
            action_candidates = []
            valid_offset = 0
            
            # 收集所有有效的动作候选
            for idx, traj in enumerate(self.trajectories):
                action = traj.get_once(query_time)
                if action is None:
                    valid_offset = idx
                    continue
                action_candidates.append(action)
            
            # 清理过期的轨迹
            self.trajectories = self.trajectories[valid_offset:]
            
            if len(action_candidates) < 1:
                return None
            
            # 时间加权融合
            all_actions = np.array(action_candidates)
            exp_weights = np.exp(-self.temporal_factor_k * np.arange(len(action_candidates) - 1, -1, -1))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = exp_weights[:, np.newaxis]
            
            fused_action = (all_actions * exp_weights).sum(axis=0)
            return fused_action

    def get_fused_action_with_meta(self, query_time):
        """
        获取时间加权融合后的动作，并返回候选元信息

        Args:
            query_time: 查询时间戳

        Returns:
            tuple: (fused_action, meta)
                   meta: {"candidate_timestamps": [...], "weights": [...], "num_candidates": int}
        """
        with self.lock:
            action_candidates = []
            candidate_timestamps = []
            valid_offset = 0

            for idx, traj in enumerate(self.trajectories):
                action, ts = traj.get_once_with_timestamp(query_time)
                if action is None:
                    valid_offset = idx
                    continue
                action_candidates.append(action)
                candidate_timestamps.append(ts)

            self.trajectories = self.trajectories[valid_offset:]

            if len(action_candidates) < 1:
                return None, {
                    "candidate_timestamps": [],
                    "weights": [],
                    "num_candidates": 0,
                }

            all_actions = np.array(action_candidates)
            exp_weights = np.exp(-self.temporal_factor_k * np.arange(len(action_candidates) - 1, -1, -1))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = exp_weights[:, np.newaxis]

            fused_action = (all_actions * exp_weights).sum(axis=0)

            return fused_action, {
                "candidate_timestamps": candidate_timestamps,
                "weights": exp_weights.squeeze(-1).tolist(),
                "num_candidates": len(action_candidates),
            }
    
    def clear(self):
        """清空所有轨迹"""
        with self.lock:
            self.trajectories.clear()
    
    def __len__(self):
        with self.lock:
            return len(self.trajectories)


# 兼容原始 VecTF 接口的包装类
class VecTF(TrajectoryInterpolator):
    """
    兼容原始 svar VecTF 接口的包装类
    """
    
    def __init__(self, config=None):
        """
        初始化 VecTF
        
        Args:
            config: 配置字典（保留兼容性，当前未使用）
        """
        super().__init__()
        self.config = config or {}
