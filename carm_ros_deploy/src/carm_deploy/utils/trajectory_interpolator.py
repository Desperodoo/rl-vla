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
        self.lock = threading.RLock()
    
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
    管理多个 TrajectoryInterpolator，支持两种执行模式：
    
    1. temporal_ensemble (原始模式): 多 chunk 时间加权融合
       - 所有活跃 chunk 参与融合
       - 平滑但响应较慢，存在 train-infer mismatch
       
    2. receding_horizon (标准 action chunking): 只执行最新 chunk
       - 每次推理生成新 chunk，只执行前 act_horizon 步
       - 响应快，与训练时语义一致，适合 RLFT
       - 支持可选的 crossfade 平滑切换
    """
    
    # 执行模式常量
    MODE_TEMPORAL_ENSEMBLE = 'temporal_ensemble'
    MODE_RECEDING_HORIZON = 'receding_horizon'
    
    def __init__(self, temporal_factor_k=0.01, execution_mode='temporal_ensemble',
                 max_active_chunks=None, crossfade_steps=0):
        """
        初始化动作块管理器
        
        Args:
            temporal_factor_k: 时间加权因子（temporal_ensemble 模式使用）
            execution_mode: 执行模式
                - 'temporal_ensemble': 多 chunk 时间加权融合（原始行为）
                - 'receding_horizon': 标准 action chunking，只用最新 chunk
            max_active_chunks: 最大活跃 chunk 数量（可选，用于限制内存）
                - temporal_ensemble: 默认 None（不限制）
                - receding_horizon: 默认 2（当前 + 下一个）
            crossfade_steps: chunk 切换时的平滑过渡步数（仅 receding_horizon 模式）
                - 0: 无平滑，直接切换
                - >0: 在新旧 chunk 之间做线性 crossfade
        """
        self.temporal_factor_k = temporal_factor_k
        self.execution_mode = execution_mode
        self.crossfade_steps = crossfade_steps
        
        # 设置默认 max_active_chunks
        if max_active_chunks is None:
            if execution_mode == self.MODE_RECEDING_HORIZON:
                max_active_chunks = 2
            # temporal_ensemble 模式默认不限制
        self.max_active_chunks = max_active_chunks
        
        self.trajectories = []  # list of (chunk_id, TrajectoryInterpolator)
        self.lock = threading.Lock()
        self._next_chunk_id = 0  # 递增 chunk ID
        self._last_fused_chunk_ids = []  # 上次融合使用的 chunk_ids
        
        # receding_horizon 模式专用状态
        self._current_chunk_id = None  # 当前正在执行的 chunk ID
        self._current_chunk_step = 0  # 当前 chunk 已执行的步数
        self._last_action = None  # 上次返回的动作（用于 hold 和 crossfade）
        self._crossfade_progress = 0  # crossfade 进度计数器
    
    def add_trajectory(self, trajectory):
        """
        添加新的轨迹
        
        Args:
            trajectory: TrajectoryInterpolator 实例
            
        Returns:
            int: 分配给该轨迹的 chunk_id
        """
        with self.lock:
            chunk_id = self._next_chunk_id
            self._next_chunk_id += 1
            self.trajectories.append((chunk_id, trajectory))
            
            # 如果设置了 max_active_chunks，移除旧的 chunk
            if self.max_active_chunks is not None:
                while len(self.trajectories) > self.max_active_chunks:
                    self.trajectories.pop(0)
            
            return chunk_id
    
    def _get_action_temporal_ensemble(self, query_time, return_meta=False):
        """
        时间加权融合模式（原始行为）
        
        Args:
            query_time: 查询时间戳
            return_meta: 是否返回元信息
            
        Returns:
            如果 return_meta=False: fused_action 或 None
            如果 return_meta=True: (fused_action, meta) 或 (None, empty_meta)
        """
        action_candidates = []
        candidate_timestamps = []
        chunk_ids_used = []
        valid_offset = 0

        for idx, (chunk_id, traj) in enumerate(self.trajectories):
            action, ts = traj.get_once_with_timestamp(query_time)
            if action is None:
                valid_offset = idx + 1
                continue
            action_candidates.append(action)
            candidate_timestamps.append(ts)
            chunk_ids_used.append(chunk_id)

        # 清理过期的轨迹
        self.trajectories = self.trajectories[valid_offset:]

        if len(action_candidates) < 1:
            self._last_fused_chunk_ids = []
            if return_meta:
                return None, {
                    "candidate_timestamps": [],
                    "weights": [],
                    "num_candidates": 0,
                    "used_chunk_ids": [],
                    "execution_mode": self.execution_mode,
                }
            return None

        all_actions = np.array(action_candidates)
        exp_weights = np.exp(-self.temporal_factor_k * np.arange(len(action_candidates) - 1, -1, -1))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = exp_weights[:, np.newaxis]

        fused_action = (all_actions * exp_weights).sum(axis=0)
        self._last_fused_chunk_ids = chunk_ids_used
        self._last_action = fused_action.copy()

        if return_meta:
            return fused_action, {
                "candidate_timestamps": candidate_timestamps,
                "weights": exp_weights.squeeze(-1).tolist(),
                "num_candidates": len(action_candidates),
                "used_chunk_ids": chunk_ids_used,
                "execution_mode": self.execution_mode,
            }
        return fused_action
    
    def _get_action_receding_horizon(self, query_time, return_meta=False):
        """
        标准 action chunking 模式（receding horizon）
        
        只使用最新的有效 chunk，执行完 act_horizon 步后等待新 chunk
        支持可选的 crossfade 平滑切换
        
        Args:
            query_time: 查询时间戳
            return_meta: 是否返回元信息
            
        Returns:
            如果 return_meta=False: action 或 None
            如果 return_meta=True: (action, meta) 或 (None, empty_meta)
        """
        # 找到最新的有效 chunk
        latest_valid_chunk_id = None
        latest_valid_action = None
        latest_valid_ts = None
        prev_valid_action = None  # 用于 crossfade
        
        valid_offset = 0
        
        for idx, (chunk_id, traj) in enumerate(self.trajectories):
            action, ts = traj.get_once_with_timestamp(query_time)
            if action is None:
                valid_offset = idx + 1
                continue
            
            # 保存前一个有效动作（用于 crossfade）
            if latest_valid_action is not None:
                prev_valid_action = latest_valid_action.copy()
            
            latest_valid_chunk_id = chunk_id
            latest_valid_action = action
            latest_valid_ts = ts
        
        # 清理过期的轨迹
        self.trajectories = self.trajectories[valid_offset:]
        
        # 如果没有有效 chunk
        if latest_valid_action is None:
            self._last_fused_chunk_ids = []
            # Option B: 返回上次的动作（hold position）
            if self._last_action is not None:
                if return_meta:
                    return self._last_action.copy(), {
                        "candidate_timestamps": [],
                        "weights": [1.0],
                        "num_candidates": 0,
                        "used_chunk_ids": [],
                        "execution_mode": self.execution_mode,
                        "is_hold": True,
                    }
                return self._last_action.copy()
            
            if return_meta:
                return None, {
                    "candidate_timestamps": [],
                    "weights": [],
                    "num_candidates": 0,
                    "used_chunk_ids": [],
                    "execution_mode": self.execution_mode,
                }
            return None
        
        # 检测 chunk 切换
        is_new_chunk = (self._current_chunk_id != latest_valid_chunk_id)
        if is_new_chunk:
            self._current_chunk_id = latest_valid_chunk_id
            self._current_chunk_step = 0
            self._crossfade_progress = 0
        else:
            self._current_chunk_step += 1
        
        # 应用 crossfade（如果启用且有前一个动作）
        final_action = latest_valid_action.copy()
        crossfade_weight = 1.0  # 新 chunk 的权重
        
        if (self.crossfade_steps > 0 and 
            is_new_chunk and 
            self._last_action is not None and
            self._crossfade_progress < self.crossfade_steps):
            
            # 线性 crossfade: 从旧动作逐渐过渡到新动作
            alpha = (self._crossfade_progress + 1) / self.crossfade_steps
            final_action = (1 - alpha) * self._last_action + alpha * latest_valid_action
            crossfade_weight = alpha
            self._crossfade_progress += 1
        
        self._last_action = final_action.copy()
        self._last_fused_chunk_ids = [latest_valid_chunk_id]
        
        if return_meta:
            return final_action, {
                "candidate_timestamps": [latest_valid_ts] if latest_valid_ts else [],
                "weights": [crossfade_weight],
                "num_candidates": 1,
                "used_chunk_ids": [latest_valid_chunk_id],
                "execution_mode": self.execution_mode,
                "current_chunk_id": self._current_chunk_id,
                "current_chunk_step": self._current_chunk_step,
                "is_new_chunk": is_new_chunk,
                "crossfade_progress": self._crossfade_progress if self.crossfade_steps > 0 else None,
            }
        return final_action
    
    def get_fused_action(self, query_time):
        """
        获取动作（根据 execution_mode 分发到不同实现）
        
        Args:
            query_time: 查询时间戳
            
        Returns:
            numpy array: 动作向量，如果没有有效动作返回 None
        """
        with self.lock:
            if self.execution_mode == self.MODE_RECEDING_HORIZON:
                return self._get_action_receding_horizon(query_time, return_meta=False)
            else:
                return self._get_action_temporal_ensemble(query_time, return_meta=False)

    def get_fused_action_with_meta(self, query_time):
        """
        获取动作并返回元信息（根据 execution_mode 分发到不同实现）

        Args:
            query_time: 查询时间戳

        Returns:
            tuple: (action, meta)
                   meta 包含 execution_mode 和其他诊断信息
        """
        with self.lock:
            if self.execution_mode == self.MODE_RECEDING_HORIZON:
                return self._get_action_receding_horizon(query_time, return_meta=True)
            else:
                return self._get_action_temporal_ensemble(query_time, return_meta=True)
    
    def clear(self):
        """清空所有轨迹"""
        with self.lock:
            self.trajectories.clear()
            self._last_fused_chunk_ids = []
    
    def get_last_fused_chunk_ids(self):
        """获取上次融合使用的 chunk_ids（供外部查询，如数据采集）"""
        with self.lock:
            return list(self._last_fused_chunk_ids)
    
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
