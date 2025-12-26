#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享内存通信模块

用于推理进程和控制进程之间的低延迟通信。
使用 multiprocessing.shared_memory 实现零拷贝数据传输。

架构:
    ┌─────────────────────┐     SharedMemory     ┌─────────────────────┐
    │   Inference Process │◄───────────────────►│   Control Process   │
    │                     │                       │                     │
    │  - 相机采集          │   SharedState        │  - 500Hz 控制循环   │
    │  - GPU 推理          │   - action_buffer    │  - robot_ctrl 独占  │
    │  - UI 显示          │   - robot_state      │  - 安全监控          │
    │                     │   - control_flags    │                     │
    └─────────────────────┘                       └─────────────────────┘
"""

import numpy as np
from multiprocessing import shared_memory, Lock
from dataclasses import dataclass
from typing import Optional, Tuple
import struct
import time


# ============================================================================
# 控制标志位定义
# ============================================================================

class ControlFlags:
    """进程间控制标志"""
    # 状态标志 (低 8 位)
    IDLE = 0
    RUNNING = 1
    PAUSED = 2
    RESETTING = 3
    EMERGENCY_STOP = 4
    DAMPING = 5
    
    # 请求标志 (8-15 位)
    REQUEST_START = 1 << 8
    REQUEST_PAUSE = 1 << 9
    REQUEST_RESET = 1 << 10
    REQUEST_STOP = 1 << 11
    
    # 确认标志 (16-23 位)
    ACK_START = 1 << 16
    ACK_PAUSE = 1 << 17
    ACK_RESET = 1 << 18
    ACK_STOP = 1 << 19
    
    # 错误标志 (24-31 位)
    ERROR_TIMEOUT = 1 << 24
    ERROR_SAFETY = 1 << 25
    ERROR_COMMUNICATION = 1 << 26


# ============================================================================
# 共享内存布局
# ============================================================================

@dataclass
class SharedMemoryLayout:
    """
    共享内存布局定义
    
    总大小: 16384 bytes (16KB, 页对齐)
    
    Layout:
        [0:4]       - version (uint32)
        [4:8]       - flags (uint32) - ControlFlags
        [8:16]      - timestamp (float64)
        [16:20]     - action_buffer_size (uint32) - 当前动作缓冲区有效长度
        [20:24]     - action_buffer_idx (uint32) - 当前执行到的动作索引
        [24:32]     - inference_timestamp (float64) - 最近一次推理时间戳
        [32:40]     - control_timestamp (float64) - 最近一次控制时间戳
        
        [64:120]    - robot_state (7 * float64 = 56 bytes)
                      [joint_pos(6) + gripper_pos(1)]
        [120:176]   - robot_vel (7 * float64 = 56 bytes)
                      [joint_vel(6) + gripper_vel(1)]
        
        [256:9216]  - action_buffer (160 * 7 * float64 = 8960 bytes)
                      最多存储 160 个动作帧 (足够 8帧@30Hz -> 500Hz 插值)
        
        [9216:9272] - target_pose (7 * float64 = 56 bytes)
                      暂停时保存的目标位置
        
        [9272:9280] - error_code (uint64)
        [9280:16384] - 扩展区域
    """
    VERSION = 2  # 版本号递增，因为布局变化
    TOTAL_SIZE = 16384
    
    # Header offsets
    OFFSET_VERSION = 0
    OFFSET_FLAGS = 4
    OFFSET_TIMESTAMP = 8
    OFFSET_ACTION_SIZE = 16
    OFFSET_ACTION_IDX = 20
    OFFSET_INFERENCE_TS = 24
    OFFSET_CONTROL_TS = 32
    
    # Robot state offsets
    OFFSET_ROBOT_STATE = 64
    OFFSET_ROBOT_VEL = 120
    
    # Action buffer - 支持 160 帧 (足够 8帧@30Hz 插值到 500Hz)
    OFFSET_ACTION_BUFFER = 256
    ACTION_BUFFER_SIZE = 8960  # 160 * 7 * 8 bytes
    MAX_ACTION_FRAMES = 160
    ACTION_DIM = 7
    
    # Target pose (for pause) - 更新偏移
    OFFSET_TARGET_POSE = 9216  # 256 + 8960 = 9216
    
    # Initial pose (from dataset)
    OFFSET_INITIAL_POSE = 9272  # 7 * float64 = 56 bytes
    OFFSET_INITIAL_POSE_VALID = 9328  # uint32, 1 if initial pose is set
    
    # Error
    OFFSET_ERROR_CODE = 9336


class SharedState:
    """
    共享状态管理器
    
    用于推理进程和控制进程之间的通信。
    
    使用方式:
        # 控制进程 (创建者)
        state = SharedState.create("arx5_control")
        
        # 推理进程 (连接者)
        state = SharedState.connect("arx5_control")
        
        # 写入动作
        state.write_actions(action_sequence)
        
        # 读取机器人状态
        robot_state = state.read_robot_state()
        
        # 请求暂停
        state.request_pause()
    """
    
    def __init__(self, shm: shared_memory.SharedMemory, is_owner: bool = False):
        self._shm = shm
        self._is_owner = is_owner
        self._buf = np.ndarray(
            (SharedMemoryLayout.TOTAL_SIZE,), 
            dtype=np.uint8, 
            buffer=shm.buf
        )
        self._layout = SharedMemoryLayout()
        
    @classmethod
    def create(cls, name: str) -> "SharedState":
        """创建共享内存 (控制进程调用)"""
        try:
            # 尝试删除已存在的共享内存
            old_shm = shared_memory.SharedMemory(name=name)
            old_shm.close()
            old_shm.unlink()
        except FileNotFoundError:
            pass
        
        shm = shared_memory.SharedMemory(
            name=name, 
            create=True, 
            size=SharedMemoryLayout.TOTAL_SIZE
        )
        
        instance = cls(shm, is_owner=True)
        instance._initialize()
        return instance
    
    @classmethod
    def connect(cls, name: str, timeout: float = 5.0) -> "SharedState":
        """连接到共享内存 (推理进程调用)"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                shm = shared_memory.SharedMemory(name=name)
                instance = cls(shm, is_owner=False)
                
                # 验证版本
                version = instance._read_uint32(SharedMemoryLayout.OFFSET_VERSION)
                if version != SharedMemoryLayout.VERSION:
                    raise ValueError(f"版本不匹配: {version} != {SharedMemoryLayout.VERSION}")
                
                return instance
            except FileNotFoundError:
                time.sleep(0.1)
        
        raise TimeoutError(f"无法连接到共享内存 '{name}'，超时 {timeout}s")
    
    def _initialize(self):
        """初始化共享内存"""
        # 清零
        self._buf[:] = 0
        
        # 写入版本号
        self._write_uint32(SharedMemoryLayout.OFFSET_VERSION, SharedMemoryLayout.VERSION)
        
        # 初始状态
        self._write_uint32(SharedMemoryLayout.OFFSET_FLAGS, ControlFlags.IDLE)
        self._write_float64(SharedMemoryLayout.OFFSET_TIMESTAMP, time.time())
    
    def close(self):
        """关闭共享内存"""
        self._shm.close()
        if self._is_owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
    
    def __del__(self):
        try:
            self.close()
        except:
            pass
    
    # ========================================================================
    # 底层读写方法
    # ========================================================================
    
    def _read_uint32(self, offset: int) -> int:
        return struct.unpack('I', bytes(self._buf[offset:offset+4]))[0]
    
    def _write_uint32(self, offset: int, value: int):
        self._buf[offset:offset+4] = np.frombuffer(
            struct.pack('I', value), dtype=np.uint8
        )
    
    def _read_uint64(self, offset: int) -> int:
        return struct.unpack('Q', bytes(self._buf[offset:offset+8]))[0]
    
    def _write_uint64(self, offset: int, value: int):
        self._buf[offset:offset+8] = np.frombuffer(
            struct.pack('Q', value), dtype=np.uint8
        )
    
    def _read_float64(self, offset: int) -> float:
        return struct.unpack('d', bytes(self._buf[offset:offset+8]))[0]
    
    def _write_float64(self, offset: int, value: float):
        self._buf[offset:offset+8] = np.frombuffer(
            struct.pack('d', value), dtype=np.uint8
        )
    
    def _read_float64_array(self, offset: int, size: int) -> np.ndarray:
        return np.frombuffer(
            bytes(self._buf[offset:offset+size*8]), 
            dtype=np.float64
        ).copy()
    
    def _write_float64_array(self, offset: int, arr: np.ndarray):
        data = arr.astype(np.float64).tobytes()
        self._buf[offset:offset+len(data)] = np.frombuffer(data, dtype=np.uint8)
    
    # ========================================================================
    # 高层 API
    # ========================================================================
    
    # --- 控制标志 ---
    
    def get_flags(self) -> int:
        """获取当前控制标志"""
        return self._read_uint32(SharedMemoryLayout.OFFSET_FLAGS)
    
    def set_flags(self, flags: int):
        """设置控制标志"""
        self._write_uint32(SharedMemoryLayout.OFFSET_FLAGS, flags)
    
    def get_state(self) -> int:
        """获取状态 (低 8 位)"""
        return self.get_flags() & 0xFF
    
    def set_state(self, state: int):
        """设置状态"""
        flags = self.get_flags()
        flags = (flags & ~0xFF) | (state & 0xFF)
        self.set_flags(flags)
    
    def request_start(self):
        """请求开始推理"""
        flags = self.get_flags()
        flags |= ControlFlags.REQUEST_START
        self.set_flags(flags)
    
    def request_pause(self):
        """请求暂停"""
        flags = self.get_flags()
        flags |= ControlFlags.REQUEST_PAUSE
        self.set_flags(flags)
    
    def request_reset(self):
        """请求复位"""
        flags = self.get_flags()
        flags |= ControlFlags.REQUEST_RESET
        self.set_flags(flags)
    
    def request_stop(self):
        """请求紧急停止"""
        flags = self.get_flags()
        flags |= ControlFlags.REQUEST_STOP
        self.set_flags(flags)
    
    def clear_requests(self):
        """清除所有请求标志"""
        flags = self.get_flags()
        flags &= ~(0xFF << 8)  # 清除 8-15 位
        self.set_flags(flags)
    
    def ack_start(self):
        """确认开始"""
        flags = self.get_flags()
        flags |= ControlFlags.ACK_START
        flags &= ~ControlFlags.REQUEST_START
        self.set_flags(flags)
    
    def ack_pause(self):
        """确认暂停"""
        flags = self.get_flags()
        flags |= ControlFlags.ACK_PAUSE
        flags &= ~ControlFlags.REQUEST_PAUSE
        self.set_flags(flags)
    
    def ack_reset(self):
        """确认复位"""
        flags = self.get_flags()
        flags |= ControlFlags.ACK_RESET
        flags &= ~ControlFlags.REQUEST_RESET
        self.set_flags(flags)
    
    def ack_stop(self):
        """确认停止"""
        flags = self.get_flags()
        flags |= ControlFlags.ACK_STOP
        flags &= ~ControlFlags.REQUEST_STOP
        self.set_flags(flags)
    
    def clear_acks(self):
        """清除所有确认标志"""
        flags = self.get_flags()
        flags &= ~(0xFF << 16)  # 清除 16-23 位
        self.set_flags(flags)
    
    def has_request(self, request: int) -> bool:
        """检查是否有指定请求"""
        return bool(self.get_flags() & request)
    
    def has_ack(self, ack: int) -> bool:
        """检查是否有指定确认"""
        return bool(self.get_flags() & ack)
    
    # --- 时间戳 ---
    
    def update_inference_timestamp(self):
        """更新推理时间戳"""
        self._write_float64(SharedMemoryLayout.OFFSET_INFERENCE_TS, time.time())
    
    def update_control_timestamp(self):
        """更新控制时间戳"""
        self._write_float64(SharedMemoryLayout.OFFSET_CONTROL_TS, time.time())
    
    def get_inference_timestamp(self) -> float:
        """获取推理时间戳"""
        return self._read_float64(SharedMemoryLayout.OFFSET_INFERENCE_TS)
    
    def get_control_timestamp(self) -> float:
        """获取控制时间戳"""
        return self._read_float64(SharedMemoryLayout.OFFSET_CONTROL_TS)
    
    # --- 机器人状态 (控制进程写入，推理进程读取) ---
    
    def write_robot_state(self, joint_pos: np.ndarray, gripper_pos: float,
                          joint_vel: np.ndarray = None, gripper_vel: float = 0.0):
        """写入机器人状态"""
        # 位置
        state = np.zeros(7, dtype=np.float64)
        state[:6] = joint_pos[:6]
        state[6] = gripper_pos
        self._write_float64_array(SharedMemoryLayout.OFFSET_ROBOT_STATE, state)
        
        # 速度
        if joint_vel is not None:
            vel = np.zeros(7, dtype=np.float64)
            vel[:6] = joint_vel[:6]
            vel[6] = gripper_vel
            self._write_float64_array(SharedMemoryLayout.OFFSET_ROBOT_VEL, vel)
        
        self._write_float64(SharedMemoryLayout.OFFSET_TIMESTAMP, time.time())
    
    def read_robot_state(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        读取机器人状态
        
        Returns:
            (position[7], velocity[7], timestamp)
        """
        pos = self._read_float64_array(SharedMemoryLayout.OFFSET_ROBOT_STATE, 7)
        vel = self._read_float64_array(SharedMemoryLayout.OFFSET_ROBOT_VEL, 7)
        ts = self._read_float64(SharedMemoryLayout.OFFSET_TIMESTAMP)
        return pos, vel, ts
    
    # --- 动作缓冲区 (推理进程写入，控制进程读取) ---
    
    def write_actions(self, actions: np.ndarray):
        """
        写入动作序列
        
        Args:
            actions: shape (N, 7), N <= MAX_ACTION_FRAMES
        """
        actions = np.asarray(actions, dtype=np.float64)
        n_frames = min(len(actions), SharedMemoryLayout.MAX_ACTION_FRAMES)
        
        # 写入动作数据
        flat_actions = actions[:n_frames].flatten()
        self._write_float64_array(SharedMemoryLayout.OFFSET_ACTION_BUFFER, flat_actions)
        
        # 更新元数据
        self._write_uint32(SharedMemoryLayout.OFFSET_ACTION_SIZE, n_frames)
        self._write_uint32(SharedMemoryLayout.OFFSET_ACTION_IDX, 0)
        self.update_inference_timestamp()
    
    def read_actions(self) -> Tuple[np.ndarray, int, int]:
        """
        读取动作序列
        
        Returns:
            (actions[N, 7], current_idx, total_size)
        """
        n_frames = self._read_uint32(SharedMemoryLayout.OFFSET_ACTION_SIZE)
        idx = self._read_uint32(SharedMemoryLayout.OFFSET_ACTION_IDX)
        
        if n_frames == 0:
            return np.zeros((0, 7), dtype=np.float64), 0, 0
        
        flat_actions = self._read_float64_array(
            SharedMemoryLayout.OFFSET_ACTION_BUFFER, 
            n_frames * 7
        )
        actions = flat_actions.reshape(n_frames, 7)
        
        return actions, idx, n_frames
    
    def get_next_action(self) -> Tuple[Optional[np.ndarray], bool]:
        """
        获取下一个要执行的动作
        
        Returns:
            (action[7] or None, is_last)
        """
        n_frames = self._read_uint32(SharedMemoryLayout.OFFSET_ACTION_SIZE)
        idx = self._read_uint32(SharedMemoryLayout.OFFSET_ACTION_IDX)
        
        if n_frames == 0 or idx >= n_frames:
            return None, True
        
        # 读取当前动作
        offset = SharedMemoryLayout.OFFSET_ACTION_BUFFER + idx * 7 * 8
        action = self._read_float64_array(offset, 7)
        
        # 更新索引
        self._write_uint32(SharedMemoryLayout.OFFSET_ACTION_IDX, idx + 1)
        
        is_last = (idx + 1 >= n_frames)
        return action, is_last
    
    def get_action_buffer_status(self) -> Tuple[int, int]:
        """获取动作缓冲区状态: (current_idx, total_size)"""
        idx = self._read_uint32(SharedMemoryLayout.OFFSET_ACTION_IDX)
        size = self._read_uint32(SharedMemoryLayout.OFFSET_ACTION_SIZE)
        return idx, size
    
    def is_action_buffer_empty(self) -> bool:
        """检查动作缓冲区是否为空"""
        idx, size = self.get_action_buffer_status()
        return size == 0 or idx >= size
    
    def should_request_new_chunk(self, threshold_ratio: float = 0.7) -> bool:
        """
        检查是否应该请求新的动作 chunk
        
        与原版 arx5_policy_inference 保持一致：当执行到 ~70% 时就提前触发新推理，
        而不是等缓冲区完全耗尽。这样可以实现无缝衔接。
        
        Args:
            threshold_ratio: 触发阈值比例 (默认 0.7 = 70%)
            
        Returns:
            True 如果应该请求新 chunk
        """
        idx, size = self.get_action_buffer_status()
        if size == 0:
            return True
        return idx >= int(size * threshold_ratio)
    
    # --- 目标位置 (暂停时使用) ---
    
    def write_target_pose(self, pose: np.ndarray):
        """写入目标位置 (暂停时保持的位置)"""
        pose = np.asarray(pose, dtype=np.float64)[:7]
        self._write_float64_array(SharedMemoryLayout.OFFSET_TARGET_POSE, pose)
    
    def read_target_pose(self) -> np.ndarray:
        """读取目标位置"""
        return self._read_float64_array(SharedMemoryLayout.OFFSET_TARGET_POSE, 7)
    
    # --- 初始位姿 (从数据集加载) ---
    
    def write_initial_pose(self, pose: np.ndarray):
        """写入初始位姿 (从数据集加载的初始位置)"""
        pose = np.asarray(pose, dtype=np.float64)[:7]
        self._write_float64_array(SharedMemoryLayout.OFFSET_INITIAL_POSE, pose)
        self._write_uint32(SharedMemoryLayout.OFFSET_INITIAL_POSE_VALID, 1)
    
    def read_initial_pose(self) -> Optional[np.ndarray]:
        """
        读取初始位姿
        
        Returns:
            pose[7] 如果设置了初始位姿，否则 None
        """
        is_valid = self._read_uint32(SharedMemoryLayout.OFFSET_INITIAL_POSE_VALID)
        if is_valid:
            return self._read_float64_array(SharedMemoryLayout.OFFSET_INITIAL_POSE, 7)
        return None
    
    def has_initial_pose(self) -> bool:
        """检查是否设置了初始位姿"""
        return self._read_uint32(SharedMemoryLayout.OFFSET_INITIAL_POSE_VALID) == 1
    
    def clear_initial_pose(self):
        """清除初始位姿标记 (已应用后调用)"""
        self._write_uint32(SharedMemoryLayout.OFFSET_INITIAL_POSE_VALID, 0)
    
    # --- 错误处理 ---
    
    def set_error(self, error_code: int):
        """设置错误码"""
        self._write_uint64(SharedMemoryLayout.OFFSET_ERROR_CODE, error_code)
        flags = self.get_flags()
        flags |= (error_code & (0xFF << 24))
        self.set_flags(flags)
    
    def get_error(self) -> int:
        """获取错误码"""
        return self._read_uint64(SharedMemoryLayout.OFFSET_ERROR_CODE)
    
    def clear_error(self):
        """清除错误"""
        self._write_uint64(SharedMemoryLayout.OFFSET_ERROR_CODE, 0)
        flags = self.get_flags()
        flags &= ~(0xFF << 24)
        self.set_flags(flags)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # 模拟控制进程
        print("启动控制进程 (服务器端)...")
        state = SharedState.create("arx5_test")
        
        print("等待连接...")
        while True:
            time.sleep(0.1)
            
            # 检查请求
            if state.has_request(ControlFlags.REQUEST_START):
                print("收到 START 请求")
                state.set_state(ControlFlags.RUNNING)
                state.ack_start()
            
            if state.has_request(ControlFlags.REQUEST_PAUSE):
                print("收到 PAUSE 请求")
                state.set_state(ControlFlags.PAUSED)
                state.ack_pause()
            
            if state.has_request(ControlFlags.REQUEST_STOP):
                print("收到 STOP 请求")
                state.set_state(ControlFlags.EMERGENCY_STOP)
                state.ack_stop()
                break
            
            # 模拟写入机器人状态
            state.write_robot_state(
                joint_pos=np.random.randn(6) * 0.1,
                gripper_pos=0.04,
                joint_vel=np.zeros(6),
            )
            
            # 检查动作缓冲区
            actions, idx, size = state.read_actions()
            if size > 0 and idx < size:
                action, is_last = state.get_next_action()
                print(f"执行动作 {idx}/{size}: {action[:3]}...")
            
            state.update_control_timestamp()
        
        state.close()
        print("控制进程退出")
        
    else:
        # 模拟推理进程
        print("启动推理进程 (客户端)...")
        state = SharedState.connect("arx5_test")
        
        print("已连接，发送 START 请求...")
        state.request_start()
        
        # 等待确认
        while not state.has_ack(ControlFlags.ACK_START):
            time.sleep(0.01)
        print("START 已确认")
        state.clear_acks()
        
        # 写入动作
        actions = np.random.randn(10, 7)
        state.write_actions(actions)
        print(f"写入 {len(actions)} 个动作")
        
        # 等待执行
        time.sleep(2)
        
        # 读取机器人状态
        pos, vel, ts = state.read_robot_state()
        print(f"机器人状态: pos={pos[:3]}..., ts={ts:.3f}")
        
        # 发送停止
        print("发送 STOP 请求...")
        state.request_stop()
        
        while not state.has_ack(ControlFlags.ACK_STOP):
            time.sleep(0.01)
        print("STOP 已确认")
        
        state.close()
        print("推理进程退出")
