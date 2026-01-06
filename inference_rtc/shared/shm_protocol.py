#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享内存协议定义 (Python 侧)

Python → C++ 单向无锁通信
"""

import os
import time
import struct
import numpy as np
from multiprocessing import shared_memory
from dataclasses import dataclass
from typing import Optional


@dataclass
class ShmProtocol:
    """共享内存布局定义"""
    
    # 版本号
    VERSION = 1
    
    # 默认参数
    DEFAULT_DOF = 7      # 6 关节 + 1 夹爪
    DEFAULT_H = 8        # 8 个关键帧
    DEFAULT_DT_KEY = 1.0 / 30.0  # 30Hz
    
    # Header 布局 (32 bytes)
    OFFSET_VERSION = 0      # uint64 (8 bytes)
    OFFSET_T_WRITE = 8      # double (8 bytes)
    OFFSET_DOF = 16         # int32 (4 bytes)
    OFFSET_H = 20           # int32 (4 bytes)
    OFFSET_DT_KEY = 24      # double (8 bytes)
    HEADER_SIZE = 32
    
    # Payload 布局
    OFFSET_Q_KEY = 32       # double[H][dof]
    
    @classmethod
    def calc_total_size(cls, H: int = DEFAULT_H, dof: int = DEFAULT_DOF) -> int:
        """计算总大小"""
        return cls.HEADER_SIZE + H * dof * 8
    
    @classmethod
    def get_default_size(cls) -> int:
        """获取默认大小"""
        return cls.calc_total_size(cls.DEFAULT_H, cls.DEFAULT_DOF)


def get_monotonic_time() -> float:
    """获取单调时钟时间 (CLOCK_MONOTONIC)"""
    return time.clock_gettime(time.CLOCK_MONOTONIC)


class ShmKeyframeWriter:
    """
    共享内存关键帧写入器 (Python 推理进程使用)
    
    用法:
        writer = ShmKeyframeWriter.create("rtc_keyframes")
        writer.write_keyframes(q_key)  # q_key: [H, dof] numpy array
    """
    
    SHM_NAME = "rtc_keyframes"
    
    def __init__(
        self, 
        shm: shared_memory.SharedMemory, 
        is_owner: bool,
        H: int,
        dof: int,
        dt_key: float
    ):
        self._shm = shm
        self._is_owner = is_owner
        self._H = H
        self._dof = dof
        self._dt_key = dt_key
        self._version = 0
        
        # 创建 numpy 视图
        total_size = ShmProtocol.calc_total_size(H, dof)
        self._buf = np.ndarray((total_size,), dtype=np.uint8, buffer=shm.buf)
    
    @classmethod
    def create(
        cls, 
        name: str = None,
        H: int = ShmProtocol.DEFAULT_H,
        dof: int = ShmProtocol.DEFAULT_DOF,
        dt_key: float = ShmProtocol.DEFAULT_DT_KEY
    ) -> "ShmKeyframeWriter":
        """创建共享内存 (Python 推理进程调用)"""
        name = name or cls.SHM_NAME
        total_size = ShmProtocol.calc_total_size(H, dof)
        
        # 删除已存在的共享内存
        try:
            old_shm = shared_memory.SharedMemory(name=name)
            old_shm.close()
            old_shm.unlink()
        except FileNotFoundError:
            pass
        
        # 创建新的
        shm = shared_memory.SharedMemory(name=name, create=True, size=total_size)
        
        instance = cls(shm, is_owner=True, H=H, dof=dof, dt_key=dt_key)
        instance._init_header()
        
        print(f"[ShmWriter] 创建共享内存: {name}")
        print(f"  H={H}, dof={dof}, dt_key={dt_key:.4f}s")
        print(f"  总大小: {total_size} bytes")
        
        return instance
    
    def _init_header(self):
        """初始化 header"""
        # version = 0 (表示还没有有效数据)
        struct.pack_into('<Q', self._buf, ShmProtocol.OFFSET_VERSION, 0)
        # t_write = 0
        struct.pack_into('<d', self._buf, ShmProtocol.OFFSET_T_WRITE, 0.0)
        # dof
        struct.pack_into('<i', self._buf, ShmProtocol.OFFSET_DOF, self._dof)
        # H
        struct.pack_into('<i', self._buf, ShmProtocol.OFFSET_H, self._H)
        # dt_key
        struct.pack_into('<d', self._buf, ShmProtocol.OFFSET_DT_KEY, self._dt_key)
    
    def write_keyframes(self, q_key: np.ndarray) -> int:
        """
        写入关键帧轨迹
        
        Args:
            q_key: [H, dof] numpy array, 关键帧关节位置
            
        Returns:
            新的版本号
        """
        assert q_key.shape == (self._H, self._dof), \
            f"Expected shape ({self._H}, {self._dof}), got {q_key.shape}"
        
        # 1. 写入 payload (q_key)
        q_key_bytes = q_key.astype(np.float64).tobytes()
        offset = ShmProtocol.OFFSET_Q_KEY
        self._buf[offset:offset + len(q_key_bytes)] = np.frombuffer(q_key_bytes, dtype=np.uint8)
        
        # 2. 写入时间戳
        t_write = get_monotonic_time()
        struct.pack_into('<d', self._buf, ShmProtocol.OFFSET_T_WRITE, t_write)
        
        # 3. 递增版本号 (原子操作，通知 C++ 端)
        self._version += 1
        struct.pack_into('<Q', self._buf, ShmProtocol.OFFSET_VERSION, self._version)
        
        return self._version
    
    @property
    def version(self) -> int:
        return self._version
    
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


class ShmRobotStateReader:
    """
    从控制进程读取机器人状态 (复用现有的 SharedState)
    
    这是一个适配器，从现有的 inference/shared_state.py 读取状态
    """
    
    def __init__(self, shared_state):
        """
        Args:
            shared_state: inference.shared_state.SharedState 实例
        """
        self._shared_state = shared_state
    
    def read_state(self) -> tuple:
        """
        读取机器人状态
        
        Returns:
            (pos, vel, timestamp): 关节位置、速度、时间戳
        """
        return self._shared_state.read_robot_state()
    
    def get_state(self) -> int:
        """获取控制状态标志"""
        return self._shared_state.get_state()
