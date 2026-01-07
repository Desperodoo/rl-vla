#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense Camera Module (多进程架构)

支持:
- D435i (腕部相机) - 用于推理
- D455 (外部相机) - 用于录制
- SharedMemoryRingBuffer 传递帧数据
- 视频录制功能
"""

import multiprocessing as mp
import time
import os
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import cv2

# 添加 umi-arx 的 shared_memory 模块路径
UMI_ARX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'umi-arx')
if UMI_ARX_PATH not in sys.path:
    sys.path.insert(0, UMI_ARX_PATH)

from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[RealSenseCamera] pyrealsense2 未安装，相机功能不可用")


# ===================== 配置类 =====================

@dataclass
class RealSenseCameraConfig:
    """RealSense 相机配置"""
    name: str
    serial_number: str
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    enable_depth: bool = False
    enable_recording: bool = False  # 是否启用视频录制
    recording_path: Optional[str] = None  # 录制视频保存路径


# 预定义的相机配置
CAMERA_CONFIGS = {
    'wrist': RealSenseCameraConfig(
        name='wrist',
        serial_number='036222071712',  # D435i
        resolution=(640, 480),
        fps=30,
        enable_depth=False,
        enable_recording=False,
    ),
    'external': RealSenseCameraConfig(
        name='external', 
        serial_number='037522250003',  # D455
        resolution=(640, 480),
        fps=30,
        enable_depth=False,
        enable_recording=True,  # 外部相机用于录制
    ),
}


class CameraCommand(Enum):
    """相机控制命令"""
    STOP = 0
    START_RECORDING = 1
    STOP_RECORDING = 2


# ===================== 相机进程类 =====================

class RealSenseCameraProcess(mp.Process):
    """
    RealSense 相机多进程类
    
    在独立进程中运行相机采集，通过 SharedMemoryRingBuffer 传递帧数据
    """
    
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        config: RealSenseCameraConfig,
        get_max_k: int = 30,
        get_time_budget: float = 0.02,
        verbose: bool = False,
    ):
        """
        Args:
            shm_manager: 共享内存管理器
            config: 相机配置
            get_max_k: RingBuffer 最大查询帧数
            get_time_budget: 获取帧的时间预算
            verbose: 是否打印详细日志
        """
        super().__init__(name=f"RealSenseCamera_{config.name}")
        
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 未安装")
        
        self.config = config
        self.verbose = verbose
        
        # 创建帧数据的共享内存 RingBuffer
        w, h = config.resolution
        example = {
            'rgb': np.zeros((h, w, 3), dtype=np.uint8),  # RGB 图像
            'timestamp': 0.0,  # 系统时间戳
            'frame_number': 0,  # 帧号
        }
        
        if config.enable_depth:
            example['depth'] = np.zeros((h, w), dtype=np.uint16)
        
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=get_time_budget,
            put_desired_frequency=float(config.fps),
        )
        
        # 进程控制
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        
        # 录制控制 (使用 Value 进行进程间通信)
        self.recording_flag = mp.Value('b', False)  # bool
        self.recording_path_value = mp.Array('c', 256)  # 录制路径
        
    # ========= 启动/停止 =========
    
    def start(self, wait: bool = True):
        """启动相机进程"""
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait: bool = True):
        """停止相机进程"""
        self.stop_event.set()
        if wait:
            self.join()
        if self.verbose:
            print(f"[{self.config.name}] 相机进程已停止")
    
    def start_wait(self, timeout: float = 10.0):
        """等待相机就绪"""
        self.ready_event.wait(timeout)
        if not self.ready_event.is_set():
            raise TimeoutError(f"[{self.config.name}] 相机启动超时")
    
    @property
    def is_ready(self) -> bool:
        return self.ready_event.is_set()
    
    # ========= 上下文管理 =========
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    # ========= 数据获取 =========
    
    def get_frame(self, out=None) -> Dict[str, np.ndarray]:
        """获取最新帧"""
        return self.ring_buffer.get(out=out)
    
    def get_frames(self, k: int, out=None) -> Dict[str, np.ndarray]:
        """获取最近 k 帧"""
        return self.ring_buffer.get_last_k(k=k, out=out)
    
    @property
    def frame_count(self) -> int:
        """已采集的帧数"""
        return self.ring_buffer.count
    
    # ========= 录制控制 =========
    
    def start_recording(self, path: str):
        """开始录制视频"""
        if not self.config.enable_recording:
            print(f"[{self.config.name}] 此相机未启用录制功能")
            return
        
        # 设置录制路径
        path_bytes = path.encode('utf-8')[:255]
        self.recording_path_value[:len(path_bytes)] = path_bytes
        self.recording_path_value[len(path_bytes)] = b'\0'
        self.recording_flag.value = True
        
        if self.verbose:
            print(f"[{self.config.name}] 开始录制: {path}")
    
    def stop_recording(self):
        """停止录制视频"""
        self.recording_flag.value = False
        if self.verbose:
            print(f"[{self.config.name}] 停止录制")
    
    # ========= 主循环 =========
    
    def run(self):
        """相机采集主循环 (在子进程中运行)"""
        try:
            # 初始化 RealSense
            pipeline = rs.pipeline()
            rs_config = rs.config()
            
            # 通过序列号启用设备
            rs_config.enable_device(self.config.serial_number)
            
            # 配置 RGB 流
            w, h = self.config.resolution
            rs_config.enable_stream(
                rs.stream.color, w, h, rs.format.bgr8, self.config.fps
            )
            
            # 配置深度流（如果启用）
            if self.config.enable_depth:
                rs_config.enable_stream(
                    rs.stream.depth, w, h, rs.format.z16, self.config.fps
                )
            
            # 启动流
            profile = pipeline.start(rs_config)
            
            # 深度对齐
            align = rs.align(rs.stream.color) if self.config.enable_depth else None
            
            # 等待自动曝光稳定
            for _ in range(30):
                pipeline.wait_for_frames()
            
            if self.verbose:
                print(f"[{self.config.name}] 相机已启动 (SN: {self.config.serial_number})")
            
            # 视频录制器
            video_writer = None
            is_recording = False
            
            # 设置就绪标志
            self.ready_event.set()
            
            frame_count = 0
            dt = 1.0 / self.config.fps
            t_start = time.monotonic()
            
            while not self.stop_event.is_set():
                try:
                    # 获取帧
                    frames = pipeline.wait_for_frames(timeout_ms=100)
                    
                    # 深度对齐
                    if align is not None:
                        frames = align.process(frames)
                    
                    # 获取彩色帧
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    
                    # BGR -> RGB (复制数据，避免内部缓冲区被覆盖)
                    bgr = np.asanyarray(color_frame.get_data()).copy()
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    
                    # 准备数据
                    data = {
                        'rgb': rgb,
                        'timestamp': time.time(),
                        'frame_number': frame_count,
                    }
                    
                    # 获取深度帧
                    if self.config.enable_depth:
                        depth_frame = frames.get_depth_frame()
                        if depth_frame:
                            data['depth'] = np.asanyarray(depth_frame.get_data()).copy()
                        else:
                            data['depth'] = np.zeros((h, w), dtype=np.uint16)
                    
                    # 写入共享内存
                    self.ring_buffer.put(data)
                    
                    # 处理录制
                    if self.config.enable_recording:
                        should_record = self.recording_flag.value
                        
                        if should_record and not is_recording:
                            # 开始录制
                            path_bytes = bytes(self.recording_path_value[:]).split(b'\0')[0]
                            recording_path = path_bytes.decode('utf-8')
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(
                                recording_path, fourcc, self.config.fps, (w, h)
                            )
                            is_recording = True
                            if self.verbose:
                                print(f"[{self.config.name}] 录制开始: {recording_path}")
                        
                        elif not should_record and is_recording:
                            # 停止录制
                            if video_writer is not None:
                                video_writer.release()
                                video_writer = None
                            is_recording = False
                            if self.verbose:
                                print(f"[{self.config.name}] 录制停止")
                        
                        elif is_recording and video_writer is not None:
                            # 写入帧 (使用 BGR 格式)
                            video_writer.write(bgr)
                    
                    frame_count += 1
                    
                    # 维持帧率
                    t_target = t_start + frame_count * dt
                    t_now = time.monotonic()
                    if t_target > t_now:
                        time.sleep(t_target - t_now)
                        
                except RuntimeError as e:
                    if "timeout" in str(e).lower():
                        continue
                    raise
            
        except Exception as e:
            print(f"[{self.config.name}] 相机错误: {e}")
            raise
        finally:
            # 清理
            if video_writer is not None:
                video_writer.release()
            try:
                pipeline.stop()
            except:
                pass
            if self.verbose:
                print(f"[{self.config.name}] 相机进程结束")


# ===================== 相机管理器 =====================

class RealSenseCameraManager:
    """
    多 RealSense 相机管理器
    
    管理多个相机进程，提供统一的接口
    """
    
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        camera_configs: Optional[Dict[str, RealSenseCameraConfig]] = None,
        verbose: bool = False,
    ):
        """
        Args:
            shm_manager: 共享内存管理器
            camera_configs: 相机配置字典，键为相机名称
            verbose: 是否打印详细日志
        """
        self.shm_manager = shm_manager
        self.camera_configs = camera_configs or CAMERA_CONFIGS
        self.verbose = verbose
        
        self.cameras: Dict[str, RealSenseCameraProcess] = {}
        self._started = False
    
    @staticmethod
    def detect_cameras() -> List[Tuple[str, str]]:
        """
        检测所有连接的 RealSense 相机
        
        Returns:
            (序列号, 产品名称) 元组列表
        """
        if not REALSENSE_AVAILABLE:
            return []
        
        ctx = rs.context()
        devices = ctx.query_devices()
        
        result = []
        for dev in devices:
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            result.append((serial, name))
        
        return result
    
    def initialize(self, camera_names: Optional[List[str]] = None) -> bool:
        """
        初始化相机
        
        Args:
            camera_names: 要初始化的相机名称列表，None 表示所有配置的相机
        
        Returns:
            是否成功初始化所有相机
        """
        if not REALSENSE_AVAILABLE:
            print("[CameraManager] pyrealsense2 未安装，跳过相机初始化")
            return False
        
        # 检测连接的相机
        detected = self.detect_cameras()
        detected_serials = {serial for serial, _ in detected}
        
        if self.verbose:
            print(f"[CameraManager] 检测到 {len(detected)} 个相机:")
            for serial, name in detected:
                print(f"  - {name} (SN: {serial})")
        
        # 确定要初始化的相机
        if camera_names is None:
            camera_names = list(self.camera_configs.keys())
        
        # 创建相机进程
        success = True
        for name in camera_names:
            if name not in self.camera_configs:
                print(f"[CameraManager] 未知相机配置: {name}")
                success = False
                continue
            
            config = self.camera_configs[name]
            
            # 检查相机是否连接
            if config.serial_number not in detected_serials:
                print(f"[CameraManager] 相机 {name} (SN: {config.serial_number}) 未连接")
                success = False
                continue
            
            # 创建相机进程
            camera = RealSenseCameraProcess(
                shm_manager=self.shm_manager,
                config=config,
                verbose=self.verbose,
            )
            self.cameras[name] = camera
        
        return success
    
    def start(self, wait: bool = True):
        """启动所有相机"""
        for name, camera in self.cameras.items():
            camera.start(wait=False)
            if self.verbose:
                print(f"[CameraManager] 启动相机 {name}")
        
        if wait:
            for name, camera in self.cameras.items():
                camera.start_wait()
        
        self._started = True
    
    def stop(self, wait: bool = True):
        """停止所有相机"""
        for camera in self.cameras.values():
            camera.stop(wait=False)
        
        if wait:
            for camera in self.cameras.values():
                camera.join()
        
        self._started = False
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def get_camera(self, name: str) -> Optional[RealSenseCameraProcess]:
        """获取指定相机"""
        return self.cameras.get(name)
    
    def get_frame(self, name: str, out=None) -> Optional[Dict[str, np.ndarray]]:
        """获取指定相机的最新帧"""
        camera = self.cameras.get(name)
        if camera is None:
            return None
        return camera.get_frame(out=out)
    
    def get_frames(self, name: str, k: int, out=None) -> Optional[Dict[str, np.ndarray]]:
        """获取指定相机的最近 k 帧"""
        camera = self.cameras.get(name)
        if camera is None:
            return None
        return camera.get_frames(k=k, out=out)
    
    def start_recording(self, name: str, path: str):
        """开始录制指定相机"""
        camera = self.cameras.get(name)
        if camera is not None:
            camera.start_recording(path)
    
    def stop_recording(self, name: str):
        """停止录制指定相机"""
        camera = self.cameras.get(name)
        if camera is not None:
            camera.stop_recording()


# ===================== 便捷函数 =====================

def create_wrist_camera(shm_manager: SharedMemoryManager, verbose: bool = False) -> RealSenseCameraProcess:
    """创建腕部相机 (D435i)"""
    return RealSenseCameraProcess(
        shm_manager=shm_manager,
        config=CAMERA_CONFIGS['wrist'],
        verbose=verbose,
    )


def create_external_camera(
    shm_manager: SharedMemoryManager, 
    recording_path: Optional[str] = None,
    verbose: bool = False
) -> RealSenseCameraProcess:
    """创建外部相机 (D455)"""
    config = RealSenseCameraConfig(
        name='external',
        serial_number='037522250003',
        resolution=(640, 480),
        fps=30,
        enable_depth=False,
        enable_recording=True,
        recording_path=recording_path,
    )
    return RealSenseCameraProcess(
        shm_manager=shm_manager,
        config=config,
        verbose=verbose,
    )
