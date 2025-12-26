#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense Camera Manager (独立版本，不依赖 ROS)

支持:
- 自动检测连接的 RealSense 相机
- 多相机同步采集
- RGB 流 30Hz
- 线程安全的帧访问
"""

import threading
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import cv2

import numpy as np

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[CameraManager] pyrealsense2 未安装，相机功能不可用")


@dataclass
class CameraConfig:
    """相机配置"""
    name: str
    serial_number: Optional[str] = None  # 如果为 None，将自动分配
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    enable_depth: bool = False
    depth_resolution: Optional[Tuple[int, int]] = None


@dataclass 
class CameraFrame:
    """单帧数据容器"""
    rgb: np.ndarray              # RGB 图像 [H, W, 3] uint8
    depth: Optional[np.ndarray]  # 深度图像 [H, W] uint16 (mm) 或 None
    timestamp: float             # 系统时间戳
    frame_number: int            # 帧计数器


class RealSenseCamera:
    """单个 RealSense 相机封装"""
    
    def __init__(self, config: CameraConfig, serial_number: str):
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 未安装")
        
        self.config = config
        self.serial_number = serial_number
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        
        # 通过序列号启用设备
        self.rs_config.enable_device(serial_number)
        
        # 配置 RGB 流
        w, h = config.resolution
        self.rs_config.enable_stream(
            rs.stream.color, w, h, rs.format.bgr8, config.fps
        )
        
        # 配置深度流（如果启用）
        if config.enable_depth:
            dw, dh = config.depth_resolution or config.resolution
            self.rs_config.enable_stream(
                rs.stream.depth, dw, dh, rs.format.z16, config.fps
            )
        
        self.align = None
        self.profile = None
        self.frame_count = 0
        self._started = False
    
    def start(self):
        """启动相机流"""
        if self._started:
            return
        
        self.profile = self.pipeline.start(self.rs_config)
        
        # 创建对齐对象（将深度对齐到彩色）
        if self.config.enable_depth:
            self.align = rs.align(rs.stream.color)
        
        # 等待自动曝光稳定
        for _ in range(30):
            self.pipeline.wait_for_frames()
        
        self._started = True
        print(f"[Camera {self.config.name}] 已启动 (SN: {self.serial_number})")
    
    def stop(self):
        """停止相机流"""
        if self._started:
            self.pipeline.stop()
            self._started = False
            print(f"[Camera {self.config.name}] 已停止")
    
    def get_frame(self) -> Optional[CameraFrame]:
        """获取当前帧（非阻塞，带超时）"""
        if not self._started:
            return None
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
        except RuntimeError:
            return None
        
        # 如果启用了深度，将深度对齐到彩色
        if self.align is not None:
            frames = self.align.process(frames)
        
        # 获取彩色帧
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        
        # 重要：必须复制！np.asanyarray 返回的是 RealSense 内部缓冲区的视图
        # RealSense 返回的是 BGR 格式 (rs.format.bgr8)，需要转换为 RGB
        bgr = np.asanyarray(color_frame.get_data()).copy()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # 获取深度帧（如果启用）
        depth = None
        if self.config.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth = np.asanyarray(depth_frame.get_data()).copy()
        
        self.frame_count += 1
        
        return CameraFrame(
            rgb=rgb,
            depth=depth,
            timestamp=time.time(),
            frame_number=self.frame_count
        )


class CameraManager:
    """多 RealSense 相机管理器"""
    
    def __init__(self, camera_configs: Dict[str, CameraConfig]):
        """
        初始化相机管理器
        
        Args:
            camera_configs: 相机名称到配置的映射字典
        """
        self.configs = camera_configs
        self.cameras: Dict[str, RealSenseCamera] = {}
        self.latest_frames: Dict[str, Optional[CameraFrame]] = {}
        self.lock = threading.Lock()
        
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
    
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
    
    def initialize(self, auto_assign: bool = True) -> bool:
        """
        初始化相机
        
        Args:
            auto_assign: 如果为 True，自动将检测到的相机分配给配置槽位
        
        Returns:
            如果所有相机初始化成功返回 True
        """
        if not REALSENSE_AVAILABLE:
            print("[CameraManager] pyrealsense2 未安装，跳过相机初始化")
            return False
        
        detected = self.detect_cameras()
        print(f"[CameraManager] 检测到 {len(detected)} 个 RealSense 相机:")
        for serial, name in detected:
            print(f"  - {name} (SN: {serial})")
        
        if len(detected) < len(self.configs):
            print(f"[CameraManager] 警告: 期望 {len(self.configs)} 个相机，找到 {len(detected)} 个")
        
        # 构建序列号映射
        serial_map: Dict[str, str] = {}
        
        if auto_assign:
            config_names = list(self.configs.keys())
            for i, (serial, _) in enumerate(detected):
                if i < len(config_names):
                    config_name = config_names[i]
                    if self.configs[config_name].serial_number:
                        serial_map[config_name] = self.configs[config_name].serial_number
                    else:
                        serial_map[config_name] = serial
                        print(f"[CameraManager] 自动分配相机 {serial} 到 '{config_name}'")
        else:
            for name, config in self.configs.items():
                if config.serial_number:
                    serial_map[name] = config.serial_number
        
        # 初始化相机
        for name, config in self.configs.items():
            if name not in serial_map:
                print(f"[CameraManager] 跳过 '{name}': 未分配序列号")
                continue
            
            serial = serial_map[name]
            try:
                camera = RealSenseCamera(config, serial)
                self.cameras[name] = camera
                self.latest_frames[name] = None
            except Exception as e:
                print(f"[CameraManager] 初始化 '{name}' 失败: {e}")
                return False
        
        return len(self.cameras) > 0
    
    def start(self):
        """启动所有相机和采集线程"""
        for camera in self.cameras.values():
            camera.start()
        
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        print("[CameraManager] 采集线程已启动")
    
    def stop(self):
        """停止采集线程和所有相机"""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        
        for camera in self.cameras.values():
            camera.stop()
        print("[CameraManager] 已停止")
    
    def _capture_loop(self):
        """后台采集循环"""
        while self._running:
            frames = {}
            for name, camera in self.cameras.items():
                frame = camera.get_frame()
                if frame is not None:
                    frames[name] = frame
            
            with self.lock:
                self.latest_frames.update(frames)
            
            time.sleep(0.001)
    
    def get_frames(self) -> Dict[str, Optional[CameraFrame]]:
        """获取所有相机的最新帧（线程安全）"""
        with self.lock:
            return dict(self.latest_frames)
    
    def get_frame(self, camera_name: str) -> Optional[CameraFrame]:
        """获取指定相机的最新帧"""
        with self.lock:
            return self.latest_frames.get(camera_name)


# 默认相机配置（根据实际硬件调整）
DEFAULT_CAMERA_CONFIGS = {
    "wrist": CameraConfig(
        name="wrist",
        serial_number="036222071712",  # D435i
        resolution=(640, 480),
        fps=30,
        enable_depth=False
    ),
    "external": CameraConfig(
        name="external", 
        serial_number="037522250003",  # D455
        resolution=(640, 480),
        fps=30,
        enable_depth=False
    )
}


if __name__ == "__main__":
    # 测试相机管理器
    print("检测相机...")
    cameras = CameraManager.detect_cameras()
    print(f"找到 {len(cameras)} 个相机")
    
    if cameras:
        manager = CameraManager(DEFAULT_CAMERA_CONFIGS)
        if manager.initialize(auto_assign=True):
            manager.start()
            
            print("\n采集 3 秒...")
            time.sleep(3)
            
            frames = manager.get_frames()
            for name, frame in frames.items():
                if frame:
                    print(f"  {name}: shape={frame.rgb.shape}, frame={frame.frame_number}")
            
            manager.stop()
