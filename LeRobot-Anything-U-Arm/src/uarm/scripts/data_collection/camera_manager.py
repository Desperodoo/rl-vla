#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Manager for RealSense RGB-D Cameras

Supports:
- Auto-detection of connected RealSense cameras
- Synchronized capture from multiple cameras
- RGB and Depth streaming at 30Hz
- Thread-safe frame access
"""

import threading
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pyrealsense2 as rs

# 支持多种导入方式
try:
    from .dataset_config import CameraConfig
except ImportError:
    try:
        from dataset_config import CameraConfig
    except ImportError:
        from data_collection.dataset_config import CameraConfig


@dataclass
class CameraFrame:
    """Container for a single camera frame"""
    rgb: np.ndarray              # RGB image [H, W, 3] uint8
    depth: Optional[np.ndarray]  # Depth image [H, W] uint16 (mm) or None
    timestamp: float             # System timestamp
    frame_number: int            # Frame counter


class RealSenseCamera:
    """Single RealSense camera wrapper"""
    
    def __init__(self, config: CameraConfig, serial_number: str):
        self.config = config
        self.serial_number = serial_number
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        
        # Enable device by serial number
        self.rs_config.enable_device(serial_number)
        
        # Configure RGB stream
        w, h = config.resolution
        self.rs_config.enable_stream(
            rs.stream.color, w, h, rs.format.bgr8, config.fps
        )
        
        # Configure depth stream if enabled
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
        """Start camera streaming"""
        if self._started:
            return
        
        self.profile = self.pipeline.start(self.rs_config)
        
        # Create alignment object (align depth to color)
        if self.config.enable_depth:
            self.align = rs.align(rs.stream.color)
        
        # Wait for auto-exposure to stabilize
        for _ in range(30):
            self.pipeline.wait_for_frames()
        
        self._started = True
        print(f"[Camera {self.config.name}] Started (SN: {self.serial_number})")
    
    def stop(self):
        """Stop camera streaming"""
        if self._started:
            self.pipeline.stop()
            self._started = False
            print(f"[Camera {self.config.name}] Stopped")
    
    def get_frame(self) -> Optional[CameraFrame]:
        """Get current frame (non-blocking with timeout)"""
        if not self._started:
            return None
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
        except RuntimeError:
            return None
        
        # Align depth to color if depth is enabled
        if self.align is not None:
            frames = self.align.process(frames)
        
        # Get color frame
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        
        # IMPORTANT: Make a copy! np.asanyarray returns a view into RealSense's
        # internal buffer, which gets overwritten when the next frame arrives.
        rgb = np.asanyarray(color_frame.get_data()).copy()
        
        # Get depth frame if enabled
        depth = None
        if self.config.enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                # Also copy depth data
                depth = np.asanyarray(depth_frame.get_data()).copy()
        
        self.frame_count += 1
        
        return CameraFrame(
            rgb=rgb,
            depth=depth,
            timestamp=time.time(),
            frame_number=self.frame_count
        )
    
    def get_intrinsics(self) -> Optional[dict]:
        """Get camera intrinsic parameters"""
        if not self._started or self.profile is None:
            return None
        
        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        result = {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.ppx,
            "cy": intrinsics.ppy,
            "distortion_model": str(intrinsics.model),
            "distortion_coeffs": list(intrinsics.coeffs),
        }
        
        if self.config.enable_depth:
            depth_stream = self.profile.get_stream(rs.stream.depth)
            depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            depth_sensor = self.profile.get_device().first_depth_sensor()
            
            result["depth_scale"] = depth_sensor.get_depth_scale()
            result["depth_intrinsics"] = {
                "width": depth_intrinsics.width,
                "height": depth_intrinsics.height,
                "fx": depth_intrinsics.fx,
                "fy": depth_intrinsics.fy,
                "cx": depth_intrinsics.ppx,
                "cy": depth_intrinsics.ppy,
            }
        
        return result


class CameraManager:
    """
    Manager for multiple RealSense cameras with synchronized capture
    """
    
    def __init__(self, camera_configs: Dict[str, CameraConfig]):
        """
        Initialize camera manager
        
        Args:
            camera_configs: Dictionary mapping camera names to their configs
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
        Detect all connected RealSense cameras
        
        Returns:
            List of (serial_number, product_name) tuples
        """
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
        Initialize cameras
        
        Args:
            auto_assign: If True, automatically assign detected cameras to config slots
                        based on detection order (first = wrist, second = external)
        
        Returns:
            True if all cameras initialized successfully
        """
        detected = self.detect_cameras()
        print(f"[CameraManager] Detected {len(detected)} RealSense camera(s):")
        for serial, name in detected:
            print(f"  - {name} (SN: {serial})")
        
        if len(detected) < len(self.configs):
            print(f"[CameraManager] Warning: Expected {len(self.configs)} cameras, found {len(detected)}")
        
        # Build serial number mapping
        serial_map: Dict[str, str] = {}  # config_name -> serial_number
        
        if auto_assign:
            # Auto-assign based on config order and detection order
            config_names = list(self.configs.keys())
            for i, (serial, _) in enumerate(detected):
                if i < len(config_names):
                    config_name = config_names[i]
                    # Check if config has explicit serial number
                    if self.configs[config_name].serial_number:
                        serial_map[config_name] = self.configs[config_name].serial_number
                    else:
                        serial_map[config_name] = serial
                        print(f"[CameraManager] Auto-assigned camera {serial} to '{config_name}'")
        else:
            # Use explicit serial numbers from config
            for name, config in self.configs.items():
                if config.serial_number:
                    serial_map[name] = config.serial_number
                else:
                    print(f"[CameraManager] Warning: No serial number for '{name}', skipping")
        
        # Initialize cameras
        for name, config in self.configs.items():
            if name not in serial_map:
                print(f"[CameraManager] Skipping '{name}': no serial number assigned")
                continue
            
            serial = serial_map[name]
            try:
                camera = RealSenseCamera(config, serial)
                self.cameras[name] = camera
                self.latest_frames[name] = None
            except Exception as e:
                print(f"[CameraManager] Failed to initialize '{name}': {e}")
                return False
        
        return len(self.cameras) > 0
    
    def start(self):
        """Start all cameras and capture thread"""
        for camera in self.cameras.values():
            camera.start()
        
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        print("[CameraManager] Capture thread started")
    
    def stop(self):
        """Stop capture thread and all cameras"""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        
        for camera in self.cameras.values():
            camera.stop()
        print("[CameraManager] Stopped")
    
    def _capture_loop(self):
        """Background capture loop"""
        while self._running:
            frames = {}
            for name, camera in self.cameras.items():
                frame = camera.get_frame()
                if frame is not None:
                    frames[name] = frame
            
            with self.lock:
                self.latest_frames.update(frames)
            
            # Small sleep to prevent busy-waiting
            time.sleep(0.001)
    
    def get_frames(self) -> Dict[str, Optional[CameraFrame]]:
        """
        Get latest frames from all cameras (thread-safe)
        
        Returns:
            Dictionary mapping camera names to their latest frames (copies)
        """
        with self.lock:
            # Return copies to avoid race conditions with capture thread
            result = {}
            for name, frame in self.latest_frames.items():
                if frame is not None:
                    # Deep copy the frame data
                    result[name] = CameraFrame(
                        rgb=frame.rgb.copy(),
                        depth=frame.depth.copy() if frame.depth is not None else None,
                        timestamp=frame.timestamp,
                        frame_number=frame.frame_number
                    )
                else:
                    result[name] = None
            return result
    
    def get_frame(self, camera_name: str) -> Optional[CameraFrame]:
        """Get latest frame from a specific camera"""
        with self.lock:
            return self.latest_frames.get(camera_name)
    
    def get_intrinsics(self) -> Dict[str, dict]:
        """Get intrinsics for all cameras"""
        result = {}
        for name, camera in self.cameras.items():
            intrinsics = camera.get_intrinsics()
            if intrinsics:
                result[name] = intrinsics
        return result
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def list_cameras():
    """Utility function to list all connected RealSense cameras"""
    print("\n=== Connected RealSense Cameras ===")
    cameras = CameraManager.detect_cameras()
    
    if not cameras:
        print("No RealSense cameras detected!")
        print("\nTroubleshooting:")
        print("1. Check USB connections")
        print("2. Try different USB ports (USB 3.0 recommended)")
        print("3. Run: rs-enumerate-devices")
        return
    
    for i, (serial, name) in enumerate(cameras, 1):
        print(f"\n[Camera {i}]")
        print(f"  Name: {name}")
        print(f"  Serial: {serial}")
    
    print("\n" + "="*40)
    print(f"Total: {len(cameras)} camera(s)")
    print("\nTo use these cameras, add the serial numbers to your config:")
    print("  wrist camera:    serial_number = '<first serial>'")
    print("  external camera: serial_number = '<second serial>'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RealSense Camera Manager")
    parser.add_argument("--list", action="store_true", help="List connected cameras")
    parser.add_argument("--test", action="store_true", help="Test camera capture")
    args = parser.parse_args()
    
    if args.list:
        list_cameras()
    elif args.test:
        import cv2
        from .dataset_config import DEFAULT_CONFIG
        
        manager = CameraManager(DEFAULT_CONFIG.cameras)
        if not manager.initialize(auto_assign=True):
            print("Failed to initialize cameras")
            exit(1)
        
        print("\nTesting camera capture (press 'q' to quit)...")
        with manager:
            while True:
                frames = manager.get_frames()
                
                for name, frame in frames.items():
                    if frame is not None:
                        cv2.imshow(f"{name} - RGB", frame.rgb)
                        if frame.depth is not None:
                            # Normalize depth for visualization
                            depth_vis = cv2.normalize(
                                frame.depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                            )
                            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                            cv2.imshow(f"{name} - Depth", depth_color)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cv2.destroyAllWindows()
    else:
        list_cameras()
