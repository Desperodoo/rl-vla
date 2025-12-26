#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Configuration for ARX5 Teleoperation Data Collection

This module defines the configuration for data collection, including:
- Camera settings (resolution, frame rate, serial numbers)
- Robot state dimensions
- Data storage paths and formats
- Compatible with ManiSkill dataset structure
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import yaml


@dataclass
class CameraConfig:
    """Configuration for a single RealSense camera"""
    name: str                          # Camera identifier (e.g., "wrist", "external")
    serial_number: Optional[str]       # Camera serial number (None for auto-detect)
    resolution: Tuple[int, int]        # (width, height)
    fps: int                           # Frame rate
    enable_depth: bool                 # Whether to capture depth
    depth_resolution: Optional[Tuple[int, int]] = None  # Depth resolution (can differ from RGB)
    
    def __post_init__(self):
        if self.enable_depth and self.depth_resolution is None:
            self.depth_resolution = self.resolution


@dataclass
class RobotConfig:
    """Configuration for ARX5 robot"""
    model: str = "X5"
    interface: str = "can0"
    joint_dof: int = 6
    gripper_range: Tuple[float, float] = (0.0, 0.08)  # meters
    
    # State dimensions
    joint_pos_dim: int = 6
    joint_vel_dim: int = 6
    gripper_dim: int = 1
    ee_pose_dim: int = 7  # xyz + quaternion (xyzw)
    
    @property
    def state_dim(self) -> int:
        """Total state dimension: joint_pos + joint_vel + gripper"""
        return self.joint_pos_dim + self.joint_vel_dim + self.gripper_dim
    
    @property
    def action_dim(self) -> int:
        """Action dimension: target joint_pos + gripper"""
        return self.joint_pos_dim + self.gripper_dim


@dataclass
class DatasetConfig:
    """Main configuration for data collection"""
    
    # Task information
    task_name: str = "arx5_teleop"
    task_description: str = "ARX5 teleoperation data collection"
    
    # Data paths
    raw_data_dir: str = "~/.arx_demos/raw"
    processed_data_dir: str = "~/.arx_demos/processed"
    
    # Collection settings
    control_freq: int = 30  # Hz - collection frequency
    max_episode_steps: int = 1000  # Maximum steps per episode
    
    # Camera configurations
    cameras: Dict[str, CameraConfig] = field(default_factory=lambda: {
        "wrist": CameraConfig(
            name="wrist",
            serial_number=None,  # Will be auto-detected
            resolution=(640, 480),
            fps=30,
            enable_depth=True,
        ),
        "external": CameraConfig(
            name="external", 
            serial_number=None,  # Will be auto-detected
            resolution=(640, 480),
            fps=30,
            enable_depth=True,
        ),
    })
    
    # Robot configuration
    robot: RobotConfig = field(default_factory=RobotConfig)
    
    # Image processing for raw storage
    image_format: str = "png"  # png for RGB, 16-bit png for depth
    depth_scale: float = 1000.0  # Convert meters to millimeters for 16-bit storage
    
    # HDF5 compression
    hdf5_compression: str = "gzip"
    hdf5_compression_level: int = 4
    
    def __post_init__(self):
        # Expand paths
        self.raw_data_dir = os.path.expanduser(self.raw_data_dir)
        self.processed_data_dir = os.path.expanduser(self.processed_data_dir)
    
    def get_episode_dir(self, timestamp: str) -> str:
        """Get directory path for a specific episode"""
        return os.path.join(self.raw_data_dir, self.task_name, timestamp)
    
    def get_processed_path(self) -> str:
        """Get path for processed dataset"""
        return os.path.join(self.processed_data_dir, self.task_name)
    
    def save(self, path: str):
        """Save configuration to YAML file"""
        config_dict = {
            "task_name": self.task_name,
            "task_description": self.task_description,
            "raw_data_dir": self.raw_data_dir,
            "processed_data_dir": self.processed_data_dir,
            "control_freq": self.control_freq,
            "max_episode_steps": self.max_episode_steps,
            "cameras": {
                name: {
                    "name": cam.name,
                    "serial_number": cam.serial_number,
                    "resolution": list(cam.resolution),
                    "fps": cam.fps,
                    "enable_depth": cam.enable_depth,
                    "depth_resolution": list(cam.depth_resolution) if cam.depth_resolution else None,
                }
                for name, cam in self.cameras.items()
            },
            "robot": {
                "model": self.robot.model,
                "interface": self.robot.interface,
                "joint_dof": self.robot.joint_dof,
                "gripper_range": list(self.robot.gripper_range),
            },
            "image_format": self.image_format,
            "depth_scale": self.depth_scale,
            "hdf5_compression": self.hdf5_compression,
            "hdf5_compression_level": self.hdf5_compression_level,
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> 'DatasetConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        cameras = {}
        for name, cam_dict in config_dict.get("cameras", {}).items():
            cameras[name] = CameraConfig(
                name=cam_dict["name"],
                serial_number=cam_dict.get("serial_number"),
                resolution=tuple(cam_dict["resolution"]),
                fps=cam_dict["fps"],
                enable_depth=cam_dict["enable_depth"],
                depth_resolution=tuple(cam_dict["depth_resolution"]) if cam_dict.get("depth_resolution") else None,
            )
        
        robot_dict = config_dict.get("robot", {})
        robot = RobotConfig(
            model=robot_dict.get("model", "X5"),
            interface=robot_dict.get("interface", "can0"),
            joint_dof=robot_dict.get("joint_dof", 6),
            gripper_range=tuple(robot_dict.get("gripper_range", [0.0, 0.08])),
        )
        
        return cls(
            task_name=config_dict.get("task_name", "arx5_teleop"),
            task_description=config_dict.get("task_description", ""),
            raw_data_dir=config_dict.get("raw_data_dir", "~/.arx_demos/raw"),
            processed_data_dir=config_dict.get("processed_data_dir", "~/.arx_demos/processed"),
            control_freq=config_dict.get("control_freq", 30),
            max_episode_steps=config_dict.get("max_episode_steps", 1000),
            cameras=cameras,
            robot=robot,
            image_format=config_dict.get("image_format", "png"),
            depth_scale=config_dict.get("depth_scale", 1000.0),
            hdf5_compression=config_dict.get("hdf5_compression", "gzip"),
            hdf5_compression_level=config_dict.get("hdf5_compression_level", 4),
        )


# Default configuration instance
DEFAULT_CONFIG = DatasetConfig()


if __name__ == "__main__":
    # Test configuration
    config = DatasetConfig(task_name="pick_cube")
    print(f"Task: {config.task_name}")
    print(f"Control frequency: {config.control_freq} Hz")
    print(f"Robot state dim: {config.robot.state_dim}")
    print(f"Robot action dim: {config.robot.action_dim}")
    print(f"Cameras: {list(config.cameras.keys())}")
    
    # Save and reload test
    test_path = "/tmp/test_config.yaml"
    config.save(test_path)
    print(f"\nSaved config to {test_path}")
    
    loaded = DatasetConfig.load(test_path)
    print(f"Loaded task: {loaded.task_name}")
