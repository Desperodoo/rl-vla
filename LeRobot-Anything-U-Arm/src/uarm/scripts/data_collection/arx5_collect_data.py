#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARX5 Teleoperation Data Collection Script

This is the main entry point for collecting teleoperation data with:
- ARX5 robot arm (6-DOF + gripper)
- Dual RealSense RGB-D cameras (wrist D435i + external D455)
- ManiSkill-compatible data format

Usage:
    python arx5_collect_data.py --task pick_cube
    python arx5_collect_data.py --task pick_cube --list-cameras
    python arx5_collect_data.py --task pick_cube --config my_config.yaml

Controls:
    Space     - Start/Pause recording
    Enter     - Save episode (mark as success)
    F         - Save episode (mark as failed)
    Backspace - Discard current episode
    Q         - Quit
"""

import argparse
import os
import sys

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from data_collection.dataset_config import DatasetConfig, CameraConfig
from data_collection.camera_manager import CameraManager, list_cameras
from data_collection.data_recorder import TeleopDataCollector, DataRecorder


def parse_args():
    parser = argparse.ArgumentParser(
        description="ARX5 Teleoperation Data Collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - start collecting data for "pick_cube" task
    python arx5_collect_data.py --task pick_cube
    
    # List connected cameras first
    python arx5_collect_data.py --list-cameras
    
    # Use custom config file
    python arx5_collect_data.py --task pick_cube --config my_config.yaml
    
    # Specify camera serial numbers manually
    python arx5_collect_data.py --task pick_cube \\
        --wrist-camera 123456789 \\
        --external-camera 987654321
        
    # Camera-only mode (no robot) for testing
    python arx5_collect_data.py --task test --camera-only
"""
    )
    
    # Task settings
    parser.add_argument("--task", type=str, default="arx5_teleop",
                       help="Task name for this collection session")
    parser.add_argument("--description", type=str, default="",
                       help="Task description")
    
    # Config file
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config YAML file (overrides other settings)")
    parser.add_argument("--save-config", type=str, default=None,
                       help="Save current config to YAML file and exit")
    
    # Camera settings
    parser.add_argument("--list-cameras", action="store_true",
                       help="List connected cameras and exit")
    parser.add_argument("--wrist-camera", type=str, default=None,
                       help="Serial number for wrist camera (D435i)")
    parser.add_argument("--external-camera", type=str, default=None,
                       help="Serial number for external camera (D455)")
    parser.add_argument("--no-depth", action="store_true",
                       help="Disable depth capture (RGB only)")
    parser.add_argument("--resolution", type=str, default="640x480",
                       help="Camera resolution (WxH), default: 640x480")
    
    # Robot settings
    parser.add_argument("--can", type=str, default="can0",
                       help="CAN interface for ARX5 (default: can0)")
    parser.add_argument("--servo-port", type=str, default=None,
                       help="Serial port for master arm servo controller")
    parser.add_argument("--camera-only", action="store_true",
                       help="Run without robot (camera testing mode)")
    
    # Collection settings
    parser.add_argument("--fps", type=int, default=30,
                       help="Collection frequency in Hz (default: 30)")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum steps per episode (default: 1000)")
    
    # Output settings
    parser.add_argument("--output-dir", type=str, default="~/.arx_demos/raw",
                       help="Output directory for raw data")
    
    # Display settings
    parser.add_argument("--headless", action="store_true",
                       help="Run without GUI (for remote/no-display environments)")
    
    return parser.parse_args()


def create_config_from_args(args) -> DatasetConfig:
    """Create DatasetConfig from command line arguments"""
    
    # Parse resolution
    try:
        w, h = map(int, args.resolution.lower().split('x'))
        resolution = (w, h)
    except:
        print(f"Warning: Invalid resolution '{args.resolution}', using 640x480")
        resolution = (640, 480)
    
    # Create camera configs
    cameras = {
        "wrist": CameraConfig(
            name="wrist",
            serial_number=args.wrist_camera,
            resolution=resolution,
            fps=args.fps,
            enable_depth=not args.no_depth,
        ),
        "external": CameraConfig(
            name="external",
            serial_number=args.external_camera,
            resolution=resolution,
            fps=args.fps,
            enable_depth=not args.no_depth,
        ),
    }
    
    config = DatasetConfig(
        task_name=args.task,
        task_description=args.description,
        raw_data_dir=args.output_dir,
        control_freq=args.fps,
        max_episode_steps=args.max_steps,
        cameras=cameras,
    )
    
    return config


def main():
    args = parse_args()
    
    # List cameras and exit if requested
    if args.list_cameras:
        list_cameras()
        return
    
    # Load or create config
    if args.config:
        print(f"Loading config from {args.config}")
        config = DatasetConfig.load(args.config)
    else:
        config = create_config_from_args(args)
    
    # Save config and exit if requested
    if args.save_config:
        config.save(args.save_config)
        print(f"Config saved to {args.save_config}")
        return
    
    # Print configuration summary
    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)
    print(f"  Task:        {config.task_name}")
    print(f"  Output:      {config.raw_data_dir}")
    print(f"  Frequency:   {config.control_freq} Hz")
    print(f"  Max steps:   {config.max_episode_steps}")
    print(f"  Cameras:     {list(config.cameras.keys())}")
    print(f"  Depth:       {'Enabled' if not args.no_depth else 'Disabled'}")
    print(f"  Robot:       {'Disabled (camera-only)' if args.camera_only else f'ARX5 via {args.can}'}")
    print("="*60 + "\n")
    
    # Create and run collector
    if args.camera_only:
        # Camera-only testing mode
        run_camera_only_test(config, headless=args.headless)
    else:
        # Full teleoperation mode
        collector = TeleopDataCollector(
            config=config,
            arx_interface=args.can,
            servo_port=args.servo_port,
            headless=args.headless
        )
        
        if collector.setup():
            collector.run(task_name=args.task)


def run_camera_only_test(config: DatasetConfig, headless: bool = False):
    """Run camera-only mode for testing without robot"""
    import cv2
    import time
    import numpy as np
    
    print("\n" + "="*60)
    print("Camera-Only Testing Mode" + (" (Headless)" if headless else ""))
    print("="*60)
    print("Press 'Q' to quit" if not headless else "Press Ctrl+C to quit")
    print("="*60 + "\n")
    
    # Initialize cameras
    camera_manager = CameraManager(config.cameras)
    if not camera_manager.initialize(auto_assign=True):
        print("No cameras available!")
        return
    
    # Initialize recorder for testing
    recorder = DataRecorder(config)
    recorder.start_session(config.task_name + "_camera_test")
    
    # Create mock robot state for testing
    from data_collection.data_recorder import RobotState
    
    def get_mock_robot_state():
        return RobotState(
            joint_pos=np.zeros(6, dtype=np.float32),
            joint_vel=np.zeros(6, dtype=np.float32),
            gripper_pos=0.04,
            timestamp=time.time()
        )
    
    cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
    
    try:
        camera_manager.start()
        dt = 1.0 / config.control_freq
        
        print("\nControls:")
        print("  [Space]     - Start/Stop recording")
        print("  [Enter]     - Save episode")
        print("  [Backspace] - Discard episode")
        print("  [Q]         - Quit\n")
        
        while True:
            loop_start = time.time()
            
            frames = camera_manager.get_frames()
            
            # Record if active
            if recorder.is_recording:
                recorder.record_step(
                    get_mock_robot_state(),
                    frames,
                    np.zeros(7, dtype=np.float32)
                )
            
            # Create preview
            preview_images = []
            for name, frame in frames.items():
                if frame is not None:
                    img = frame.rgb.copy()
                    
                    # Recording indicator
                    if recorder.is_recording:
                        cv2.circle(img, (30, 30), 15, (0, 0, 255), -1)
                        cv2.putText(img, "REC", (50, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    cv2.putText(img, name, (10, img.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Show depth as inset if available
                    if frame.depth is not None:
                        depth_vis = cv2.normalize(
                            frame.depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                        )
                        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                        depth_small = cv2.resize(depth_color, (160, 120))
                        img[10:130, img.shape[1]-170:img.shape[1]-10] = depth_small
                    
                    preview_images.append(img)
            
            if preview_images:
                combined = np.hstack(preview_images) if len(preview_images) > 1 else preview_images[0]
                
                stats = recorder.get_stats()
                status_text = (f"Episode: {stats['current_episode_id']} | "
                              f"Steps: {stats['current_episode_steps']} | "
                              f"Saved: {stats['total_episodes_saved']}")
                cv2.putText(combined, status_text, (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("Camera Test", combined)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                if not recorder.is_recording:
                    recorder.start_episode()
                else:
                    recorder.stop_episode()
            elif key == 13:  # Enter
                if recorder.current_episode:
                    recorder.stop_episode()
                    recorder.save_episode(success=True)
            elif key == 8:  # Backspace
                if recorder.current_episode:
                    recorder.stop_episode()
                    recorder.save_episode(discard=True)
            elif key == ord('q') or key == ord('Q'):
                break
            
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    finally:
        camera_manager.stop()
        cv2.destroyAllWindows()
        
        stats = recorder.get_stats()
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        print(f"  Session: {stats['session_dir']}")
        print(f"  Episodes saved: {stats['total_episodes_saved']}")
        print(f"  Total steps: {stats['total_steps_recorded']}")
        print("="*60)


if __name__ == "__main__":
    main()
