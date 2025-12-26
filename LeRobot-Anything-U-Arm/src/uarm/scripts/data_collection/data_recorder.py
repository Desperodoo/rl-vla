#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Recorder for ARX5 Teleoperation

Records synchronized data from:
- ARX5 robot state (joint positions, velocities, gripper)
- Dual RealSense cameras (RGB + Depth)

Saves raw data in ManiSkill-compatible format.
"""

import os
import sys
import json
import time
import select
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import cv2
import h5py

from .dataset_config import DatasetConfig, DEFAULT_CONFIG
from .camera_manager import CameraManager, CameraFrame


@dataclass
class RobotState:
    """Container for robot state at a single timestep"""
    joint_pos: np.ndarray       # [6] joint positions in radians
    joint_vel: np.ndarray       # [6] joint velocities in rad/s
    gripper_pos: float          # gripper position in meters
    timestamp: float            # system timestamp


@dataclass
class Timestep:
    """Container for a single timestep of data"""
    # Robot state
    robot_state: RobotState
    
    # Camera frames (name -> CameraFrame)
    camera_frames: Dict[str, CameraFrame]
    
    # Action (target command sent to robot)
    action: Optional[np.ndarray] = None  # [7] joint_pos[6] + gripper[1]
    
    # Metadata
    step_idx: int = 0
    timestamp: float = 0.0


class DataRecorder:
    """
    Records teleoperation data for ARX5 robot
    
    Features:
    - Thread-safe data recording
    - Keyboard control for recording sessions
    - Automatic episode management
    - Raw data storage in organized directory structure
    """
    
    def __init__(self, config: DatasetConfig = None):
        self.config = config or DEFAULT_CONFIG
        
        # Current episode data
        self.current_episode: List[Timestep] = []
        self.episode_metadata: Dict = {}
        
        # Recording state
        self.is_recording = False
        self.current_episode_id = 0
        self.current_session_dir: Optional[str] = None
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.total_episodes_saved = 0
        self.total_steps_recorded = 0
    
    def start_session(self, task_name: Optional[str] = None) -> str:
        """
        Start a new recording session
        
        Args:
            task_name: Optional override for task name
        
        Returns:
            Path to session directory
        """
        if task_name:
            self.config.task_name = task_name
        
        # Create session directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_dir = os.path.join(
            self.config.raw_data_dir,
            self.config.task_name,
            timestamp
        )
        os.makedirs(self.current_session_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(self.current_session_dir, "config.yaml")
        self.config.save(config_path)
        
        # Reset counters
        self.current_episode_id = 0
        self.total_episodes_saved = 0
        
        print(f"[DataRecorder] Started session: {self.current_session_dir}")
        return self.current_session_dir
    
    def start_episode(self, seed: Optional[int] = None, metadata: Dict = None):
        """Start recording a new episode"""
        with self.lock:
            self.current_episode = []
            self.episode_metadata = {
                "episode_id": self.current_episode_id,
                "start_time": time.time(),
                "seed": seed,
                "control_mode": "joint_pos",
                **(metadata or {})
            }
            self.is_recording = True
        
        print(f"[DataRecorder] Started episode {self.current_episode_id}")
    
    def stop_episode(self) -> int:
        """Stop recording current episode"""
        with self.lock:
            self.is_recording = False
            num_steps = len(self.current_episode)
            self.episode_metadata["end_time"] = time.time()
            self.episode_metadata["elapsed_steps"] = num_steps
        
        print(f"[DataRecorder] Stopped episode {self.current_episode_id} ({num_steps} steps)")
        return num_steps
    
    def record_step(
        self,
        robot_state: RobotState,
        camera_frames: Dict[str, CameraFrame],
        action: Optional[np.ndarray] = None
    ):
        """
        Record a single timestep
        
        Args:
            robot_state: Current robot state
            camera_frames: Dictionary of camera frames
            action: Action command (optional)
        """
        if not self.is_recording:
            return
        
        with self.lock:
            step_idx = len(self.current_episode)
            timestep = Timestep(
                robot_state=robot_state,
                camera_frames=camera_frames,
                action=action,
                step_idx=step_idx,
                timestamp=time.time()
            )
            self.current_episode.append(timestep)
            self.total_steps_recorded += 1
    
    def save_episode(self, success: bool = True, discard: bool = False) -> Optional[str]:
        """
        Save current episode to disk
        
        Args:
            success: Whether the episode was successful
            discard: If True, discard the episode without saving
        
        Returns:
            Path to saved episode directory, or None if discarded
        """
        with self.lock:
            if not self.current_episode:
                print("[DataRecorder] No data to save")
                return None
            
            if discard:
                print(f"[DataRecorder] Discarded episode {self.current_episode_id}")
                self.current_episode = []
                return None
            
            # Create episode directory
            episode_dir = os.path.join(
                self.current_session_dir,
                f"episode_{self.current_episode_id:04d}"
            )
            os.makedirs(episode_dir, exist_ok=True)
            
            # Update metadata
            self.episode_metadata["success"] = success
            self.episode_metadata["num_steps"] = len(self.current_episode)
            
            # Save data
            self._save_episode_raw(episode_dir)
            
            # Update counters
            saved_id = self.current_episode_id
            self.current_episode_id += 1
            self.total_episodes_saved += 1
            self.current_episode = []
        
        print(f"[DataRecorder] Saved episode {saved_id} to {episode_dir}")
        return episode_dir
    
    def _save_episode_raw(self, episode_dir: str):
        """Save episode data in raw format"""
        
        # Create subdirectories for images
        for cam_name in self.config.cameras.keys():
            os.makedirs(os.path.join(episode_dir, f"{cam_name}_rgb"), exist_ok=True)
            if self.config.cameras[cam_name].enable_depth:
                os.makedirs(os.path.join(episode_dir, f"{cam_name}_depth"), exist_ok=True)
        
        # Prepare arrays for HDF5
        num_steps = len(self.current_episode)
        
        joint_pos = np.zeros((num_steps, self.config.robot.joint_pos_dim), dtype=np.float32)
        joint_vel = np.zeros((num_steps, self.config.robot.joint_vel_dim), dtype=np.float32)
        gripper_pos = np.zeros((num_steps, 1), dtype=np.float32)
        actions = np.zeros((num_steps, self.config.robot.action_dim), dtype=np.float32)
        timestamps = np.zeros(num_steps, dtype=np.float64)
        
        # Process each timestep
        for i, ts in enumerate(self.current_episode):
            # Robot state
            joint_pos[i] = ts.robot_state.joint_pos
            joint_vel[i] = ts.robot_state.joint_vel
            gripper_pos[i] = ts.robot_state.gripper_pos
            timestamps[i] = ts.timestamp
            
            if ts.action is not None:
                actions[i] = ts.action
            
            # Save images
            for cam_name, frame in ts.camera_frames.items():
                if frame is None:
                    continue
                
                # Save RGB as PNG
                rgb_path = os.path.join(
                    episode_dir, f"{cam_name}_rgb", f"{i:06d}.png"
                )
                cv2.imwrite(rgb_path, frame.rgb)
                
                # Save depth as 16-bit PNG (millimeters)
                if frame.depth is not None:
                    depth_path = os.path.join(
                        episode_dir, f"{cam_name}_depth", f"{i:06d}.png"
                    )
                    # Depth is already in uint16 mm from RealSense
                    cv2.imwrite(depth_path, frame.depth)
        
        # Save robot data to HDF5
        h5_path = os.path.join(episode_dir, "robot_data.h5")
        with h5py.File(h5_path, 'w') as f:
            # Observation group
            obs = f.create_group("obs")
            obs.create_dataset("joint_pos", data=joint_pos, compression="gzip")
            obs.create_dataset("joint_vel", data=joint_vel, compression="gzip")
            obs.create_dataset("gripper_pos", data=gripper_pos, compression="gzip")
            obs.create_dataset("timestamps", data=timestamps, compression="gzip")
            
            # Actions
            f.create_dataset("actions", data=actions, compression="gzip")
            
            # Metadata as attributes
            f.attrs["num_steps"] = num_steps
            f.attrs["control_freq"] = self.config.control_freq
            f.attrs["success"] = self.episode_metadata.get("success", False)
        
        # Save metadata as JSON
        meta_path = os.path.join(episode_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            # Convert numpy types for JSON serialization
            meta = {k: (v.item() if isinstance(v, np.generic) else v) 
                    for k, v in self.episode_metadata.items()}
            json.dump(meta, f, indent=2)
        
        print(f"[DataRecorder] Raw data saved: {num_steps} steps, "
              f"{len(self.config.cameras)} cameras")
    
    def get_stats(self) -> Dict:
        """Get recording statistics"""
        return {
            "session_dir": self.current_session_dir,
            "current_episode_id": self.current_episode_id,
            "total_episodes_saved": self.total_episodes_saved,
            "total_steps_recorded": self.total_steps_recorded,
            "current_episode_steps": len(self.current_episode),
            "is_recording": self.is_recording,
        }


class TeleopDataCollector:
    """
    Main data collection class that combines:
    - ARX5 robot teleoperation
    - Camera capture
    - Data recording
    - Interactive control via keyboard
    """
    
    def __init__(
        self,
        config: DatasetConfig = None,
        arx_model: str = "X5",
        arx_interface: str = "can0",
        servo_port: Optional[str] = None,
        headless: bool = False
    ):
        self.config = config or DEFAULT_CONFIG
        self.arx_model = arx_model
        self.arx_interface = arx_interface
        self.servo_port = servo_port
        self.headless = headless
        
        # Components (initialized in setup())
        self.camera_manager: Optional[CameraManager] = None
        self.recorder: Optional[DataRecorder] = None
        self.teleop = None  # ArxTeleop instance
        self.servo_reader = None  # ServoReader instance
        
        # Control timing (will be set from ARX5 controller in setup())
        self.controller_dt = 1.0 / self.config.control_freq  # Default fallback
        
        # Control state
        self._running = False
        self._paused = False
    
    def setup(self) -> bool:
        """Initialize all components"""
        print("\n" + "="*60)
        print("ARX5 Data Collection System Setup")
        print("="*60)
        
        # 1. Initialize cameras
        print("\n[1/3] Initializing cameras...")
        self.camera_manager = CameraManager(self.config.cameras)
        if not self.camera_manager.initialize(auto_assign=True):
            print("[Setup] Warning: Camera initialization failed, continuing without cameras")
        
        # 2. Initialize robot (import here to avoid circular deps)
        print("\n[2/3] Initializing robot...")
        try:
            import sys
            # Add paths for ARX5 SDK and local scripts
            arx_sdk_path = "/home/lizh/arx5-sdk/python"
            scripts_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            if arx_sdk_path not in sys.path:
                sys.path.insert(0, arx_sdk_path)
            if scripts_path not in sys.path:
                sys.path.insert(0, scripts_path)
            
            # Change to ARX SDK directory for proper module loading
            original_cwd = os.getcwd()
            os.chdir(arx_sdk_path)
            
            # Import from local repository's arx_teleop.py
            from Follower_Arm.ARX.arx_teleop import ServoReader, ArxTeleop
            
            self.servo_reader = ServoReader(port=self.servo_port, baudrate=115200)
            self.teleop = ArxTeleop(model=self.arx_model, interface=self.arx_interface)
            
            # Get the actual controller dt from ARX5 (critical for smooth control!)
            self.controller_dt = float(self.teleop.ctrl_cfg.controller_dt)
            print(f"[Setup] ARX5 controller dt: {self.controller_dt:.4f}s ({1.0/self.controller_dt:.1f}Hz)")
            
            # Restore original working directory
            os.chdir(original_cwd)
            print("[Setup] Robot initialized successfully")
        except Exception as e:
            print(f"[Setup] Robot initialization failed: {e}")
            print("[Setup] Running in camera-only mode for testing")
            self.controller_dt = 1.0 / self.config.control_freq  # Fallback
        
        # 3. Initialize recorder
        print("\n[3/3] Initializing data recorder...")
        self.recorder = DataRecorder(self.config)
        
        print("\n" + "="*60)
        print("Setup complete!")
        print("="*60)
        
        return True
    
    def get_robot_state(self) -> Optional[RobotState]:
        """Get current robot state from ARX5"""
        if self.teleop is None:
            return None
        
        try:
            state = self.teleop.ctrl.get_state()
            return RobotState(
                joint_pos=np.array(state.pos(), dtype=np.float32),
                joint_vel=np.array(state.vel(), dtype=np.float32),
                gripper_pos=float(state.gripper_pos),
                timestamp=time.time()
            )
        except Exception as e:
            print(f"[Warning] Failed to get robot state: {e}")
            return None
    
    def run(self, task_name: str = "arx5_teleop"):
        """
        Main data collection loop
        
        Controls:
            Space:     Start/Pause recording
            Enter:     Save current episode (mark as success)
            Backspace: Discard current episode
            F:         Save as failed episode
            Q:         Quit
        """
        print("\n" + "="*60)
        print("Data Collection Controls" + (" (Headless Mode)" if self.headless else ""))
        print("="*60)
        if self.headless:
            print("  Type command + Enter:")
            print("    s         - Start/Pause recording")
            print("    save      - Save episode (success)")
            print("    f         - Save episode (failed)")
            print("    d         - Discard episode")
            print("    q         - Quit")
            print("    h         - Show help")
        else:
            print("  In OpenCV Window (click window first):")
            print("    [Space]     - Start/Pause recording")
            print("    [Enter]     - Save episode (success)")
            print("    [F]         - Save episode (failed)")
            print("    [Backspace] - Discard episode")
            print("    [Q]         - Quit")
            print("  Or type in terminal: s, save, f, d, q, h")
        print("="*60 + "\n")
        
        # Start session
        self.recorder.start_session(task_name)
        
        # Start camera capture
        if self.camera_manager and self.camera_manager.cameras:
            self.camera_manager.start()
        
        # Start servo reading thread if available
        if self.servo_reader:
            servo_thread = threading.Thread(
                target=self.servo_reader.read_loop,
                kwargs={"hz": 100},
                daemon=True
            )
            servo_thread.start()
        
        self._running = True
        
        # Shared state between control thread and main thread
        self._latest_robot_state: Optional[RobotState] = None
        self._latest_action: Optional[np.ndarray] = None
        self._state_lock = threading.Lock()
        
        # Start dedicated control thread (HIGH PRIORITY - no blocking operations!)
        control_thread = threading.Thread(
            target=self._control_loop,
            daemon=True
        )
        control_thread.start()
        
        # Main thread handles: UI, keyboard, data recording (can be slower)
        self._ui_and_recording_loop()
    
    def _control_loop(self):
        """
        Dedicated control loop thread - runs at ARX5 controller frequency.
        
        CRITICAL: This thread MUST NOT do any blocking I/O operations:
        - No camera access
        - No file I/O
        - No cv2 operations
        - No print (except for periodic debug)
        """
        if not self.servo_reader or not self.teleop:
            print("[Control] No robot connected, control loop exiting")
            return
        
        control_dt = self.controller_dt
        loop_count = 0
        
        print(f"[Control] Starting control loop at {1.0/control_dt:.1f}Hz")
        
        while self._running:
            loop_start = time.time()
            
            # Read master arm angles
            master_angles = self.servo_reader.get_angles()
            
            # Send command to slave arm
            self.teleop.send_cmd(master_angles)
            
            # Get robot state (fast operation, just reads from controller)
            try:
                state = self.teleop.ctrl.get_state()
                robot_state = RobotState(
                    joint_pos=np.array(state.pos(), dtype=np.float32),
                    joint_vel=np.array(state.vel(), dtype=np.float32),
                    gripper_pos=float(state.gripper_pos),
                    timestamp=time.time()
                )
                
                action = np.concatenate([
                    self.teleop._last_cmd_pos,
                    [self.teleop._last_cmd_grip]
                ]).astype(np.float32)
                
                # Update shared state (thread-safe)
                with self._state_lock:
                    self._latest_robot_state = robot_state
                    self._latest_action = action
                    
            except Exception as e:
                if loop_count % 100 == 0:
                    print(f"[Control] Warning: {e}")
            
            # Debug output only when there's significant change or issue
            # Removed frequent logging to reduce noise
            
            loop_count += 1
            
            # Precise timing for control loop
            elapsed = time.time() - loop_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif loop_count % 50 == 0 and elapsed > control_dt * 1.5:
                print(f"[Control] Warning: Loop overrun {elapsed*1000:.1f}ms > {control_dt*1000:.1f}ms")
    
    def _ui_and_recording_loop(self):
        """
        Main thread loop for UI and data recording.
        Runs at config.control_freq (30Hz) - slower than control loop.
        """
        # Create preview windows (only if not headless)
        if not self.headless:
            cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
        
        record_dt = 1.0 / self.config.control_freq
        loop_count = 0
        
        try:
            while self._running:
                loop_start = time.time()
                
                # Get latest robot state (thread-safe copy)
                robot_state = None
                action = None
                with self._state_lock:
                    if self._latest_robot_state is not None:
                        robot_state = self._latest_robot_state
                        action = self._latest_action
                
                # Get camera frames (can be slow, but doesn't affect control)
                camera_frames = {}
                if self.camera_manager:
                    camera_frames = self.camera_manager.get_frames()
                
                # Record step if recording
                if self.recorder.is_recording and robot_state:
                    self.recorder.record_step(robot_state, camera_frames, action)
                
                # UI updates (only if not headless)
                if not self.headless:
                    # Update preview
                    self._update_preview(camera_frames, robot_state)
                    
                    # Handle keyboard input from OpenCV window
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:  # 255 means no key pressed
                        self._handle_key(key)
                    
                    # Also check stdin for terminal input (in case window doesn't have focus)
                    stdin_key = self._check_stdin()
                    if stdin_key:
                        self._handle_headless_key(stdin_key)
                    
                    # Print status periodically
                    if loop_count % 30 == 0:
                        status = "REC" if self.recorder.is_recording else "IDLE"
                        step_count = len(self.recorder.current_episode) if self.recorder.is_recording else 0
                        print(f"\r[Status] {status} | Steps: {step_count} | "
                              f"Episodes: {self.recorder.total_episodes_saved}", end="", flush=True)
                else:
                    # Headless mode: print status periodically and check stdin
                    if loop_count % 30 == 0:  # Every ~1 second
                        status = "REC" if self.recorder.is_recording else "IDLE"
                        step_count = len(self.recorder.current_episode) if self.recorder.is_recording else 0
                        print(f"[Status] {status} | Steps: {step_count} | "
                              f"Episodes: {self.recorder.total_episodes_saved}")
                    
                    # Check for keyboard input in headless mode
                    key = self._check_stdin()
                    if key:
                        self._handle_headless_key(key)
                
                loop_count += 1
                
                # Maintain recording loop frequency
                elapsed = time.time() - loop_start
                if elapsed < record_dt:
                    time.sleep(record_dt - elapsed)
        
        except KeyboardInterrupt:
            print("\n[Main] Interrupted by user")
        
        finally:
            self._cleanup()
    
    def _update_preview(
        self,
        camera_frames: Dict[str, CameraFrame],
        robot_state: Optional[RobotState]
    ):
        """Update preview window"""
        # Create preview image
        preview_images = []
        
        for name, frame in camera_frames.items():
            if frame is not None:
                img = frame.rgb.copy()
                
                # Add recording indicator
                if self.recorder.is_recording:
                    cv2.circle(img, (30, 30), 15, (0, 0, 255), -1)
                    cv2.putText(img, "REC", (50, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Add camera name
                cv2.putText(img, name, (10, img.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                preview_images.append(img)
        
        if preview_images:
            # Stack images horizontally
            combined = np.hstack(preview_images) if len(preview_images) > 1 else preview_images[0]
            
            # Add status bar
            status = self.recorder.get_stats()
            status_text = (f"Episode: {status['current_episode_id']} | "
                          f"Steps: {status['current_episode_steps']} | "
                          f"Total saved: {status['total_episodes_saved']}")
            
            cv2.putText(combined, status_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Data Collection", combined)
    
    def _handle_key(self, key: int):
        """Handle keyboard input"""
        if key == ord(' '):  # Space - toggle recording
            if not self.recorder.is_recording:
                self.recorder.start_episode()
            else:
                self.recorder.stop_episode()
        
        elif key == 13:  # Enter - save as success
            if self.recorder.current_episode:
                self.recorder.stop_episode()
                self.recorder.save_episode(success=True)
        
        elif key == ord('f') or key == ord('F'):  # F - save as failed
            if self.recorder.current_episode:
                self.recorder.stop_episode()
                self.recorder.save_episode(success=False)
        
        elif key == 8:  # Backspace - discard
            if self.recorder.current_episode:
                self.recorder.stop_episode()
                self.recorder.save_episode(discard=True)
        
        elif key == ord('q') or key == ord('Q'):  # Q - quit
            self._running = False
    
    def _check_stdin(self) -> Optional[str]:
        """
        Non-blocking check for keyboard input from stdin (for headless mode).
        Returns the key pressed or None if no input.
        """
        import sys
        import select
        
        # Check if there's input available on stdin
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.readline().strip().lower()
        return None
    
    def _handle_headless_key(self, key: str):
        """Handle keyboard input in headless mode"""
        if key == 'space' or key == 's' or key == '':  # Space or 's' or just Enter
            if not self.recorder.is_recording:
                self.recorder.start_episode()
                print("[Action] Started recording")
            else:
                self.recorder.stop_episode()
                print("[Action] Paused recording")
        
        elif key == 'enter' or key == 'save':  # Save as success
            if self.recorder.current_episode:
                self.recorder.stop_episode()
                self.recorder.save_episode(success=True)
                print("[Action] Saved episode as SUCCESS")
        
        elif key == 'f' or key == 'fail':  # Save as failed
            if self.recorder.current_episode:
                self.recorder.stop_episode()
                self.recorder.save_episode(success=False)
                print("[Action] Saved episode as FAILED")
        
        elif key == 'd' or key == 'discard':  # Discard
            if self.recorder.current_episode:
                self.recorder.stop_episode()
                self.recorder.save_episode(discard=True)
                print("[Action] Discarded episode")
        
        elif key == 'q' or key == 'quit':  # Quit
            print("[Action] Quitting...")
            self._running = False
        
        elif key == 'h' or key == 'help':  # Help
            print("\n[Headless Mode Commands]")
            print("  s/space/Enter - Start/Pause recording")
            print("  save/enter    - Save episode (success)")
            print("  f/fail        - Save episode (failed)")
            print("  d/discard     - Discard episode")
            print("  q/quit        - Quit")
            print("  h/help        - Show this help\n")
    
    def _cleanup(self):
        """Cleanup resources"""
        print("\n[Main] Cleaning up...")
        
        # Save any unsaved episode
        if self.recorder and self.recorder.current_episode:
            print("[Main] Saving current episode before exit...")
            self.recorder.stop_episode()
            self.recorder.save_episode(success=False)
        
        # Stop cameras
        if self.camera_manager:
            self.camera_manager.stop()
        
        # Reset robot
        if self.teleop:
            try:
                self.teleop.ctrl.reset_to_home()
            except:
                pass
        
        cv2.destroyAllWindows()
        
        # Print summary
        if self.recorder:
            stats = self.recorder.get_stats()
            print("\n" + "="*60)
            print("Session Summary")
            print("="*60)
            print(f"  Session directory: {stats['session_dir']}")
            print(f"  Total episodes saved: {stats['total_episodes_saved']}")
            print(f"  Total steps recorded: {stats['total_steps_recorded']}")
            print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ARX5 Teleoperation Data Collection")
    parser.add_argument("--task", type=str, default="arx5_teleop",
                       help="Task name for this collection session")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config YAML file")
    parser.add_argument("--can", type=str, default="can0",
                       help="CAN interface for ARX5")
    parser.add_argument("--servo-port", type=str, default=None,
                       help="Serial port for servo controller (auto-detect if not specified)")
    args = parser.parse_args()
    
    # Load config if specified
    config = None
    if args.config:
        config = DatasetConfig.load(args.config)
    
    # Create collector and run
    collector = TeleopDataCollector(
        config=config,
        arx_interface=args.can,
        servo_port=args.servo_port
    )
    
    if collector.setup():
        collector.run(task_name=args.task)
