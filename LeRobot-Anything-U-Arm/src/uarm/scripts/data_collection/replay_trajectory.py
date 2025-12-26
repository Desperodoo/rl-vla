#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Replay and Verification for ARX5 Dataset

This script allows:
- Visual replay of recorded trajectories (video playback)
- Physical replay on ARX5 robot (execute recorded actions)
- Data integrity verification
- Statistics visualization

Inspired by ManiSkill's trajectory.replay_trajectory module.

Usage:
    # Visual replay only
    python replay_trajectory.py --traj-path ~/.arx_demos/processed/pick_cube/trajectory.h5 --visual-only
    
    # Physical replay on robot
    python replay_trajectory.py --traj-path ~/.arx_demos/processed/pick_cube/trajectory.h5 --execute
    
    # Verify data integrity
    python replay_trajectory.py --traj-path ~/.arx_demos/processed/pick_cube/trajectory.h5 --verify
"""

import argparse
import os
import sys
import time
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import cv2
import h5py
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d


class TrajectoryPlayer:
    """
    Plays back recorded trajectories for visualization and verification
    """
    
    def __init__(self, traj_path: str):
        """
        Initialize trajectory player
        
        Args:
            traj_path: Path to trajectory HDF5 file
        """
        self.traj_path = os.path.expanduser(traj_path)
        self.traj_dir = os.path.dirname(self.traj_path)
        
        # Load trajectory data
        self.h5_file = h5py.File(self.traj_path, 'r')
        self.trajectory_ids = [k for k in self.h5_file.keys() if k.startswith("traj_")]
        self.num_trajectories = len(self.trajectory_ids)
        
        # Load metadata if exists
        self.metadata = {}
        meta_path = os.path.join(self.traj_dir, "trajectory.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Load statistics if exists
        self.stats = {}
        stats_path = os.path.join(self.traj_dir, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
        
        print(f"Loaded trajectory file: {self.traj_path}")
        print(f"Number of trajectories: {self.num_trajectories}")
    
    def get_trajectory(self, traj_idx: int) -> Dict:
        """Load a single trajectory by index"""
        
        traj_id = f"traj_{traj_idx}"
        if traj_id not in self.h5_file:
            raise ValueError(f"Trajectory {traj_idx} not found")
        
        grp = self.h5_file[traj_id]
        
        data = {
            "joint_pos": np.array(grp["obs/joint_pos"]),
            "joint_vel": np.array(grp["obs/joint_vel"]),
            "gripper_pos": np.array(grp["obs/gripper_pos"]),
            "actions": np.array(grp["actions"]),
            "success": bool(grp.attrs.get("success", False)),
            "num_steps": int(grp.attrs.get("num_steps", len(grp["actions"]))),
        }
        
        # Load images if available
        if "obs/images" in grp:
            data["images"] = {}
            for cam_name in grp["obs/images"].keys():
                data["images"][cam_name] = {}
                cam_grp = grp[f"obs/images/{cam_name}"]
                if "rgb" in cam_grp:
                    data["images"][cam_name]["rgb"] = np.array(cam_grp["rgb"])
                if "depth" in cam_grp:
                    data["images"][cam_name]["depth"] = np.array(cam_grp["depth"])
        
        return data
    
    def get_summary(self) -> Dict:
        """Get summary statistics of the dataset"""
        
        total_steps = 0
        success_count = 0
        episode_lengths = []
        
        for traj_id in self.trajectory_ids:
            grp = self.h5_file[traj_id]
            num_steps = int(grp.attrs.get("num_steps", len(grp["actions"])))
            success = bool(grp.attrs.get("success", False))
            
            total_steps += num_steps
            episode_lengths.append(num_steps)
            if success:
                success_count += 1
        
        return {
            "num_trajectories": self.num_trajectories,
            "total_steps": total_steps,
            "success_count": success_count,
            "success_rate": success_count / self.num_trajectories if self.num_trajectories > 0 else 0,
            "avg_episode_length": np.mean(episode_lengths) if episode_lengths else 0,
            "min_episode_length": min(episode_lengths) if episode_lengths else 0,
            "max_episode_length": max(episode_lengths) if episode_lengths else 0,
        }
    
    def verify_integrity(self) -> Dict:
        """Verify data integrity of all trajectories"""
        
        print("\nVerifying data integrity...")
        issues = []
        
        for i, traj_id in enumerate(tqdm(self.trajectory_ids)):
            grp = self.h5_file[traj_id]
            
            # Check required fields
            required_fields = ["obs/joint_pos", "obs/joint_vel", "obs/gripper_pos", "actions"]
            for field in required_fields:
                if field not in grp:
                    issues.append(f"{traj_id}: Missing field '{field}'")
            
            # Check dimensions
            if "obs/joint_pos" in grp and "actions" in grp:
                joint_pos = np.array(grp["obs/joint_pos"])
                actions = np.array(grp["actions"])
                
                if len(joint_pos) != len(actions):
                    issues.append(f"{traj_id}: Length mismatch (joint_pos={len(joint_pos)}, actions={len(actions)})")
                
                # Check for NaN/Inf
                if np.any(np.isnan(joint_pos)) or np.any(np.isinf(joint_pos)):
                    issues.append(f"{traj_id}: NaN/Inf in joint_pos")
                
                if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
                    issues.append(f"{traj_id}: NaN/Inf in actions")
                
                # Check joint limits (rough bounds)
                if np.any(np.abs(joint_pos) > 2 * np.pi):
                    issues.append(f"{traj_id}: Joint positions out of range")
        
        result = {
            "status": "OK" if not issues else "ISSUES_FOUND",
            "num_issues": len(issues),
            "issues": issues[:20] if len(issues) > 20 else issues  # Limit to 20
        }
        
        if issues:
            print(f"\nFound {len(issues)} issues:")
            for issue in issues[:10]:
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print("\n✓ Data integrity OK")
        
        return result
    
    def visual_replay(
        self,
        traj_indices: Optional[List[int]] = None,
        fps: int = 30,
        save_video: Optional[str] = None
    ):
        """
        Visual replay of trajectories
        
        Args:
            traj_indices: List of trajectory indices to replay (None = all)
            fps: Playback frame rate
            save_video: Path to save video (optional)
        """
        
        if traj_indices is None:
            traj_indices = list(range(self.num_trajectories))
        
        print(f"\nVisual replay of {len(traj_indices)} trajectories")
        print("Controls: [Space] Pause/Resume, [N] Next, [Q] Quit")
        
        cv2.namedWindow("Trajectory Replay", cv2.WINDOW_NORMAL)
        
        video_writer = None
        if save_video:
            # Will be initialized when we know the frame size
            pass
        
        for traj_idx in traj_indices:
            print(f"\nTrajectory {traj_idx}")
            
            try:
                traj_data = self.get_trajectory(traj_idx)
            except Exception as e:
                print(f"Error loading trajectory {traj_idx}: {e}")
                continue
            
            num_steps = traj_data["num_steps"]
            success = traj_data["success"]
            
            print(f"  Steps: {num_steps}, Success: {success}")
            
            paused = False
            frame_idx = 0
            
            while frame_idx < num_steps:
                # Create visualization frame
                frame = self._create_viz_frame(traj_data, frame_idx, traj_idx)
                
                # Initialize video writer if needed
                if save_video and video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(save_video, fourcc, fps, (w, h))
                
                if video_writer:
                    video_writer.write(frame)
                
                cv2.imshow("Trajectory Replay", frame)
                
                # Handle input
                key = cv2.waitKey(int(1000 / fps) if not paused else 50) & 0xFF
                
                if key == ord(' '):
                    paused = not paused
                elif key == ord('n'):
                    break  # Next trajectory
                elif key == ord('q'):
                    if video_writer:
                        video_writer.release()
                    cv2.destroyAllWindows()
                    return
                
                if not paused:
                    frame_idx += 1
        
        if video_writer:
            video_writer.release()
            print(f"\nVideo saved to {save_video}")
        
        cv2.destroyAllWindows()
    
    def _create_viz_frame(self, traj_data: Dict, frame_idx: int, traj_idx: int) -> np.ndarray:
        """Create visualization frame for a single timestep"""
        
        # Check if we have images
        if "images" in traj_data and traj_data["images"]:
            # Use actual images
            images = []
            for cam_name, cam_data in traj_data["images"].items():
                if "rgb" in cam_data:
                    img = cam_data["rgb"][frame_idx].copy()
                    # Convert BGR to RGB if needed (OpenCV uses BGR)
                    cv2.putText(img, cam_name, (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    images.append(img)
            
            if images:
                frame = np.hstack(images) if len(images) > 1 else images[0]
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            # Create synthetic visualization
            frame = self._create_state_viz(traj_data, frame_idx)
        
        # Add overlay information
        h, w = frame.shape[:2]
        
        # Status bar at top
        cv2.rectangle(frame, (0, 0), (w, 40), (40, 40, 40), -1)
        
        status = f"Traj: {traj_idx} | Frame: {frame_idx}/{traj_data['num_steps']} | "
        status += "SUCCESS" if traj_data["success"] else "FAILED"
        cv2.putText(frame, status, (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if traj_data["success"] else (0, 0, 255), 2)
        
        return frame
    
    def _create_state_viz(self, traj_data: Dict, frame_idx: int) -> np.ndarray:
        """Create visualization from state data (when no images available)"""
        
        frame = np.zeros((480, 800, 3), dtype=np.uint8)
        
        # Get current state
        joint_pos = traj_data["joint_pos"][frame_idx]
        gripper_pos = traj_data["gripper_pos"][frame_idx]
        action = traj_data["actions"][frame_idx]
        
        # Draw joint positions as bar graph
        bar_width = 60
        bar_max_height = 200
        start_x = 50
        start_y = 350
        
        cv2.putText(frame, "Joint Positions (rad)", (start_x, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for i, pos in enumerate(joint_pos):
            # Normalize to [0, 1] assuming range [-pi, pi]
            normalized = (pos + np.pi) / (2 * np.pi)
            bar_height = int(normalized * bar_max_height)
            
            x = start_x + i * (bar_width + 20)
            
            # Draw bar
            color = (0, 200, 100)
            cv2.rectangle(frame, (x, start_y - bar_height), (x + bar_width, start_y), color, -1)
            
            # Draw label
            cv2.putText(frame, f"J{i}", (x + 15, start_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{pos:.2f}", (x + 5, start_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw gripper
        gripper_x = start_x + 6 * (bar_width + 20)
        gripper_normalized = gripper_pos[0] / 0.08  # Assuming max 0.08m
        gripper_height = int(gripper_normalized * bar_max_height)
        cv2.rectangle(frame, (gripper_x, start_y - gripper_height), 
                     (gripper_x + bar_width, start_y), (200, 100, 0), -1)
        cv2.putText(frame, "Grip", (gripper_x + 10, start_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _interpolate_trajectory(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        target_dt: float = 0.002,
        smooth: bool = True,
        smooth_window: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate trajectory to higher frequency using cubic spline
        
        Args:
            actions: Original action array [N, 7] (6 joints + gripper)
            timestamps: Original timestamps [N]
            target_dt: Target time interval (0.002s = 500Hz)
            smooth: Apply smoothing before interpolation
            smooth_window: Smoothing window size
        
        Returns:
            interp_actions: Interpolated actions [M, 7]
            interp_times: Interpolated timestamps [M]
        """
        num_frames = len(actions)
        
        # Apply smoothing to original trajectory if requested
        if smooth and num_frames > smooth_window:
            smoothed_actions = np.zeros_like(actions)
            for j in range(actions.shape[1]):
                smoothed_actions[:, j] = uniform_filter1d(actions[:, j], size=smooth_window, mode='nearest')
            actions = smoothed_actions
        
        # Create relative time array
        if timestamps is not None:
            t_orig = timestamps - timestamps[0]  # Start from 0
        else:
            # Fallback: assume uniform 30Hz
            t_orig = np.arange(num_frames) / 30.0
        
        total_time = t_orig[-1]
        
        # Create target time array at 500Hz
        t_interp = np.arange(0, total_time, target_dt)
        
        # Interpolate each dimension using cubic spline
        interp_actions = np.zeros((len(t_interp), actions.shape[1]))
        
        for j in range(actions.shape[1]):
            # Use cubic spline for smooth interpolation
            # For gripper (index 6), use linear interpolation to avoid overshoot
            if j == 6:
                # Linear interpolation for gripper
                interp_actions[:, j] = np.interp(t_interp, t_orig, actions[:, j])
            else:
                # Cubic spline for joints
                cs = CubicSpline(t_orig, actions[:, j], bc_type='clamped')
                interp_actions[:, j] = cs(t_interp)
        
        # Clamp gripper to valid range [0, 0.08]
        interp_actions[:, 6] = np.clip(interp_actions[:, 6], 0.0, 0.08)
        
        print(f"Interpolation: {num_frames} frames @ ~30Hz -> {len(t_interp)} frames @ {1.0/target_dt:.0f}Hz")
        
        return interp_actions, t_interp
    
    def physical_replay(
        self,
        traj_idx: int,
        speed_factor: float = 1.0,
        dry_run: bool = False,
        interpolate: bool = True,
        target_freq: float = 500.0,
        smooth: bool = True
    ):
        """
        Execute trajectory on physical ARX5 robot with interpolation and smoothing
        
        TIMING NOTE:
        ============
        - Original data is recorded at ~30Hz
        - Robot controller runs at 500Hz internally
        - We interpolate the trajectory to 500Hz for smooth execution
        - Cubic spline is used for joints, linear for gripper
        
        GRIPPER CONTROL:
        ================
        - Gripper uses POSITION control (not velocity or binary)
        - Range: [0.0, 0.08] meters (0=closed, 0.08=open)
        - We do NOT use binary (0/1) control - continuous position is better
        
        Args:
            traj_idx: Trajectory index to replay
            speed_factor: Speed multiplier (1.0 = original speed)
            dry_run: If True, only print commands without executing
            interpolate: If True, interpolate to 500Hz (default: True)
            target_freq: Target control frequency for interpolation (default: 500Hz)
            smooth: Apply smoothing before interpolation (default: True)
        """
        
        print(f"\n{'='*60}")
        print(f"Physical Replay - Trajectory {traj_idx}")
        print(f"{'='*60}")
        
        # Load trajectory
        traj_data = self.get_trajectory(traj_idx)
        num_steps = traj_data["num_steps"]
        actions = traj_data["actions"]
        
        # Get timestamps from trajectory
        traj_grp = self.h5_file[self.trajectory_ids[traj_idx]]
        timestamps = None
        record_dt_mean = 1.0 / 30.0  # Default
        
        if "obs/timestamps" in traj_grp:
            timestamps = np.array(traj_grp["obs/timestamps"])
            if len(timestamps) > 1:
                record_dt_mean = float(np.diff(timestamps).mean())
        elif "record_dt_mean" in traj_grp.attrs:
            record_dt_mean = float(traj_grp.attrs["record_dt_mean"])
        elif "control_dt" in traj_grp.attrs:
            record_dt_mean = float(traj_grp.attrs["control_dt"])
        
        print(f"Original: {num_steps} steps @ {1.0/record_dt_mean:.1f}Hz")
        
        # Calculate original duration
        if timestamps is not None:
            original_duration = timestamps[-1] - timestamps[0]
        else:
            original_duration = (num_steps - 1) * record_dt_mean
        
        print(f"Original duration: {original_duration:.1f}s")
        
        # Interpolate if requested
        target_dt = 1.0 / target_freq
        if interpolate:
            interp_actions, interp_times = self._interpolate_trajectory(
                actions, timestamps, target_dt=target_dt, smooth=smooth
            )
            replay_actions = interp_actions
            replay_dt = target_dt
            replay_steps = len(interp_actions)
        else:
            replay_actions = actions
            replay_dt = record_dt_mean
            replay_steps = num_steps
        
        # Adjusted for speed factor
        effective_dt = replay_dt / speed_factor
        expected_duration = original_duration / speed_factor
        
        print(f"Replay: {replay_steps} steps @ {1.0/effective_dt:.0f}Hz (speed: {speed_factor}x)")
        print(f"Expected duration: {expected_duration:.1f}s")
        
        # Show gripper range
        gripper_cmds = replay_actions[:, 6]
        print(f"Gripper range: [{gripper_cmds.min():.4f}, {gripper_cmds.max():.4f}] m")
        
        if not dry_run:
            # Initialize robot
            try:
                sys.path.append("/home/lizh/arx5-sdk/python")
                os.chdir("/home/lizh/arx5-sdk/python")
                import arx5_interface as arx5
                
                ctrl = arx5.Arx5JointController("X5", "can0")
                ctrl_cfg = ctrl.get_controller_config()
                controller_dt = float(ctrl_cfg.controller_dt)
                
                print(f"\nRobot controller dt: {controller_dt*1000:.1f}ms ({1.0/controller_dt:.0f}Hz)")
                
                # Reset to home first
                print("Resetting to home position...")
                ctrl.reset_to_home()
                time.sleep(1.0)
                
            except Exception as e:
                print(f"Failed to initialize robot: {e}")
                print("Running in dry-run mode instead")
                dry_run = True
        
        if not dry_run:
            input("\nPress Enter to start replay (Ctrl+C to cancel)...")
        
        replay_start_time = time.time()
        
        try:
            # Use tqdm with miniters to reduce overhead at high frequency
            progress = tqdm(range(replay_steps), miniters=100, desc="Replaying")
            
            for i in progress:
                action = replay_actions[i]
                joint_cmd = action[:6]
                gripper_cmd = action[6]
                
                if dry_run:
                    # Print less frequently in dry run
                    if i % 500 == 0:
                        print(f"  Step {i}/{replay_steps}: joints[0]={joint_cmd[0]:.3f}, gripper={gripper_cmd:.4f}m")
                else:
                    # Send command to robot
                    js = arx5.JointState(6)
                    js.pos()[:] = joint_cmd
                    js.gripper_pos = gripper_cmd
                    ctrl.set_joint_cmd(js)
                    ctrl.send_recv_once()
                
                # Sleep for the target dt (adjusted by speed factor)
                # Note: At 500Hz, sleep(0.002/speed) might not be accurate
                # The arx5 controller handles timing internally via send_recv_once()
                if i < replay_steps - 1:
                    time.sleep(effective_dt)
        
        except KeyboardInterrupt:
            print("\nReplay interrupted!")
        
        finally:
            replay_duration = time.time() - replay_start_time
            print(f"\n{'='*60}")
            print(f"Replay complete!")
            print(f"Expected duration: {expected_duration:.1f}s")
            print(f"Actual duration:   {replay_duration:.1f}s")
            print(f"Timing error:      {abs(replay_duration - expected_duration):.2f}s ({abs(replay_duration - expected_duration)/expected_duration*100:.1f}%)")
            print(f"{'='*60}")
            
            if not dry_run:
                print("Resetting to home...")
                ctrl.reset_to_home()
    
    def close(self):
        """Close HDF5 file"""
        self.h5_file.close()


def main():
    parser = argparse.ArgumentParser(description="Replay and verify ARX5 trajectories")
    
    parser.add_argument("--traj-path", "-t", type=str, required=True,
                       help="Path to trajectory HDF5 file")
    
    # Mode selection
    parser.add_argument("--visual-only", action="store_true",
                       help="Visual replay only (no robot)")
    parser.add_argument("--execute", action="store_true",
                       help="Execute on physical robot")
    parser.add_argument("--verify", action="store_true",
                       help="Verify data integrity")
    parser.add_argument("--summary", action="store_true",
                       help="Print dataset summary")
    
    # Replay options
    parser.add_argument("--traj-idx", type=int, default=None,
                       help="Specific trajectory index to replay")
    parser.add_argument("--num-trajs", "-n", type=int, default=5,
                       help="Number of trajectories to replay")
    parser.add_argument("--fps", type=int, default=30,
                       help="Playback FPS for visual replay")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Speed factor for physical replay")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run (print commands without executing)")
    
    # Interpolation and smoothing options
    parser.add_argument("--no-interp", action="store_true",
                       help="Disable interpolation (replay at original ~30Hz)")
    parser.add_argument("--target-freq", type=float, default=500.0,
                       help="Target control frequency for interpolation (default: 500Hz)")
    parser.add_argument("--no-smooth", action="store_true",
                       help="Disable trajectory smoothing before interpolation")
    
    # Output
    parser.add_argument("--save-video", type=str, default=None,
                       help="Save replay to video file")
    
    args = parser.parse_args()
    
    # Load trajectory player
    player = TrajectoryPlayer(args.traj_path)
    
    try:
        if args.summary:
            summary = player.get_summary()
            print("\n" + "="*50)
            print("Dataset Summary")
            print("="*50)
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
            print("="*50)
        
        if args.verify:
            player.verify_integrity()
        
        if args.visual_only or (not args.execute and not args.verify and not args.summary):
            # Visual replay
            if args.traj_idx is not None:
                traj_indices = [args.traj_idx]
            else:
                traj_indices = list(range(min(args.num_trajs, player.num_trajectories)))
            
            player.visual_replay(
                traj_indices=traj_indices,
                fps=args.fps,
                save_video=args.save_video
            )
        
        if args.execute:
            # Physical replay
            traj_idx = args.traj_idx if args.traj_idx is not None else 0
            player.physical_replay(
                traj_idx=traj_idx,
                speed_factor=args.speed,
                dry_run=args.dry_run,
                interpolate=not args.no_interp,
                target_freq=args.target_freq,
                smooth=not args.no_smooth
            )
    
    finally:
        player.close()


if __name__ == "__main__":
    main()
