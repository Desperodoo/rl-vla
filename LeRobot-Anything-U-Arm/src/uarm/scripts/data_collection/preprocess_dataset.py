#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing and Cleaning for ARX5 Teleoperation Dataset

This script converts raw collected data to ManiSkill-compatible HDF5 format:
- Loads raw episodes from disk
- Applies data cleaning (remove static frames, filter outliers, smooth trajectories)
- Resizes/compresses images
- Outputs consolidated HDF5 + JSON metadata

Usage:
    python preprocess_dataset.py --input ~/.arx_demos/raw/pick_cube/20231215_120000 --output ~/.arx_demos/processed/pick_cube
    python preprocess_dataset.py --input ~/.arx_demos/raw/pick_cube --output ~/.arx_demos/processed/pick_cube --all-sessions
"""

import argparse
import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import shutil

import numpy as np
import cv2
import h5py
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing"""
    
    # Image processing
    target_rgb_size: Tuple[int, int] = (256, 256)  # Output RGB resolution
    target_depth_size: Tuple[int, int] = (256, 256)  # Output depth resolution
    jpeg_quality: int = 95  # JPEG compression quality (1-100)
    store_images_in_hdf5: bool = True  # Store images in HDF5 or as separate files
    
    # Trajectory cleaning
    min_episode_length: int = 10  # Minimum steps to keep episode
    remove_static_frames: bool = False  # Remove frames where robot didn't move (disabled by default for teleop)
    static_threshold: float = 0.0001  # Joint position change threshold (rad) - lowered from 0.001
    
    # Smoothing
    smooth_trajectory: bool = True
    smooth_window: int = 5  # Smoothing window size
    smooth_method: str = "savgol"  # "moving_avg" or "savgol"
    
    # Outlier removal
    remove_velocity_outliers: bool = False  # Disabled by default - teleop data may have slow movements
    max_joint_velocity: float = 3.0  # rad/s
    max_gripper_velocity: float = 0.5  # m/s
    
    # Subsampling
    subsample: bool = False
    subsample_factor: int = 3  # Keep every N-th frame
    
    # Normalization (store stats for inference)
    compute_stats: bool = True


class DataPreprocessor:
    """
    Preprocesses raw teleoperation data into clean ManiSkill-compatible format
    """
    
    def __init__(self, config: PreprocessConfig = None):
        self.config = config or PreprocessConfig()
        
        # Statistics accumulators
        self.all_joint_pos = []
        self.all_joint_vel = []
        self.all_gripper_pos = []
        self.all_actions = []
    
    def process_session(
        self,
        input_dir: str,
        output_dir: str,
        session_name: Optional[str] = None
    ) -> Dict:
        """
        Process all episodes in a session directory
        
        Args:
            input_dir: Path to raw session directory
            output_dir: Path for processed output
            session_name: Optional name override
        
        Returns:
            Processing statistics
        """
        input_dir = os.path.expanduser(input_dir)
        output_dir = os.path.expanduser(output_dir)
        
        # Find all episode directories
        episode_dirs = sorted(glob.glob(os.path.join(input_dir, "episode_*")))
        
        if not episode_dirs:
            print(f"No episodes found in {input_dir}")
            return {"status": "error", "message": "No episodes found"}
        
        print(f"\n{'='*60}")
        print(f"Processing session: {input_dir}")
        print(f"Found {len(episode_dirs)} episodes")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load session config if exists
        session_config_path = os.path.join(input_dir, "config.yaml")
        session_config = None
        if os.path.exists(session_config_path):
            shutil.copy(session_config_path, os.path.join(output_dir, "config.yaml"))
        
        # Process episodes
        processed_episodes = []
        total_steps = 0
        
        for episode_dir in tqdm(episode_dirs, desc="Processing episodes"):
            result = self._process_episode(episode_dir)
            if result is not None:
                processed_episodes.append(result)
                total_steps += result["num_steps"]
        
        print(f"\nProcessed {len(processed_episodes)}/{len(episode_dirs)} episodes")
        print(f"Total steps: {total_steps}")
        
        # Compute and save statistics
        stats = {}
        if self.config.compute_stats and processed_episodes:
            stats = self._compute_statistics()
            stats_path = os.path.join(output_dir, "stats.json")
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Saved statistics to {stats_path}")
        
        # Save consolidated HDF5
        h5_path = os.path.join(output_dir, "trajectory.h5")
        self._save_consolidated_hdf5(h5_path, processed_episodes)
        
        # Save metadata JSON (ManiSkill format) - only if we have valid episodes
        if processed_episodes:
            meta_path = os.path.join(output_dir, "trajectory.json")
            self._save_metadata_json(meta_path, processed_episodes, session_name or os.path.basename(input_dir))
        else:
            print("\n⚠️  Warning: No valid episodes to process!")
            print("    Episodes may be too short (< min_length frames).")
            print("    Try using --min-length 2 or collect longer episodes.")
        
        return {
            "status": "success",
            "input_dir": input_dir,
            "output_dir": output_dir,
            "total_episodes": len(episode_dirs),
            "processed_episodes": len(processed_episodes),
            "total_steps": total_steps,
            "stats": stats
        }
    
    def _process_episode(self, episode_dir: str) -> Optional[Dict]:
        """Process a single episode"""
        
        # Load robot data
        robot_h5_path = os.path.join(episode_dir, "robot_data.h5")
        if not os.path.exists(robot_h5_path):
            print(f"Warning: No robot data in {episode_dir}")
            return None
        
        with h5py.File(robot_h5_path, 'r') as f:
            joint_pos = np.array(f['obs/joint_pos'])
            joint_vel = np.array(f['obs/joint_vel'])
            gripper_pos = np.array(f['obs/gripper_pos'])
            timestamps = np.array(f['obs/timestamps'])
            actions = np.array(f['actions'])
            success = bool(f.attrs.get('success', False))
        
        # Load metadata
        meta_path = os.path.join(episode_dir, "metadata.json")
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        
        # Get valid frame indices
        valid_indices = self._get_valid_indices(
            joint_pos, joint_vel, gripper_pos, timestamps
        )
        
        if len(valid_indices) < self.config.min_episode_length:
            print(f"Warning: Episode too short after cleaning ({len(valid_indices)} frames)")
            return None
        
        # Apply filtering
        joint_pos_clean = joint_pos[valid_indices]
        joint_vel_clean = joint_vel[valid_indices]
        gripper_pos_clean = gripper_pos[valid_indices]
        timestamps_clean = timestamps[valid_indices]
        actions_clean = actions[valid_indices]
        
        # Apply smoothing
        if self.config.smooth_trajectory:
            joint_pos_clean = self._smooth_trajectory(joint_pos_clean)
            actions_clean = self._smooth_trajectory(actions_clean)
        
        # Apply subsampling
        if self.config.subsample:
            indices = np.arange(0, len(joint_pos_clean), self.config.subsample_factor)
            joint_pos_clean = joint_pos_clean[indices]
            joint_vel_clean = joint_vel_clean[indices]
            gripper_pos_clean = gripper_pos_clean[indices]
            timestamps_clean = timestamps_clean[indices]
            actions_clean = actions_clean[indices]
        
        # Load and process images
        images = self._load_and_process_images(episode_dir, valid_indices)
        
        # Accumulate for statistics
        self.all_joint_pos.append(joint_pos_clean)
        self.all_joint_vel.append(joint_vel_clean)
        self.all_gripper_pos.append(gripper_pos_clean)
        self.all_actions.append(actions_clean)
        
        return {
            "episode_id": metadata.get("episode_id", 0),
            "num_steps": len(joint_pos_clean),
            "success": success,
            "source_dir": episode_dir,
            "joint_pos": joint_pos_clean,
            "joint_vel": joint_vel_clean,
            "gripper_pos": gripper_pos_clean,
            "timestamps": timestamps_clean,
            "actions": actions_clean,
            "images": images,
            "metadata": metadata
        }
    
    def _get_valid_indices(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        gripper_pos: np.ndarray,
        timestamps: np.ndarray
    ) -> np.ndarray:
        """Get indices of valid frames after cleaning"""
        
        num_frames = len(joint_pos)
        valid = np.ones(num_frames, dtype=bool)
        
        # Remove static frames
        if self.config.remove_static_frames:
            for i in range(1, num_frames):
                joint_diff = np.abs(joint_pos[i] - joint_pos[i-1]).max()
                if joint_diff < self.config.static_threshold:
                    valid[i] = False
        
        # Remove velocity outliers
        if self.config.remove_velocity_outliers:
            for i in range(num_frames):
                if np.abs(joint_vel[i]).max() > self.config.max_joint_velocity:
                    valid[i] = False
        
        # Always keep first and last frames
        valid[0] = True
        valid[-1] = True
        
        return np.where(valid)[0]
    
    def _smooth_trajectory(self, data: np.ndarray) -> np.ndarray:
        """Apply smoothing to trajectory data"""
        
        if len(data) < self.config.smooth_window:
            return data
        
        if self.config.smooth_method == "moving_avg":
            # Moving average
            smoothed = uniform_filter1d(data, size=self.config.smooth_window, axis=0)
        elif self.config.smooth_method == "savgol":
            # Savitzky-Golay filter (preserves edges better)
            window = min(self.config.smooth_window, len(data))
            if window % 2 == 0:
                window -= 1
            if window >= 3:
                smoothed = savgol_filter(data, window, polyorder=2, axis=0)
            else:
                smoothed = data
        else:
            smoothed = data
        
        return smoothed.astype(np.float32)
    
    def _load_and_process_images(
        self,
        episode_dir: str,
        valid_indices: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Load and process images for valid frames"""
        
        images = {}
        
        # Find camera directories
        for subdir in os.listdir(episode_dir):
            subdir_path = os.path.join(episode_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            
            if subdir.endswith("_rgb"):
                cam_name = subdir[:-4]  # Remove "_rgb"
                images.setdefault(cam_name, {})
                
                rgb_images = []
                for idx in valid_indices:
                    img_path = os.path.join(subdir_path, f"{idx:06d}.png")
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Resize
                            img = cv2.resize(img, self.config.target_rgb_size)
                            rgb_images.append(img)
                        else:
                            rgb_images.append(np.zeros((*self.config.target_rgb_size[::-1], 3), dtype=np.uint8))
                    else:
                        rgb_images.append(np.zeros((*self.config.target_rgb_size[::-1], 3), dtype=np.uint8))
                
                if rgb_images:
                    images[cam_name]["rgb"] = np.stack(rgb_images)
            
            elif subdir.endswith("_depth"):
                cam_name = subdir[:-6]  # Remove "_depth"
                images.setdefault(cam_name, {})
                
                depth_images = []
                for idx in valid_indices:
                    img_path = os.path.join(subdir_path, f"{idx:06d}.png")
                    if os.path.exists(img_path):
                        depth = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                        if depth is not None:
                            # Resize
                            depth = cv2.resize(depth, self.config.target_depth_size, 
                                             interpolation=cv2.INTER_NEAREST)
                            depth_images.append(depth)
                        else:
                            depth_images.append(np.zeros(self.config.target_depth_size[::-1], dtype=np.uint16))
                    else:
                        depth_images.append(np.zeros(self.config.target_depth_size[::-1], dtype=np.uint16))
                
                if depth_images:
                    images[cam_name]["depth"] = np.stack(depth_images)
        
        return images
    
    def _compute_statistics(self) -> Dict:
        """Compute normalization statistics from all processed data"""
        
        all_jp = np.concatenate(self.all_joint_pos)
        all_jv = np.concatenate(self.all_joint_vel)
        all_gp = np.concatenate(self.all_gripper_pos)
        all_act = np.concatenate(self.all_actions)
        
        stats = {
            "joint_pos": {
                "mean": all_jp.mean(axis=0).tolist(),
                "std": all_jp.std(axis=0).tolist(),
                "min": all_jp.min(axis=0).tolist(),
                "max": all_jp.max(axis=0).tolist(),
            },
            "joint_vel": {
                "mean": all_jv.mean(axis=0).tolist(),
                "std": all_jv.std(axis=0).tolist(),
                "min": all_jv.min(axis=0).tolist(),
                "max": all_jv.max(axis=0).tolist(),
            },
            "gripper_pos": {
                "mean": float(all_gp.mean()),
                "std": float(all_gp.std()),
                "min": float(all_gp.min()),
                "max": float(all_gp.max()),
            },
            "actions": {
                "mean": all_act.mean(axis=0).tolist(),
                "std": all_act.std(axis=0).tolist(),
                "min": all_act.min(axis=0).tolist(),
                "max": all_act.max(axis=0).tolist(),
            }
        }
        
        return stats
    
    def _save_consolidated_hdf5(self, h5_path: str, episodes: List[Dict]):
        """Save all episodes to a single HDF5 file (ManiSkill format)"""
        
        with h5py.File(h5_path, 'w') as f:
            for i, ep in enumerate(tqdm(episodes, desc="Writing HDF5")):
                grp = f.create_group(f"traj_{i}")
                
                # Observations
                obs = grp.create_group("obs")
                obs.create_dataset("joint_pos", data=ep["joint_pos"], compression="gzip")
                obs.create_dataset("joint_vel", data=ep["joint_vel"], compression="gzip")
                obs.create_dataset("gripper_pos", data=ep["gripper_pos"], compression="gzip")
                
                # Save timestamps for proper replay timing
                if "timestamps" in ep:
                    obs.create_dataset("timestamps", data=ep["timestamps"], compression="gzip")
                
                # Images (optional, can be large)
                if self.config.store_images_in_hdf5 and ep.get("images"):
                    img_grp = obs.create_group("images")
                    for cam_name, cam_data in ep["images"].items():
                        cam_grp = img_grp.create_group(cam_name)
                        if "rgb" in cam_data:
                            cam_grp.create_dataset("rgb", data=cam_data["rgb"], 
                                                  compression="gzip", compression_opts=4)
                        if "depth" in cam_data:
                            cam_grp.create_dataset("depth", data=cam_data["depth"],
                                                  compression="gzip", compression_opts=4)
                
                # Actions
                grp.create_dataset("actions", data=ep["actions"], compression="gzip")
                
                # Metadata
                grp.attrs["success"] = ep["success"]
                grp.attrs["num_steps"] = ep["num_steps"]
                grp.attrs["episode_id"] = ep["episode_id"]
                
                # ====================================================================
                # TIMING METADATA - Critical for trajectory replay
                # ====================================================================
                # These attributes store the RECORDING timing, NOT the control loop timing.
                # - Recording frequency: ~30Hz (dt ≈ 33ms) - rate at which frames were saved
                # - Control frequency:   ~500Hz (dt ≈ 2ms) - internal robot control loop (DO NOT USE FOR REPLAY)
                # 
                # When replaying, we must use record_dt_mean to match the original recording speed.
                # Using controller_dt would replay ~16x faster than intended!
                # ====================================================================
                
                if "timestamps" in ep and len(ep["timestamps"]) > 1:
                    dt_array = np.diff(ep["timestamps"])
                    
                    # Frame timing statistics
                    grp.attrs["record_dt_mean"] = float(dt_array.mean())      # Average time between frames
                    grp.attrs["record_dt_std"] = float(dt_array.std())        # Timing jitter
                    grp.attrs["record_fps_nominal"] = float(1.0 / dt_array.mean())  # Effective FPS
                    
                    # Legacy attributes for compatibility (will be deprecated)
                    grp.attrs["control_dt"] = float(dt_array.mean())
                    grp.attrs["control_freq"] = float(1.0 / dt_array.mean())
                else:
                    print(f"Warning: Episode {i} has no timestamps, using default 30Hz")
                    grp.attrs["record_dt_mean"] = 1.0 / 30.0
                    grp.attrs["record_dt_std"] = 0.0
                    grp.attrs["record_fps_nominal"] = 30.0
        
        print(f"Saved HDF5 to {h5_path}")
    
    def _save_metadata_json(self, json_path: str, episodes: List[Dict], env_id: str):
        """Save metadata in ManiSkill-compatible JSON format"""
        
        metadata = {
            "env_info": {
                "env_id": env_id,
                "max_episode_steps": max(ep["num_steps"] for ep in episodes),
                "env_kwargs": {}
            },
            "source_type": "teleoperation",
            "source_desc": "ARX5 teleoperation data collected with UArm",
            "episodes": [
                {
                    "episode_id": ep["episode_id"],
                    "reset_kwargs": {},
                    "control_mode": "joint_pos",
                    "elapsed_steps": ep["num_steps"],
                    "info": {
                        "success": ep["success"]
                    }
                }
                for ep in episodes
            ]
        }
        
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess ARX5 teleoperation data")
    
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input directory (session directory)")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output directory for processed data")
    parser.add_argument("--all-sessions", action="store_true",
                       help="Process all sessions in input directory")
    
    # Image settings
    parser.add_argument("--rgb-size", type=str, default="256x256",
                       help="Target RGB image size (WxH)")
    parser.add_argument("--depth-size", type=str, default="256x256",
                       help="Target depth image size (WxH)")
    parser.add_argument("--no-images", action="store_true",
                       help="Don't include images in HDF5 (saves space)")
    
    # Cleaning settings
    parser.add_argument("--min-length", type=int, default=10,
                       help="Minimum episode length to keep")
    parser.add_argument("--clean", action="store_true",
                       help="Enable trajectory cleaning (remove static frames and velocity outliers)")
    parser.add_argument("--no-smooth", action="store_true",
                       help="Disable trajectory smoothing")
    parser.add_argument("--remove-static-frames", action="store_true",
                       help="Remove static frames where robot didn't move")
    
    # Subsampling
    parser.add_argument("--subsample", type=int, default=1,
                       help="Subsample factor (keep every N-th frame)")
    
    args = parser.parse_args()
    
    # Parse sizes
    def parse_size(s):
        w, h = map(int, s.lower().split('x'))
        return (w, h)
    
    config = PreprocessConfig(
        target_rgb_size=parse_size(args.rgb_size),
        target_depth_size=parse_size(args.depth_size),
        store_images_in_hdf5=not args.no_images,
        min_episode_length=args.min_length,
        remove_static_frames=args.remove_static_frames,  # Only enable if --clean is specified
        remove_velocity_outliers=args.clean,  # Only enable if --clean is specified
        smooth_trajectory=not args.no_smooth,
        subsample=args.subsample > 1,
        subsample_factor=args.subsample,
    )
    
    preprocessor = DataPreprocessor(config)
    
    input_dir = os.path.expanduser(args.input)
    output_dir = os.path.expanduser(args.output)
    
    if args.all_sessions:
        # Process all session directories
        session_dirs = sorted(glob.glob(os.path.join(input_dir, "*")))
        session_dirs = [d for d in session_dirs if os.path.isdir(d)]
        
        for session_dir in session_dirs:
            session_name = os.path.basename(session_dir)
            session_output = os.path.join(output_dir, session_name)
            preprocessor.process_session(session_dir, session_output, session_name)
    else:
        # Process single session
        preprocessor.process_session(input_dir, output_dir)


if __name__ == "__main__":
    main()
