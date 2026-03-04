#!/usr/bin/env python3
"""
D_syn+ 数据准备脚本:
1. VAE 解码合成轨迹 latent → RGB
2. 转换为 raw ManiSkill H5 格式
3. 创建合并 H5 (demos + D_syn+)

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/vlaw/prepare_dsyn_plus_combined.py
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ============================================================================
# Configuration
# ============================================================================

# D_syn+ trajectory indices (α=0.4 threshold)
DSYN_PLUS_002A = ["traj_0043", "traj_0073", "traj_0104", "traj_0116", "traj_0146", "traj_0166"]
DSYN_PLUS_002B = ["traj_0043", "traj_0099", "traj_0116", "traj_0146", "traj_0166", "traj_0171", "traj_0199"]

# Paths
SYNTHETIC_002A = "data/vlaw/synthetic/iter1_002a_aligned/synthetic_iter1_final_1772461191.h5"
SYNTHETIC_002B = "data/vlaw/synthetic/iter1_002b_sliding/synthetic_iter1_final_1772418849.h5"
RAW_DEMO_PATH = os.path.expanduser("~/.maniskill/demos/LiftPegUpright-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5")
VLM_LABELS_002A = "data/vlaw/labeled/synthetic_iter1_002a_aligned/LiftPegUpright-v1_vlm_rewards.h5"
VLM_LABELS_002B = "data/vlaw/labeled/synthetic_iter1_002b_sliding/LiftPegUpright-v1_vlm_rewards.h5"

# Output paths
OUTPUT_DIR_100 = "data/vlaw/combined/flywheel_b_100demos"
OUTPUT_DIR_669 = "data/vlaw/combined/flywheel_b_669demos"

# VAE config
VAE_SCALE_FACTOR = 0.18215  # SVD VAE scaling factor
TARGET_RGB_SIZE = 128  # Raw ManiSkill demo image size
DECODE_CHUNK_SIZE = 8  # Decode in chunks to manage VRAM


# ============================================================================
# VAE Decode
# ============================================================================

def load_vae(device: str = "cuda") -> torch.nn.Module:
    """Load the SVD VAE decoder from HuggingFace cache."""
    from diffusers import AutoencoderKL
    
    # The SVD pipeline uses the same VAE as sd-vae-ft-mse
    # Try loading from local cache first
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    # Load from SVD pipeline's VAE subcomponent
    print("Loading VAE from stabilityai/stable-video-diffusion-img2vid-xt (vae subfolder)...")
    try:
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            subfolder="vae",
            local_files_only=True,
        )
    except Exception as e:
        print(f"SVD VAE not found in cache, trying sd-vae-ft-mse: {e}")
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            local_files_only=True,
        )
    
    vae = vae.to(device).to(torch.float16)
    vae.eval()
    print(f"VAE loaded: scaling_factor={vae.config.scaling_factor}, "
          f"latent_channels={vae.config.latent_channels}")
    return vae


def decode_latents_to_rgb(
    vae: torch.nn.Module,
    latents: np.ndarray,
    device: str = "cuda",
    chunk_size: int = DECODE_CHUNK_SIZE,
) -> np.ndarray:
    """Decode VAE latents to RGB images.
    
    Args:
        vae: AutoencoderKL model
        latents: (T, 4, 48, 24) float16 latents
        device: CUDA device
        chunk_size: Batch size for decoding
        
    Returns:
        rgb: (T, 128, 128, 3) uint8 RGB images
    """
    T = latents.shape[0]
    all_rgb = []
    
    for i in range(0, T, chunk_size):
        chunk = torch.from_numpy(latents[i:i+chunk_size]).to(device).to(torch.float16)
        
        # Scale latents for VAE decoder
        chunk = chunk / vae.config.scaling_factor
        
        with torch.no_grad():
            # Decode: input (B, 4, 48, 24) → output (B, 3, 384, 192)
            decoded = vae.decode(chunk).sample
        
        # Post-process: [-1, 1] → [0, 1] → [0, 255]
        decoded = (decoded / 2.0 + 0.5).clamp(0, 1)
        
        # The decoded 384×192 is two 192×192 images stacked vertically
        # (base_camera on top, render_camera on bottom)
        # For raw ManiSkill format, we only need base_camera (single camera)
        # Take top half: (B, 3, 0:192, 0:192)
        base_cam = decoded[:, :, :192, :]  # (B, 3, 192, 192)
        
        # Resize 192×192 → 128×128 to match raw ManiSkill demo resolution
        base_cam = F.interpolate(
            base_cam.float(), size=(TARGET_RGB_SIZE, TARGET_RGB_SIZE),
            mode='bilinear', align_corners=False,
        )
        
        # Convert to uint8 NHWC
        base_cam = (base_cam * 255).byte()
        base_cam = base_cam.permute(0, 2, 3, 1).cpu().numpy()  # (B, 128, 128, 3)
        all_rgb.append(base_cam)
    
    return np.concatenate(all_rgb, axis=0)  # (T, 128, 128, 3)


# ============================================================================
# H5 Format Conversion
# ============================================================================

def convert_synthetic_to_maniskill_format(
    latent: np.ndarray,
    state: np.ndarray,
    actions: np.ndarray,
    rgb: np.ndarray,
    vlm_yes_prob: float,
) -> Dict:
    """Convert decoded synthetic trajectory to raw ManiSkill H5 format.
    
    Raw ManiSkill format: T_obs = T_actions + 1
    Synthetic has T_obs = T_actions, so we use obs[:T] and actions[:T-1]
    
    Args:
        latent: (T, 4, 48, 24) - unused here but kept for reference
        state: (T, 25) float32 - [qpos(9), qvel(9), tcp_pose(7)]
        actions: (T, 7) float32
        rgb: (T, 128, 128, 3) uint8 - decoded from latent
        vlm_yes_prob: VLM score for this trajectory
        
    Returns:
        Dict with raw ManiSkill trajectory structure
    """
    T = state.shape[0]
    T_actions = T - 1  # Match T+1/T convention
    
    # Split state into components
    qpos = state[:T, :9].astype(np.float32)       # (T, 9)
    qvel = state[:T, 9:18].astype(np.float32)      # (T, 9)
    tcp_pose = state[:T, 18:25].astype(np.float32)  # (T, 7)
    
    # Observations (T timesteps)
    obs = {
        "agent": {
            "qpos": qpos,           # (T, 9)
            "qvel": qvel,           # (T, 9)
        },
        "extra": {
            "tcp_pose": tcp_pose,   # (T, 7)
        },
        "sensor_data": {
            "base_camera": {
                "rgb": rgb[:T],     # (T, 128, 128, 3)
            },
        },
    }
    
    # Actions (T-1 timesteps)
    act = actions[:T_actions].astype(np.float32)  # (T-1, 7)
    
    # Success / rewards / terminated / truncated (T-1 timesteps)
    # Since VLM approved (p_yes > 0.4), mark as success
    success = np.ones(T_actions, dtype=bool)
    rewards = np.ones(T_actions, dtype=np.float32)
    terminated = np.zeros(T_actions, dtype=bool)
    truncated = np.zeros(T_actions, dtype=bool)
    
    return {
        "obs": obs,
        "actions": act,
        "success": success,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
    }


def write_traj_to_h5(group: h5py.Group, traj_data: Dict, prefix: str = ""):
    """Recursively write trajectory data to H5 group."""
    for key, value in traj_data.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            write_traj_to_h5(subgroup, value, prefix=f"{prefix}{key}/")
        elif isinstance(value, np.ndarray):
            group.create_dataset(key, data=value, compression="gzip", compression_opts=4)
        else:
            raise ValueError(f"Unexpected type {type(value)} for key {prefix}{key}")


# ============================================================================
# Main Pipeline
# ============================================================================

def step2_decode_dsyn_plus(device: str = "cuda") -> Tuple[List[Dict], List[Dict]]:
    """Step 2: Decode D_syn+ latents to RGB using VAE.
    
    Returns:
        decoded_002a: List of dicts with rgb, state, actions, vlm_yes_prob
        decoded_002b: List of dicts with rgb, state, actions, vlm_yes_prob
    """
    print("\n" + "="*60)
    print("Step 2: VAE Decode D_syn+ latents → RGB")
    print("="*60)
    
    vae = load_vae(device)
    
    results = {}
    for label, syn_path, vlm_path, traj_ids in [
        ("002a", SYNTHETIC_002A, VLM_LABELS_002A, DSYN_PLUS_002A),
        ("002b", SYNTHETIC_002B, VLM_LABELS_002B, DSYN_PLUS_002B),
    ]:
        print(f"\n--- Processing {label}: {len(traj_ids)} trajectories ---")
        decoded_trajs = []
        
        with h5py.File(syn_path, 'r') as syn_f, h5py.File(vlm_path, 'r') as vlm_f:
            for traj_id in traj_ids:
                print(f"  Decoding {traj_id}...", end=" ")
                t0 = time.time()
                
                latent = syn_f[traj_id]["latent"][:]     # (60, 4, 48, 24) float16
                state = syn_f[traj_id]["state"][:]       # (60, 25) float32
                actions = syn_f[traj_id]["actions"][:]   # (60, 7) float32
                vlm_yes_prob = float(vlm_f[traj_id].attrs['vlm_yes_prob'])
                
                # VAE decode
                rgb = decode_latents_to_rgb(vae, latent, device=device)
                
                decoded_trajs.append({
                    "latent": latent,
                    "state": state,
                    "actions": actions,
                    "rgb": rgb,
                    "vlm_yes_prob": vlm_yes_prob,
                    "traj_id": traj_id,
                    "source": label,
                })
                
                dt = time.time() - t0
                print(f"rgb={rgb.shape}, range=[{rgb.min()},{rgb.max()}], p_yes={vlm_yes_prob:.3f}, {dt:.1f}s")
        
        results[label] = decoded_trajs
    
    # Cleanup VAE
    del vae
    torch.cuda.empty_cache()
    
    return results["002a"], results["002b"]


def step3_create_combined_h5(
    decoded_002a: List[Dict],
    decoded_002b: List[Dict],
    num_demos_list: List[int] = [100, 669],
):
    """Step 3: Create combined H5 files (demos + D_syn+).
    
    Merges D_syn+ from both 002a and 002b (total 13 unique trajectories).
    Creates combined H5 for each demo count.
    """
    print("\n" + "="*60)
    print("Step 3: Create combined H5 files")
    print("="*60)
    
    # Merge all D_syn+ (002a: 6 trajs, 002b: 7 trajs = 13 total)
    all_dsyn_plus = decoded_002a + decoded_002b
    print(f"Total D_syn+ trajectories: {len(all_dsyn_plus)}")
    print(f"  002a: {[t['traj_id'] for t in decoded_002a]}")
    print(f"  002b: {[t['traj_id'] for t in decoded_002b]}")
    
    # Convert all to ManiSkill format
    converted_trajs = []
    for traj in all_dsyn_plus:
        converted = convert_synthetic_to_maniskill_format(
            latent=traj["latent"],
            state=traj["state"],
            actions=traj["actions"],
            rgb=traj["rgb"],
            vlm_yes_prob=traj["vlm_yes_prob"],
        )
        converted["_meta"] = {
            "source": f"dsyn_plus_{traj['source']}",
            "orig_traj_id": traj["traj_id"],
            "vlm_yes_prob": traj["vlm_yes_prob"],
        }
        converted_trajs.append(converted)
    
    # For each demo count, create combined H5
    for num_demos in num_demos_list:
        output_dir = f"data/vlaw/combined/flywheel_b_{num_demos}demos"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "combined.h5")
        
        print(f"\n--- Creating {output_path} ({num_demos} demos + {len(converted_trajs)} D_syn+) ---")
        
        with h5py.File(output_path, 'w') as out_f:
            traj_count = 0
            
            # First: copy demos from raw ManiSkill H5
            print(f"  Copying {num_demos} demos from raw ManiSkill H5...")
            with h5py.File(RAW_DEMO_PATH, 'r') as demo_f:
                demo_keys = sorted(
                    [k for k in demo_f.keys() if k.startswith('traj')],
                    key=lambda x: int(x.split('_')[-1])
                )[:num_demos]
                
                for dk in demo_keys:
                    out_key = f"traj_{traj_count}"
                    demo_f.copy(dk, out_f, name=out_key)
                    traj_count += 1
                
                print(f"  Copied {len(demo_keys)} demo trajectories")
            
            # Then: add D_syn+ trajectories
            print(f"  Adding {len(converted_trajs)} D_syn+ trajectories...")
            for conv_traj in converted_trajs:
                out_key = f"traj_{traj_count}"
                traj_group = out_f.create_group(out_key)
                
                # Write trajectory data (excluding _meta)
                traj_data = {k: v for k, v in conv_traj.items() if k != "_meta"}
                write_traj_to_h5(traj_group, traj_data)
                
                # Store metadata as attributes
                meta = conv_traj["_meta"]
                traj_group.attrs["source"] = meta["source"]
                traj_group.attrs["orig_traj_id"] = meta["orig_traj_id"]
                traj_group.attrs["vlm_yes_prob"] = meta["vlm_yes_prob"]
                
                traj_count += 1
            
            print(f"  Total trajectories in combined H5: {traj_count}")
        
        # Verify file
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  File size: {file_size:.1f} MB")


def step4_validate(num_demos_list: List[int] = [100, 669]):
    """Step 4: Validate combined H5 files can be loaded by training code."""
    print("\n" + "="*60)
    print("Step 4: Validate combined H5 files")
    print("="*60)
    
    for num_demos in num_demos_list:
        output_path = f"data/vlaw/combined/flywheel_b_{num_demos}demos/combined.h5"
        print(f"\n--- Validating {output_path} ---")
        
        with h5py.File(output_path, 'r') as f:
            traj_keys = sorted(
                [k for k in f.keys() if k.startswith('traj')],
                key=lambda x: int(x.split('_')[-1])
            )
            print(f"  Total trajectories: {len(traj_keys)}")
            
            # Check a demo trajectory
            demo_traj = f[traj_keys[0]]
            print(f"  Demo traj ({traj_keys[0]}):")
            print(f"    obs/agent/qpos: {demo_traj['obs/agent/qpos'].shape}")
            print(f"    obs/sensor_data/base_camera/rgb: {demo_traj['obs/sensor_data/base_camera/rgb'].shape}")
            print(f"    actions: {demo_traj['actions'].shape}")
            obs_t = demo_traj['obs/agent/qpos'].shape[0]
            act_t = demo_traj['actions'].shape[0]
            assert obs_t == act_t + 1, f"T+1/T mismatch: obs={obs_t}, actions={act_t}"
            print(f"    ✓ T+1/T format correct (obs={obs_t}, actions={act_t})")
            
            # Check a D_syn+ trajectory (last one)
            syn_key = traj_keys[-1]
            syn_traj = f[syn_key]
            print(f"  D_syn+ traj ({syn_key}):")
            print(f"    obs/agent/qpos: {syn_traj['obs/agent/qpos'].shape}")
            print(f"    obs/sensor_data/base_camera/rgb: {syn_traj['obs/sensor_data/base_camera/rgb'].shape}")
            print(f"    actions: {syn_traj['actions'].shape}")
            obs_t = syn_traj['obs/agent/qpos'].shape[0]
            act_t = syn_traj['actions'].shape[0]
            assert obs_t == act_t + 1, f"T+1/T mismatch: obs={obs_t}, actions={act_t}"
            print(f"    ✓ T+1/T format correct (obs={obs_t}, actions={act_t})")
            
            # Check rgb range
            rgb = syn_traj['obs/sensor_data/base_camera/rgb'][:]
            print(f"    RGB range: [{rgb.min()}, {rgb.max()}], dtype={rgb.dtype}")
            assert rgb.dtype == np.uint8
            print(f"    ✓ RGB dtype correct (uint8)")
            
            # Check source attribute
            if 'source' in syn_traj.attrs:
                print(f"    source: {syn_traj.attrs['source']}")
                print(f"    vlm_yes_prob: {syn_traj.attrs['vlm_yes_prob']:.4f}")
            
            # Count demo vs synthetic
            n_demo = 0
            n_syn = 0
            for k in traj_keys:
                if 'source' in f[k].attrs and 'dsyn' in f[k].attrs['source']:
                    n_syn += 1
                else:
                    n_demo += 1
            print(f"  Summary: {n_demo} demos + {n_syn} D_syn+ = {n_demo + n_syn} total")
    
    # Quick load test via the actual training code's data_utils
    print("\n--- Testing load with data_utils.load_traj_hdf5 ---")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rlft.datasets.data_utils import load_traj_hdf5
    for num_demos in num_demos_list:
        output_path = f"data/vlaw/combined/flywheel_b_{num_demos}demos/combined.h5"
        data = load_traj_hdf5(output_path, num_traj=min(5, num_demos + 13))
        # Check 'obs' key exists in loaded data
        first_key = sorted(data.keys())[0]
        assert "obs" in data[first_key], f"Missing 'obs' key in loaded data"
        assert "actions" in data[first_key], f"Missing 'actions' key in loaded data"
        print(f"  ✓ {output_path}: load_traj_hdf5 passed, "
              f"first traj has keys: {list(data[first_key].keys())}")
    
    print("\n✅ All validations passed!")


def save_decoded_frames_sample(decoded_trajs: List[Dict], output_dir: str, max_frames: int = 5):
    """Save a few decoded frames as PNG for visual inspection."""
    os.makedirs(output_dir, exist_ok=True)
    for traj in decoded_trajs:
        traj_id = traj["traj_id"]
        source = traj["source"]
        rgb = traj["rgb"]
        for frame_idx in [0, rgb.shape[0]//4, rgb.shape[0]//2, 3*rgb.shape[0]//4, rgb.shape[0]-1]:
            img = Image.fromarray(rgb[frame_idx])
            img.save(os.path.join(output_dir, f"{source}_{traj_id}_frame{frame_idx:03d}.png"))
    print(f"  Saved sample frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="D_syn+ data preparation")
    parser.add_argument("--device", default="cuda", help="CUDA device")
    parser.add_argument("--skip-decode", action="store_true", help="Skip VAE decoding (use cached)")
    parser.add_argument("--step", type=int, default=0, help="Start from step (2/3/4)")
    args = parser.parse_args()
    
    print(f"Device: {args.device}")
    print(f"D_syn+ counts: 002a={len(DSYN_PLUS_002A)}, 002b={len(DSYN_PLUS_002B)}, total=13")
    
    # Step 2: VAE Decode
    decoded_002a, decoded_002b = step2_decode_dsyn_plus(args.device)
    
    # Save sample frames
    sample_dir = "results/vlaw/dsyn_plus_decoded_frames"
    save_decoded_frames_sample(decoded_002a + decoded_002b, sample_dir)
    
    # Step 3: Create combined H5
    step3_create_combined_h5(decoded_002a, decoded_002b)
    
    # Step 4: Validate
    step4_validate()
    
    print("\n" + "="*60)
    print("✅ All steps completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
