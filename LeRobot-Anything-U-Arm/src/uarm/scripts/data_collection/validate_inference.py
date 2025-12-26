#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate Policy Inference Script

This script validates the arx5_policy_inference.py by:
1. Loading a recorded dataset (rlft trajectory.h5 format)
2. Running inference on dataset observations
3. Comparing predicted actions with ground truth actions
4. Identifying potential issues

Usage:
    python -m data_collection.validate_inference \
        --checkpoint ~/rlft/.../checkpoints/iter_42000.pt \
        --dataset ~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5
"""

import argparse
import os
import sys
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import h5py
import cv2
import torch

# Add rlft to path
RLFT_PATH = os.path.expanduser("~/rlft/diffusion_policy")
if RLFT_PATH not in sys.path:
    sys.path.insert(0, RLFT_PATH)


@dataclass
class ValidationConfig:
    """Config matching arx5_policy_inference.py"""
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    image_size: Tuple[int, int] = (128, 128)  # (H, W)


def load_rlft_dataset(dataset_path: str, traj_idx: int = 0) -> Dict:
    """
    Load rlft trajectory.h5 dataset.
    
    Format:
        traj_0/
            actions: [T, 7]
            obs/
                images/
                    wrist/
                        rgb: [T, H, W, 3]
                        depth: [T, H, W]
                    external/
                        rgb: [T, H, W, 3]
                        depth: [T, H, W]
                joint_pos: [T, 6]
                joint_vel: [T, 6]
                gripper_pos: [T, 1]
    """
    print(f"\nLoading dataset: {dataset_path}")
    
    with h5py.File(dataset_path, 'r') as f:
        # List trajectories
        traj_keys = [k for k in f.keys() if k.startswith('traj_')]
        print(f"  Found {len(traj_keys)} trajectories")
        
        traj_key = f'traj_{traj_idx}'
        if traj_key not in f:
            traj_key = traj_keys[0]
        
        print(f"  Loading: {traj_key}")
        traj = f[traj_key]
        
        data = {}
        
        # Actions
        if 'actions' in traj:
            data['action'] = np.array(traj['actions'])
            print(f"    actions: {data['action'].shape}")
        
        # Observations
        if 'obs' in traj:
            obs = traj['obs']
            
            # Images - handle nested structure
            if 'images' in obs:
                images = obs['images']
                for cam in images.keys():
                    cam_group = images[cam]
                    if isinstance(cam_group, h5py.Group):
                        # Nested structure: images/wrist/rgb
                        if 'rgb' in cam_group:
                            data[f'image_{cam}'] = np.array(cam_group['rgb'])
                            print(f"    image_{cam}: {data[f'image_{cam}'].shape}")
                    else:
                        # Direct structure: images/wrist = rgb array
                        data[f'image_{cam}'] = np.array(cam_group)
                        print(f"    image_{cam}: {data[f'image_{cam}'].shape}")
            
            # Build state from components
            if 'joint_pos' in obs and 'joint_vel' in obs and 'gripper_pos' in obs:
                joint_pos = np.array(obs['joint_pos'])  # [T, 6]
                joint_vel = np.array(obs['joint_vel'])  # [T, 6]
                gripper_pos = np.array(obs['gripper_pos'])  # [T, 1]
                
                # Concatenate to [T, 13] state vector
                data['state'] = np.concatenate([
                    joint_pos,
                    joint_vel,
                    gripper_pos
                ], axis=1).astype(np.float32)
                print(f"    state: {data['state'].shape} (from joint_pos, joint_vel, gripper_pos)")
            elif 'state' in obs:
                data['state'] = np.array(obs['state'])
                print(f"    state: {data['state'].shape}")
    
    return data


def load_dataset(dataset_path: str, traj_idx: int = 0) -> Dict:
    """Load HDF5 dataset (auto-detect format)."""
    print(f"\nLoading dataset: {dataset_path}")
    
    with h5py.File(dataset_path, 'r') as f:
        keys = list(f.keys())
        
        # Check if rlft format (traj_0, traj_1, ...)
        if any(k.startswith('traj_') for k in keys):
            return load_rlft_dataset(dataset_path, traj_idx)
        
        # Otherwise try generic format
        data = {}
        
        # Load observations
        if 'observations' in f:
            obs_group = f['observations']
            
            # Images
            if 'images' in obs_group:
                img_group = obs_group['images']
                for cam_name in img_group.keys():
                    data[f'image_{cam_name}'] = np.array(img_group[cam_name])
                    print(f"  image_{cam_name}: {data[f'image_{cam_name}'].shape}")
            
            # State
            if 'state' in obs_group:
                data['state'] = np.array(obs_group['state'])
                print(f"  state: {data['state'].shape}")
            
            # Qpos/Qvel
            if 'qpos' in obs_group:
                data['qpos'] = np.array(obs_group['qpos'])
                print(f"  qpos: {data['qpos'].shape}")
            if 'qvel' in obs_group:
                data['qvel'] = np.array(obs_group['qvel'])
                print(f"  qvel: {data['qvel'].shape}")
        
        # Load actions
        if 'action' in f:
            data['action'] = np.array(f['action'])
            print(f"  action: {data['action'].shape}")
        
    return data


def build_state_vector(data: Dict, idx: int) -> np.ndarray:
    """
    Build 13D state vector from dataset.
    
    State format: [joint_pos(6), joint_vel(6), gripper_pos(1)]
    """
    if 'state' in data:
        # Already in correct format
        return data['state'][idx].astype(np.float32)
    
    # Build from qpos/qvel
    state = np.zeros(13, dtype=np.float32)
    
    if 'qpos' in data:
        qpos = data['qpos'][idx]
        state[:6] = qpos[:6]  # joint_pos
        if len(qpos) > 6:
            state[12] = qpos[6]  # gripper_pos
    
    if 'qvel' in data:
        qvel = data['qvel'][idx]
        state[6:12] = qvel[:6]  # joint_vel
    
    return state


def load_policy(checkpoint_path: str, config: ValidationConfig, device: str = "cuda"):
    """Load policy components (same as arx5_policy_inference.py)."""
    print(f"\nLoading policy from: {checkpoint_path}")
    
    from diffusion_policy.plain_conv import PlainConv
    from diffusion_policy.utils import StateEncoder
    from diffusion_policy.algorithms import (
        DiffusionPolicyAgent,
        FlowMatchingAgent,
        ConsistencyFlowAgent,
    )
    from diffusion_policy.algorithms.networks import VelocityUNet1D
    from diffusion_policy.conditional_unet1d import ConditionalUnet1D
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Detect algorithm
    agent_keys = list(ckpt.get('agent', ckpt.get('ema_agent', {})).keys())
    if any('velocity_net' in k for k in agent_keys):
        if any('velocity_net_ema' in k for k in agent_keys):
            algorithm = "consistency_flow"
        else:
            algorithm = "flow_matching"
    else:
        algorithm = "diffusion_policy"
    
    print(f"  Algorithm: {algorithm}")
    
    # Architecture params
    ACTION_DIM = 7
    STATE_DIM = 13
    visual_feature_dim = 256
    state_encoder_hidden_dim = 128
    state_encoder_out_dim = 256
    diffusion_step_embed_dim = 64
    unet_dims = (64, 128, 256)
    n_groups = 8
    
    global_cond_dim = config.obs_horizon * (visual_feature_dim + state_encoder_out_dim)
    
    # Create encoders
    visual_encoder = PlainConv(
        in_channels=3,
        out_dim=visual_feature_dim,
        pool_feature_map=True,
    ).to(device)
    
    state_encoder = StateEncoder(
        state_dim=STATE_DIM,
        hidden_dim=state_encoder_hidden_dim,
        out_dim=state_encoder_out_dim,
    ).to(device)
    
    # Create agent
    if algorithm == "consistency_flow":
        velocity_net = VelocityUNet1D(
            input_dim=ACTION_DIM,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_dims,
            n_groups=n_groups,
        )
        agent = ConsistencyFlowAgent(
            velocity_net=velocity_net,
            action_dim=ACTION_DIM,
            obs_horizon=config.obs_horizon,
            pred_horizon=config.pred_horizon,
            num_flow_steps=10,
            ema_decay=0.999,
            action_bounds=None,
            device=str(device),
        )
    elif algorithm == "flow_matching":
        velocity_net = VelocityUNet1D(
            input_dim=ACTION_DIM,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_dims,
            n_groups=n_groups,
        )
        agent = FlowMatchingAgent(
            velocity_net=velocity_net,
            action_dim=ACTION_DIM,
            obs_horizon=config.obs_horizon,
            pred_horizon=config.pred_horizon,
            num_flow_steps=10,
            action_bounds=None,
            device=str(device),
        )
    else:
        noise_pred_net = ConditionalUnet1D(
            input_dim=ACTION_DIM,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_dims,
            n_groups=n_groups,
        )
        agent = DiffusionPolicyAgent(
            noise_pred_net=noise_pred_net,
            action_dim=ACTION_DIM,
            obs_horizon=config.obs_horizon,
            pred_horizon=config.pred_horizon,
            num_diffusion_iters=100,
            device=str(device),
        )
    
    agent = agent.to(device)
    
    # Load weights
    agent_key = "ema_agent" if "ema_agent" in ckpt else "agent"
    agent.load_state_dict(ckpt[agent_key])
    visual_encoder.load_state_dict(ckpt["visual_encoder"])
    state_encoder.load_state_dict(ckpt["state_encoder"])
    
    # Eval mode
    agent.eval()
    visual_encoder.eval()
    state_encoder.eval()
    
    print(f"  Loaded weights from: {agent_key}")
    
    return {
        'agent': agent,
        'visual_encoder': visual_encoder,
        'state_encoder': state_encoder,
        'device': device,
        'algorithm': algorithm
    }


def run_inference(
    policy: Dict,
    rgb_stack: np.ndarray,  # [obs_horizon, H, W, 3] uint8
    state_stack: np.ndarray,  # [obs_horizon, 13] float32
    config: ValidationConfig,
    verbose: bool = True
) -> np.ndarray:
    """
    Run policy inference.
    
    Returns:
        actions: [act_horizon, 7] predicted actions
    """
    device = policy['device']
    
    # Preprocess RGB: resize, NCHW, normalize
    rgb_resized = []
    for i in range(len(rgb_stack)):
        rgb = cv2.resize(rgb_stack[i], 
                        (config.image_size[1], config.image_size[0]),
                        interpolation=cv2.INTER_LINEAR)
        rgb_resized.append(rgb)
    rgb_stack = np.stack(rgb_resized)  # [T, H, W, 3]
    
    rgb_nchw = np.transpose(rgb_stack, (0, 3, 1, 2))  # [T, 3, H, W]
    rgb_nchw = rgb_nchw.astype(np.float32) / 255.0
    
    if verbose:
        print(f"\n  Input RGB: shape={rgb_nchw.shape}, range=[{rgb_nchw.min():.3f}, {rgb_nchw.max():.3f}]")
        print(f"  Input State: shape={state_stack.shape}")
        print(f"    joint_pos: {state_stack[-1, :6]}")
        print(f"    joint_vel: {state_stack[-1, 6:12]}")
        print(f"    gripper:   {state_stack[-1, 12]:.4f}")
    
    # To tensors
    rgb = torch.from_numpy(rgb_nchw).unsqueeze(0).to(device)  # [1, T, C, H, W]
    state = torch.from_numpy(state_stack).unsqueeze(0).to(device)  # [1, T, 13]
    
    B, T = rgb.shape[0], rgb.shape[1]
    
    # Encode
    rgb_flat = rgb.view(B * T, *rgb.shape[2:])
    state_flat = state.view(B * T, -1)
    
    with torch.no_grad():
        visual_feat = policy['visual_encoder'](rgb_flat)
        visual_feat = visual_feat.view(B, T, -1)
        
        state_feat = policy['state_encoder'](state_flat)
        state_feat = state_feat.view(B, T, -1)
        
        if verbose:
            print(f"\n  Visual features: {visual_feat.shape}, mean={visual_feat.mean():.4f}, std={visual_feat.std():.4f}")
            print(f"  State features:  {state_feat.shape}, mean={state_feat.mean():.4f}, std={state_feat.std():.4f}")
        
        # Global conditioning
        obs_features = torch.cat([visual_feat, state_feat], dim=-1)
        obs_cond = obs_features.view(B, -1)
        
        if verbose:
            print(f"  Obs conditioning: {obs_cond.shape}, mean={obs_cond.mean():.4f}, std={obs_cond.std():.4f}")
        
        # Inference
        action_seq = policy['agent'].get_action(obs_cond)
        if isinstance(action_seq, tuple):
            action_seq = action_seq[0]
    
    action_seq = action_seq[0].cpu().numpy()  # [pred_horizon, 7]
    
    if verbose:
        print(f"\n  Raw action output: {action_seq.shape}")
        print(f"    Full sequence range: [{action_seq.min():.4f}, {action_seq.max():.4f}]")
        print(f"    Per-dim stats:")
        for i in range(7):
            print(f"      dim[{i}]: mean={action_seq[:, i].mean():.4f}, std={action_seq[:, i].std():.4f}, range=[{action_seq[:, i].min():.4f}, {action_seq[:, i].max():.4f}]")
    
    # Extract executable actions
    start = config.obs_horizon - 1
    end = start + config.act_horizon
    executable_actions = action_seq[start:end]
    
    return executable_actions, action_seq


def validate_on_dataset(
    policy: Dict,
    data: Dict,
    config: ValidationConfig,
    num_samples: int = 5,
    visualize: bool = True
):
    """Validate policy on dataset samples."""
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    
    # Get dataset length
    if 'action' in data:
        n_frames = len(data['action'])
    elif 'state' in data:
        n_frames = len(data['state'])
    else:
        print("ERROR: No action or state data found!")
        return []
    
    print(f"Dataset length: {n_frames} frames")
    
    # Sample indices (skip first few to have obs_horizon)
    sample_indices = np.linspace(
        config.obs_horizon, 
        min(n_frames - config.act_horizon - 1, n_frames - 10), 
        num_samples
    ).astype(int)
    
    all_errors = []
    
    for sample_idx, idx in enumerate(sample_indices):
        print(f"\n{'='*40}")
        print(f"Sample {sample_idx + 1}/{num_samples} (frame {idx})")
        print(f"{'='*40}")
        
        # Build observation stack
        rgb_stack = []
        state_stack = []
        
        for t in range(config.obs_horizon):
            obs_idx = idx - config.obs_horizon + 1 + t
            
            # Get image
            if 'image_wrist' in data:
                rgb = data['image_wrist'][obs_idx]
            elif 'image_external' in data:
                rgb = data['image_external'][obs_idx]
            else:
                # First available image key
                img_keys = [k for k in data.keys() if k.startswith('image_')]
                if img_keys:
                    rgb = data[img_keys[0]][obs_idx]
                else:
                    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Get state
            state = build_state_vector(data, obs_idx)
            
            rgb_stack.append(rgb)
            state_stack.append(state)
        
        rgb_stack = np.stack(rgb_stack)
        state_stack = np.stack(state_stack)
        
        # Run inference
        pred_actions, full_pred = run_inference(policy, rgb_stack, state_stack, config, verbose=True)
        
        # Get ground truth actions
        end_idx = min(idx + config.act_horizon, n_frames)
        gt_actions = data['action'][idx:end_idx]
        
        print(f"\n  Ground truth actions: {gt_actions.shape}")
        print(f"    Per-dim stats:")
        for i in range(min(7, gt_actions.shape[1])):
            print(f"      dim[{i}]: mean={gt_actions[:, i].mean():.4f}, range=[{gt_actions[:, i].min():.4f}, {gt_actions[:, i].max():.4f}]")
        
        # Compare
        min_len = min(len(pred_actions), len(gt_actions))
        pred_actions = pred_actions[:min_len]
        gt_actions = gt_actions[:min_len, :7]  # Take first 7 dims if more
        
        error = np.abs(pred_actions - gt_actions)
        mse = np.mean((pred_actions - gt_actions) ** 2)
        mae = np.mean(error)
        
        print(f"\n  Comparison:")
        print(f"    MSE: {mse:.6f}")
        print(f"    MAE: {mae:.6f}")
        print(f"    Per-dim MAE:")
        for i in range(7):
            print(f"      dim[{i}]: {error[:, i].mean():.4f}")
        
        all_errors.append({
            'idx': idx,
            'mse': mse,
            'mae': mae,
            'pred': pred_actions,
            'gt': gt_actions,
            'error': error
        })
        
        # Check if actions are too small
        pred_magnitude = np.abs(pred_actions).mean()
        gt_magnitude = np.abs(gt_actions).mean()
        print(f"\n  Action magnitude:")
        print(f"    Predicted: {pred_magnitude:.6f}")
        print(f"    Ground truth: {gt_magnitude:.6f}")
        print(f"    Ratio (pred/gt): {pred_magnitude / (gt_magnitude + 1e-8):.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    avg_mse = np.mean([e['mse'] for e in all_errors])
    avg_mae = np.mean([e['mae'] for e in all_errors])
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average MAE: {avg_mae:.6f}")
    
    # Visualize
    if visualize and len(all_errors) > 0:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            # Plot first sample
            err = all_errors[0]
            
            for i in range(7):
                ax = axes[i // 4, i % 4]
                ax.plot(err['gt'][:, i], 'b-', label='GT', linewidth=2)
                ax.plot(err['pred'][:, i], 'r--', label='Pred', linewidth=2)
                ax.set_title(f'Dim {i} ({"gripper" if i == 6 else f"joint{i}"})')
                ax.legend()
                ax.grid(True)
            
            axes[1, 3].axis('off')
            axes[1, 3].text(0.5, 0.5, 
                           f"MSE: {err['mse']:.6f}\nMAE: {err['mae']:.6f}",
                           ha='center', va='center', fontsize=14)
            
            plt.tight_layout()
            plt.savefig('/tmp/inference_validation.png', dpi=150)
            print(f"\nSaved plot to /tmp/inference_validation.png")
            plt.show()
        except Exception as e:
            print(f"Visualization error: {e}")
    
    return all_errors


def check_data_statistics(data: Dict):
    """Check dataset statistics."""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    if 'action' in data:
        actions = data['action']
        print(f"\nActions: {actions.shape}")
        print(f"  Range: [{actions.min():.4f}, {actions.max():.4f}]")
        print(f"  Mean:  {actions.mean():.4f}")
        print(f"  Std:   {actions.std():.4f}")
        
        print(f"\n  Per-dimension:")
        for i in range(min(7, actions.shape[1])):
            dim_name = "gripper" if i == 6 else f"joint{i}"
            print(f"    {dim_name}: mean={actions[:, i].mean():.4f}, std={actions[:, i].std():.4f}, range=[{actions[:, i].min():.4f}, {actions[:, i].max():.4f}]")
    
    if 'state' in data:
        states = data['state']
        print(f"\nStates: {states.shape}")
        print(f"  joint_pos[0:6]:  range=[{states[:, :6].min():.4f}, {states[:, :6].max():.4f}]")
        print(f"  joint_vel[6:12]: range=[{states[:, 6:12].min():.4f}, {states[:, 6:12].max():.4f}]")
        if states.shape[1] > 12:
            print(f"  gripper[12]:     range=[{states[:, 12].min():.4f}, {states[:, 12].max():.4f}]")
    
    if 'qpos' in data:
        qpos = data['qpos']
        print(f"\nQpos: {qpos.shape}")
        for i in range(min(7, qpos.shape[1])):
            print(f"  dim[{i}]: range=[{qpos[:, i].min():.4f}, {qpos[:, i].max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Validate Policy Inference")
    
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--dataset", "-d", type=str, 
                       default="~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5",
                       help="Path to HDF5 dataset file")
    parser.add_argument("--traj-idx", type=int, default=0,
                       help="Trajectory index to load")
    parser.add_argument("--num-samples", "-n", type=int, default=5,
                       help="Number of samples to validate")
    parser.add_argument("--no-viz", action="store_true",
                       help="Disable visualization")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for inference")
    
    args = parser.parse_args()
    
    # Load dataset
    data = load_dataset(os.path.expanduser(args.dataset), args.traj_idx)
    
    # Check statistics
    check_data_statistics(data)
    
    # Load config
    config = ValidationConfig()
    
    # Load policy
    policy = load_policy(
        os.path.expanduser(args.checkpoint),
        config,
        args.device
    )
    
    # Validate
    errors = validate_on_dataset(
        policy,
        data,
        config,
        num_samples=args.num_samples,
        visualize=not args.no_viz
    )
    
    # Diagnosis
    print("\n" + "=" * 60)
    print("Diagnosis")
    print("=" * 60)
    
    if len(errors) > 0:
        avg_pred_mag = np.mean([np.abs(e['pred']).mean() for e in errors])
        avg_gt_mag = np.mean([np.abs(e['gt']).mean() for e in errors])
        ratio = avg_pred_mag / (avg_gt_mag + 1e-8)
        
        print(f"\nAction magnitude analysis:")
        print(f"  Predicted avg magnitude: {avg_pred_mag:.6f}")
        print(f"  Ground truth avg magnitude: {avg_gt_mag:.6f}")
        print(f"  Ratio: {ratio:.4f}")
        
        if ratio < 0.1:
            print("\n⚠️  ISSUE: Predicted actions are much smaller than ground truth!")
            print("   Possible causes:")
            print("   1. Model not trained properly")
            print("   2. Action normalization mismatch")
            print("   3. Wrong checkpoint loaded")
        elif ratio > 10:
            print("\n⚠️  ISSUE: Predicted actions are much larger than ground truth!")
            print("   Possible causes:")
            print("   1. Action denormalization issue")
            print("   2. Wrong model architecture")
        elif np.mean([e['mae'] for e in errors]) > 0.5:
            print("\n⚠️  WARNING: High prediction error!")
            print("   Possible causes:")
            print("   1. Observation preprocessing mismatch")
            print("   2. State encoding issue")
            print("   3. Model undertrained")
        else:
            print("\n✓ Action magnitudes look reasonable")
            print("  If robot still doesn't move, check:")
            print("  1. Safety limits in apply_safety_limits()")
            print("  2. max_joint_delta setting (currently 0.1 rad)")
            print("  3. Action execution in execute_action()")


if __name__ == "__main__":
    main()
