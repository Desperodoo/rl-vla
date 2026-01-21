#!/usr/bin/env python3
"""
数据预处理对齐验证脚本

对比 train_carm.py (CARMDataset) 和 train_finetune_from_realbot.py (InferenceDataset)
的数据预处理是否对齐，确保 finetune 时数据格式一致。

检查项目：
1. 观测数据格式：rgb shape, state shape
2. Action 格式：连续动作维度，relative pose 计算
3. Gripper 离散化：阈值和 label 分布
4. 数值范围：各维度的统计特性
"""

import os
import sys
import numpy as np
import torch
import h5py
import glob
from typing import Dict, Any, Optional

# 添加路径
sys.path.insert(0, '/home/lizh/rl-vla/rlft/diffusion_policy')

from diffusion_policy.carm_utils import (
    create_carm_obs_process_fn,
    compute_relative_pose_transform,
)
from train_carm import CARMDataset
from train_finetune_from_realbot import InferenceDataset


def print_separator(title: str = ""):
    """打印分隔线"""
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def analyze_sample(sample: Dict[str, Any], name: str):
    """分析单个样本的结构和数值"""
    print(f"\n--- {name} ---")
    
    # Observations
    obs = sample['observations']
    rgb = obs['rgb']
    state = obs['state']
    
    print(f"  RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"  RGB range: [{rgb.float().min():.2f}, {rgb.float().max():.2f}]")
    print(f"  State shape: {state.shape}, dtype: {state.dtype}")
    print(f"  State (qpos_joint) range: [{state.min():.4f}, {state.max():.4f}]")
    print(f"  State mean: {state.float().mean():.4f}")
    
    # Actions
    actions_cont = sample['actions_cont']
    gripper_label = sample['gripper_label']
    
    print(f"\n  Actions_cont shape: {actions_cont.shape}, dtype: {actions_cont.dtype}")
    
    # 分维度分析
    if actions_cont.shape[-1] == 13:  # full mode
        joints = actions_cont[:, :6]
        rel_xyz = actions_cont[:, 6:9]
        rel_quat = actions_cont[:, 9:13]
        
        print(f"  Joints [0:6]:")
        print(f"    range: [{joints.min():.4f}, {joints.max():.4f}]")
        print(f"    mean: {joints.mean():.4f}, std: {joints.std():.4f}")
        
        print(f"  Relative XYZ [6:9]:")
        print(f"    range: [{rel_xyz.min():.6f}, {rel_xyz.max():.6f}]")
        print(f"    mean: {rel_xyz.mean():.6f}, std: {rel_xyz.std():.6f}")
        
        print(f"  Relative Quat [9:13]:")
        print(f"    range: [{rel_quat.min():.4f}, {rel_quat.max():.4f}]")
        print(f"    mean: {rel_quat.mean():.4f}, std: {rel_quat.std():.4f}")
    
    print(f"\n  Gripper_label shape: {gripper_label.shape}")
    print(f"  Gripper_label distribution: close={gripper_label.sum()}, open={len(gripper_label) - gripper_label.sum()}")
    
    return {
        'rgb_shape': tuple(rgb.shape),
        'rgb_range': (rgb.float().min().item(), rgb.float().max().item()),
        'state_shape': tuple(state.shape),
        'state_range': (state.min().item(), state.max().item()),
        'actions_shape': tuple(actions_cont.shape),
        'joints_stats': (actions_cont[:, :6].mean().item(), actions_cont[:, :6].std().item()) if actions_cont.shape[-1] >= 6 else None,
        'rel_xyz_stats': (actions_cont[:, 6:9].mean().item(), actions_cont[:, 6:9].std().item()) if actions_cont.shape[-1] >= 9 else None,
        'rel_quat_stats': (actions_cont[:, 9:13].mean().item(), actions_cont[:, 9:13].std().item()) if actions_cont.shape[-1] >= 13 else None,
        'gripper_close_ratio': gripper_label.sum().item() / len(gripper_label),
    }


def analyze_raw_data_diff():
    """分析原始数据的差异"""
    print_separator("原始数据格式对比")
    
    # 遥操作数据
    teleop_file = "/home/lizh/rl-vla/recorded_data/mix/episode_0001_20260112_223900.hdf5"
    # 真机推理数据
    inference_file = sorted(glob.glob("/home/lizh/rl-vla/inference_logs/inference_episode_*.hdf5"))[0]
    
    print(f"\n遥操作数据: {os.path.basename(teleop_file)}")
    with h5py.File(teleop_file, 'r') as f:
        print(f"  Keys: {list(f.keys())}")
        print(f"  observations keys: {list(f['observations'].keys())}")
        
        action = np.array(f['action'])
        qpos_end = np.array(f['observations/qpos_end'])
        
        print(f"\n  action shape: {action.shape}")
        print(f"  action[0, 7:14] (绝对 end_pose): {action[0, 7:14]}")
        print(f"  qpos_end[0, :7]: {qpos_end[0, :7]}")
        
        # 计算相对位姿 (模拟 CARMDataset 的处理)
        ref_pose = qpos_end[0, :7]
        target_pose = action[1, 7:14]  # 下一帧的目标位姿
        relative_pose = compute_relative_pose_transform(ref_pose, target_pose)
        print(f"\n  相对位姿 (computed): {relative_pose}")
        print(f"    rel_xyz: {relative_pose[:3]}")
        print(f"    rel_quat: {relative_pose[3:]}")
    
    print(f"\n真机推理数据: {os.path.basename(inference_file)}")
    with h5py.File(inference_file, 'r') as f:
        print(f"  Keys: {list(f.keys())}")
        print(f"  observations keys: {list(f['observations'].keys())}")
        
        if 'action_intervened' in f:
            action = np.array(f['action_intervened'])[:, 0, :]  # [T, 15]
        else:
            action = np.array(f['action'])
        
        qpos_end = np.array(f['observations/qpos_end'])
        
        print(f"\n  action shape (from action_intervened[:, 0, :]): {action.shape}")
        print(f"  action[0, 7:14] (已经是相对位姿): {action[0, 7:14]}")
        print(f"  qpos_end[0, :7]: {qpos_end[0, :7]}")
        print(f"    rel_xyz: {action[0, 7:10]}")
        print(f"    rel_quat: {action[0, 10:14]}")


def compare_datasets():
    """对比两个 Dataset 的输出"""
    print_separator("Dataset 输出对比")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建相同的 obs_process_fn
    obs_process_fn = create_carm_obs_process_fn(
        output_format="NCHW",
        target_size=(128, 128),
        normalize_images=True,
    )
    
    # 加载 CARMDataset (遥操作数据)
    print("\n正在加载 CARMDataset (遥操作数据)...")
    teleop_dataset = CARMDataset(
        data_path="/home/lizh/rl-vla/recorded_data/mix",
        obs_process_fn=obs_process_fn,
        device=device,
        num_episodes=2,  # 只加载 2 个 episode 加速测试
        obs_horizon=2,
        pred_horizon=16,
        action_mode="full",
        precompute_actions=False,
        action_normalizer=None,
        gripper_threshold=0.05,
    )
    
    # 加载 InferenceDataset (真机推理数据)
    print("\n正在加载 InferenceDataset (真机推理数据)...")
    inference_dataset = InferenceDataset(
        data_path="/home/lizh/rl-vla/inference_logs",
        obs_process_fn=obs_process_fn,
        device=device,
        num_episodes=2,
        obs_horizon=2,
        pred_horizon=16,
        action_mode="full",
        use_intervened_action=True,
        gripper_threshold=0.05,
        action_normalizer=None,
    )
    
    print(f"\nCARMDataset: {len(teleop_dataset)} 个样本")
    print(f"InferenceDataset: {len(inference_dataset)} 个样本")
    
    # 分析样本
    print_separator("样本结构对比")
    
    teleop_sample = teleop_dataset[0]
    inference_sample = inference_dataset[0]
    
    teleop_stats = analyze_sample(teleop_sample, "CARMDataset (遥操作)")
    inference_stats = analyze_sample(inference_sample, "InferenceDataset (真机推理)")
    
    # 对比
    print_separator("关键差异检查")
    
    issues = []
    
    # 1. 检查形状是否一致
    if teleop_stats['rgb_shape'] != inference_stats['rgb_shape']:
        issues.append(f"RGB shape 不一致: {teleop_stats['rgb_shape']} vs {inference_stats['rgb_shape']}")
    else:
        print(f"✓ RGB shape 一致: {teleop_stats['rgb_shape']}")
    
    if teleop_stats['state_shape'] != inference_stats['state_shape']:
        issues.append(f"State shape 不一致: {teleop_stats['state_shape']} vs {inference_stats['state_shape']}")
    else:
        print(f"✓ State shape 一致: {teleop_stats['state_shape']}")
    
    if teleop_stats['actions_shape'] != inference_stats['actions_shape']:
        issues.append(f"Actions shape 不一致: {teleop_stats['actions_shape']} vs {inference_stats['actions_shape']}")
    else:
        print(f"✓ Actions shape 一致: {teleop_stats['actions_shape']}")
    
    # 2. 检查数值范围
    rgb_range_diff = abs(teleop_stats['rgb_range'][1] - inference_stats['rgb_range'][1])
    if rgb_range_diff > 1:
        issues.append(f"RGB range 差异较大: {teleop_stats['rgb_range']} vs {inference_stats['rgb_range']}")
    else:
        print(f"✓ RGB range 相近: teleop={teleop_stats['rgb_range']}, inference={inference_stats['rgb_range']}")
    
    # 3. 检查 relative pose 数量级
    if teleop_stats['rel_xyz_stats'] and inference_stats['rel_xyz_stats']:
        teleop_xyz_std = teleop_stats['rel_xyz_stats'][1]
        inference_xyz_std = inference_stats['rel_xyz_stats'][1]
        ratio = inference_xyz_std / teleop_xyz_std if teleop_xyz_std > 1e-8 else float('inf')
        
        if ratio > 5 or ratio < 0.2:
            issues.append(f"Relative XYZ std 差异较大: teleop={teleop_xyz_std:.6f}, inference={inference_xyz_std:.6f}, ratio={ratio:.2f}")
        else:
            print(f"✓ Relative XYZ std 数量级相近: teleop={teleop_xyz_std:.6f}, inference={inference_xyz_std:.6f}, ratio={ratio:.2f}")
    
    # 4. 检查 quaternion 范围
    if teleop_stats['rel_quat_stats'] and inference_stats['rel_quat_stats']:
        teleop_quat_mean = teleop_stats['rel_quat_stats'][0]
        inference_quat_mean = inference_stats['rel_quat_stats'][0]
        
        # 对于单位四元数，w 分量应该接近 1
        print(f"  Relative Quat mean: teleop={teleop_quat_mean:.4f}, inference={inference_quat_mean:.4f}")
    
    if issues:
        print("\n⚠️  发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ 数据预处理对齐检查通过!")
    
    return teleop_dataset, inference_dataset


def analyze_action_distribution(teleop_dataset, inference_dataset, num_samples=100):
    """分析 action 分布"""
    print_separator("Action 分布统计")
    
    # 收集样本
    teleop_actions = []
    inference_actions = []
    
    for i in range(min(num_samples, len(teleop_dataset))):
        sample = teleop_dataset[i]
        teleop_actions.append(sample['actions_cont'].cpu().numpy())
    
    for i in range(min(num_samples, len(inference_dataset))):
        sample = inference_dataset[i]
        inference_actions.append(sample['actions_cont'].cpu().numpy())
    
    teleop_actions = np.concatenate(teleop_actions, axis=0)  # [N*pred_horizon, 13]
    inference_actions = np.concatenate(inference_actions, axis=0)
    
    print(f"\n收集了 {len(teleop_actions)} 个 teleop actions")
    print(f"收集了 {len(inference_actions)} 个 inference actions")
    
    # 统计各维度
    dims = [
        ("Joint 0", 0),
        ("Joint 1", 1),
        ("Joint 2", 2),
        ("Joint 3", 3),
        ("Joint 4", 4),
        ("Joint 5", 5),
        ("Rel X", 6),
        ("Rel Y", 7),
        ("Rel Z", 8),
        ("Rel qx", 9),
        ("Rel qy", 10),
        ("Rel qz", 11),
        ("Rel qw", 12),
    ]
    
    print(f"\n{'Dimension':<12} {'Teleop Mean':>12} {'Teleop Std':>12} {'Infer Mean':>12} {'Infer Std':>12} {'Ratio':>8}")
    print("-" * 70)
    
    for name, idx in dims:
        t_mean = teleop_actions[:, idx].mean()
        t_std = teleop_actions[:, idx].std()
        i_mean = inference_actions[:, idx].mean()
        i_std = inference_actions[:, idx].std()
        ratio = i_std / t_std if t_std > 1e-8 else float('inf')
        
        print(f"{name:<12} {t_mean:>12.6f} {t_std:>12.6f} {i_mean:>12.6f} {i_std:>12.6f} {ratio:>8.2f}")


def test_model_forward():
    """测试模型前向传播是否兼容"""
    print_separator("模型前向传播测试")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载 checkpoint
    ckpt_path = "/home/lizh/rl-vla/rlft/diffusion_policy/runs/consistency_flow_discrete_gripper_weight_0.02/checkpoints/latest.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    print(f"加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # 检查 checkpoint 内容
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    
    # 加载 args
    import json
    args_path = os.path.dirname(ckpt_path) + "/args.json"
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            args = json.load(f)
        print(f"\n训练参数:")
        print(f"  algorithm: {args.get('algorithm')}")
        print(f"  action_mode: {args.get('action_mode')}")
        print(f"  obs_horizon: {args.get('obs_horizon')}")
        print(f"  pred_horizon: {args.get('pred_horizon')}")
        print(f"  visual_encoder_type: {args.get('visual_encoder_type')}")
        print(f"  gripper_threshold: {args.get('gripper_threshold')}")
    
    # 创建模型并加载
    from diffusion_policy.resnet_encoder import ResNetEncoder
    from diffusion_policy.carm_utils import StateEncoder
    from train_carm import GripperHead, create_agent
    from dataclasses import dataclass, field
    from typing import List, Literal
    
    # 使用保存的参数创建模型
    visual_encoder = ResNetEncoder(
        backbone_name=args.get('visual_encoder_type', 'resnet18'),
        out_dim=args.get('visual_feature_dim', 256),
        pretrained=False,
        freeze_backbone=False,
        freeze_bn=True,
    ).to(device)
    
    state_encoder = StateEncoder(
        state_dim=7,
        hidden_dim=args.get('state_encoder_hidden_dim', 128),
        out_dim=args.get('state_encoder_out_dim', 256),
    ).to(device)
    
    # 加载权重
    visual_encoder.load_state_dict(ckpt['visual_encoder'])
    state_encoder.load_state_dict(ckpt['state_encoder'])
    
    print("\n模型加载成功!")
    
    # 创建测试数据
    obs_process_fn = create_carm_obs_process_fn(
        output_format="NCHW",
        target_size=(224, 224),  # ResNet 使用 224
        normalize_images=True,
    )
    
    # 测试两种数据的前向传播
    print("\n测试前向传播...")
    
    # 从两个数据源各取一个样本
    teleop_dataset = CARMDataset(
        data_path="/home/lizh/rl-vla/recorded_data/mix",
        obs_process_fn=obs_process_fn,
        device=device,
        num_episodes=1,
        obs_horizon=args.get('obs_horizon', 2),
        pred_horizon=args.get('pred_horizon', 16),
        action_mode="full",
        precompute_actions=False,
        gripper_threshold=args.get('gripper_threshold', 0.05),
    )
    
    inference_dataset = InferenceDataset(
        data_path="/home/lizh/rl-vla/inference_logs",
        obs_process_fn=obs_process_fn,
        device=device,
        num_episodes=1,
        obs_horizon=args.get('obs_horizon', 2),
        pred_horizon=args.get('pred_horizon', 16),
        action_mode="full",
        use_intervened_action=True,
        gripper_threshold=args.get('gripper_threshold', 0.05),
    )
    
    teleop_sample = teleop_dataset[0]
    inference_sample = inference_dataset[0]
    
    visual_encoder.eval()
    state_encoder.eval()
    
    with torch.no_grad():
        # Teleop
        rgb_t = teleop_sample['observations']['rgb'].unsqueeze(0)  # [1, T, C, H, W]
        state_t = teleop_sample['observations']['state'].unsqueeze(0)  # [1, T, 7]
        
        B, T = rgb_t.shape[:2]
        rgb_flat = rgb_t.view(B * T, *rgb_t.shape[2:]).float() / 255.0
        visual_feat_t = visual_encoder(rgb_flat).view(B, T, -1)
        
        state_flat = state_t.view(B * T, -1).float()
        state_feat_t = state_encoder(state_flat).view(B, T, -1)
        
        obs_feat_t = torch.cat([visual_feat_t, state_feat_t], dim=-1)
        print(f"\nTeleop obs_features shape: {obs_feat_t.shape}")
        print(f"  visual_feat: {visual_feat_t.shape}, state_feat: {state_feat_t.shape}")
        
        # Inference
        rgb_i = inference_sample['observations']['rgb'].unsqueeze(0)
        state_i = inference_sample['observations']['state'].unsqueeze(0)
        
        rgb_flat = rgb_i.view(B * T, *rgb_i.shape[2:]).float() / 255.0
        visual_feat_i = visual_encoder(rgb_flat).view(B, T, -1)
        
        state_flat = state_i.view(B * T, -1).float()
        state_feat_i = state_encoder(state_flat).view(B, T, -1)
        
        obs_feat_i = torch.cat([visual_feat_i, state_feat_i], dim=-1)
        print(f"Inference obs_features shape: {obs_feat_i.shape}")
        print(f"  visual_feat: {visual_feat_i.shape}, state_feat: {state_feat_i.shape}")
        
        # 检查特征是否在合理范围
        print(f"\nTeleop visual_feat stats: mean={visual_feat_t.mean():.4f}, std={visual_feat_t.std():.4f}")
        print(f"Inference visual_feat stats: mean={visual_feat_i.mean():.4f}, std={visual_feat_i.std():.4f}")
        
        print(f"\nTeleop state_feat stats: mean={state_feat_t.mean():.4f}, std={state_feat_t.std():.4f}")
        print(f"Inference state_feat stats: mean={state_feat_i.mean():.4f}, std={state_feat_i.std():.4f}")
    
    print("\n✅ 前向传播测试通过!")


def deep_dive_relative_pose_issue():
    """深入分析相对位姿数值差异问题"""
    print_separator("深入分析：相对位姿数值差异")
    
    # 遥操作数据
    teleop_file = "/home/lizh/rl-vla/recorded_data/mix/episode_0001_20260112_223900.hdf5"
    # 真机推理数据
    inference_files = sorted(glob.glob("/home/lizh/rl-vla/inference_logs/inference_episode_*.hdf5"))
    inference_file = inference_files[0]
    
    print("\n1. 分析遥操作数据的相对位姿分布:")
    print(f"   文件: {os.path.basename(teleop_file)}")
    
    teleop_rel_xyz = []
    with h5py.File(teleop_file, 'r') as f:
        action = np.array(f['action'])
        qpos_end = np.array(f['observations/qpos_end'])
        
        # 计算所有帧的相对位姿
        for t in range(len(action) - 1):
            ref_pose = qpos_end[t, :7]
            target_pose = action[t + 1, 7:14]  # 下一帧的目标位姿（绝对）
            relative_pose = compute_relative_pose_transform(ref_pose, target_pose)
            teleop_rel_xyz.append(relative_pose[:3])
    
    teleop_rel_xyz = np.array(teleop_rel_xyz)
    print(f"   样本数: {len(teleop_rel_xyz)}")
    print(f"   Rel X: mean={teleop_rel_xyz[:,0].mean():.6f}, std={teleop_rel_xyz[:,0].std():.6f}")
    print(f"   Rel Y: mean={teleop_rel_xyz[:,1].mean():.6f}, std={teleop_rel_xyz[:,1].std():.6f}")
    print(f"   Rel Z: mean={teleop_rel_xyz[:,2].mean():.6f}, std={teleop_rel_xyz[:,2].std():.6f}")
    print(f"   ||xyz|| mean: {np.linalg.norm(teleop_rel_xyz, axis=1).mean():.6f}")
    
    print("\n2. 分析推理数据的相对位姿分布:")
    print(f"   文件: {os.path.basename(inference_file)}")
    
    inference_rel_xyz = []
    with h5py.File(inference_file, 'r') as f:
        action_intervened = np.array(f['action_intervened'])[:, 0, :]  # [T, 15]
        
        # 直接使用保存的相对位姿
        inference_rel_xyz = action_intervened[:, 7:10]  # [T, 3]
    
    print(f"   样本数: {len(inference_rel_xyz)}")
    print(f"   Rel X: mean={inference_rel_xyz[:,0].mean():.6f}, std={inference_rel_xyz[:,0].std():.6f}")
    print(f"   Rel Y: mean={inference_rel_xyz[:,1].mean():.6f}, std={inference_rel_xyz[:,1].std():.6f}")
    print(f"   Rel Z: mean={inference_rel_xyz[:,2].mean():.6f}, std={inference_rel_xyz[:,2].std():.6f}")
    print(f"   ||xyz|| mean: {np.linalg.norm(inference_rel_xyz, axis=1).mean():.6f}")
    
    print("\n3. 数值范围对比:")
    print(f"   Teleop ||xyz|| range: [{np.linalg.norm(teleop_rel_xyz, axis=1).min():.6f}, {np.linalg.norm(teleop_rel_xyz, axis=1).max():.6f}]")
    print(f"   Inference ||xyz|| range: [{np.linalg.norm(inference_rel_xyz, axis=1).min():.6f}, {np.linalg.norm(inference_rel_xyz, axis=1).max():.6f}]")
    
    # 比较四元数
    print("\n4. 四元数对比:")
    
    teleop_rel_quat = []
    with h5py.File(teleop_file, 'r') as f:
        action = np.array(f['action'])
        qpos_end = np.array(f['observations/qpos_end'])
        for t in range(len(action) - 1):
            ref_pose = qpos_end[t, :7]
            target_pose = action[t + 1, 7:14]
            relative_pose = compute_relative_pose_transform(ref_pose, target_pose)
            teleop_rel_quat.append(relative_pose[3:])
    teleop_rel_quat = np.array(teleop_rel_quat)
    
    with h5py.File(inference_file, 'r') as f:
        action_intervened = np.array(f['action_intervened'])[:, 0, :]
        inference_rel_quat = action_intervened[:, 10:14]
    
    print(f"   Teleop qw: mean={teleop_rel_quat[:,3].mean():.4f}, std={teleop_rel_quat[:,3].std():.4f}")
    print(f"   Inference qw: mean={inference_rel_quat[:,3].mean():.4f}, std={inference_rel_quat[:,3].std():.4f}")
    
    # 检查是否存在 qw > 1 的情况
    teleop_qw_invalid = (np.abs(teleop_rel_quat[:,3]) > 1.01).sum()
    infer_qw_invalid = (np.abs(inference_rel_quat[:,3]) > 1.01).sum()
    print(f"   Teleop |qw| > 1.01: {teleop_qw_invalid}/{len(teleop_rel_quat)}")
    print(f"   Inference |qw| > 1.01: {infer_qw_invalid}/{len(inference_rel_quat)}")
    
    print("\n" + "="*70)
    print("  分析结论:")
    print("="*70)
    
    teleop_std = teleop_rel_xyz.std()
    infer_std = inference_rel_xyz.std()
    ratio = infer_std / teleop_std if teleop_std > 1e-8 else float('inf')
    
    if ratio > 10:
        print("""
⚠️  发现关键问题：Relative XYZ 数值范围差异过大！

原因分析：
  - 遥操作数据：每帧之间的位移非常小 (std ~0.00004m)
    这是因为遥操作数据帧率较高，相邻帧之间的位移本来就很小
  
  - 推理数据：模型输出的相对位姿更大 (std ~0.006m)
    这是因为模型是在 pred_horizon 上预测的，不是相邻帧

这可能不是一个 Bug，而是数据语义的差异：
  - CARMDataset: 计算的是 obs_frame_pose 到 target_pose 的相对变换
    对于 pred_horizon=16，第一个 action 相对位移小，后面逐渐增大
  
  - InferenceDataset: 直接使用模型输出的 action_intervened[:, 0, :]
    每帧保存的是该帧对应的模型输出的第一个 action

关键问题：是否应该保持一致？
""")
    else:
        print("✅ 数值范围在合理范围内")
    
    return teleop_rel_xyz, inference_rel_xyz


def analyze_pred_horizon_effect():
    """分析预测时域对相对位姿数值的影响"""
    print_separator("分析：预测时域对相对位姿的影响")
    
    from diffusion_policy.carm_utils import compute_relative_pose_transform
    
    teleop_file = "/home/lizh/rl-vla/recorded_data/mix/episode_0001_20260112_223900.hdf5"
    
    with h5py.File(teleop_file, 'r') as f:
        action = np.array(f['action'])
        qpos_end = np.array(f['observations/qpos_end'])
        
        print("分析不同预测时域下相对位姿的数值范围:")
        print(f"{'Horizon':<10} {'Mean ||xyz||':<15} {'Std ||xyz||':<15} {'Max ||xyz||':<15}")
        print("-" * 55)
        
        for horizon in [1, 4, 8, 16, 32]:
            rel_xyz_list = []
            for t in range(0, len(action) - horizon, horizon):
                ref_pose = qpos_end[t, :7]  # 观测帧位姿
                for k in range(horizon):
                    target_pose = action[t + k, 7:14]
                    relative_pose = compute_relative_pose_transform(ref_pose, target_pose)
                    rel_xyz_list.append(relative_pose[:3])
            
            rel_xyz = np.array(rel_xyz_list)
            norms = np.linalg.norm(rel_xyz, axis=1)
            print(f"{horizon:<10} {norms.mean():<15.6f} {norms.std():<15.6f} {norms.max():<15.6f}")
    
    print("\n说明：随着 horizon 增大，相对位姿的数值范围会增大")
    print("      因为目标帧离观测帧越远，位姿变化越大")


if __name__ == "__main__":
    print("=" * 70)
    print("  数据预处理对齐验证")
    print("=" * 70)
    
    # 1. 分析原始数据差异
    analyze_raw_data_diff()
    
    # 2. 对比 Dataset 输出
    teleop_ds, inference_ds = compare_datasets()
    
    # 3. 分析 action 分布
    analyze_action_distribution(teleop_ds, inference_ds)
    
    # 4. 深入分析相对位姿问题
    deep_dive_relative_pose_issue()
    
    # 5. 分析预测时域的影响
    analyze_pred_horizon_effect()
    
    # 6. 测试模型前向传播
    test_model_forward()
    
    print_separator("分析完成")
