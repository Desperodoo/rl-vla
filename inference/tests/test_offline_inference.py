#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线推理测试

使用数据集进行离线推理测试，对比预测输出和 ground truth。
验证多进程版本的数据预处理是否与原版一致。

用法:
    python -m inference.tests.test_offline_inference \
        -c /path/to/checkpoint.pt \
        -d ~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5
"""

import os
import sys
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import cv2

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.config import setup_rlft


def load_trajectory_h5(path: str) -> Dict:
    """加载 trajectory.h5 文件"""
    path = os.path.expanduser(path)
    print(f"加载数据集: {path}")
    
    with h5py.File(path, 'r') as f:
        # 列出所有轨迹
        traj_keys = [k for k in f.keys() if k.startswith('traj_')]
        print(f"  找到 {len(traj_keys)} 条轨迹")
        
        data = {}
        for traj_key in sorted(traj_keys, key=lambda x: int(x.split('_')[1])):
            traj = f[traj_key]
            
            # 读取观测数据
            obs = traj['obs']
            
            traj_data = {
                'rgb': np.array(obs['images']['wrist']['rgb']),  # (T, H, W, 3)
                'joint_pos': np.array(obs['joint_pos']),  # (T, 6)
                'joint_vel': np.array(obs['joint_vel']),  # (T, 6)
                'gripper_pos': np.array(obs['gripper_pos']),  # (T, 1)
                'actions': np.array(traj['actions']),  # (T, 7)
            }
            
            # 检查是否有深度数据
            if 'depth' in obs['images']['wrist']:
                traj_data['depth'] = np.array(obs['images']['wrist']['depth'])
            
            data[traj_key] = traj_data
            
            print(f"  {traj_key}: rgb={traj_data['rgb'].shape}, actions={traj_data['actions'].shape}")
    
    return data


def preprocess_obs_like_inference_node(
    rgb: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    gripper_pos: np.ndarray,
    target_size: Tuple[int, int] = (128, 128),
    obs_horizon: int = 2,
) -> Dict[str, np.ndarray]:
    """
    按照 inference_node.py 的方式预处理观测数据
    
    这是多进程版本的预处理逻辑
    """
    T = len(rgb)
    
    # 调整图像大小
    rgb_resized = np.zeros((T, target_size[0], target_size[1], 3), dtype=np.uint8)
    for i in range(T):
        rgb_resized[i] = cv2.resize(
            rgb[i], 
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    # 转换为 NCHW 格式并归一化
    rgb_nchw = np.transpose(rgb_resized, (0, 3, 1, 2))  # (T, C, H, W)
    rgb_nchw = rgb_nchw.astype(np.float32) / 255.0
    
    # 构建状态向量
    gripper = gripper_pos.reshape(-1, 1) if gripper_pos.ndim == 1 else gripper_pos
    state = np.concatenate([joint_pos, joint_vel, gripper], axis=-1)  # (T, 13)
    state = state.astype(np.float32)
    
    return {
        'rgb': rgb_nchw,
        'state': state,
    }


def preprocess_obs_like_rlft(
    rgb: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    gripper_pos: np.ndarray,
    target_size: Tuple[int, int] = (128, 128),
) -> Dict[str, np.ndarray]:
    """
    按照 rlft/diffusion_policy 的方式预处理观测数据
    
    这是原版训练时的预处理逻辑
    """
    T = len(rgb)
    
    # 调整图像大小
    rgb_resized = np.zeros((T, target_size[0], target_size[1], 3), dtype=np.uint8)
    for i in range(T):
        rgb_resized[i] = cv2.resize(
            rgb[i], 
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    # 转换为 NCHW 格式 (注意: 不在这里归一化，在 encode_observation 中做)
    rgb_nchw = np.transpose(rgb_resized, (0, 3, 1, 2))  # (T, C, H, W)
    
    # 构建状态向量
    gripper = gripper_pos.reshape(-1, 1) if gripper_pos.ndim == 1 else gripper_pos
    state = np.concatenate([joint_pos, joint_vel, gripper], axis=-1)  # (T, 13)
    state = state.astype(np.float32)
    
    return {
        'rgb': rgb_nchw,  # uint8, (T, C, H, W)
        'state': state,   # float32, (T, 13)
    }


def detect_algorithm_from_checkpoint(checkpoint_path: str) -> str:
    """从 checkpoint 路径或内容检测算法类型"""
    path_lower = checkpoint_path.lower()
    
    # 优先从路径名检测
    if "consistency_flow" in path_lower:
        return "consistency_flow"
    elif "shortcut_flow" in path_lower:
        return "shortcut_flow"
    elif "reflected_flow" in path_lower:
        return "reflected_flow"
    elif "flow_matching" in path_lower:
        return "flow_matching"
    elif "diffusion_policy" in path_lower:
        return "diffusion_policy"
    
    # 从 checkpoint 内容检测
    import torch
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    agent_state = ckpt.get('agent', {}) or ckpt.get('ema_agent', {})
    agent_keys = list(agent_state.keys())
    
    if any('velocity_net_ema' in k for k in agent_keys):
        return "consistency_flow"
    elif any('velocity_net' in k for k in agent_keys):
        return "flow_matching"
    elif any('noise_pred_net' in k for k in agent_keys):
        return "diffusion_policy"
    
    return "consistency_flow"  # 默认


def load_policy(checkpoint_path: str, device: str = "cuda", use_ema: bool = True):
    """加载策略模型"""
    setup_rlft()
    
    from diffusion_policy.plain_conv import PlainConv
    from diffusion_policy.utils import StateEncoder
    from diffusion_policy.algorithms import (
        DiffusionPolicyAgent,
        FlowMatchingAgent,
        ConsistencyFlowAgent,
    )
    from diffusion_policy.algorithms.networks import VelocityUNet1D
    from diffusion_policy.conditional_unet1d import ConditionalUnet1D
    
    # 配置
    obs_horizon = 2
    pred_horizon = 16
    action_dim = 7
    state_dim = 13
    visual_feature_dim = 256
    state_encoder_out_dim = 256
    global_cond_dim = obs_horizon * (visual_feature_dim + state_encoder_out_dim)
    
    algorithm = detect_algorithm_from_checkpoint(checkpoint_path)
    print(f"检测到算法: {algorithm}")
    
    # 加载 checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型
    visual_encoder = PlainConv(
        in_channels=3,
        out_dim=visual_feature_dim,
        pool_feature_map=True,
    ).to(device)
    
    state_encoder = StateEncoder(
        state_dim=state_dim,
        hidden_dim=128,
        out_dim=state_encoder_out_dim,
    ).to(device)
    
    # 创建 agent
    if algorithm == "consistency_flow":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=64,
            down_dims=(64, 128, 256),
            n_groups=8,
        )
        agent = ConsistencyFlowAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            num_flow_steps=10,
            ema_decay=0.999,
            action_bounds=None,
            device=device,
        ).to(device)
    elif algorithm == "flow_matching":
        velocity_net = VelocityUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=64,
            down_dims=(64, 128, 256),
            n_groups=8,
        )
        agent = FlowMatchingAgent(
            velocity_net=velocity_net,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            num_flow_steps=10,
            action_bounds=None,
            device=device,
        ).to(device)
    elif algorithm == "diffusion_policy":
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=64,
            down_dims=(64, 128, 256),
            n_groups=8,
        )
        agent = DiffusionPolicyAgent(
            noise_pred_net=noise_pred_net,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            num_diffusion_iters=100,
            device=device,
        ).to(device)
    else:
        raise ValueError(f"未知算法: {algorithm}")
    
    # 加载权重
    agent_key = "ema_agent" if use_ema else "agent"
    agent.load_state_dict(ckpt[agent_key])
    visual_encoder.load_state_dict(ckpt["visual_encoder"])
    state_encoder.load_state_dict(ckpt["state_encoder"])
    
    # 设置 eval 模式
    agent.eval()
    visual_encoder.eval()
    state_encoder.eval()
    
    return visual_encoder, state_encoder, agent, obs_horizon


@torch.no_grad()
def rollout_trajectory(
    visual_encoder,
    state_encoder,
    agent,
    processed_obs: Dict[str, np.ndarray],
    obs_horizon: int = 2,
    device: str = "cuda",
    preprocess_mode: str = "inference_node",  # "inference_node" or "rlft"
) -> np.ndarray:
    """
    沿轨迹进行推理
    
    preprocess_mode:
        - "inference_node": 使用多进程版本的预处理 (已归一化的 float32)
        - "rlft": 使用原版的预处理 (uint8, 在 encode 时归一化)
    """
    rgb_all = processed_obs["rgb"]
    state_all = processed_obs["state"]
    T = len(rgb_all)
    action_dim = 7
    
    predictions = np.zeros((T, action_dim), dtype=np.float32)
    
    for t in tqdm(range(T), desc="Rolling out trajectory"):
        # 获取观测索引
        obs_start = max(0, t - obs_horizon + 1)
        obs_indices = list(range(obs_start, t + 1))
        
        # 填充
        while len(obs_indices) < obs_horizon:
            obs_indices.insert(0, obs_indices[0])
        
        # 准备输入
        rgb = torch.from_numpy(rgb_all[obs_indices]).unsqueeze(0).to(device)
        state = torch.from_numpy(state_all[obs_indices]).unsqueeze(0).to(device)
        
        B, T_obs = rgb.shape[0], rgb.shape[1]
        
        # 根据预处理模式处理 RGB
        if preprocess_mode == "rlft":
            # RLFT 方式: uint8 -> float32 并归一化
            rgb_flat = rgb.view(B * T_obs, *rgb.shape[2:]).float() / 255.0
        else:
            # inference_node 方式: 已经是归一化的 float32
            rgb_flat = rgb.view(B * T_obs, *rgb.shape[2:])
        
        state_flat = state.view(B * T_obs, -1).float()
        
        # 编码
        visual_feat = visual_encoder(rgb_flat)
        visual_feat = visual_feat.view(B, T_obs, -1)
        
        state_feat = state_encoder(state_flat)
        state_feat = state_feat.view(B, T_obs, -1)
        
        # 拼接
        obs_features = torch.cat([visual_feat, state_feat], dim=-1)
        obs_cond = obs_features.view(B, -1)
        
        # 预测
        action_seq = agent.get_action(obs_cond)
        if isinstance(action_seq, tuple):
            action_seq = action_seq[0]
        
        # 取当前时刻的动作 (obs_horizon - 1)
        action_at_t = action_seq[0, obs_horizon - 1].cpu().numpy()
        predictions[t] = action_at_t
    
    return predictions


def plot_trajectory_comparison(
    gt_actions: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
    title: str = "Trajectory Rollout",
):
    """绘制轨迹对比图"""
    T = len(gt_actions)
    joint_names = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'Gripper']
    
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(title)
    
    for i, (ax, name) in enumerate(zip(axes, joint_names)):
        ax.plot(gt_actions[:, i], 'b-', label='Ground Truth', alpha=0.8)
        ax.plot(predictions[:, i], 'r-', label='Prediction', alpha=0.8)
        ax.set_ylabel(name)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Timestep')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"保存图片: {save_path}")


def compute_metrics(gt_actions: np.ndarray, predictions: np.ndarray):
    """计算评估指标"""
    mse = np.mean((gt_actions - predictions) ** 2, axis=0)
    mae = np.mean(np.abs(gt_actions - predictions), axis=0)
    
    joint_names = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'Gripper']
    
    print("\n" + "=" * 60)
    print("评估指标")
    print("=" * 60)
    print(f"{'Joint':<10} {'MSE':<12} {'MAE':<12}")
    print("-" * 34)
    for i, name in enumerate(joint_names):
        print(f"{name:<10} {mse[i]:<12.6f} {mae[i]:<12.6f}")
    print("-" * 34)
    print(f"{'Total':<10} {mse.mean():<12.6f} {mae.mean():<12.6f}")


def main():
    parser = argparse.ArgumentParser(description="离线推理测试")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="Checkpoint 文件路径")
    parser.add_argument("--dataset", "-d", type=str, required=True,
                       help="数据集路径 (trajectory.h5)")
    parser.add_argument("--traj-idx", type=int, default=0,
                       help="测试哪条轨迹 (默认: 0)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备")
    parser.add_argument("--no-ema", action="store_true",
                       help="不使用 EMA 权重")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                       help="输出目录")
    parser.add_argument("--compare-preprocess", action="store_true",
                       help="对比两种预处理方式的结果")
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    data = load_trajectory_h5(args.dataset)
    traj_key = f"traj_{args.traj_idx}"
    
    if traj_key not in data:
        print(f"错误: 轨迹 '{traj_key}' 不存在")
        print(f"可用轨迹: {list(data.keys())}")
        return
    
    traj_data = data[traj_key]
    
    # 加载模型
    visual_encoder, state_encoder, agent, obs_horizon = load_policy(
        args.checkpoint, args.device, use_ema=not args.no_ema
    )
    
    # ===== 测试 inference_node 预处理方式 =====
    print("\n" + "=" * 60)
    print("测试: inference_node 预处理方式")
    print("=" * 60)
    
    processed_obs = preprocess_obs_like_inference_node(
        traj_data['rgb'],
        traj_data['joint_pos'],
        traj_data['joint_vel'],
        traj_data['gripper_pos'],
    )
    
    print(f"RGB 数据类型: {processed_obs['rgb'].dtype}, 范围: [{processed_obs['rgb'].min():.3f}, {processed_obs['rgb'].max():.3f}]")
    print(f"State 数据类型: {processed_obs['state'].dtype}")
    
    predictions_inf = rollout_trajectory(
        visual_encoder, state_encoder, agent,
        processed_obs, obs_horizon, args.device,
        preprocess_mode="inference_node"
    )
    
    gt_actions = traj_data['actions']
    
    print("\n[inference_node 预处理]")
    compute_metrics(gt_actions, predictions_inf)
    
    plot_trajectory_comparison(
        gt_actions, predictions_inf,
        os.path.join(args.output_dir, f"trajectory_rollout_inference_node_traj{args.traj_idx}.png"),
        title=f"Trajectory Rollout (inference_node preprocess) - Traj {args.traj_idx}"
    )
    
    # ===== 测试 rlft 预处理方式 =====
    if args.compare_preprocess:
        print("\n" + "=" * 60)
        print("测试: rlft 预处理方式")
        print("=" * 60)
        
        processed_obs_rlft = preprocess_obs_like_rlft(
            traj_data['rgb'],
            traj_data['joint_pos'],
            traj_data['joint_vel'],
            traj_data['gripper_pos'],
        )
        
        print(f"RGB 数据类型: {processed_obs_rlft['rgb'].dtype}, 范围: [{processed_obs_rlft['rgb'].min()}, {processed_obs_rlft['rgb'].max()}]")
        
        predictions_rlft = rollout_trajectory(
            visual_encoder, state_encoder, agent,
            processed_obs_rlft, obs_horizon, args.device,
            preprocess_mode="rlft"
        )
        
        print("\n[rlft 预处理]")
        compute_metrics(gt_actions, predictions_rlft)
        
        plot_trajectory_comparison(
            gt_actions, predictions_rlft,
            os.path.join(args.output_dir, f"trajectory_rollout_rlft_traj{args.traj_idx}.png"),
            title=f"Trajectory Rollout (rlft preprocess) - Traj {args.traj_idx}"
        )
        
        # 对比两种预处理的差异
        print("\n" + "=" * 60)
        print("两种预处理方式的预测差异")
        print("=" * 60)
        diff = np.abs(predictions_inf - predictions_rlft)
        print(f"最大差异: {diff.max():.6f}")
        print(f"平均差异: {diff.mean():.6f}")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
