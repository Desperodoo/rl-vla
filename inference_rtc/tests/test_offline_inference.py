#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线推理测试

验证策略推理的正确性（无需机器人）:
1. 加载 demo 数据 (trajectory.h5)
2. 加载 checkpoint
3. 对比预测 action vs ground truth
4. 输出 MSE/MAE 指标
5. 保存可视化对比图

用法:
    python -m inference_rtc.tests.test_offline_inference \
        --checkpoint /path/to/checkpoint.pt \
        --demo ~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5
"""

import os
import sys
import argparse
import numpy as np
from typing import Optional, List
from tqdm import tqdm

import torch
import matplotlib
matplotlib.use('Agg')  # 非 GUI 后端
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.config import setup_rlft


class OfflineInferenceTest:
    """离线推理测试器"""
    
    def __init__(
        self,
        checkpoint_path: str,
        demo_path: str,
        device: str = "cuda",
        use_ema: bool = True,
        verbose: bool = True,
    ):
        self.checkpoint_path = os.path.expanduser(checkpoint_path)
        self.demo_path = os.path.expanduser(demo_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_ema = use_ema
        self.verbose = verbose
        
        # 设置 rlft 路径
        setup_rlft()
        
        # 加载模块
        from diffusion_policy.utils import (
            load_traj_hdf5,
            create_real_robot_obs_process_fn,
            get_real_robot_data_info,
        )
        
        self.load_traj_hdf5 = load_traj_hdf5
        self.create_real_robot_obs_process_fn = create_real_robot_obs_process_fn
        self.get_real_robot_data_info = get_real_robot_data_info
        
        # 配置参数 (与训练一致)
        self.obs_horizon = 2
        self.act_horizon = 8
        self.pred_horizon = 16
        self.image_size = (128, 128)
        self.num_flow_steps = 10
        
        # 模型组件
        self.visual_encoder = None
        self.state_encoder = None
        self.agent = None
        
    def load_demo(self, traj_idx: int = 0):
        """加载 demo 数据"""
        print(f"\n[1] 加载 demo 数据: {self.demo_path}")
        
        raw_data = self.load_traj_hdf5(self.demo_path, num_traj=traj_idx + 1)
        traj_key = f"traj_{traj_idx}"
        self.traj = raw_data[traj_key]
        
        self.T = len(self.traj['actions'])
        print(f"  轨迹长度: {self.T} 帧")
        print(f"  动作维度: {self.traj['actions'].shape}")
        
        # 创建观测处理函数
        self.obs_process_fn = self.create_real_robot_obs_process_fn(
            output_format="NCHW",
            camera_name="wrist",
            include_depth=False,
            target_size=self.image_size,
        )
        
        # 处理观测
        self.processed_obs = self.obs_process_fn(self.traj["obs"])
        print(f"  RGB shape: {self.processed_obs['rgb'].shape}")
        print(f"  State shape: {self.processed_obs['state'].shape}")
        
        return self.traj['actions']
    
    def load_model(self):
        """加载模型"""
        print(f"\n[2] 加载模型: {self.checkpoint_path}")
        
        from diffusion_policy.plain_conv import PlainConv
        from diffusion_policy.utils import StateEncoder
        from diffusion_policy.algorithms import ConsistencyFlowAgent
        from diffusion_policy.algorithms.networks import VelocityUNet1D
        
        # 获取数据信息
        data_info = self.get_real_robot_data_info(self.demo_path)
        self.action_dim = data_info["action_dim"]
        self.state_dim = data_info["state_dim"]
        
        print(f"  Action dim: {self.action_dim}")
        print(f"  State dim: {self.state_dim}")
        
        # 计算 global condition 维度
        visual_feature_dim = 256
        state_encoder_out_dim = 256
        global_cond_dim = self.obs_horizon * (visual_feature_dim + state_encoder_out_dim)
        
        # 加载 checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 检测算法类型
        agent_keys = list(ckpt.get('agent', ckpt.get('ema_agent', {})).keys())
        if any('velocity_net_ema' in k for k in agent_keys):
            algorithm = "consistency_flow"
        elif any('velocity_net' in k for k in agent_keys):
            algorithm = "flow_matching"
        else:
            algorithm = "diffusion_policy"
        print(f"  算法: {algorithm}")
        
        # 创建视觉编码器
        self.visual_encoder = PlainConv(
            in_channels=3,
            out_dim=visual_feature_dim,
            pool_feature_map=True,
        ).to(self.device)
        
        # 创建状态编码器
        self.state_encoder = StateEncoder(
            state_dim=self.state_dim,
            hidden_dim=128,
            out_dim=state_encoder_out_dim,
        ).to(self.device)
        
        # 创建 agent
        velocity_net = VelocityUNet1D(
            input_dim=self.action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=64,
            down_dims=(64, 128, 256),
            n_groups=8,
        )
        
        self.agent = ConsistencyFlowAgent(
            velocity_net=velocity_net,
            action_dim=self.action_dim,
            obs_horizon=self.obs_horizon,
            pred_horizon=self.pred_horizon,
            num_flow_steps=self.num_flow_steps,
            ema_decay=0.999,
            action_bounds=None,
            device=str(self.device),
        ).to(self.device)
        
        # 加载权重
        agent_key = "ema_agent" if self.use_ema else "agent"
        self.agent.load_state_dict(ckpt[agent_key])
        self.visual_encoder.load_state_dict(ckpt["visual_encoder"])
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        
        # 设置为评估模式
        self.agent.eval()
        self.visual_encoder.eval()
        self.state_encoder.eval()
        
        param_count = sum(p.numel() for p in self.agent.parameters())
        print(f"  参数量: {param_count / 1e6:.2f}M")
        print(f"  EMA 权重: {self.use_ema}")
    
    @torch.no_grad()
    def predict(self, rgb: torch.Tensor, state: torch.Tensor) -> np.ndarray:
        """单步预测"""
        B, T = rgb.shape[0], rgb.shape[1]
        
        # 编码观测
        rgb_flat = rgb.view(B * T, *rgb.shape[2:]).float() / 255.0
        visual_feat = self.visual_encoder(rgb_flat)
        visual_feat = visual_feat.view(B, T, -1)
        
        state_flat = state.view(B * T, -1).float()
        state_feat = self.state_encoder(state_flat)
        state_feat = state_feat.view(B, T, -1)
        
        obs_features = torch.cat([visual_feat, state_feat], dim=-1)
        obs_cond = obs_features.view(B, -1)
        
        # 推理
        action_seq = self.agent.get_action(obs_cond)
        if isinstance(action_seq, tuple):
            action_seq = action_seq[0]
        
        # 取第一个动作
        first_action = action_seq[0, self.obs_horizon - 1].cpu().numpy()
        return first_action
    
    def rollout(self, step_size: int = 1) -> np.ndarray:
        """沿轨迹进行推理"""
        print(f"\n[3] 进行轨迹推理 (step_size={step_size})...")
        
        rgb_all = self.processed_obs["rgb"]
        state_all = self.processed_obs["state"]
        
        predictions = np.zeros((self.T, self.action_dim), dtype=np.float32)
        
        for t in tqdm(range(0, self.T, step_size), desc="推理中"):
            # 获取观测索引
            obs_start = max(0, t - self.obs_horizon + 1)
            obs_indices = list(range(obs_start, t + 1))
            
            # 填充
            while len(obs_indices) < self.obs_horizon:
                obs_indices.insert(0, obs_indices[0])
            
            # 准备输入
            rgb = torch.from_numpy(rgb_all[obs_indices]).unsqueeze(0).to(self.device)
            state = torch.from_numpy(state_all[obs_indices]).unsqueeze(0).to(self.device)
            
            # 预测
            pred_action = self.predict(rgb, state)
            predictions[t] = pred_action
            
            # 填充跳过的步
            for i in range(1, step_size):
                if t + i < self.T:
                    predictions[t + i] = pred_action
        
        return predictions
    
    def compute_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """计算指标"""
        print(f"\n[4] 计算指标...")
        
        mse = np.mean((predictions - ground_truth) ** 2)
        mae = np.mean(np.abs(predictions - ground_truth))
        
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        
        # 每个维度的 MAE
        action_names = [f"Joint{i}" for i in range(6)] + ["Gripper"]
        print(f"\n  Per-Dimension MAE:")
        for i, name in enumerate(action_names):
            dim_mae = np.mean(np.abs(predictions[:, i] - ground_truth[:, i]))
            print(f"    {name}: {dim_mae:.6f}")
        
        return {"mse": mse, "mae": mae}
    
    def visualize(
        self, 
        predictions: np.ndarray, 
        ground_truth: np.ndarray,
        save_path: str,
    ):
        """可视化对比"""
        print(f"\n[5] 保存可视化图: {save_path}")
        
        action_names = [f"Joint{i}" for i in range(6)] + ["Gripper"]
        T, action_dim = ground_truth.shape
        
        fig, axes = plt.subplots(action_dim, 1, figsize=(14, 12))
        timesteps = np.arange(T)
        
        for i, (ax, name) in enumerate(zip(axes, action_names)):
            ax.plot(timesteps, ground_truth[:, i], 'b-', label='Ground Truth', alpha=0.8)
            ax.plot(timesteps, predictions[:, i], 'r--', label='Prediction', alpha=0.8)
            ax.set_ylabel(name, fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 计算该维度的 MAE
            dim_mae = np.mean(np.abs(predictions[:, i] - ground_truth[:, i]))
            ax.set_title(f"{name} (MAE: {dim_mae:.4f})", fontsize=10)
        
        axes[-1].set_xlabel('Timestep', fontsize=10)
        
        mse = np.mean((predictions - ground_truth) ** 2)
        mae = np.mean(np.abs(predictions - ground_truth))
        fig.suptitle(f"Offline Inference Test\nMSE: {mse:.6f}, MAE: {mae:.6f}", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 图片已保存")


def main():
    parser = argparse.ArgumentParser(description="离线推理测试")
    parser.add_argument("-c", "--checkpoint", required=True, help="Checkpoint 路径")
    parser.add_argument("-d", "--demo", 
                       default="~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5",
                       help="Demo 数据路径")
    parser.add_argument("--traj-idx", type=int, default=0, help="轨迹索引")
    parser.add_argument("--step-size", type=int, default=1, help="推理步长")
    parser.add_argument("--no-ema", action="store_true", help="不使用 EMA 权重")
    parser.add_argument("-o", "--output", default=None, help="输出图片路径")
    parser.add_argument("--device", default="cuda", help="设备")
    
    args = parser.parse_args()
    
    # 默认输出路径
    if args.output is None:
        checkpoint_name = os.path.basename(args.checkpoint).replace('.pt', '')
        args.output = f"offline_inference_{checkpoint_name}.png"
    
    print("=" * 60)
    print("离线推理测试")
    print("=" * 60)
    
    # 创建测试器
    tester = OfflineInferenceTest(
        checkpoint_path=args.checkpoint,
        demo_path=args.demo,
        device=args.device,
        use_ema=not args.no_ema,
    )
    
    # 加载 demo
    ground_truth = tester.load_demo(traj_idx=args.traj_idx)
    
    # 加载模型
    tester.load_model()
    
    # 推理
    predictions = tester.rollout(step_size=args.step_size)
    
    # 计算指标
    metrics = tester.compute_metrics(predictions, ground_truth)
    
    # 可视化
    tester.visualize(predictions, ground_truth, args.output)
    
    print("\n" + "=" * 60)
    if metrics["mae"] < 0.1:
        print("✓ 测试通过! MAE < 0.1")
    else:
        print("⚠ 警告: MAE 较大，请检查数据预处理")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
