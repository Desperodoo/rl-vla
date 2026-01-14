#!/usr/bin/env python3
"""
CARM 离线测试脚本

使用数据集进行离线评估，验证推理 pipeline 并评估模型性能。
不需要机械臂，只需要 PyTorch 环境。

使用方法:
    conda activate arx-py310
    python offline_test.py --model_path runs/consistency_flow/checkpoints/latest.pt \
                           --data_dir ~/rl-vla/recorded_data \
                           --output_dir offline_results
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime
import h5py
import cv2

# 添加训练代码路径
script_dir = os.path.dirname(os.path.abspath(__file__))
rl_vla_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, os.path.join(rl_vla_root, 'rlft', 'diffusion_policy'))

import torch
import torch.nn as nn
from einops import rearrange

# 训练代码中的模块
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.carm_utils import StateEncoder, load_carm_episode, compute_relative_actions
from diffusion_policy.algorithms.networks import VelocityUNet1D
from diffusion_policy.algorithms import (
    ConsistencyFlowAgent,
    FlowMatchingAgent,
    DiffusionPolicyAgent,
)


class OfflinePolicy:
    """
    离线策略推理类
    与 RealPolicy 类似，但专门用于离线测试
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # 默认参数
        self.obs_horizon = 2
        self.pred_horizon = 16
        self.action_dim = 15
        self.state_dim = 7
        self.target_image_size = (128, 128)
        self.visual_feature_dim = 256
        self.state_encoder_hidden_dim = 128
        self.state_encoder_out_dim = 256
        self.use_state_encoder = True
        self.algorithm = 'consistency_flow'
        
        # 模型组件
        self.visual_encoder = None
        self.state_encoder = None
        self.agent = None
        
        # 观测历史
        self.obs_history = {'rgb': [], 'state': []}
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.model_path}")
        
        checkpoint_dir = os.path.dirname(self.model_path)
        
        # 加载配置
        args_path = os.path.join(checkpoint_dir, "args.json")
        if os.path.exists(args_path):
            print(f"Loading config from: {args_path}")
            with open(args_path, 'r') as f:
                args = json.load(f)
            
            self.obs_horizon = args.get('obs_horizon', self.obs_horizon)
            self.pred_horizon = args.get('pred_horizon', self.pred_horizon)
            self.visual_feature_dim = args.get('visual_feature_dim', self.visual_feature_dim)
            self.state_encoder_hidden_dim = args.get('state_encoder_hidden_dim', self.state_encoder_hidden_dim)
            self.state_encoder_out_dim = args.get('state_encoder_out_dim', self.state_encoder_out_dim)
            self.use_state_encoder = args.get('use_state_encoder', self.use_state_encoder)
            self.algorithm = args.get('algorithm', self.algorithm)
            
            target_size = args.get('target_image_size', self.target_image_size)
            if isinstance(target_size, str):
                target_size = eval(target_size)
            self.target_image_size = target_size
            
            action_mode = args.get('action_mode', 'full')
            self.action_dim = 15 if action_mode == 'full' else 8
            
            print(f"Config: algorithm={self.algorithm}, action_dim={self.action_dim}, "
                  f"obs_horizon={self.obs_horizon}, pred_horizon={self.pred_horizon}")
        
        # 创建模型
        self.visual_encoder = PlainConv(
            in_channels=3,
            out_dim=self.visual_feature_dim,
            pool_feature_map=True,
        ).to(self.device)
        
        encoded_state_dim = self.state_dim
        if self.use_state_encoder:
            self.state_encoder = StateEncoder(
                state_dim=self.state_dim,
                hidden_dim=self.state_encoder_hidden_dim,
                out_dim=self.state_encoder_out_dim,
            ).to(self.device)
            encoded_state_dim = self.state_encoder_out_dim
        
        global_cond_dim = self.obs_horizon * (self.visual_feature_dim + encoded_state_dim)
        self.agent = self._create_agent(global_cond_dim)
        
        # 加载权重
        print(f"Loading checkpoint from: {self.model_path}")
        ckpt = torch.load(self.model_path, map_location=self.device)
        
        if "visual_encoder" in ckpt:
            self.visual_encoder.load_state_dict(ckpt["visual_encoder"])
        if self.state_encoder is not None and "state_encoder" in ckpt:
            self.state_encoder.load_state_dict(ckpt["state_encoder"])
        if "ema_agent" in ckpt:
            self.agent.load_state_dict(ckpt["ema_agent"])
        elif "agent" in ckpt:
            self.agent.load_state_dict(ckpt["agent"])
        
        # 评估模式
        self.visual_encoder.eval()
        if self.state_encoder is not None:
            self.state_encoder.eval()
        self.agent.eval()
        
        print(f"Model loaded successfully!")
    
    def _create_agent(self, global_cond_dim: int) -> nn.Module:
        """创建 agent"""
        diffusion_step_embed_dim = 64
        unet_dims = [64, 128, 256]
        n_groups = 8
        
        if self.algorithm == "consistency_flow":
            velocity_net = VelocityUNet1D(
                input_dim=self.action_dim,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=tuple(unet_dims),
                n_groups=n_groups,
            )
            return ConsistencyFlowAgent(
                velocity_net=velocity_net,
                action_dim=self.action_dim,
                obs_horizon=self.obs_horizon,
                pred_horizon=self.pred_horizon,
                num_flow_steps=10,
                device=str(self.device),
            ).to(self.device)
        elif self.algorithm == "flow_matching":
            velocity_net = VelocityUNet1D(
                input_dim=self.action_dim,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=tuple(unet_dims),
                n_groups=n_groups,
            )
            return FlowMatchingAgent(
                velocity_net=velocity_net,
                action_dim=self.action_dim,
                obs_horizon=self.obs_horizon,
                pred_horizon=self.pred_horizon,
                num_flow_steps=10,
                device=str(self.device),
            ).to(self.device)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def reset(self):
        """重置观测历史"""
        self.obs_history = {'rgb': [], 'state': []}
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        h, w = self.target_image_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        image = rearrange(image, 'h w c -> c h w')
        return image
    
    def _update_obs_history(self, rgb: np.ndarray, state: np.ndarray):
        """更新观测历史"""
        self.obs_history['rgb'].append(rgb)
        self.obs_history['state'].append(state)
        
        if len(self.obs_history['rgb']) > self.obs_horizon:
            self.obs_history['rgb'].pop(0)
            self.obs_history['state'].pop(0)
        
        while len(self.obs_history['rgb']) < self.obs_horizon:
            self.obs_history['rgb'].insert(0, self.obs_history['rgb'][0])
            self.obs_history['state'].insert(0, self.obs_history['state'][0])
    
    def _encode_observations(self) -> torch.Tensor:
        """编码观测"""
        B, T = 1, self.obs_horizon
        
        rgb_list = [torch.from_numpy(r).float() for r in self.obs_history['rgb']]
        rgb = torch.stack(rgb_list, dim=0).unsqueeze(0).to(self.device)
        rgb_flat = rgb.view(B * T, *rgb.shape[2:]) / 255.0
        visual_feat = self.visual_encoder(rgb_flat)
        visual_feat = visual_feat.view(B, T, -1)
        
        state_list = [torch.from_numpy(s).float() for s in self.obs_history['state']]
        state = torch.stack(state_list, dim=0).unsqueeze(0).to(self.device)
        
        if self.state_encoder is not None:
            state_flat = state.view(B * T, -1)
            state_feat = self.state_encoder(state_flat)
            state_feat = state_feat.view(B, T, -1)
        else:
            state_feat = state
        
        obs_features = torch.cat([visual_feat, state_feat], dim=-1)
        return obs_features
    
    @torch.no_grad()
    def predict(self, image: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """
        执行推理
        
        Args:
            image: RGB 图像 [H, W, C]
            qpos: 关节状态 [7]
            
        Returns:
            actions: 预测动作 [pred_horizon, action_dim]
        """
        # 预处理
        image_processed = self._preprocess_image(image)
        self._update_obs_history(image_processed, qpos)
        
        # 推理
        obs_features = self._encode_observations()
        actions = self.agent.get_action_deterministic(obs_features)
        
        return actions.squeeze(0).cpu().numpy()


class OfflineEvaluator:
    """
    离线评估器
    使用数据集评估模型性能
    """
    
    def __init__(self, model_path: str, data_dir: str, output_dir: str = 'offline_results'):
        self.model_path = model_path
        self.data_dir = os.path.expanduser(data_dir)
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型
        print("Loading model...")
        self.policy = OfflinePolicy(model_path)
        
        # 获取数据集文件列表
        self.episode_files = sorted([
            f for f in os.listdir(self.data_dir) 
            if f.startswith('episode_') and f.endswith('.hdf5')
        ])
        print(f"Found {len(self.episode_files)} episodes in {self.data_dir}")
    
    def evaluate_episode(self, ep_idx: int, verbose: bool = False) -> Dict:
        """
        评估单个 episode
        
        Returns:
            Dict with predicted_actions, gt_actions, metrics
        """
        filepath = os.path.join(self.data_dir, self.episode_files[ep_idx])
        episode = load_carm_episode(filepath)
        
        self.policy.reset()
        
        predicted_actions = []
        gt_actions = []
        
        num_steps = len(episode['qpos_joint'])
        
        # 关键修复：将原始 action（绝对位姿）转换为相对位姿
        # 训练时使用的是 compute_relative_actions()，这里也要同样处理
        raw_actions = episode['action']  # [T, 15] 原始动作（末端是绝对位姿）
        qpos_end = episode['qpos_end']   # [T, 8] 末端位姿
        gripper = episode['gripper']     # [T] 夹爪
        
        # 转换为相对动作（与训练数据一致）
        relative_actions = compute_relative_actions(qpos_end, raw_actions, gripper)
        
        iterator = tqdm(range(num_steps), desc=f"Episode {ep_idx}") if verbose else range(num_steps)
        
        for t in iterator:
            # 获取当前帧数据
            image = episode['images'][t]  # [H, W, C]
            qpos = episode['qpos_joint'][t]  # [7]
            gt_action = relative_actions[t]  # [15] 相对位姿动作（与训练一致）
            
            # 推理
            pred_actions = self.policy.predict(image, qpos)  # [pred_horizon, action_dim]
            pred_action = pred_actions[0]  # 取第一个预测动作
            
            predicted_actions.append(pred_action)
            gt_actions.append(gt_action)
        
        predicted_actions = np.array(predicted_actions)
        gt_actions = np.array(gt_actions)
        
        # 计算指标
        metrics = self._compute_metrics(predicted_actions, gt_actions)
        
        return {
            'predicted_actions': predicted_actions,
            'gt_actions': gt_actions,
            'metrics': metrics,
            'qpos_joint': episode['qpos_joint'],
            'qpos_end': episode['qpos_end'],
        }
    
    def _compute_metrics(self, pred: np.ndarray, gt: np.ndarray) -> Dict:
        """计算评估指标"""
        # 分解动作空间
        # full mode: [joint(6), gripper(1), relative_end_pose(7), gripper(1)]
        joint_pred = pred[:, :6]
        joint_gt = gt[:, :6]
        gripper_pred = pred[:, 6]
        gripper_gt = gt[:, 6]
        pose_pred = pred[:, 7:14]
        pose_gt = gt[:, 7:14]
        
        # 计算 MSE
        joint_mse = np.mean((joint_pred - joint_gt) ** 2)
        gripper_mse = np.mean((gripper_pred - gripper_gt) ** 2)
        pose_mse = np.mean((pose_pred - pose_gt) ** 2)
        total_mse = np.mean((pred - gt) ** 2)
        
        # 计算 MAE
        joint_mae = np.mean(np.abs(joint_pred - joint_gt))
        gripper_mae = np.mean(np.abs(gripper_pred - gripper_gt))
        pose_mae = np.mean(np.abs(pose_pred - pose_gt))
        total_mae = np.mean(np.abs(pred - gt))
        
        # 计算各关节的误差
        joint_errors = []
        for i in range(6):
            joint_errors.append({
                'mse': np.mean((joint_pred[:, i] - joint_gt[:, i]) ** 2),
                'mae': np.mean(np.abs(joint_pred[:, i] - joint_gt[:, i])),
                'max': np.max(np.abs(joint_pred[:, i] - joint_gt[:, i])),
            })
        
        return {
            'joint_mse': joint_mse,
            'joint_mae': joint_mae,
            'gripper_mse': gripper_mse,
            'gripper_mae': gripper_mae,
            'pose_mse': pose_mse,
            'pose_mae': pose_mae,
            'total_mse': total_mse,
            'total_mae': total_mae,
            'joint_errors': joint_errors,
        }
    
    def plot_episode_comparison(self, result: Dict, ep_idx: int, save: bool = True):
        """绘制单个 episode 的对比曲线"""
        pred = result['predicted_actions']
        gt = result['gt_actions']
        num_steps = len(pred)
        time_steps = np.arange(num_steps)
        
        # 创建图形
        fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        fig.suptitle(f'Episode {ep_idx}: Predicted vs Ground Truth', fontsize=14)
        
        # 1. 关节 1-3
        for i in range(3):
            ax = axes[i, 0]
            ax.plot(time_steps, gt[:, i], 'b-', label='Ground Truth', alpha=0.7)
            ax.plot(time_steps, pred[:, i], 'r--', label='Predicted', alpha=0.7)
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'Joint {i+1} (rad)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Joint {i+1}')
        
        # 2. 关节 4-6
        for i in range(3):
            ax = axes[i, 1]
            ax.plot(time_steps, gt[:, i+3], 'b-', label='Ground Truth', alpha=0.7)
            ax.plot(time_steps, pred[:, i+3], 'r--', label='Predicted', alpha=0.7)
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'Joint {i+4} (rad)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Joint {i+4}')
        
        # 3. 夹爪
        ax = axes[3, 0]
        ax.plot(time_steps, gt[:, 6], 'b-', label='Ground Truth', alpha=0.7)
        ax.plot(time_steps, pred[:, 6], 'r--', label='Predicted', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Gripper (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Gripper')
        
        # 4. 末端位置 (x, y, z) - 相对位姿
        ax = axes[3, 1]
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']
        for i, (c, l) in enumerate(zip(colors, labels)):
            ax.plot(time_steps, gt[:, 7+i], f'{c}-', label=f'GT {l}', alpha=0.5)
            ax.plot(time_steps, pred[:, 7+i], f'{c}--', label=f'Pred {l}', alpha=0.5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Relative Position (m)')
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_title('End Effector Relative Position (should be near 0)')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'comparison_ep{ep_idx:03d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_error_distribution(self, all_results: List[Dict], save: bool = True):
        """绘制误差分布直方图"""
        all_pred = np.concatenate([r['predicted_actions'] for r in all_results], axis=0)
        all_gt = np.concatenate([r['gt_actions'] for r in all_results], axis=0)
        errors = all_pred - all_gt
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Prediction Error Distribution', fontsize=14)
        
        # 关节误差
        for i in range(6):
            ax = axes[i // 4, i % 4]
            ax.hist(errors[:, i], bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Error (rad)')
            ax.set_ylabel('Count')
            ax.set_title(f'Joint {i+1} Error')
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        # 夹爪误差
        ax = axes[1, 2]
        ax.hist(errors[:, 6], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Error (m)')
        ax.set_ylabel('Count')
        ax.set_title('Gripper Error')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        # 总误差
        ax = axes[1, 3]
        total_error = np.mean(np.abs(errors), axis=1)
        ax.hist(total_error, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Mean Absolute Error')
        ax.set_ylabel('Count')
        ax.set_title('Total MAE per Step')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, 'error_distribution.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_cumulative_error(self, result: Dict, ep_idx: int, save: bool = True):
        """绘制累积误差曲线"""
        pred = result['predicted_actions']
        gt = result['gt_actions']
        num_steps = len(pred)
        
        # 计算每步误差
        step_errors = np.mean(np.abs(pred - gt), axis=1)
        cumulative_errors = np.cumsum(step_errors)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 每步误差
        ax = axes[0]
        ax.plot(np.arange(num_steps), step_errors, 'b-', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title(f'Episode {ep_idx}: Step-wise Error')
        ax.grid(True, alpha=0.3)
        
        # 累积误差
        ax = axes[1]
        ax.plot(np.arange(num_steps), cumulative_errors, 'r-', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative Error')
        ax.set_title(f'Episode {ep_idx}: Cumulative Error')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f'cumulative_error_ep{ep_idx:03d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def run_evaluation(self, num_episodes: Optional[int] = None, 
                       plot_individual: bool = True,
                       verbose: bool = True):
        """
        运行完整评估
        
        Args:
            num_episodes: 评估的 episode 数量 (None = all)
            plot_individual: 是否绘制每个 episode 的对比图
            verbose: 是否显示进度
        """
        if num_episodes is None:
            num_episodes = len(self.episode_files)
        
        all_results = []
        all_metrics = []
        
        print(f"\n{'='*60}")
        print(f"Offline Evaluation: {num_episodes} episodes")
        print(f"{'='*60}\n")
        
        for ep_idx in range(num_episodes):
            print(f"\nEvaluating episode {ep_idx + 1}/{num_episodes}...")
            result = self.evaluate_episode(ep_idx, verbose=verbose)
            all_results.append(result)
            all_metrics.append(result['metrics'])
            
            # 绘制单个 episode 对比图
            if plot_individual:
                self.plot_episode_comparison(result, ep_idx)
                self.plot_cumulative_error(result, ep_idx)
            
            # 打印当前 episode 指标
            m = result['metrics']
            print(f"  Joint MAE: {m['joint_mae']:.4f}, Gripper MAE: {m['gripper_mae']:.4f}, "
                  f"Pose MAE: {m['pose_mae']:.4f}, Total MAE: {m['total_mae']:.4f}")
        
        # 计算整体指标
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics]) 
            for key in all_metrics[0].keys() if key != 'joint_errors'
        }
        
        # 绘制误差分布
        self.plot_error_distribution(all_results)
        
        # 保存结果
        self._save_results(all_results, avg_metrics)
        
        # 打印总结
        print(f"\n{'='*60}")
        print("Evaluation Summary")
        print(f"{'='*60}")
        print(f"Total Episodes: {num_episodes}")
        print(f"Average Joint MAE:   {avg_metrics['joint_mae']:.4f} rad")
        print(f"Average Gripper MAE: {avg_metrics['gripper_mae']:.4f} m")
        print(f"Average Pose MAE:    {avg_metrics['pose_mae']:.4f}")
        print(f"Average Total MAE:   {avg_metrics['total_mae']:.4f}")
        print(f"\nResults saved to: {self.output_dir}")
        print(f"{'='*60}\n")
        
        return avg_metrics
    
    def _save_results(self, all_results: List[Dict], avg_metrics: Dict):
        """保存评估结果"""
        # 将 numpy 类型转换为 Python 原生类型
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 保存指标
        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'avg_metrics': convert_to_native(avg_metrics),
                'model_path': self.model_path,
                'data_dir': self.data_dir,
                'num_episodes': len(all_results),
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        
        # 保存详细结果到 HDF5
        results_path = os.path.join(self.output_dir, 'evaluation_results.hdf5')
        with h5py.File(results_path, 'w') as f:
            for i, result in enumerate(all_results):
                grp = f.create_group(f'episode_{i:03d}')
                grp.create_dataset('predicted_actions', data=result['predicted_actions'])
                grp.create_dataset('gt_actions', data=result['gt_actions'])
                
                metrics_grp = grp.create_group('metrics')
                for k, v in result['metrics'].items():
                    if k != 'joint_errors':
                        metrics_grp.attrs[k] = float(v) if isinstance(v, (np.floating, float)) else v


def parse_args():
    parser = argparse.ArgumentParser(description='CARM Offline Evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='~/rl-vla/recorded_data',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='offline_results',
                        help='Output directory for results')
    parser.add_argument('--num_episodes', type=int, default=None,
                        help='Number of episodes to evaluate (default: all)')
    parser.add_argument('--no_individual_plots', action='store_true',
                        help='Skip individual episode plots')
    parser.add_argument('--quiet', action='store_true',
                        help='Less verbose output')
    return parser.parse_args()


def main():
    args = parse_args()
    
    evaluator = OfflineEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    
    evaluator.run_evaluation(
        num_episodes=args.num_episodes,
        plot_individual=not args.no_individual_plots,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
