#!/usr/bin/env python3
"""
Quick evaluation test for Critic module

快速验证 eval_critic 的各项功能（使用随机初始化的网络）
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_policy.critic import CriticDataset, CriticSMDPDataset
from diffusion_policy.critic.critic_network import create_critic_network


def compute_pairwise_ranking_accuracy(values_t1, values_t2, progress_t1, progress_t2):
    """计算 pairwise ranking 准确率"""
    # Ground truth: progress_t2 > progress_t1 => value_t2 > value_t1
    gt_greater = progress_t2 > progress_t1
    pred_greater = values_t2 > values_t1
    
    correct = (gt_greater == pred_greater).float()
    return correct.mean().item()


def compute_spearman_correlation(values, progress):
    """计算 Spearman 相关系数"""
    rho, pval = spearmanr(values, progress)
    return rho, pval


def test_eval_metrics():
    """测试评估指标计算"""
    print("=" * 60)
    print("Test: Evaluation Metrics")
    print("=" * 60)
    
    # 创建网络（随机初始化）
    device = torch.device("cpu")
    critic = create_critic_network(
        backbone_name="resnet18",
        pretrained=True,
        freeze_visual=True,
        visual_feature_dim=128,
        hidden_dims=(128, 64),
        obs_horizon=2,
        state_dim=7,
        device="cpu"
    )
    critic.eval()
    
    # 加载少量数据用于测试
    data_path = "../../recorded_data/mix_with_reward"
    if not os.path.exists(data_path):
        print(f"  ⚠ Data path not found: {data_path}")
        return
    
    print("  Loading dataset...")
    dataset = CriticDataset(
        data_path=data_path,
        obs_horizon=2,
        filter_mode="success_only",
        verbose=False
    )
    print(f"  Loaded {len(dataset)} samples from {len(dataset.episodes)} episodes")
    
    # 随机采样一些样本
    num_samples = min(100, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    print(f"  Evaluating {num_samples} samples...")
    
    values = []
    progresses = []
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            rgb = sample['rgb'].unsqueeze(0)  # [1, obs_horizon, C, H, W]
            state = sample['state'].unsqueeze(0)  # [1, obs_horizon, state_dim]
            progress = sample['progress'].item()
            
            value = critic(rgb, state).item()
            values.append(value)
            progresses.append(progress)
    
    values = np.array(values)
    progresses = np.array(progresses)
    
    # 1. Spearman correlation
    rho, pval = compute_spearman_correlation(values, progresses)
    print(f"  Spearman ρ(V, p): {rho:.4f} (p={pval:.4e})")
    
    # 2. Pairwise ranking accuracy on SMDP pairs
    print("  Loading SMDP dataset...")
    smdp_dataset = CriticSMDPDataset(
        data_path=data_path,
        chunk_size=8,
        gap_options=[8, 16],
        filter_mode="success_only",
        verbose=False
    )
    print(f"  Built {len(smdp_dataset)} SMDP pairs")
    
    # 随机采样一些 pair
    num_pairs = min(200, len(smdp_dataset))
    pair_indices = np.random.choice(len(smdp_dataset), num_pairs, replace=False)
    
    v_t1_list = []
    v_t2_list = []
    p_t1_list = []
    p_t2_list = []
    
    with torch.no_grad():
        for idx in pair_indices:
            sample = smdp_dataset[idx]
            
            v_t1 = critic(sample['rgb_t1'].unsqueeze(0), sample['state_t1'].unsqueeze(0)).item()
            v_t2 = critic(sample['rgb_t2'].unsqueeze(0), sample['state_t2'].unsqueeze(0)).item()
            
            v_t1_list.append(v_t1)
            v_t2_list.append(v_t2)
            p_t1_list.append(sample['p_t1'].item())
            p_t2_list.append(sample['p_t2'].item())
    
    v_t1 = torch.tensor(v_t1_list)
    v_t2 = torch.tensor(v_t2_list)
    p_t1 = torch.tensor(p_t1_list)
    p_t2 = torch.tensor(p_t2_list)
    
    rank_acc = compute_pairwise_ranking_accuracy(v_t1, v_t2, p_t1, p_t2)
    print(f"  Pairwise Ranking Accuracy: {rank_acc*100:.2f}%")
    
    # 3. Advantage 方向一致性
    advantage = v_t2 - v_t1  # V(s_{t+K}) - V(s_t)
    progress_diff = p_t2 - p_t1  # p_{t+K} - p_t
    
    adv_sign_correct = ((advantage > 0) == (progress_diff > 0)).float().mean().item()
    print(f"  Advantage Direction Accuracy: {adv_sign_correct*100:.2f}%")
    
    print("\n  ✓ Metrics computation OK\n")
    
    return values, progresses


def test_visualization(values=None, progresses=None):
    """测试可视化功能"""
    print("=" * 60)
    print("Test: Visualization")
    print("=" * 60)
    
    output_dir = "/tmp/critic_eval_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用随机数据如果没有提供
    if values is None or progresses is None:
        values = np.random.rand(100) * 0.5 + 0.25
        progresses = np.linspace(0, 1, 100) + np.random.randn(100) * 0.1
        progresses = np.clip(progresses, 0, 1)
    
    # 1. V vs p 散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(progresses, values, alpha=0.5, s=20)
    plt.xlabel('Progress p(s)')
    plt.ylabel('Value V(s)')
    plt.title('V(s) vs p(s) Scatter Plot')
    plt.grid(True, alpha=0.3)
    
    # 添加拟合线
    z = np.polyfit(progresses, values, 1)
    p_line = np.poly1d(z)
    plt.plot(np.linspace(0, 1, 100), p_line(np.linspace(0, 1, 100)), 
             'r--', label=f'Linear fit (slope={z[0]:.3f})')
    plt.legend()
    
    scatter_path = os.path.join(output_dir, "v_vs_p_scatter.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved scatter plot: {scatter_path}")
    
    # 2. Value 分布直方图
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Value V(s)')
    plt.ylabel('Count')
    plt.title('Value Distribution (untrained model)')
    plt.axvline(x=np.mean(values), color='r', linestyle='--', 
                label=f'Mean: {np.mean(values):.3f}')
    plt.legend()
    
    hist_path = os.path.join(output_dir, "value_distribution.png")
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved histogram: {hist_path}")
    
    # 3. 模拟单条轨迹的 V(s) 曲线
    plt.figure(figsize=(10, 6))
    
    T = 50
    progress_traj = np.linspace(0, 1, T)
    value_traj = 0.5 * np.ones(T) + 0.1 * np.random.randn(T)  # 未训练模型接近 0.5
    
    plt.subplot(1, 2, 1)
    plt.plot(progress_traj, label='Progress p(s)', color='blue')
    plt.plot(value_traj, label='Value V(s)', color='red')
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.title('Trajectory: V(s) and p(s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    advantage = np.diff(value_traj)
    plt.bar(range(len(advantage)), advantage, alpha=0.7)
    plt.xlabel('Frame')
    plt.ylabel('Advantage A')
    plt.title('Advantage: V(s_{t+1}) - V(s_t)')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    traj_path = os.path.join(output_dir, "trajectory_analysis.png")
    plt.savefig(traj_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved trajectory analysis: {traj_path}")
    
    print(f"\n  All visualizations saved to: {output_dir}")
    print("  ✓ Visualization OK\n")


def main():
    print("\n" + "=" * 60)
    print("   CRITIC EVALUATION TEST (CPU)")
    print("=" * 60 + "\n")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. 测试评估指标
    values, progresses = test_eval_metrics()
    
    # 2. 测试可视化
    test_visualization(values, progresses)
    
    print("=" * 60)
    print("   ALL EVALUATION TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
