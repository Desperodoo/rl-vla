"""
Critic Evaluation Script

评估训练好的 Critic V(s) 的性能，计算多种指标并生成可视化。

评估指标：
1. Pairwise Ranking Accuracy - V(s_t2) > V(s_t1) when p_t2 > p_t1
2. Spearman ρ(V, p) - V 与真实进度 p 的相关性
3. Advantage 方向一致性 - A^(K) = V(s_{t+K}) - V(s_t) 的符号准确率
4. Terminal 可分性 (A2) - 成功/失败终态的 V 分布 AUC

可视化：
- 单条轨迹 V(s) vs p(s) 曲线
- Advantage 分布直方图
- 终态 V 分布对比

Usage:
    python eval_critic.py --checkpoint runs/critic_xxx/checkpoints/best.pt \
        --data-path ~/rl-vla/recorded_data/mix_with_reward
"""

import os
import sys
import json
import argparse
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_policy.critic import CriticDataset, CriticSMDPDataset
from diffusion_policy.critic.critic_network import create_critic_network


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Critic V(s)")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to labeled dataset")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: checkpoint dir)")
    
    # Evaluation settings
    parser.add_argument("--num-samples", type=int, default=2000,
                        help="Number of samples for evaluation")
    parser.add_argument("--num-trajectory-vis", type=int, default=5,
                        help="Number of trajectories to visualize")
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def load_critic_from_checkpoint(checkpoint_path: str, device: torch.device):
    """从检查点加载 Critic"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt.get("args", {})
    
    # 获取配置
    state_mode = args.get("state_mode", "joint_only")
    state_dim = {"joint_only": 7, "ee_only": 8, "both": 14}[state_mode]
    
    # 创建网络
    critic = create_critic_network(
        state_dim=state_dim,
        visual_feature_dim=args.get("visual_dim", 256),
        obs_horizon=args.get("obs_horizon", 2),
        hidden_dims=tuple(args.get("hidden_dims", [256, 256])),
        backbone_name=args.get("backbone", "resnet18"),
        pretrained=True,
        freeze_visual=True,
        freeze_bn=True,
        device=str(device),
    )
    
    # 加载权重
    critic.load_state_dict(ckpt["critic"])
    critic.eval()
    
    return critic, args


def evaluate_pairwise_ranking(
    critic,
    dataset: CriticSMDPDataset,
    device: torch.device,
    num_samples: int = 2000,
) -> dict:
    """评估 Pairwise Ranking 性能"""
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    v_t1_list, v_t2_list = [], []
    p_t1_list, p_t2_list = [], []
    gap_list = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating pairwise ranking"):
            sample = dataset[idx]
            
            rgb_t1 = sample["rgb_t1"].unsqueeze(0).to(device)
            state_t1 = sample["state_t1"].unsqueeze(0).to(device)
            rgb_t2 = sample["rgb_t2"].unsqueeze(0).to(device)
            state_t2 = sample["state_t2"].unsqueeze(0).to(device)
            
            v_t1 = critic(rgb_t1, state_t1).squeeze().cpu().item()
            v_t2 = critic(rgb_t2, state_t2).squeeze().cpu().item()
            
            v_t1_list.append(v_t1)
            v_t2_list.append(v_t2)
            p_t1_list.append(sample["p_t1"].item())
            p_t2_list.append(sample["p_t2"].item())
            gap_list.append(sample["gap"])
    
    v_t1_arr = np.array(v_t1_list)
    v_t2_arr = np.array(v_t2_list)
    p_t1_arr = np.array(p_t1_list)
    p_t2_arr = np.array(p_t2_list)
    gap_arr = np.array(gap_list)
    
    # Overall ranking accuracy
    pred_rank = v_t2_arr > v_t1_arr
    true_rank = p_t2_arr > p_t1_arr
    rank_accuracy = (pred_rank == true_rank).mean()
    
    # Per-gap accuracy
    unique_gaps = np.unique(gap_arr)
    gap_accuracies = {}
    for gap in unique_gaps:
        mask = gap_arr == gap
        if mask.sum() > 0:
            gap_acc = (pred_rank[mask] == true_rank[mask]).mean()
            gap_accuracies[int(gap)] = float(gap_acc)
    
    # Advantage analysis
    pred_advantage = v_t2_arr - v_t1_arr
    true_delta_p = p_t2_arr - p_t1_arr
    
    # Sign accuracy
    pred_sign = np.sign(pred_advantage)
    true_sign = np.sign(true_delta_p)
    sign_accuracy = (pred_sign == true_sign).mean()
    
    # Spearman correlations
    spearman_vp, _ = spearmanr(np.concatenate([v_t1_arr, v_t2_arr]),
                                np.concatenate([p_t1_arr, p_t2_arr]))
    spearman_adv, _ = spearmanr(pred_advantage, true_delta_p)
    
    return {
        "rank_accuracy": float(rank_accuracy),
        "sign_accuracy": float(sign_accuracy),
        "spearman_vp": float(spearman_vp),
        "spearman_advantage": float(spearman_adv),
        "gap_accuracies": gap_accuracies,
        # Raw data for visualization
        "_v_t1": v_t1_arr,
        "_v_t2": v_t2_arr,
        "_p_t1": p_t1_arr,
        "_p_t2": p_t2_arr,
        "_advantage_pred": pred_advantage,
        "_advantage_true": true_delta_p,
    }


def evaluate_terminal_separability(
    critic,
    dataset: CriticDataset,
    device: torch.device,
) -> dict:
    """评估终态可分性 (A2)"""
    success_v = []
    failure_v = []
    
    with torch.no_grad():
        for episode_idx, episode in enumerate(tqdm(dataset.episodes, desc="Evaluating terminals")):
            # 获取最后一帧
            T = episode["num_frames"]
            last_frame_idx = T - 1
            obs_horizon = dataset.obs_horizon
            
            # 获取观测
            start_idx = max(0, last_frame_idx - obs_horizon + 1)
            
            # 图像
            images = []
            for t in range(start_idx, last_frame_idx + 1):
                img = dataset._preprocess_image(episode["images"][t])
                images.append(img)
            # Pad if needed
            while len(images) < obs_horizon:
                images.insert(0, images[0])
            images = np.stack(images[-obs_horizon:], axis=0)
            
            # 状态
            state = episode["state"][start_idx:last_frame_idx + 1]
            while len(state) < obs_horizon:
                state = np.concatenate([state[:1], state], axis=0)
            state = state[-obs_horizon:]
            
            # 推理
            rgb = torch.from_numpy(images).unsqueeze(0).to(device)
            state_t = torch.from_numpy(state).unsqueeze(0).to(device)
            v = critic(rgb, state_t).squeeze().cpu().item()
            
            if episode["is_success"]:
                success_v.append(v)
            elif episode["is_failure"]:
                failure_v.append(v)
    
    success_v = np.array(success_v)
    failure_v = np.array(failure_v)
    
    # Compute AUC
    if len(success_v) > 0 and len(failure_v) > 0:
        labels = np.concatenate([np.ones(len(success_v)), np.zeros(len(failure_v))])
        scores = np.concatenate([success_v, failure_v])
        auc = roc_auc_score(labels, scores)
    else:
        auc = 0.0
    
    return {
        "terminal_auc": float(auc),
        "success_v_mean": float(success_v.mean()) if len(success_v) > 0 else 0.0,
        "success_v_std": float(success_v.std()) if len(success_v) > 0 else 0.0,
        "failure_v_mean": float(failure_v.mean()) if len(failure_v) > 0 else 0.0,
        "failure_v_std": float(failure_v.std()) if len(failure_v) > 0 else 0.0,
        "num_success": len(success_v),
        "num_failure": len(failure_v),
        "_success_v": success_v,
        "_failure_v": failure_v,
    }


def evaluate_trajectory(
    critic,
    episode: dict,
    obs_horizon: int,
    device: torch.device,
    preprocess_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    """评估单条轨迹，返回 V(s) 序列"""
    T = episode["num_frames"]
    v_values = []
    
    with torch.no_grad():
        for t in range(obs_horizon - 1, T):
            start_idx = t - obs_horizon + 1
            
            # 图像
            images = []
            for i in range(start_idx, t + 1):
                img = preprocess_fn(episode["images"][i])
                images.append(img)
            images = np.stack(images, axis=0)
            
            # 状态
            state = episode["state"][start_idx:t + 1]
            
            # 推理
            rgb = torch.from_numpy(images).unsqueeze(0).to(device)
            state_t = torch.from_numpy(state).unsqueeze(0).to(device)
            v = critic(rgb, state_t).squeeze().cpu().item()
            v_values.append(v)
    
    # 对齐 progress（从 obs_horizon-1 开始）
    progress = episode["progress"][obs_horizon - 1:]
    
    return np.array(v_values), progress


def plot_trajectory_comparison(
    v_values: np.ndarray,
    progress: np.ndarray,
    episode_info: dict,
    save_path: str,
):
    """绘制单条轨迹的 V(s) vs p(s) 对比图"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    t = np.arange(len(v_values))
    
    # 上图：V(s) 和 p(s) 曲线
    ax1 = axes[0]
    ax1.plot(t, v_values, 'b-', linewidth=2, label='V(s) predicted')
    ax1.plot(t, progress[:len(v_values)], 'r--', linewidth=2, label='p(s) ground truth')
    ax1.set_ylabel('Value / Progress')
    ax1.legend(loc='upper left')
    ax1.set_title(f"Episode: {episode_info.get('filepath', 'Unknown')} (reward={episode_info.get('reward', '?')})")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # 下图：Advantage（V变化率）
    ax2 = axes[1]
    advantage = np.diff(v_values)
    delta_p = np.diff(progress[:len(v_values)])
    ax2.fill_between(t[1:], advantage, alpha=0.5, label='Advantage (ΔV)')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Advantage')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_advantage_distribution(
    advantage_pred: np.ndarray,
    advantage_true: np.ndarray,
    save_path: str,
):
    """绘制 Advantage 分布对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：直方图对比
    ax1 = axes[0]
    ax1.hist(advantage_pred, bins=50, alpha=0.7, label='Predicted Advantage', density=True)
    ax1.hist(advantage_true, bins=50, alpha=0.7, label='True Δp', density=True)
    ax1.set_xlabel('Advantage')
    ax1.set_ylabel('Density')
    ax1.set_title('Advantage Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：散点图
    ax2 = axes[1]
    ax2.scatter(advantage_true, advantage_pred, alpha=0.3, s=10)
    
    # 添加对角线
    lim_min = min(advantage_true.min(), advantage_pred.min())
    lim_max = max(advantage_true.max(), advantage_pred.max())
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='y=x')
    
    ax2.set_xlabel('True Δp')
    ax2.set_ylabel('Predicted Advantage')
    ax2.set_title('Advantage Correlation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_terminal_distribution(
    success_v: np.ndarray,
    failure_v: np.ndarray,
    auc: float,
    save_path: str,
):
    """绘制终态 V 分布对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(success_v) > 0:
        ax.hist(success_v, bins=20, alpha=0.7, label=f'Success (n={len(success_v)}, μ={success_v.mean():.3f})',
                color='green', density=True)
    if len(failure_v) > 0:
        ax.hist(failure_v, bins=20, alpha=0.7, label=f'Failure (n={len(failure_v)}, μ={failure_v.mean():.3f})',
                color='red', density=True)
    
    ax.set_xlabel('V(s_terminal)')
    ax.set_ylabel('Density')
    ax.set_title(f'Terminal State Value Distribution (AUC={auc:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 输出目录
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Critic Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    
    # 加载模型
    print("\nLoading model...")
    critic, model_args = load_critic_from_checkpoint(args.checkpoint, device)
    
    # 加载数据集
    print("\nLoading dataset...")
    smdp_dataset = CriticSMDPDataset(
        data_path=args.data_path,
        chunk_size=model_args.get("chunk_size", 8),
        obs_horizon=model_args.get("obs_horizon", 2),
        image_size=(model_args.get("image_size", 224), model_args.get("image_size", 224)),
        state_mode=model_args.get("state_mode", "joint_only"),
        success_threshold=model_args.get("success_threshold", 5),
        failure_threshold=model_args.get("failure_threshold", 3),
        filter_mode="all",  # 评估时用所有数据
        interpolation=model_args.get("interpolation", "step"),
        verbose=True,
    )
    base_dataset = smdp_dataset.base_dataset
    
    # 评估
    results = {}
    
    # 1. Pairwise Ranking
    print("\n[1/3] Evaluating pairwise ranking...")
    ranking_results = evaluate_pairwise_ranking(
        critic, smdp_dataset, device, args.num_samples
    )
    results.update({k: v for k, v in ranking_results.items() if not k.startswith('_')})
    
    print(f"  Rank Accuracy: {ranking_results['rank_accuracy']:.4f}")
    print(f"  Sign Accuracy: {ranking_results['sign_accuracy']:.4f}")
    print(f"  Spearman ρ(V, p): {ranking_results['spearman_vp']:.4f}")
    print(f"  Advantage Spearman: {ranking_results['spearman_advantage']:.4f}")
    print(f"  Per-gap accuracies: {ranking_results['gap_accuracies']}")
    
    # 2. Terminal Separability
    print("\n[2/3] Evaluating terminal separability...")
    terminal_results = evaluate_terminal_separability(critic, base_dataset, device)
    results.update({k: v for k, v in terminal_results.items() if not k.startswith('_')})
    
    print(f"  Terminal AUC: {terminal_results['terminal_auc']:.4f}")
    print(f"  Success V: {terminal_results['success_v_mean']:.4f} ± {terminal_results['success_v_std']:.4f}")
    print(f"  Failure V: {terminal_results['failure_v_mean']:.4f} ± {terminal_results['failure_v_std']:.4f}")
    
    # 3. Trajectory Visualization
    print(f"\n[3/3] Generating trajectory visualizations ({args.num_trajectory_vis} episodes)...")
    
    # 选择要可视化的轨迹（混合成功和失败）
    success_episodes = [i for i, ep in enumerate(base_dataset.episodes) if ep["is_success"]]
    failure_episodes = [i for i, ep in enumerate(base_dataset.episodes) if ep["is_failure"]]
    
    vis_indices = []
    if success_episodes:
        vis_indices.extend(random.sample(success_episodes, min(args.num_trajectory_vis // 2, len(success_episodes))))
    if failure_episodes:
        vis_indices.extend(random.sample(failure_episodes, min(args.num_trajectory_vis // 2, len(failure_episodes))))
    
    for i, ep_idx in enumerate(vis_indices):
        episode = base_dataset.episodes[ep_idx]
        v_values, progress = evaluate_trajectory(
            critic, episode, model_args.get("obs_horizon", 2),
            device, base_dataset._preprocess_image
        )
        
        save_path = os.path.join(args.output_dir, f"trajectory_{i+1}_ep{ep_idx}.png")
        plot_trajectory_comparison(
            v_values, progress,
            {"filepath": os.path.basename(episode["filepath"]), "reward": episode["reward"]},
            save_path
        )
        print(f"  Saved: {save_path}")
    
    # 绘制 Advantage 分布
    print("\nGenerating advantage distribution plot...")
    plot_advantage_distribution(
        ranking_results["_advantage_pred"],
        ranking_results["_advantage_true"],
        os.path.join(args.output_dir, "advantage_distribution.png")
    )
    
    # 绘制终态分布
    print("Generating terminal distribution plot...")
    plot_terminal_distribution(
        terminal_results["_success_v"],
        terminal_results["_failure_v"],
        terminal_results["terminal_auc"],
        os.path.join(args.output_dir, "terminal_distribution.png")
    )
    
    # 保存结果
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations saved to: {args.output_dir}")
    
    # 打印摘要
    print("\n--- Summary ---")
    print(f"Rank Accuracy:     {results['rank_accuracy']:.4f}")
    print(f"Sign Accuracy:     {results['sign_accuracy']:.4f}")
    print(f"Spearman ρ(V, p):  {results['spearman_vp']:.4f}")
    print(f"Terminal AUC:      {results['terminal_auc']:.4f}")


if __name__ == "__main__":
    main()
