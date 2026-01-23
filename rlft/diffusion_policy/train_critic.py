"""
Critic Training Script

训练排序式 Critic V(s)，用于计算 chunk-level advantage。

支持两种训练模式：
- A1 (success_only): 仅使用成功 episode (reward >= 5)
- A2 (success_failure): 使用成功 + 失败 episode

Usage:
    # A1: 仅成功数据
    python train_critic.py --data-path ~/rl-vla/recorded_data/mix_with_reward \
        --mode success_only --epochs 100
    
    # A2: 成功 + 失败数据
    python train_critic.py --data-path ~/rl-vla/recorded_data/mix_with_reward \
        --mode success_failure --epochs 100
"""

import os
import sys
import json
import time
import argparse
import random
from datetime import datetime
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_policy.critic import (
    CriticNetwork,
    CriticDataset,
    CriticSMDPDataset,
    pairwise_ranking_loss,
    anchor_loss,
    temporal_smoothness_loss,
    inter_episode_ranking_loss,
    critic_total_loss,
)
from diffusion_policy.critic.critic_dataset import build_anchor_samples, build_terminal_samples
from diffusion_policy.critic.critic_network import create_critic_network
from diffusion_policy.resnet_encoder import ResNetEncoder
from diffusion_policy.carm_utils import StateEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train Critic V(s)")
    
    # Data settings
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to labeled dataset (with reward_timeline)")
    parser.add_argument("--mode", type=str, choices=["success_only", "success_failure"],
                        default="success_only", help="Training mode: A1 or A2")
    parser.add_argument("--success-threshold", type=int, default=5,
                        help="Reward threshold for success (>=)")
    parser.add_argument("--failure-threshold", type=int, default=3,
                        help="Reward threshold for failure (<=)")
    
    # Model settings
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--visual-dim", type=int, default=256,
                        help="Visual feature dimension")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256],
                        help="MLP hidden dimensions")
    parser.add_argument("--obs-horizon", type=int, default=2,
                        help="Observation stacking horizon")
    parser.add_argument("--state-mode", type=str, default="joint_only",
                        choices=["joint_only", "ee_only", "both"])
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Action chunk size K for SMDP")
    
    # Loss weights
    parser.add_argument("--lambda-anchor", type=float, default=1.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.1)
    parser.add_argument("--lambda-inter", type=float, default=0.5,
                        help="Inter-episode ranking weight (A2 only)")
    parser.add_argument("--rank-tau", type=float, default=0.1,
                        help="Temperature for ranking loss")
    
    # Progress label settings
    parser.add_argument("--interpolation", type=str, default="step",
                        choices=["step", "linear"])
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_state_dim(state_mode: str) -> int:
    """根据 state_mode 返回状态维度"""
    if state_mode == "joint_only":
        return 7
    elif state_mode == "ee_only":
        return 8
    else:  # both
        return 14


class AnchorSampler:
    """辅助类：从数据集中采样 anchor 样本"""
    
    def __init__(self, dataset: CriticDataset):
        self.dataset = dataset
        self.done_indices, self.low_indices = build_anchor_samples(dataset)
        self.success_terminal, self.failure_terminal = build_terminal_samples(dataset)
        
        print(f"Anchor samples: done={len(self.done_indices)}, low={len(self.low_indices)}")
        print(f"Terminal samples: success={len(self.success_terminal)}, failure={len(self.failure_terminal)}")
    
    def sample_anchor_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Optional[dict], Optional[dict]]:
        """采样 anchor batch"""
        done_batch = None
        low_batch = None
        
        # 采样 done 样本
        if len(self.done_indices) > 0:
            n_done = min(batch_size // 2, len(self.done_indices))
            indices = random.sample(self.done_indices, n_done)
            done_batch = self._collate([self.dataset[i] for i in indices], device)
        
        # 采样 low progress 样本
        if len(self.low_indices) > 0:
            n_low = min(batch_size // 2, len(self.low_indices))
            indices = random.sample(self.low_indices, n_low)
            low_batch = self._collate([self.dataset[i] for i in indices], device)
        
        return done_batch, low_batch
    
    def sample_terminal_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Optional[dict], Optional[dict]]:
        """采样终态样本（用于 inter-episode ranking）"""
        success_batch = None
        failure_batch = None
        
        if len(self.success_terminal) > 0:
            n_success = min(batch_size // 2, len(self.success_terminal))
            indices = random.sample(self.success_terminal, n_success)
            success_batch = self._collate([self.dataset[i] for i in indices], device)
        
        if len(self.failure_terminal) > 0:
            n_failure = min(batch_size // 2, len(self.failure_terminal))
            indices = random.sample(self.failure_terminal, n_failure)
            failure_batch = self._collate([self.dataset[i] for i in indices], device)
        
        return success_batch, failure_batch
    
    def _collate(self, samples, device):
        """Collate samples into batch"""
        batch = {
            "rgb": torch.stack([s["rgb"] for s in samples]).to(device),
            "state": torch.stack([s["state"] for s in samples]).to(device),
            "progress": torch.stack([s["progress"] for s in samples]).to(device),
        }
        return batch


def train_epoch(
    critic: CriticNetwork,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    anchor_sampler: AnchorSampler,
    args,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
) -> dict:
    """训练一个 epoch"""
    critic.train()
    
    total_loss = 0.0
    loss_components = defaultdict(float)
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # 移动数据到设备
        rgb_t1 = batch["rgb_t1"].to(device)
        state_t1 = batch["state_t1"].to(device)
        rgb_t2 = batch["rgb_t2"].to(device)
        state_t2 = batch["state_t2"].to(device)
        p_t1 = batch["p_t1"].to(device)
        p_t2 = batch["p_t2"].to(device)
        
        # 前向传播
        v_t1 = critic(rgb_t1, state_t1)
        v_t2 = critic(rgb_t2, state_t2)
        
        # 1. Ranking Loss
        loss_rank = pairwise_ranking_loss(v_t1, v_t2, p_t1, p_t2, tau=args.rank_tau)
        loss = loss_rank
        loss_components["loss_rank"] += loss_rank.item()
        
        # 2. Anchor Loss
        done_batch, low_batch = anchor_sampler.sample_anchor_batch(
            batch_size=len(rgb_t1) // 2, device=device
        )
        v_done = None
        v_low = None
        if done_batch is not None:
            v_done = critic(done_batch["rgb"], done_batch["state"])
        if low_batch is not None:
            v_low = critic(low_batch["rgb"], low_batch["state"])
        
        if v_done is not None or v_low is not None:
            loss_anc = anchor_loss(v_done, v_low)
            loss = loss + args.lambda_anchor * loss_anc
            loss_components["loss_anchor"] += loss_anc.item()
        
        # 3. Inter-Episode Ranking (A2 only)
        if args.mode == "success_failure":
            success_batch, failure_batch = anchor_sampler.sample_terminal_batch(
                batch_size=len(rgb_t1) // 2, device=device
            )
            if success_batch is not None and failure_batch is not None:
                v_success = critic(success_batch["rgb"], success_batch["state"])
                v_failure = critic(failure_batch["rgb"], failure_batch["state"])
                loss_inter = inter_episode_ranking_loss(v_success, v_failure, tau=args.rank_tau)
                loss = loss + args.lambda_inter * loss_inter
                loss_components["loss_inter"] += loss_inter.item()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            "loss": loss.item(),
            "rank": loss_components["loss_rank"] / num_batches,
        })
    
    # 计算平均损失
    avg_loss = total_loss / num_batches
    for k in loss_components:
        loss_components[k] /= num_batches
    
    # 记录到 TensorBoard
    global_step = epoch * len(dataloader)
    writer.add_scalar("train/total_loss", avg_loss, global_step)
    for k, v in loss_components.items():
        writer.add_scalar(f"train/{k}", v, global_step)
    
    return {"total_loss": avg_loss, **loss_components}


def evaluate(
    critic: CriticNetwork,
    dataset: CriticSMDPDataset,
    device: torch.device,
    num_samples: int = 1000,
) -> dict:
    """评估 Critic 性能"""
    critic.eval()
    
    # 随机采样
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    v_t1_list = []
    v_t2_list = []
    p_t1_list = []
    p_t2_list = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating", leave=False):
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
    
    v_t1_arr = np.array(v_t1_list)
    v_t2_arr = np.array(v_t2_list)
    p_t1_arr = np.array(p_t1_list)
    p_t2_arr = np.array(p_t2_list)
    
    # 计算指标
    # 1. Pairwise Ranking Accuracy
    # 预测: V(s_t2) > V(s_t1)，真实: p_t2 > p_t1
    pred_rank = v_t2_arr > v_t1_arr
    true_rank = p_t2_arr > p_t1_arr
    rank_accuracy = (pred_rank == true_rank).mean()
    
    # 2. Advantage 方向一致性
    pred_advantage = v_t2_arr - v_t1_arr
    true_delta_p = p_t2_arr - p_t1_arr
    # sign accuracy
    pred_sign = np.sign(pred_advantage)
    true_sign = np.sign(true_delta_p)
    sign_accuracy = (pred_sign == true_sign).mean()
    
    # 3. Spearman 相关系数
    from scipy.stats import spearmanr
    # V vs p
    all_v = np.concatenate([v_t1_arr, v_t2_arr])
    all_p = np.concatenate([p_t1_arr, p_t2_arr])
    spearman_rho, _ = spearmanr(all_v, all_p)
    
    # Advantage correlation
    adv_spearman, _ = spearmanr(pred_advantage, true_delta_p)
    
    return {
        "rank_accuracy": rank_accuracy,
        "sign_accuracy": sign_accuracy,
        "spearman_rho": spearman_rho,
        "advantage_spearman": adv_spearman,
        "v_mean": all_v.mean(),
        "v_std": all_v.std(),
    }


def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建实验目录
    if args.exp_name is None:
        args.exp_name = f"critic_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"runs/{args.exp_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
    
    # 保存配置
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 60)
    print(f"Critic Training - {args.mode.upper()}")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Mode: {args.mode}")
    print(f"Log dir: {log_dir}")
    
    # 确定 filter_mode
    filter_mode = "success_only" if args.mode == "success_only" else "success_failure"
    
    # 创建数据集
    print("\nLoading dataset...")
    smdp_dataset = CriticSMDPDataset(
        data_path=args.data_path,
        chunk_size=args.chunk_size,
        obs_horizon=args.obs_horizon,
        image_size=(args.image_size, args.image_size),
        state_mode=args.state_mode,
        success_threshold=args.success_threshold,
        failure_threshold=args.failure_threshold,
        filter_mode=filter_mode,
        interpolation=args.interpolation,
        do_ema_smooth=True,
        ema_alpha=args.ema_alpha,
        verbose=True,
    )
    
    # 为 anchor sampling 创建基础数据集引用
    base_dataset = smdp_dataset.base_dataset
    anchor_sampler = AnchorSampler(base_dataset)
    
    # 创建 DataLoader
    dataloader = DataLoader(
        smdp_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # 创建 Critic 网络
    print("\nCreating Critic network...")
    state_dim = get_state_dim(args.state_mode)
    critic = create_critic_network(
        state_dim=state_dim,
        visual_feature_dim=args.visual_dim,
        obs_horizon=args.obs_horizon,
        hidden_dims=tuple(args.hidden_dims),
        backbone_name=args.backbone,
        pretrained=True,
        freeze_visual=True,  # 冻结 ResNet backbone
        freeze_bn=True,
        device=str(device),
    )
    
    # 打印参数信息
    total_params = sum(p.numel() for p in critic.parameters())
    trainable_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # 创建优化器（只优化可训练参数）
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, critic.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 训练循环
    print("\nStarting training...")
    best_rank_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_metrics = train_epoch(
            critic, dataloader, optimizer, anchor_sampler,
            args, device, epoch, writer
        )
        
        # 更新学习率
        scheduler.step()
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
        
        # 定期评估
        if epoch % args.log_interval == 0 or epoch == args.epochs:
            eval_metrics = evaluate(critic, smdp_dataset, device)
            
            print(f"\nEpoch {epoch}/{args.epochs}:")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Rank Accuracy: {eval_metrics['rank_accuracy']:.4f}")
            print(f"  Sign Accuracy: {eval_metrics['sign_accuracy']:.4f}")
            print(f"  Spearman ρ(V, p): {eval_metrics['spearman_rho']:.4f}")
            print(f"  Advantage Spearman: {eval_metrics['advantage_spearman']:.4f}")
            
            # 记录到 TensorBoard
            for k, v in eval_metrics.items():
                writer.add_scalar(f"eval/{k}", v, epoch)
            
            # 保存最佳模型
            if eval_metrics['rank_accuracy'] > best_rank_acc:
                best_rank_acc = eval_metrics['rank_accuracy']
                torch.save({
                    "epoch": epoch,
                    "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": eval_metrics,
                    "args": vars(args),
                }, f"{log_dir}/checkpoints/best.pt")
                print(f"  -> New best model saved! (rank_acc={best_rank_acc:.4f})")
        
        # 定期保存检查点
        if epoch % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }, f"{log_dir}/checkpoints/epoch_{epoch}.pt")
    
    # 保存最终模型
    torch.save({
        "epoch": args.epochs,
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }, f"{log_dir}/checkpoints/final.pt")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best rank accuracy: {best_rank_acc:.4f}")
    print(f"Checkpoints saved to: {log_dir}/checkpoints/")
    print("=" * 60)
    
    writer.close()


if __name__ == "__main__":
    main()
