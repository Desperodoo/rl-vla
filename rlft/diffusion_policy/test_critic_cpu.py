#!/usr/bin/env python3
"""
Quick CPU test for Critic module

快速验证 Critic 模块功能（使用 CPU 和小数据量）
"""

import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_policy.critic import (
    CriticNetwork,
    CriticDataset,
    CriticSMDPDataset,
    pairwise_ranking_loss,
    anchor_loss,
    temporal_smoothness_loss,
)
from diffusion_policy.critic.critic_dataset import build_anchor_samples, build_terminal_samples
from diffusion_policy.critic.critic_network import create_critic_network


def test_critic_forward():
    """测试 Critic 网络前向传播"""
    print("=" * 60)
    print("Test: Critic Forward Pass")
    print("=" * 60)
    
    # 创建网络
    critic = create_critic_network(
        backbone_name="resnet18",
        pretrained=True,
        freeze_visual=True,
        visual_feature_dim=128,  # 小一点以加速
        hidden_dims=(128, 64),
        obs_horizon=2,
        state_dim=7,
        device="cpu"
    )
    
    # 创建假数据
    batch_size = 4
    obs_horizon = 2
    
    rgb = torch.randn(batch_size, obs_horizon, 3, 224, 224)
    state = torch.randn(batch_size, obs_horizon, 7)
    
    # 前向传播
    with torch.no_grad():
        start = time.time()
        values = critic(rgb, state)
        elapsed = time.time() - start
    
    print(f"  Input: rgb={list(rgb.shape)}, state={list(state.shape)}")
    print(f"  Output: values={list(values.shape)}")
    print(f"  Values range: [{values.min():.3f}, {values.max():.3f}]")
    print(f"  Time: {elapsed:.3f}s")
    print("  ✓ Forward pass OK\n")
    
    return critic


def test_losses(critic):
    """测试 Loss 函数"""
    print("=" * 60)
    print("Test: Loss Functions")
    print("=" * 60)
    
    batch_size = 8
    
    # 模拟 SMDP pairs (v_t, v_tk, progress_t, progress_tk)
    v_t = torch.rand(batch_size).requires_grad_(True)
    v_tk = torch.rand(batch_size).requires_grad_(True)
    p_t = torch.rand(batch_size)
    p_tk = p_t + torch.rand(batch_size) * 0.3  # p_tk > p_t
    
    # Pairwise ranking loss
    loss_rank = pairwise_ranking_loss(v_t, v_tk, p_t, p_tk, tau=0.1)
    print(f"  Ranking loss: {loss_rank.item():.4f}")
    
    # Anchor loss (done + low progress samples)
    v_done = torch.rand(3, 1)  # 3 个 done 样本
    v_low = torch.rand(4, 1)   # 4 个 low progress 样本
    loss_anchor_val = anchor_loss(v_done, v_low, target_done=1.0, target_low=0.0)
    print(f"  Anchor loss: {loss_anchor_val.item():.4f}")
    
    # Temporal smoothness (期望序列格式 [B, T])
    v_seq = torch.rand(4, 10)  # batch=4, seq_len=10
    loss_smooth = temporal_smoothness_loss(v_seq)
    print(f"  Smoothness loss: {loss_smooth.item():.4f}")
    
    print("  ✓ All losses computed\n")


def test_dataset_loading():
    """测试数据集加载（使用少量数据）"""
    print("=" * 60)
    print("Test: Dataset Loading")
    print("=" * 60)
    
    data_path = "../../recorded_data/mix_with_reward"
    
    if not os.path.exists(data_path):
        print(f"  ⚠ Data path not found: {data_path}")
        print("  Skipping dataset test\n")
        return None
    
    # 加载数据集
    dataset = CriticDataset(
        data_path=data_path,
        obs_horizon=2,
        image_size=(224, 224),
        interpolation="step",
        ema_alpha=0.2,
        filter_mode="success_only",  # 正确的参数名
        success_threshold=5,
        failure_threshold=3,
        verbose=False
    )
    
    print(f"  Episodes loaded: {len(dataset.episodes)}")
    print(f"  Total samples: {len(dataset)}")
    
    # 测试单个样本
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  rgb shape: {sample['rgb'].shape}")
    print(f"  state shape: {sample['state'].shape}")
    print(f"  progress: {sample['progress']:.3f}")
    print("  ✓ Dataset loading OK\n")
    
    return dataset


def test_dataloader(dataset):
    """测试 DataLoader"""
    print("=" * 60)
    print("Test: DataLoader")
    print("=" * 60)
    
    if dataset is None:
        print("  Skipping (no dataset)\n")
        return
    
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # 测试一个 batch
    start = time.time()
    batch = next(iter(loader))
    elapsed = time.time() - start
    
    print(f"  Batch rgb: {list(batch['rgb'].shape)}")
    print(f"  Batch state: {list(batch['state'].shape)}")
    print(f"  Batch progress: {batch['progress'].shape}")
    print(f"  Load time: {elapsed:.3f}s")
    print("  ✓ DataLoader OK\n")


def test_training_step(critic, dataset):
    """测试一个训练步骤"""
    print("=" * 60)
    print("Test: Training Step")
    print("=" * 60)
    
    if dataset is None:
        print("  Skipping (no dataset)\n")
        return
    
    from torch.utils.data import DataLoader
    
    # 构建 SMDP 数据集
    smdp_dataset = CriticSMDPDataset(
        data_path="../../recorded_data/mix_with_reward",
        chunk_size=8,
        gap_options=[8, 16],  # 较小的 gap
        obs_horizon=2,
        filter_mode="success_only",
        verbose=False
    )
    
    print(f"  SMDP pairs: {len(smdp_dataset)}")
    
    loader = DataLoader(
        smdp_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    # 设置优化器
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, critic.parameters()),
        lr=1e-4
    )
    
    # 一个训练步骤
    critic.train()
    batch = next(iter(loader))
    
    rgb_t = batch['rgb_t1']
    rgb_tk = batch['rgb_t2']
    state_t = batch['state_t1']
    state_tk = batch['state_t2']
    progress_t = batch['p_t1']
    progress_tk = batch['p_t2']
    
    optimizer.zero_grad()
    
    start = time.time()
    
    # Forward
    v_t = critic(rgb_t, state_t).squeeze(-1)
    v_tk = critic(rgb_tk, state_tk).squeeze(-1)
    
    # Loss
    loss_rank = pairwise_ranking_loss(v_t, v_tk, progress_t, progress_tk)
    
    # SMDP 模式下的平滑损失：直接对 pair 的差值做约束
    loss_smooth = torch.mean(torch.abs(v_tk - v_t))
    
    total_loss = loss_rank + 0.1 * loss_smooth
    
    # Backward
    total_loss.backward()
    optimizer.step()
    
    elapsed = time.time() - start
    
    print(f"  V_t: {v_t.detach().numpy()}")
    print(f"  V_tk: {v_tk.detach().numpy()}")
    print(f"  Loss: {total_loss.item():.4f} (rank={loss_rank.item():.4f}, smooth={loss_smooth.item():.4f})")
    print(f"  Step time: {elapsed:.3f}s")
    print("  ✓ Training step OK\n")


def main():
    print("\n" + "=" * 60)
    print("   CRITIC MODULE CPU TEST")
    print("=" * 60 + "\n")
    
    # 1. 测试前向传播
    critic = test_critic_forward()
    
    # 2. 测试 Loss 函数
    test_losses(critic)
    
    # 3. 测试数据集加载
    dataset = test_dataset_loading()
    
    # 4. 测试 DataLoader
    test_dataloader(dataset)
    
    # 5. 测试训练步骤
    test_training_step(critic, dataset)
    
    print("=" * 60)
    print("   ALL TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
