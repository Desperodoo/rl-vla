"""
Critic Module for Advantage Estimation

基于 RoboReward 标注的离散 1-5 reward + done 标签，
训练排序式（ranking-based）Critic V(s)，计算 chunk-level advantage。

Components:
- progress_labels: 从 reward_timeline 计算逐帧进度标签 p_t
- critic_network: V(s) 网络，复用 ResNet + StateEncoder
- losses: Ranking Loss, Anchor Loss, Smoothness Loss
- critic_dataset: SMDP 数据集，构建状态对用于训练

Usage:
    # A1: 仅成功数据训练
    python train_critic.py --data-path ./data --mode success_only
    
    # A2: 成功+失败数据训练
    python train_critic.py --data-path ./data --mode success_failure
    
    # 评估
    python eval_critic.py --checkpoint runs/critic_xxx/checkpoints/best.pt --data-path ./data
"""

from .progress_labels import (
    compute_progress_labels,
    interpolate_reward_timeline,
    load_progress_labels_from_hdf5,
)
from .critic_network import CriticNetwork, create_critic_network
from .losses import (
    pairwise_ranking_loss,
    anchor_loss,
    temporal_smoothness_loss,
    inter_episode_ranking_loss,
    critic_total_loss,
)
from .critic_dataset import (
    CriticDataset,
    CriticSMDPDataset,
    build_anchor_samples,
    build_terminal_samples,
)

__all__ = [
    # Progress labels
    "compute_progress_labels",
    "interpolate_reward_timeline",
    "load_progress_labels_from_hdf5",
    # Network
    "CriticNetwork",
    "create_critic_network",
    # Losses
    "pairwise_ranking_loss",
    "anchor_loss", 
    "temporal_smoothness_loss",
    "inter_episode_ranking_loss",
    "critic_total_loss",
    # Dataset
    "CriticDataset",
    "CriticSMDPDataset",
    "build_anchor_samples",
    "build_terminal_samples",
]
