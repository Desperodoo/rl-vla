"""
Critic Training Loss Functions

实现 ToDo.md 中定义的 Critic 训练损失函数：

1. Pairwise Ranking Loss (核心)
   - Episode 内采样 t1 < t2，要求 V(s_t2) > V(s_t1)
   
2. Anchor Loss (尺度稳定)
   - 成功终态 V ≈ 1
   - 低进度状态 V ≈ 0
   
3. Temporal Smoothness Loss
   - 相邻帧价值变化平滑
   
4. Inter-Episode Ranking Loss (A2)
   - 成功 episode 终态 > 失败 episode 终态
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def pairwise_ranking_loss(
    v_t1: torch.Tensor,
    v_t2: torch.Tensor,
    p_t1: torch.Tensor,
    p_t2: torch.Tensor,
    tau: float = 0.1,
    margin: float = 0.0,
) -> torch.Tensor:
    """Pairwise Ranking Loss
    
    对于 t1 < t2（即 p_t2 >= p_t1），要求 V(s_t2) >= V(s_t1)
    
    L_rank = log(1 + exp(-(V(s_t2) - V(s_t1) - margin) / tau)) * w
    
    其中权重 w = clip(p_t2 - p_t1, 0, 1)
    
    Args:
        v_t1: [B] 或 [B, 1] 时刻 t1 的预测价值
        v_t2: [B] 或 [B, 1] 时刻 t2 的预测价值 (t2 > t1)
        p_t1: [B] 时刻 t1 的真实进度标签
        p_t2: [B] 时刻 t2 的真实进度标签
        tau: 温度参数，越小越严格
        margin: ranking margin
    
    Returns:
        loss: scalar 损失值
    """
    v_t1 = v_t1.squeeze(-1)
    v_t2 = v_t2.squeeze(-1)
    
    # 计算权重：进度差越大，权重越大
    w = torch.clamp(p_t2 - p_t1, min=0.0, max=1.0)
    
    # Ranking loss: softplus(-diff / tau)
    diff = v_t2 - v_t1 - margin
    loss = F.softplus(-diff / tau) * w
    
    # 返回加权平均
    return loss.mean()


def anchor_loss(
    v_done: torch.Tensor,
    v_low: torch.Tensor,
    target_done: float = 1.0,
    target_low: float = 0.0,
) -> torch.Tensor:
    """Anchor Loss (尺度锚定)
    
    将特定状态的价值锚定到目标值：
    - 成功终态: V(s_done) ≈ 1
    - 低进度状态: V(s_low) ≈ 0
    
    L_anchor = ||V(s_done) - 1||^2 + ||V(s_low) - 0||^2
    
    Args:
        v_done: [N_done, 1] 成功终态的预测价值
        v_low: [N_low, 1] 低进度状态的预测价值
        target_done: 成功终态目标值
        target_low: 低进度状态目标值
    
    Returns:
        loss: scalar 损失值
    """
    loss = 0.0
    
    if v_done is not None and v_done.numel() > 0:
        loss_done = F.mse_loss(v_done.squeeze(-1), 
                               torch.full_like(v_done.squeeze(-1), target_done))
        loss = loss + loss_done
    
    if v_low is not None and v_low.numel() > 0:
        loss_low = F.mse_loss(v_low.squeeze(-1),
                              torch.full_like(v_low.squeeze(-1), target_low))
        loss = loss + loss_low
    
    return loss


def temporal_smoothness_loss(
    v_seq: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Temporal Smoothness Loss
    
    鼓励相邻帧的价值变化平滑：
    
    L_smooth = E[|V(s_{t+1}) - V(s_t)|]
    
    Args:
        v_seq: [B, T] 或 [B, T, 1] 序列价值预测
        mask: [B, T-1] 可选的 mask（例如排除 teleop 接管区域）
    
    Returns:
        loss: scalar 损失值
    """
    if v_seq.dim() == 3:
        v_seq = v_seq.squeeze(-1)  # [B, T]
    
    # 计算相邻帧差值
    diff = torch.abs(v_seq[:, 1:] - v_seq[:, :-1])  # [B, T-1]
    
    if mask is not None:
        diff = diff * mask
        return diff.sum() / (mask.sum() + 1e-8)
    
    return diff.mean()


def inter_episode_ranking_loss(
    v_success: torch.Tensor,
    v_failure: torch.Tensor,
    tau: float = 0.1,
    margin: float = 0.2,
) -> torch.Tensor:
    """Inter-Episode Ranking Loss (A2)
    
    跨 episode ranking：成功终态价值 > 失败终态价值
    
    L_inter = log(1 + exp(-(V(s_pos) - V(s_neg) - m) / tau))
    
    Args:
        v_success: [N_success, 1] 成功 episode 终态价值
        v_failure: [N_failure, 1] 失败 episode 终态价值
        tau: 温度参数
        margin: ranking margin
    
    Returns:
        loss: scalar 损失值
    """
    if v_success.numel() == 0 or v_failure.numel() == 0:
        return torch.tensor(0.0, device=v_success.device)
    
    v_success = v_success.squeeze(-1)  # [N_success]
    v_failure = v_failure.squeeze(-1)  # [N_failure]
    
    # 计算所有成功-失败对的 ranking loss
    # v_success[:, None] - v_failure[None, :] -> [N_success, N_failure]
    diff = v_success.unsqueeze(1) - v_failure.unsqueeze(0) - margin
    loss = F.softplus(-diff / tau)
    
    return loss.mean()


def success_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Success/Failure Classification Loss (A2 可选)
    
    在终态上训练一个分类头，预测 episode 是否成功
    
    L_cls = BCE(q(s_done), is_success)
    
    Args:
        logits: [B] 分类 logits
        labels: [B] 成功标签 (1=success, 0=failure)
    
    Returns:
        loss: scalar BCE 损失
    """
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def critic_total_loss(
    # Ranking loss inputs
    v_t1: torch.Tensor,
    v_t2: torch.Tensor,
    p_t1: torch.Tensor,
    p_t2: torch.Tensor,
    # Anchor loss inputs
    v_done: Optional[torch.Tensor] = None,
    v_low: Optional[torch.Tensor] = None,
    # Smoothness loss inputs (optional)
    v_seq: Optional[torch.Tensor] = None,
    smooth_mask: Optional[torch.Tensor] = None,
    # Inter-episode ranking inputs (A2, optional)
    v_success: Optional[torch.Tensor] = None,
    v_failure: Optional[torch.Tensor] = None,
    # Loss weights
    lambda_anchor: float = 1.0,
    lambda_smooth: float = 0.1,
    lambda_inter: float = 0.5,
    # Hyperparameters
    rank_tau: float = 0.1,
    rank_margin: float = 0.0,
    inter_margin: float = 0.2,
) -> Tuple[torch.Tensor, dict]:
    """计算 Critic 总损失
    
    L = L_rank + λ_a * L_anchor + λ_s * L_smooth + λ_inter * L_inter
    
    Args:
        v_t1, v_t2, p_t1, p_t2: Pairwise ranking loss 输入
        v_done, v_low: Anchor loss 输入
        v_seq, smooth_mask: Smoothness loss 输入
        v_success, v_failure: Inter-episode ranking 输入 (A2)
        lambda_*: 损失权重
        *_tau, *_margin: 超参数
    
    Returns:
        total_loss: 总损失
        loss_dict: 各项损失的字典
    """
    loss_dict = {}
    
    # 1. Pairwise Ranking Loss (核心)
    loss_rank = pairwise_ranking_loss(
        v_t1, v_t2, p_t1, p_t2,
        tau=rank_tau, margin=rank_margin
    )
    loss_dict["loss_rank"] = loss_rank.item()
    total_loss = loss_rank
    
    # 2. Anchor Loss
    if v_done is not None or v_low is not None:
        loss_anc = anchor_loss(v_done, v_low)
        loss_dict["loss_anchor"] = loss_anc.item()
        total_loss = total_loss + lambda_anchor * loss_anc
    
    # 3. Temporal Smoothness Loss
    if v_seq is not None:
        loss_smooth = temporal_smoothness_loss(v_seq, smooth_mask)
        loss_dict["loss_smooth"] = loss_smooth.item()
        total_loss = total_loss + lambda_smooth * loss_smooth
    
    # 4. Inter-Episode Ranking Loss (A2)
    if v_success is not None and v_failure is not None:
        loss_inter = inter_episode_ranking_loss(
            v_success, v_failure,
            tau=rank_tau, margin=inter_margin
        )
        loss_dict["loss_inter"] = loss_inter.item()
        total_loss = total_loss + lambda_inter * loss_inter
    
    loss_dict["total_loss"] = total_loss.item()
    
    return total_loss, loss_dict
