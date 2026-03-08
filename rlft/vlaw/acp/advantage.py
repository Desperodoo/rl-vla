"""ACP advantage 计算（Phase P6.A3）

N-step advantage 计算、per-task quantile 阈值、二值化、连续权重归一化。
移植自 Evo-RL: lerobot_value_infer.py 的核心算法函数。
"""

from __future__ import annotations

import numpy as np

from rlft.vlaw.acp.config import AdvantageConfig


def compute_dense_rewards(
    targets: np.ndarray,
) -> np.ndarray:
    """从 value targets 推导 per-frame dense reward。

    r[t] = target[t] - target[t+1]  （同一条轨迹内）
    r[T-1] = target[T-1]            （末帧）

    Args:
        targets: (T,) float32 — 单条轨迹的 value targets

    Returns:
        (T,) float32 — per-frame rewards
    """
    T = targets.shape[0]
    rewards = np.zeros(T, dtype=np.float32)
    for t in range(T - 1):
        rewards[t] = float(targets[t] - targets[t + 1])
    rewards[T - 1] = float(targets[T - 1])
    return rewards


def compute_n_step_advantage(
    rewards: np.ndarray,
    values: np.ndarray,
    n_step: int,
) -> np.ndarray:
    """计算 N-step advantage（单条轨迹）。

    A(t) = Σ_{k=0}^{n-1} r[t+k] + V(t+n) - V(t)
    当 t+n 超出边界时 bootstrap=0。

    Args:
        rewards: (T,) float32 — dense rewards
        values: (T,) float32 — predicted values
        n_step: N-step 步数（>0）

    Returns:
        (T,) float32 — per-frame advantage
    """
    if n_step <= 0:
        raise ValueError(f"n_step 必须 > 0, got {n_step}")

    T = rewards.shape[0]
    advantages = np.zeros(T, dtype=np.float32)

    for t in range(T):
        # 累积 n-step rewards
        discounted_sum = 0.0
        steps = min(n_step, T - t)
        for k in range(steps):
            discounted_sum += float(rewards[t + k])

        # Bootstrap value at t+n
        if t + n_step < T:
            bootstrap = float(values[t + n_step])
        else:
            bootstrap = 0.0

        advantages[t] = discounted_sum + bootstrap - float(values[t])

    return advantages


def compute_task_threshold(
    advantages: np.ndarray,
    positive_ratio: float,
) -> float:
    """计算单个 task 的 advantage 阈值（quantile）。

    threshold = quantile(advantages, 1 - positive_ratio)
    advantage >= threshold 的帧标记为 positive。

    Args:
        advantages: (N,) float32 — 该 task 所有帧的 advantage
        positive_ratio: positive 帧比例目标 (0, 1)

    Returns:
        float — 阈值
    """
    if not 0.0 <= positive_ratio <= 1.0:
        raise ValueError(f"positive_ratio 必须在 [0,1], got {positive_ratio}")
    if advantages.size == 0:
        return float("inf")
    quantile = 1.0 - positive_ratio
    return float(np.quantile(advantages, quantile))


def binarize_advantages(
    advantages: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """将连续 advantage 二值化为 indicator。

    Args:
        advantages: (T,) float32
        threshold: advantage >= threshold → 1, else → 0

    Returns:
        (T,) int32 — binary indicator
    """
    return (advantages >= threshold).astype(np.int32)


def normalize_advantages_to_weights(
    advantages: np.ndarray,
    cfg: AdvantageConfig,
) -> np.ndarray:
    """将连续 advantage 归一化为 [0, weight_clip_max] 连续权重。

    归一化策略：
    1. 先将 advantage shift 到 >=0 范围（减去最小值）
    2. 除以 (max - min) 归一化到 [0, 1]
    3. clip 到 [weight_clip_min, weight_clip_max]

    若 advantage 全部相同（方差=0），返回全 1.0 权重。

    Args:
        advantages: (T,) float32
        cfg: AdvantageConfig

    Returns:
        (T,) float32 — 归一化权重
    """
    a_min = float(np.min(advantages))
    a_max = float(np.max(advantages))

    if a_max - a_min < 1e-8:
        return np.ones_like(advantages, dtype=np.float32)

    normalized = (advantages - a_min) / (a_max - a_min)
    return np.clip(normalized, cfg.weight_clip_min, cfg.weight_clip_max).astype(
        np.float32
    )


def compute_trajectory_weights(
    value_targets: np.ndarray,
    predicted_values: np.ndarray,
    cfg: AdvantageConfig,
) -> dict[str, np.ndarray]:
    """完整的单条轨迹 advantage → weight 计算流水线。

    Args:
        value_targets: (T,) float32 — GT value targets
        predicted_values: (T,) float32 — model predicted values
        cfg: AdvantageConfig

    Returns:
        dict:
            "rewards": (T,) float32
            "advantages": (T,) float32
            "indicators": (T,) int32
            "weights": (T,) float32
    """
    rewards = compute_dense_rewards(value_targets)
    advantages = compute_n_step_advantage(rewards, predicted_values, cfg.n_step)
    threshold = compute_task_threshold(advantages, cfg.positive_ratio)
    indicators = binarize_advantages(advantages, threshold)

    if cfg.use_continuous_weights:
        weights = normalize_advantages_to_weights(advantages, cfg)
    else:
        weights = indicators.astype(np.float32)

    return {
        "rewards": rewards,
        "advantages": advantages,
        "indicators": indicators,
        "weights": weights,
    }
