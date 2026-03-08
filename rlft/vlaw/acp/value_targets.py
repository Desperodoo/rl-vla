"""ACP value target 计算（Phase P6.A2）

从 HDF5 轨迹数据的 env_success GT 计算 per-frame normalized value target。
公式：target = clip((-remaining_steps - c_fail * (1-success)) / (max_len + c_fail), -1, 0)

移植自 Evo-RL: modeling_pistar06.py::compute_normalized_value_targets
适配 VLAW HDF5 schema（每条轨迹 env_success (T,) bool）。
"""

from __future__ import annotations

import numpy as np

from rlft.vlaw.acp.config import ValueTargetConfig


def compute_value_targets(
    env_success: np.ndarray,
    episode_length: int,
    max_episode_length: int,
    cfg: ValueTargetConfig,
) -> np.ndarray:
    """为单条轨迹计算 per-frame value target。

    Args:
        env_success: (T,) bool — 每帧的 env_success 标志
        episode_length: 轨迹实际长度（= T）
        max_episode_length: 该 task 下所有轨迹的最大长度
        cfg: ValueTargetConfig

    Returns:
        (T,) float32 — per-frame value target，范围 [clip_min, clip_max]
    """
    T = env_success.shape[0]
    if T != episode_length:
        raise ValueError(
            f"env_success 长度 ({T}) 与 episode_length ({episode_length}) 不匹配"
        )
    if max_episode_length <= 0:
        raise ValueError(f"max_episode_length 必须 > 0, got {max_episode_length}")

    # 判断轨迹是否最终成功（任意帧 success=True 即为成功）
    is_success = bool(np.any(env_success))
    c_fail = float(max_episode_length) * cfg.c_fail_coef

    targets = np.zeros(T, dtype=np.float32)
    denom = float(max_episode_length) + c_fail

    for t in range(T):
        remaining_steps = episode_length - t - 1
        g = -float(remaining_steps)
        if not is_success:
            g -= c_fail
        targets[t] = np.clip(g / denom, cfg.clip_min, cfg.clip_max)

    return targets


def compute_value_targets_batch(
    trajectories: list[dict],
    max_episode_length: int,
    cfg: ValueTargetConfig,
) -> list[np.ndarray]:
    """为一批轨迹计算 value targets。

    Args:
        trajectories: 列表，每项需包含:
            - "env_success": (T,) bool
            - "length": int
        max_episode_length: 全局最大轨迹长度
        cfg: ValueTargetConfig

    Returns:
        列表，每项 (T_i,) float32 value targets
    """
    results = []
    for traj in trajectories:
        env_success = np.asarray(traj["env_success"], dtype=bool)
        length = int(traj["length"])
        target = compute_value_targets(
            env_success=env_success,
            episode_length=length,
            max_episode_length=max_episode_length,
            cfg=cfg,
        )
        results.append(target)
    return results
