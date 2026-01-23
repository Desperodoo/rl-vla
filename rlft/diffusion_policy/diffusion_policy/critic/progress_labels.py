"""
Progress Labels Computation

从 RoboReward 的逐帧 reward（per_frame_reward）或 reward_timeline 生成进度标签 p_t。

进度标签 p_t ∈ [0, 1] 表示任务完成程度：
- p_t = 0: 任务刚开始
- p_t = 1: 任务完成

支持两种数据格式：
1. per_frame_reward（推荐）：直接使用逐帧标注的 reward，无需插值
2. reward_timeline（旧格式）：从10个检查点分数插值生成逐帧 reward

计算步骤：
1. 获取逐帧 reward (1-5)：直接使用 per_frame_reward 或从 reward_timeline 插值
2. 归一化到 [0, 1]: r_tilde = (reward - 1) / 4
3. Episode 内归一化（可选）
4. EMA 时间平滑（可选）
"""

import numpy as np
from typing import Tuple, Optional, Literal


def interpolate_reward_timeline(
    checkpoints: np.ndarray,
    scores: np.ndarray,
    total_frames: int,
    method: Literal["step", "linear"] = "step",
) -> np.ndarray:
    """从 reward_timeline 检查点插值生成逐帧 reward
    
    Args:
        checkpoints: 检查点帧位置 [num_checkpoints]，1-indexed 截止帧
        scores: 对应分数 [num_checkpoints]，值为 1-5
        total_frames: 总帧数
        method: 插值方法
            - "step": 阶梯函数，每个区间使用该区间结束时的分数
            - "linear": 线性插值
    
    Returns:
        per_frame_reward: [total_frames] 每帧的 reward (1-5)
    
    Example:
        checkpoints = [46, 93, 140, ..., 469]
        scores = [1, 2, 2, ..., 5]
        
        step 方法:
            frame 0-45: score 1 (第一个checkpoint的分数)
            frame 46-92: score 2
            ...
        
        linear 方法:
            在相邻 checkpoint 之间线性插值
    """
    per_frame_reward = np.zeros(total_frames, dtype=np.float32)
    
    # 确保 checkpoints 是 0-indexed 的结束位置（exclusive）
    # 原始 checkpoints 是 1-indexed 的截止帧，需要转换
    ckpts = np.array(checkpoints)  # [46, 93, ..., 469]
    
    if method == "step":
        # 阶梯函数：每个区间使用该区间的分数
        prev_end = 0
        for i, (end, score) in enumerate(zip(ckpts, scores)):
            per_frame_reward[prev_end:end] = score
            prev_end = end
        # 处理最后可能的剩余帧
        if prev_end < total_frames:
            per_frame_reward[prev_end:] = scores[-1]
    
    elif method == "linear":
        # 线性插值
        # 将 checkpoints 视为评估点（到该帧为止的表现）
        # 创建插值点：帧0的分数用第一个checkpoint的分数
        x_points = np.concatenate([[0], ckpts - 1])  # 转为 0-indexed
        y_points = np.concatenate([[scores[0]], scores])  # 帧0使用第一个分数
        
        # 确保最后一个点覆盖 total_frames
        if x_points[-1] < total_frames - 1:
            x_points = np.append(x_points, total_frames - 1)
            y_points = np.append(y_points, scores[-1])
        
        # 线性插值
        frame_indices = np.arange(total_frames)
        per_frame_reward = np.interp(frame_indices, x_points, y_points)
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return per_frame_reward.astype(np.float32)


def normalize_to_01(reward: np.ndarray) -> np.ndarray:
    """将 1-5 reward 归一化到 [0, 1]
    
    r_tilde = (reward - 1) / 4
    """
    return (reward - 1.0) / 4.0


def episode_normalize(
    p: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Episode 内归一化，使 p 的范围在 [0, 1]
    
    p_norm = (p - p.min()) / (p.max() - p.min() + eps)
    
    这使得每个 episode 的进度都从接近 0 开始，到接近 1 结束（如果任务完成）
    """
    p_min = p.min()
    p_max = p.max()
    return (p - p_min) / (p_max - p_min + eps)


def ema_smooth(
    p: np.ndarray,
    alpha: float = 0.2,
) -> np.ndarray:
    """指数移动平均平滑
    
    p_smooth[t] = alpha * p[t] + (1 - alpha) * p_smooth[t-1]
    
    Args:
        p: 输入序列
        alpha: 平滑系数，越大越接近原始值
    
    Returns:
        平滑后的序列
    """
    p_smooth = np.zeros_like(p)
    p_smooth[0] = p[0]
    for t in range(1, len(p)):
        p_smooth[t] = alpha * p[t] + (1 - alpha) * p_smooth[t - 1]
    return p_smooth


def compute_progress_labels(
    checkpoints: np.ndarray,
    scores: np.ndarray,
    total_frames: int,
    done_array: Optional[np.ndarray] = None,
    is_success: bool = True,
    interpolation: Literal["step", "linear"] = "step",
    do_episode_normalize: bool = True,
    do_ema_smooth: bool = True,
    ema_alpha: float = 0.2,
    failure_scale: float = 0.3,
    per_frame_reward: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算逐帧进度标签 p_t
    
    完整流程：
    1. 获取逐帧 reward：直接使用 per_frame_reward 或从 reward_timeline 插值
    2. 归一化到 [0, 1]
    3. 对失败 episode 应用缩放因子（A2）
    4. Episode 内归一化（可选）
    5. EMA 时间平滑（可选）
    6. 强制锚定终态：成功 p_T=1，失败 p_T=0
    
    Args:
        checkpoints: 检查点帧位置（当 per_frame_reward 为 None 时使用）
        scores: 对应分数 (1-5)（当 per_frame_reward 为 None 时使用）
        total_frames: 总帧数
        done_array: done 标记数组（可选，用于确定成功/失败）
        is_success: 是否为成功 episode（reward >= 5）
        interpolation: 插值方法 "step" 或 "linear"（当 per_frame_reward 为 None 时使用）
        do_episode_normalize: 是否进行 episode 内归一化
        do_ema_smooth: 是否进行 EMA 平滑
        ema_alpha: EMA 平滑系数
        failure_scale: 失败 episode 的进度缩放因子（A2）
        per_frame_reward: 直接提供的逐帧 reward [total_frames]（优先使用，无需插值）
    
    Returns:
        Tuple[per_frame_reward, progress_labels]:
            - per_frame_reward: [total_frames] 逐帧 reward (1-5)
            - progress_labels: [total_frames] 进度标签 p_t ∈ [0, 1]
    """
    # Step 1: 获取逐帧 reward
    if per_frame_reward is not None:
        # 直接使用提供的逐帧 reward（推荐方式）
        per_frame_reward = np.asarray(per_frame_reward, dtype=np.float32)
        # 如果需要，裁剪或填充到 total_frames
        if len(per_frame_reward) != total_frames:
            if len(per_frame_reward) > total_frames:
                per_frame_reward = per_frame_reward[:total_frames]
            else:
                # 用最后一个值填充
                pad_length = total_frames - len(per_frame_reward)
                per_frame_reward = np.concatenate([
                    per_frame_reward,
                    np.full(pad_length, per_frame_reward[-1])
                ])
    else:
        # 从 reward_timeline 插值生成逐帧 reward（旧方式）
        per_frame_reward = interpolate_reward_timeline(
            checkpoints, scores, total_frames, method=interpolation
        )
    
    # Step 2: 归一化到 [0, 1]
    p = normalize_to_01(per_frame_reward)
    
    # Step 3: 对失败 episode 应用缩放因子（A2 方案）
    if not is_success:
        p = p * failure_scale
    
    # Step 4: Episode 内归一化（可选）
    if do_episode_normalize:
        p = episode_normalize(p)
    
    # Step 5: EMA 时间平滑（可选）
    if do_ema_smooth:
        p = ema_smooth(p, alpha=ema_alpha)
    
    # Step 6: 强制锚定终态
    if is_success:
        # 成功 episode: 终止帧强制 p_T = 1
        # 如果有 done_array，从 done 开始的帧都设为 1
        if done_array is not None and done_array.any():
            done_start = np.argmax(done_array)
            p[done_start:] = 1.0
        else:
            p[-1] = 1.0
    else:
        # 失败 episode: 终止帧强制 p_T = 0
        p[-1] = 0.0
    
    return per_frame_reward, p.astype(np.float32)


def get_progress_stats(
    progress_labels: np.ndarray,
    per_frame_reward: np.ndarray,
) -> dict:
    """获取进度标签的统计信息"""
    return {
        "p_min": float(progress_labels.min()),
        "p_max": float(progress_labels.max()),
        "p_mean": float(progress_labels.mean()),
        "p_std": float(progress_labels.std()),
        "reward_min": float(per_frame_reward.min()),
        "reward_max": float(per_frame_reward.max()),
        "reward_mean": float(per_frame_reward.mean()),
    }


# ============================================================================
# 工具函数：从 HDF5 文件加载并计算进度标签
# ============================================================================

def load_progress_labels_from_hdf5(
    filepath: str,
    success_threshold: int = 5,
    failure_threshold: int = 3,
    prefer_per_frame: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """从已标注的 HDF5 文件加载并计算进度标签
    
    支持两种数据格式：
    1. per_frame_reward（推荐）：直接使用逐帧标注的 reward，无需插值
    2. reward_timeline（旧格式）：从10个检查点分数插值生成逐帧 reward
    
    Args:
        filepath: HDF5 文件路径
        success_threshold: 成功阈值（reward >= threshold 为成功）
        failure_threshold: 失败阈值（reward <= threshold 为失败）
        prefer_per_frame: 优先使用 per_frame_reward（如果存在）
        **kwargs: 传递给 compute_progress_labels 的参数
    
    Returns:
        Tuple[per_frame_reward, progress_labels, info]:
            - per_frame_reward: 逐帧 reward
            - progress_labels: 进度标签 p_t
            - info: 元信息 dict
    """
    import h5py
    
    with h5py.File(filepath, 'r') as f:
        # 读取总帧数
        total_frames = f.attrs['num_steps'] if 'num_steps' in f.attrs else len(f['done'])
        
        # 读取 done 数组
        done_array = f['done'][:] if 'done' in f else None
        
        # 尝试读取 per_frame_reward（推荐方式）
        per_frame_reward_raw = None
        has_per_frame = 'per_frame_reward' in f
        
        if prefer_per_frame and has_per_frame:
            per_frame_reward_raw = f['per_frame_reward'][:]
            reward = per_frame_reward_raw.max()
            max_score = int(per_frame_reward_raw.max())
            final_score = int(per_frame_reward_raw[-1])
        else:
            # 读取 reward_timeline（旧方式）
            if 'reward_timeline/checkpoints' in f and 'reward_timeline/scores' in f:
                checkpoints = f['reward_timeline/checkpoints'][:]
                scores = f['reward_timeline/scores'][:]
            else:
                raise ValueError(f"No per_frame_reward or reward_timeline found in {filepath}")
            
            reward = f.attrs.get('reward', scores.max())
            max_score = int(f.attrs.get('max_score', scores.max()))
            final_score = int(f.attrs.get('final_score', scores[-1]))
        
        # 判断成功/失败
        is_success = reward >= success_threshold
        is_failure = reward <= failure_threshold
        
        info = {
            "filepath": filepath,
            "reward": int(reward),
            "max_score": max_score,
            "final_score": final_score,
            "is_success": is_success,
            "is_failure": is_failure,
            "total_frames": int(total_frames),
            "done_frame": int(f.attrs.get('done_frame', -1)),
            "has_per_frame_reward": has_per_frame,
            "used_per_frame_reward": per_frame_reward_raw is not None,
        }
        
        # 如果使用 per_frame_reward，设置 checkpoints 和 scores 为空
        if per_frame_reward_raw is not None:
            checkpoints = np.array([])
            scores = np.array([])
        
    # 计算进度标签
    per_frame_reward, progress_labels = compute_progress_labels(
        checkpoints=checkpoints,
        scores=scores,
        total_frames=total_frames,
        done_array=done_array,
        is_success=is_success,
        per_frame_reward=per_frame_reward_raw,  # 传入逐帧 reward
        **kwargs,
    )
    
    return per_frame_reward, progress_labels, info
