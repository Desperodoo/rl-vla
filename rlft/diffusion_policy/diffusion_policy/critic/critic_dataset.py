"""
Critic Training Dataset

从 RoboReward 标注的数据集加载数据，构建 SMDP pairs 用于 Critic 训练。

数据格式沿用现有 CARM 数据的格式：
- observations/images: [T, H, W, 3] RGB 图像
- observations/qpos_joint: [T, 7] 关节状态
- done: [T] done 标记
- reward_timeline/checkpoints: [num_ckpt] 检查点帧位置
- reward_timeline/scores: [num_ckpt] 对应分数

SMDP Item 结构（沿用现有代码格式）：
{
    "s_t": (rgb[t], state[t]),
    "s_tpK": (rgb[t+K], state[t+K]),
    "p_t": progress[t],
    "p_tpK": progress[t+K],
    "done_tpK": done[t+K],
    "episode_idx": int,
    "is_success": bool,
}
"""

import os
import glob
import numpy as np
import torch
import cv2
import h5py
from typing import Dict, List, Optional, Tuple, Literal
from torch.utils.data import Dataset
from tqdm import tqdm

from .progress_labels import compute_progress_labels, load_progress_labels_from_hdf5


class CriticDataset(Dataset):
    """Critic 训练数据集
    
    加载所有 episode，计算进度标签，构建训练样本。
    
    每个样本包含：
    - rgb: [obs_horizon, C, H, W] RGB 图像
    - state: [obs_horizon, state_dim] 状态
    - progress: scalar 进度标签 p_t
    - done: bool 是否终止
    - is_success: bool episode 是否成功
    - episode_idx: int episode 索引
    - frame_idx: int 帧索引
    
    Args:
        data_path: 数据目录（包含 episode_*.hdf5）
        obs_horizon: 观测堆叠帧数
        image_size: 输出图像尺寸 (H, W)
        state_mode: 状态模式 ('joint_only', 'ee_only', 'both')
        success_threshold: 成功阈值（reward >= threshold）
        failure_threshold: 失败阈值（reward <= threshold）
        filter_mode: 筛选模式
            - 'all': 所有 episode
            - 'success_only': 仅成功 episode (A1)
            - 'success_failure': 成功 + 失败 episode (A2)
        interpolation: 进度标签插值方式 ('step', 'linear')
        verbose: 是否打印详情
    """
    
    def __init__(
        self,
        data_path: str,
        obs_horizon: int = 2,
        image_size: Tuple[int, int] = (224, 224),
        state_mode: Literal["joint_only", "ee_only", "both"] = "joint_only",
        success_threshold: int = 5,
        failure_threshold: int = 3,
        filter_mode: Literal["all", "success_only", "success_failure"] = "all",
        interpolation: Literal["step", "linear"] = "step",
        do_ema_smooth: bool = True,
        ema_alpha: float = 0.2,
        verbose: bool = True,
    ):
        super().__init__()
        
        self.data_path = data_path
        self.obs_horizon = obs_horizon
        self.image_size = image_size
        self.state_mode = state_mode
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self.filter_mode = filter_mode
        self.interpolation = interpolation
        self.do_ema_smooth = do_ema_smooth
        self.ema_alpha = ema_alpha
        
        # 扫描并加载所有 episode
        self.episodes = []
        self.samples = []  # (episode_idx, frame_idx)
        
        self._load_episodes(verbose)
    
    def _load_episodes(self, verbose: bool):
        """加载所有 episode 数据"""
        # 查找所有 HDF5 文件
        pattern = os.path.join(self.data_path, "episode_*.hdf5")
        files = sorted(glob.glob(pattern))
        
        if verbose:
            print(f"Found {len(files)} episode files")
        
        success_count = 0
        failure_count = 0
        neutral_count = 0
        
        for filepath in tqdm(files, desc="Loading episodes", disable=not verbose):
            try:
                episode = self._load_single_episode(filepath)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
                continue
            
            # 根据 filter_mode 筛选
            is_success = episode["is_success"]
            is_failure = episode["is_failure"]
            
            if self.filter_mode == "success_only" and not is_success:
                continue
            elif self.filter_mode == "success_failure" and not (is_success or is_failure):
                neutral_count += 1
                continue
            
            episode_idx = len(self.episodes)
            self.episodes.append(episode)
            
            if is_success:
                success_count += 1
            elif is_failure:
                failure_count += 1
            
            # 构建样本索引（确保有足够的 obs_horizon）
            T = episode["num_frames"]
            for t in range(self.obs_horizon - 1, T):
                self.samples.append((episode_idx, t))
        
        if verbose:
            print(f"Loaded {len(self.episodes)} episodes:")
            print(f"  - Success (reward >= {self.success_threshold}): {success_count}")
            print(f"  - Failure (reward <= {self.failure_threshold}): {failure_count}")
            if self.filter_mode == "success_failure":
                print(f"  - Neutral (skipped): {neutral_count}")
            print(f"Total samples: {len(self.samples)}")
    
    def _load_single_episode(self, filepath: str) -> dict:
        """加载单个 episode
        
        支持两种数据格式：
        1. per_frame_reward（推荐）：直接使用逐帧标注的 reward
        2. reward_timeline（旧格式）：从10个检查点分数插值
        """
        with h5py.File(filepath, 'r') as f:
            # 读取图像
            images = f['observations/images'][:]  # [T, H, W, 3]
            
            # 读取状态
            if self.state_mode == "joint_only":
                state = f['observations/qpos_joint'][:]  # [T, 7]
            elif self.state_mode == "ee_only":
                state = f['observations/qpos_end'][:]  # [T, 8]
            else:  # both
                qpos_joint = f['observations/qpos_joint'][:]  # [T, 7]
                qpos_end = f['observations/qpos_end'][:, :7]  # [T, 7] (不含 gripper)
                state = np.concatenate([qpos_joint, qpos_end], axis=-1)  # [T, 14]
            
            # 读取 done
            done = f['done'][:] if 'done' in f else np.zeros(len(images), dtype=bool)
            
            # 优先使用 per_frame_reward（推荐），否则使用 reward_timeline
            per_frame_reward_raw = None
            if 'per_frame_reward' in f:
                per_frame_reward_raw = f['per_frame_reward'][:]
                reward = per_frame_reward_raw.max()
                checkpoints = np.array([])
                scores = np.array([])
            elif 'reward_timeline/checkpoints' in f and 'reward_timeline/scores' in f:
                checkpoints = f['reward_timeline/checkpoints'][:]
                scores = f['reward_timeline/scores'][:]
                reward = f.attrs.get('reward', scores.max())
            else:
                raise ValueError(f"No per_frame_reward or reward_timeline found in {filepath}")
            
        # 判断成功/失败
        is_success = reward >= self.success_threshold
        is_failure = reward <= self.failure_threshold
        
        # 计算进度标签
        _, progress = compute_progress_labels(
            checkpoints=checkpoints,
            scores=scores,
            total_frames=len(images),
            done_array=done,
            is_success=is_success,
            interpolation=self.interpolation,
            do_episode_normalize=True,
            do_ema_smooth=self.do_ema_smooth,
            ema_alpha=self.ema_alpha,
            per_frame_reward=per_frame_reward_raw,  # 传入逐帧 reward（如果有）
        )
        
        return {
            "filepath": filepath,
            "images": images,
            "state": state.astype(np.float32),
            "done": done,
            "progress": progress,
            "reward": int(reward),
            "is_success": is_success,
            "is_failure": is_failure,
            "num_frames": len(images),
            "has_per_frame_reward": per_frame_reward_raw is not None,
        }
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """预处理图像: resize + normalize to [0, 1] + NCHW"""
        # Resize
        if img.shape[:2] != self.image_size:
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        
        # Normalize to [0, 1] and convert to CHW
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        
        return img
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        episode_idx, frame_idx = self.samples[idx]
        episode = self.episodes[episode_idx]
        
        # 获取 obs_horizon 帧的观测
        start_idx = frame_idx - self.obs_horizon + 1
        
        # 堆叠图像
        images = []
        for t in range(start_idx, frame_idx + 1):
            img = self._preprocess_image(episode["images"][t])
            images.append(img)
        images = np.stack(images, axis=0)  # [obs_horizon, C, H, W]
        
        # 堆叠状态
        state = episode["state"][start_idx:frame_idx + 1]  # [obs_horizon, state_dim]
        
        return {
            "rgb": torch.from_numpy(images),
            "state": torch.from_numpy(state),
            "progress": torch.tensor(episode["progress"][frame_idx], dtype=torch.float32),
            "done": torch.tensor(episode["done"][frame_idx], dtype=torch.bool),
            "is_success": torch.tensor(episode["is_success"], dtype=torch.bool),
            "is_failure": torch.tensor(episode["is_failure"], dtype=torch.bool),
            "episode_idx": episode_idx,
            "frame_idx": frame_idx,
        }
    
    def get_episode_info(self, episode_idx: int) -> dict:
        """获取 episode 元信息"""
        ep = self.episodes[episode_idx]
        return {
            "filepath": ep["filepath"],
            "num_frames": ep["num_frames"],
            "reward": ep["reward"],
            "is_success": ep["is_success"],
            "is_failure": ep["is_failure"],
        }


class CriticSMDPDataset(Dataset):
    """SMDP 形式的 Critic 训练数据集
    
    在 chunk 粒度上定义转移，用于 pairwise ranking 训练。
    
    每个样本包含一对状态 (s_t, s_{t+K})，用于计算 ranking loss。
    
    采样规则：
    - 在同一 episode 内采样 t1, t2，使得 t2 - t1 ∈ {K, 2K, 4K}
    - 确保有足够的 obs_horizon
    
    Args:
        data_path: 数据目录
        chunk_size: chunk 长度 K（action horizon）
        obs_horizon: 观测堆叠帧数
        gap_options: 采样间隔选项，默认 [K, 2K, 4K]
        其他参数同 CriticDataset
    """
    
    def __init__(
        self,
        data_path: str,
        chunk_size: int = 8,
        obs_horizon: int = 2,
        gap_options: Optional[List[int]] = None,
        image_size: Tuple[int, int] = (224, 224),
        state_mode: Literal["joint_only", "ee_only", "both"] = "joint_only",
        success_threshold: int = 5,
        failure_threshold: int = 3,
        filter_mode: Literal["all", "success_only", "success_failure"] = "all",
        interpolation: Literal["step", "linear"] = "step",
        do_ema_smooth: bool = True,
        ema_alpha: float = 0.2,
        verbose: bool = True,
    ):
        super().__init__()
        
        self.chunk_size = chunk_size
        self.obs_horizon = obs_horizon
        self.gap_options = gap_options or [chunk_size, 2 * chunk_size, 4 * chunk_size]
        self.image_size = image_size
        self.state_mode = state_mode
        
        # 首先加载基础数据集
        self.base_dataset = CriticDataset(
            data_path=data_path,
            obs_horizon=obs_horizon,
            image_size=image_size,
            state_mode=state_mode,
            success_threshold=success_threshold,
            failure_threshold=failure_threshold,
            filter_mode=filter_mode,
            interpolation=interpolation,
            do_ema_smooth=do_ema_smooth,
            ema_alpha=ema_alpha,
            verbose=verbose,
        )
        
        # 构建 SMDP pairs
        self.pairs = []  # (episode_idx, t1, t2)
        self._build_pairs(verbose)
    
    def _build_pairs(self, verbose: bool):
        """构建 SMDP 状态对"""
        max_gap = max(self.gap_options)
        
        for episode_idx, episode in enumerate(self.base_dataset.episodes):
            T = episode["num_frames"]
            
            # 遍历所有可能的起始点
            for t1 in range(self.obs_horizon - 1, T - max_gap):
                # 对每个 gap 选项添加一个 pair
                for gap in self.gap_options:
                    t2 = t1 + gap
                    if t2 < T:
                        self.pairs.append((episode_idx, t1, t2))
        
        if verbose:
            print(f"Built {len(self.pairs)} SMDP pairs")
            print(f"  Gap options: {self.gap_options}")
    
    def __len__(self):
        return len(self.pairs)
    
    def _get_obs_at(self, episode_idx: int, frame_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定位置的观测（obs_horizon 帧）"""
        episode = self.base_dataset.episodes[episode_idx]
        start_idx = frame_idx - self.obs_horizon + 1
        
        # 图像
        images = []
        for t in range(start_idx, frame_idx + 1):
            img = self.base_dataset._preprocess_image(episode["images"][t])
            images.append(img)
        images = np.stack(images, axis=0)
        
        # 状态
        state = episode["state"][start_idx:frame_idx + 1]
        
        return torch.from_numpy(images), torch.from_numpy(state)
    
    def __getitem__(self, idx: int) -> dict:
        episode_idx, t1, t2 = self.pairs[idx]
        episode = self.base_dataset.episodes[episode_idx]
        
        # 获取 t1 时刻的观测
        rgb_t1, state_t1 = self._get_obs_at(episode_idx, t1)
        
        # 获取 t2 时刻的观测
        rgb_t2, state_t2 = self._get_obs_at(episode_idx, t2)
        
        return {
            # t1 观测
            "rgb_t1": rgb_t1,  # [obs_horizon, C, H, W]
            "state_t1": state_t1,  # [obs_horizon, state_dim]
            "p_t1": torch.tensor(episode["progress"][t1], dtype=torch.float32),
            
            # t2 观测
            "rgb_t2": rgb_t2,
            "state_t2": state_t2,
            "p_t2": torch.tensor(episode["progress"][t2], dtype=torch.float32),
            
            # 元信息
            "done_t2": torch.tensor(episode["done"][t2], dtype=torch.bool),
            "is_success": torch.tensor(episode["is_success"], dtype=torch.bool),
            "is_failure": torch.tensor(episode["is_failure"], dtype=torch.bool),
            "episode_idx": episode_idx,
            "t1": t1,
            "t2": t2,
            "gap": t2 - t1,
        }
    
    @property
    def episodes(self):
        return self.base_dataset.episodes


def build_anchor_samples(
    dataset: CriticDataset,
    low_progress_percentile: float = 0.1,
) -> Tuple[List[int], List[int]]:
    """构建 anchor loss 的样本索引
    
    返回：
    - done_indices: 成功终态样本的索引
    - low_indices: 低进度样本的索引
    """
    done_indices = []
    low_indices = []
    
    for i, (episode_idx, frame_idx) in enumerate(dataset.samples):
        episode = dataset.episodes[episode_idx]
        
        # 成功终态：is_success 且 done
        if episode["is_success"] and episode["done"][frame_idx]:
            done_indices.append(i)
        
        # 低进度：progress 在 episode 内最低的 percentile
        progress = episode["progress"][frame_idx]
        threshold = np.percentile(episode["progress"], low_progress_percentile * 100)
        if progress <= threshold:
            low_indices.append(i)
    
    return done_indices, low_indices


def build_terminal_samples(
    dataset: CriticDataset,
) -> Tuple[List[int], List[int]]:
    """构建 inter-episode ranking 的终态样本
    
    返回：
    - success_terminal_indices: 成功 episode 终态样本索引
    - failure_terminal_indices: 失败 episode 终态样本索引
    """
    success_indices = []
    failure_indices = []
    
    for i, (episode_idx, frame_idx) in enumerate(dataset.samples):
        episode = dataset.episodes[episode_idx]
        
        # 只考虑 episode 最后一帧
        if frame_idx != episode["num_frames"] - 1:
            continue
        
        if episode["is_success"]:
            success_indices.append(i)
        elif episode["is_failure"]:
            failure_indices.append(i)
    
    return success_indices, failure_indices
