"""
数据集格式转换器

将 CARM HDF5 数据集格式转换为 RoboReward 模型所需的输入格式。
"""

import os
import h5py
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Generator
from PIL import Image

from .config import RoboRewardConfig


class DatasetConverter:
    """CARM 数据集转换器"""
    
    def __init__(
        self, 
        config: Optional[RoboRewardConfig] = None,
        sample_frames: Optional[int] = None,
    ):
        """
        初始化转换器
        
        Args:
            config: 配置对象
            sample_frames: 每个 episode 采样的帧数
                          -1 或 None 表示使用所有帧（默认）
                          正整数表示采样指定数量的帧
        """
        self.config = config or RoboRewardConfig()
        # 优先使用传入的 sample_frames，否则使用 config 中的值
        if sample_frames is not None:
            self.sample_frames = sample_frames
        else:
            self.sample_frames = self.config.sample_frames
    
    def _uniform_sample_indices(
        self, 
        total_frames: int, 
        num_samples: int
    ) -> np.ndarray:
        """
        均匀采样帧索引
        
        Args:
            total_frames: 总帧数
            num_samples: 采样帧数
            
        Returns:
            采样索引数组
        """
        if total_frames <= num_samples:
            return np.arange(total_frames)
        
        # 均匀采样
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        return indices
    
    def _keyframe_sample_indices(
        self,
        total_frames: int,
        num_samples: int,
        include_first_last: bool = True
    ) -> np.ndarray:
        """
        关键帧采样（确保包含首尾帧）
        
        Args:
            total_frames: 总帧数
            num_samples: 采样帧数
            include_first_last: 是否确保包含首尾帧
            
        Returns:
            采样索引数组
        """
        if total_frames <= num_samples:
            return np.arange(total_frames)
        
        if not include_first_last:
            return self._uniform_sample_indices(total_frames, num_samples)
        
        # 确保首尾帧
        if num_samples <= 2:
            return np.array([0, total_frames - 1])[:num_samples]
        
        # 中间帧均匀采样
        middle_samples = num_samples - 2
        middle_indices = np.linspace(1, total_frames - 2, middle_samples, dtype=int)
        
        indices = np.concatenate([[0], middle_indices, [total_frames - 1]])
        return indices
    
    def load_episode_frames(
        self,
        hdf5_path: str,
        sample_method: str = "keyframe"
    ) -> Tuple[List[Image.Image], Dict]:
        """
        从 HDF5 文件加载 episode 的视频帧
        
        Args:
            hdf5_path: HDF5 文件路径
            sample_method: 采样方法 ("uniform" 或 "keyframe")
            
        Returns:
            (PIL.Image 列表, 元数据字典) 元组
        """
        frames = []
        metadata = {}
        
        with h5py.File(hdf5_path, 'r') as f:
            # 读取图像数据
            images = f['observations/images'][:]  # [T, H, W, C]
            total_frames = len(images)
            
            # 确定实际使用的帧数
            # sample_frames <= 0 表示使用所有帧
            if self.sample_frames <= 0:
                actual_sample_frames = total_frames
            else:
                actual_sample_frames = min(self.sample_frames, total_frames)
            
            # 限制最大帧数以防止 OOM
            max_frames = getattr(self.config, 'max_frames', 512)
            if actual_sample_frames > max_frames:
                actual_sample_frames = max_frames
            
            # 选择采样方法
            if actual_sample_frames >= total_frames:
                # 使用所有帧
                indices = np.arange(total_frames)
            elif sample_method == "keyframe":
                indices = self._keyframe_sample_indices(total_frames, actual_sample_frames)
            else:
                indices = self._uniform_sample_indices(total_frames, actual_sample_frames)
            
            # 转换为 PIL.Image
            for idx in indices:
                img_array = images[idx]
                # 确保是 uint8 格式
                if img_array.dtype != np.uint8:
                    img_array = (img_array * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_array)
                frames.append(pil_img)
            
            # 提取元数据
            metadata['num_steps'] = total_frames
            metadata['sampled_frames'] = len(indices)
            metadata['sample_indices'] = indices.tolist()
            metadata['hdf5_path'] = hdf5_path
            
            # 读取 HDF5 属性
            for key in f.attrs:
                metadata[key] = f.attrs[key]
                # 处理 bytes 类型
                if isinstance(metadata[key], bytes):
                    metadata[key] = metadata[key].decode('utf-8')
        
        return frames, metadata
    
    def save_episode_with_reward(
        self,
        src_hdf5_path: str,
        dst_hdf5_path: str,
        reward: int,
        raw_output: Optional[str] = None
    ):
        """
        将原始 HDF5 数据加上 reward 保存到新文件
        
        Args:
            src_hdf5_path: 源 HDF5 文件路径
            dst_hdf5_path: 目标 HDF5 文件路径
            reward: reward 评分 (1-5)
            raw_output: 模型原始输出（可选）
        """
        # 确保目标目录存在
        os.makedirs(os.path.dirname(dst_hdf5_path), exist_ok=True)
        
        # 复制并添加 reward
        with h5py.File(src_hdf5_path, 'r') as src:
            with h5py.File(dst_hdf5_path, 'w') as dst:
                # 复制所有数据集和属性
                for key in src.keys():
                    src.copy(key, dst)
                
                # 复制根属性
                for key in src.attrs:
                    dst.attrs[key] = src.attrs[key]
                
                # 添加 reward 相关数据
                dst.attrs['reward'] = reward
                dst.attrs['reward_model'] = 'RoboReward-8B'
                
                if raw_output:
                    dst.attrs['reward_raw_output'] = raw_output
    
    def scan_episodes(self, data_dir: str) -> List[str]:
        """
        扫描目录下所有 episode HDF5 文件
        
        Args:
            data_dir: 数据目录
            
        Returns:
            HDF5 文件路径列表（已排序）
        """
        files = []
        for f in sorted(os.listdir(data_dir)):
            if f.startswith('episode_') and f.endswith('.hdf5'):
                files.append(os.path.join(data_dir, f))
        return files
    
    def iter_episodes(
        self, 
        data_dir: str,
        sample_method: str = "keyframe"
    ) -> Generator[Tuple[str, List[Image.Image], Dict], None, None]:
        """
        迭代数据目录中的所有 episodes
        
        Args:
            data_dir: 数据目录
            sample_method: 采样方法
            
        Yields:
            (文件路径, 帧列表, 元数据) 元组
        """
        files = self.scan_episodes(data_dir)
        
        for filepath in files:
            frames, metadata = self.load_episode_frames(filepath, sample_method)
            yield filepath, frames, metadata


class TaskDescriptionManager:
    """任务描述管理器"""
    
    def __init__(
        self,
        default_task: str = "complete the manipulation task",
        task_file: Optional[str] = None
    ):
        """
        初始化任务描述管理器
        
        Args:
            default_task: 默认任务描述
            task_file: 任务描述 JSON 文件路径（可选）
        """
        self.default_task = default_task
        self.task_mapping = {}
        
        if task_file and os.path.exists(task_file):
            self._load_task_file(task_file)
    
    def _load_task_file(self, task_file: str):
        """
        加载任务描述文件
        
        文件格式示例:
        {
            "default": "pick up the object and place it in the target area",
            "episode_patterns": {
                "episode_00*": "grasp the red cube",
                "episode_01*": "push the blue block"
            },
            "episodes": {
                "episode_0001": "specific task for episode 0001"
            }
        }
        """
        with open(task_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'default' in data:
            self.default_task = data['default']
        
        if 'episodes' in data:
            self.task_mapping = data['episodes']
        
        if 'episode_patterns' in data:
            self.patterns = data['episode_patterns']
        else:
            self.patterns = {}
    
    def get_task(self, episode_name: str) -> str:
        """
        获取特定 episode 的任务描述
        
        Args:
            episode_name: episode 名称或文件名
            
        Returns:
            任务描述字符串
        """
        # 提取 episode 名称（去掉路径和扩展名）
        name = os.path.basename(episode_name)
        if name.endswith('.hdf5'):
            name = name[:-5]
        
        # 精确匹配
        if name in self.task_mapping:
            return self.task_mapping[name]
        
        # 模式匹配
        import fnmatch
        for pattern, task in getattr(self, 'patterns', {}).items():
            if fnmatch.fnmatch(name, pattern):
                return task
        
        return self.default_task
