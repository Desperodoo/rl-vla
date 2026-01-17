#!/usr/bin/env python3
"""
CARM 数据集加载工具
提供便捷的数据集加载和处理函数
"""

import os
import json
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class CARMDatasetLoader:
    """CARM 数据集加载器"""
    
    def __init__(self, data_dir: str):
        """
        初始化加载器
        
        Args:
            data_dir: 数据集目录路径
        """
        self.data_dir = data_dir
        self.episodes = []
        self.metadata = {}
        self._load_metadata()
        self._scan_episodes()
    
    def _load_metadata(self):
        """加载数据集元信息"""
        info_path = os.path.join(self.data_dir, 'dataset_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.metadata = json.load(f)
    
    def _scan_episodes(self):
        """扫描所有 episode 文件"""
        files = sorted([
            f for f in os.listdir(self.data_dir) 
            if f.startswith('episode_') and f.endswith('.hdf5')
        ])
        self.episodes = [
            os.path.join(self.data_dir, f) for f in files
        ]
    
    def get_episode(self, idx: int) -> Dict:
        """
        获取单个 episode 的全部数据
        
        Args:
            idx: Episode 索引
            
        Returns:
            包含所有数据的字典
        """
        if idx < 0 or idx >= len(self.episodes):
            raise IndexError(f"Episode index {idx} out of range [0, {len(self.episodes)-1}]")
        
        filepath = self.episodes[idx]
        data = {}
        
        with h5py.File(filepath, 'r') as f:
            # 读取观测数据
            data['images'] = f['observations/images'][:]
            data['qpos_joint'] = f['observations/qpos_joint'][:]
            data['qpos_end'] = f['observations/qpos_end'][:]
            data['qpos'] = f['observations/qpos'][:]
            data['gripper'] = f['observations/gripper'][:]
            data['timestamps'] = f['observations/timestamps'][:]
            
            # 读取动作
            if 'action' in f:
                data['action'] = f['action'][:]
            
            # 读取元数据
            data['meta'] = dict(f.attrs)
        
        return data
    
    def get_trajectory(self, idx: int, start: int = 0, end: Optional[int] = None) -> Dict:
        """
        获取单个 episode 的轨迹片段
        
        Args:
            idx: Episode 索引
            start: 起始步数
            end: 结束步数（None 表示到末尾）
            
        Returns:
            轨迹片段数据
        """
        data = self.get_episode(idx)
        
        if end is None:
            end = len(data['qpos_joint'])
        
        return {
            k: v[start:end] if isinstance(v, np.ndarray) and v.ndim > 0 
            else v
            for k, v in data.items()
        }
    
    def get_frame(self, ep_idx: int, frame_idx: int) -> Dict:
        """
        获取单个帧的数据
        
        Args:
            ep_idx: Episode 索引
            frame_idx: 帧索引
            
        Returns:
            单帧数据 (图像、状态、动作)
        """
        data = self.get_episode(ep_idx)
        
        return {
            'image': data['images'][frame_idx],
            'qpos_joint': data['qpos_joint'][frame_idx],
            'qpos_end': data['qpos_end'][frame_idx],
            'qpos': data['qpos'][frame_idx],
            'gripper': data['gripper'][frame_idx],
            'timestamp': data['timestamps'][frame_idx],
            'action': data['action'][frame_idx] if 'action' in data else None,
        }
    
    def iter_episodes(self, shuffle: bool = False):
        """
        迭代所有 episodes
        
        Args:
            shuffle: 是否随机打乱顺序
            
        Yields:
            Episode 数据
        """
        indices = list(range(len(self.episodes)))
        if shuffle:
            import random
            random.shuffle(indices)
        
        for idx in indices:
            yield idx, self.get_episode(idx)
    
    def iter_frames(self, ep_idx: int, stride: int = 1):
        """
        迭代单个 episode 的帧
        
        Args:
            ep_idx: Episode 索引
            stride: 帧步长（大于1时跳帧）
            
        Yields:
            帧索引和帧数据
        """
        data = self.get_episode(ep_idx)
        num_frames = len(data['qpos_joint'])
        
        for frame_idx in range(0, num_frames, stride):
            yield frame_idx, self.get_frame(ep_idx, frame_idx)
    
    def plot_trajectory(self, ep_idx: int, joints: List[int] = [0, 1, 2, 3, 4, 5],
                        save_path: Optional[str] = None):
        """
        绘制关节轨迹
        
        Args:
            ep_idx: Episode 索引
            joints: 要绘制的关节列表
            save_path: 保存路径（None 表示显示）
        """
        data = self.get_episode(ep_idx)
        qpos = data['qpos_joint']
        
        fig, axes = plt.subplots(len(joints), 1, figsize=(12, 2*len(joints)))
        if len(joints) == 1:
            axes = [axes]
        
        for i, j in enumerate(joints):
            axes[i].plot(qpos[:, j], 'b-', linewidth=1.5)
            axes[i].set_title(f'Episode {ep_idx:04d} - Joint {j+1}')
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel('Position (rad)')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_gripper(self, ep_idx: int, save_path: Optional[str] = None):
        """绘制夹爪轨迹"""
        data = self.get_episode(ep_idx)
        gripper = data['gripper']
        
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(gripper, 'r-', linewidth=1.5)
        ax.set_title(f'Episode {ep_idx:04d} - Gripper')
        ax.set_xlabel('Step')
        ax.set_ylabel('Opening (m)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()
        
        plt.close()
    
    def plot_3d_trajectory(self, ep_idx: int, save_path: Optional[str] = None):
        """绘制末端位姿的 3D 轨迹"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("3D plotting requires mpl_toolkits")
            return
        
        data = self.get_episode(ep_idx)
        qpos_end = data['qpos_end']
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制末端位置轨迹
        ax.plot(qpos_end[:, 0], qpos_end[:, 1], qpos_end[:, 2], 'b-', linewidth=1.5)
        ax.scatter(qpos_end[0, 0], qpos_end[0, 1], qpos_end[0, 2], 
                  c='green', s=100, label='Start')
        ax.scatter(qpos_end[-1, 0], qpos_end[-1, 1], qpos_end[-1, 2], 
                  c='red', s=100, label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Episode {ep_idx:04d} - End Effector 3D Trajectory')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()
        
        plt.close()
    
    def show_frame(self, ep_idx: int, frame_idx: int):
        """显示单个帧"""
        frame = self.get_frame(ep_idx, frame_idx)
        img = frame['image']
        
        # RGB 转 BGR for OpenCV（如果需要）
        import cv2
        cv2.imshow(f'Episode {ep_idx:04d} - Frame {frame_idx}', 
                  cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        if self.metadata and 'summary' in self.metadata:
            return self.metadata['summary']
        return {}
    
    def get_joint_limits(self) -> Dict:
        """获取关节范围统计"""
        if self.metadata and 'joint_statistics' in self.metadata:
            return self.metadata['joint_statistics']
        return {}


def main():
    """示例用法"""
    import sys
    
    # 初始化加载器
    loader = CARMDatasetLoader('./recorded_data')
    
    print(f"Loaded {len(loader.episodes)} episodes")
    print(f"Dataset stats: {loader.get_statistics()}")
    print()
    
    # 例 1: 加载第一个 episode
    print("Example 1: Load first episode")
    ep0 = loader.get_episode(0)
    print(f"  Images shape: {ep0['images'].shape}")
    print(f"  qpos_joint shape: {ep0['qpos_joint'].shape}")
    print()
    
    # 例 2: 迭代所有 episodes
    print("Example 2: Iterate all episodes")
    for ep_idx, ep_data in loader.iter_episodes():
        print(f"  Episode {ep_idx:04d}: {len(ep_data['qpos_joint'])} steps")
        if ep_idx >= 2:  # 仅显示前 3 个
            break
    print()
    
    # 例 3: 获取单个帧
    print("Example 3: Get single frame")
    frame = loader.get_frame(0, 0)
    print(f"  Frame image shape: {frame['image'].shape}")
    print(f"  Frame joint position: {frame['qpos_joint']}")
    print()
    
    # 例 4: 获取关节范围
    print("Example 4: Joint limits")
    limits = loader.get_joint_limits()
    if limits:
        print(f"  Joint min: {limits.get('joint_min', 'N/A')}")
        print(f"  Joint max: {limits.get('joint_max', 'N/A')}")


if __name__ == '__main__':
    main()
