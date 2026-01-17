#!/usr/bin/env python3
"""
CARM 数据集分析脚本
分析 recorded_data 目录中的 HDF5 数据集

功能:
- 统计数据集基本信息
- 检查数据完整性
- 可视化轨迹和图像
- 生成数据集报告
"""

import argparse
import os
import sys
import json
import numpy as np
import h5py
from datetime import datetime
from collections import defaultdict

# 可选的可视化库
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class DatasetAnalyzer:
    """数据集分析器"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.episodes = []
        self.stats = {}
        
    def scan_episodes(self):
        """扫描所有 episode 文件"""
        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.hdf5')])
        self.episodes = []
        
        for filename in files:
            filepath = os.path.join(self.data_dir, filename)
            try:
                with h5py.File(filepath, 'r') as f:
                    episode_info = {
                        'filename': filename,
                        'filepath': filepath,
                        'num_steps': f.attrs.get('num_steps', 0),
                        'record_freq': f.attrs.get('record_freq', 30),
                        'image_width': f.attrs.get('image_width', 0),
                        'image_height': f.attrs.get('image_height', 0),
                        'robot_ip': f.attrs.get('robot_ip', ''),
                        'created_at': f.attrs.get('created_at', ''),
                    }
                    
                    # 检查数据集内容
                    episode_info['has_images'] = 'observations/images' in f
                    episode_info['has_qpos_joint'] = 'observations/qpos_joint' in f
                    episode_info['has_qpos_end'] = 'observations/qpos_end' in f
                    episode_info['has_qpos'] = 'observations/qpos' in f
                    episode_info['has_gripper'] = 'observations/gripper' in f
                    episode_info['has_action'] = 'action' in f
                    episode_info['has_timestamps'] = 'observations/timestamps' in f
                    
                    # 获取数据形状
                    if episode_info['has_images']:
                        episode_info['image_shape'] = f['observations/images'].shape
                    if episode_info['has_qpos_joint']:
                        episode_info['qpos_joint_shape'] = f['observations/qpos_joint'].shape
                    if episode_info['has_action']:
                        episode_info['action_shape'] = f['action'].shape
                    
                    # 计算时长
                    if episode_info['has_timestamps']:
                        timestamps = f['observations/timestamps'][:]
                        if len(timestamps) > 1:
                            episode_info['duration'] = timestamps[-1] - timestamps[0]
                            episode_info['actual_freq'] = len(timestamps) / episode_info['duration']
                        else:
                            episode_info['duration'] = 0
                            episode_info['actual_freq'] = 0
                    
                    self.episodes.append(episode_info)
                    
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
        return self.episodes
    
    def compute_statistics(self):
        """计算数据集统计信息"""
        if not self.episodes:
            self.scan_episodes()
        
        stats = {
            'total_episodes': len(self.episodes),
            'total_steps': sum(ep['num_steps'] for ep in self.episodes),
            'total_duration': sum(ep.get('duration', 0) for ep in self.episodes),
        }
        
        # 步数统计
        steps = [ep['num_steps'] for ep in self.episodes]
        if steps:
            stats['steps_min'] = int(min(steps))
            stats['steps_max'] = int(max(steps))
            stats['steps_mean'] = float(np.mean(steps))
            stats['steps_std'] = float(np.std(steps))
        
        # 时长统计
        durations = [ep.get('duration', 0) for ep in self.episodes if ep.get('duration', 0) > 0]
        if durations:
            stats['duration_min'] = float(min(durations))
            stats['duration_max'] = float(max(durations))
            stats['duration_mean'] = float(np.mean(durations))
        
        # 频率统计
        freqs = [ep.get('actual_freq', 0) for ep in self.episodes if ep.get('actual_freq', 0) > 0]
        if freqs:
            stats['freq_mean'] = float(np.mean(freqs))
            stats['freq_std'] = float(np.std(freqs))
        
        # 图像信息
        if self.episodes and self.episodes[0].get('image_shape'):
            shape = self.episodes[0]['image_shape']
            stats['image_shape'] = list(shape[1:])  # [H, W, C]
        
        # 数据完整性
        stats['complete_episodes'] = sum(
            1 for ep in self.episodes 
            if ep['has_images'] and ep['has_qpos_joint'] and ep['has_action']
        )
        
        self.stats = stats
        return stats
    
    def analyze_joint_ranges(self):
        """分析关节角范围"""
        all_qpos = []
        all_gripper = []
        
        for ep in self.episodes[:min(10, len(self.episodes))]:  # 采样前10个
            try:
                with h5py.File(ep['filepath'], 'r') as f:
                    if 'observations/qpos_joint' in f:
                        qpos = f['observations/qpos_joint'][:]
                        all_qpos.append(qpos)
                    if 'observations/gripper' in f:
                        gripper = f['observations/gripper'][:]
                        all_gripper.append(gripper)
            except:
                pass
        
        joint_stats = {}
        if all_qpos:
            qpos = np.concatenate(all_qpos, axis=0)
            joint_stats['joint_min'] = qpos.min(axis=0).tolist()
            joint_stats['joint_max'] = qpos.max(axis=0).tolist()
            joint_stats['joint_mean'] = qpos.mean(axis=0).tolist()
            joint_stats['joint_std'] = qpos.std(axis=0).tolist()
        
        if all_gripper:
            gripper = np.concatenate(all_gripper)
            joint_stats['gripper_min'] = float(gripper.min())
            joint_stats['gripper_max'] = float(gripper.max())
            joint_stats['gripper_mean'] = float(gripper.mean())
        
        return joint_stats
    
    def generate_report(self, output_path=None):
        """生成数据集报告"""
        if not self.stats:
            self.compute_statistics()
        
        joint_stats = self.analyze_joint_ranges()
        
        # 转换为 Python 原生类型（避免 numpy 类型序列化问题）
        def to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_native(v) for v in obj]
            return obj
        
        report = {
            'dataset_info': {
                'path': self.data_dir,
                'analyzed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            'summary': to_native(self.stats),
            'joint_statistics': to_native(joint_stats),
            'episodes': [
                {
                    'filename': ep['filename'],
                    'num_steps': int(ep['num_steps']),
                    'duration': round(float(ep.get('duration', 0)), 2),
                    'created_at': str(ep['created_at']),
                }
                for ep in self.episodes
            ]
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Report saved to: {output_path}")
        
        return report
    
    def print_summary(self):
        """打印摘要信息"""
        if not self.stats:
            self.compute_statistics()
        
        print("\n" + "="*60)
        print("CARM Dataset Analysis Report")
        print("="*60)
        print(f"Data directory: {self.data_dir}")
        print(f"Total episodes: {self.stats['total_episodes']}")
        print(f"Total steps: {self.stats['total_steps']}")
        print(f"Total duration: {self.stats['total_duration']:.1f} seconds")
        print()
        print("Episode Statistics:")
        print(f"  Steps per episode: {self.stats['steps_mean']:.1f} ± {self.stats['steps_std']:.1f}")
        print(f"  Steps range: [{self.stats['steps_min']}, {self.stats['steps_max']}]")
        if 'duration_mean' in self.stats:
            print(f"  Duration per episode: {self.stats['duration_mean']:.2f}s")
        if 'freq_mean' in self.stats:
            print(f"  Actual frequency: {self.stats['freq_mean']:.1f} ± {self.stats['freq_std']:.1f} Hz")
        print()
        if 'image_shape' in self.stats:
            print(f"Image shape: {self.stats['image_shape']}")
        print(f"Complete episodes: {self.stats['complete_episodes']}/{self.stats['total_episodes']}")
        print("="*60)
        
        # 关节统计
        joint_stats = self.analyze_joint_ranges()
        if joint_stats:
            print("\nJoint Statistics:")
            if 'joint_min' in joint_stats:
                print(f"  Joint min: {[f'{v:.3f}' for v in joint_stats['joint_min']]}")
                print(f"  Joint max: {[f'{v:.3f}' for v in joint_stats['joint_max']]}")
            if 'gripper_min' in joint_stats:
                print(f"  Gripper range: [{joint_stats['gripper_min']:.4f}, {joint_stats['gripper_max']:.4f}]")
            print()
    
    def visualize_trajectories(self, output_path=None, num_episodes=5):
        """可视化轨迹"""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available, skipping visualization")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        colors = plt.cm.viridis(np.linspace(0, 1, num_episodes))
        
        for idx, ep in enumerate(self.episodes[:num_episodes]):
            try:
                with h5py.File(ep['filepath'], 'r') as f:
                    qpos = f['observations/qpos_joint'][:]
                    gripper = f['observations/gripper'][:]
                    
                    # 绘制各关节轨迹
                    for j in range(6):
                        ax = axes[j // 3, j % 3]
                        ax.plot(qpos[:, j], color=colors[idx], alpha=0.7, 
                               label=f'ep{idx+1}' if j == 0 else None)
                        ax.set_title(f'Joint {j+1}')
                        ax.set_xlabel('Step')
                        ax.set_ylabel('Position (rad)')
                        ax.grid(True, alpha=0.3)
            except Exception as e:
                print(f"Error visualizing {ep['filename']}: {e}")
        
        axes[0, 0].legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Trajectory plot saved to: {output_path}")
        else:
            plt.savefig(os.path.join(self.data_dir, 'trajectory_plot.png'), dpi=150)
        
        plt.close()
    
    def export_sample_images(self, output_dir=None, num_episodes=5, num_frames=5):
        """导出样本图像"""
        if not HAS_CV2:
            print("OpenCV not available, skipping image export")
            return
        
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, 'sample_images')
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, ep in enumerate(self.episodes[:num_episodes]):
            try:
                with h5py.File(ep['filepath'], 'r') as f:
                    images = f['observations/images']
                    total_frames = images.shape[0]
                    
                    # 均匀采样帧
                    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
                    
                    for fi, frame_idx in enumerate(frame_indices):
                        img = images[frame_idx]
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        
                        # 添加帧信息
                        cv2.putText(img_bgr, f"Ep:{idx+1} Frame:{frame_idx}", 
                                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        output_path = os.path.join(output_dir, f"ep{idx+1:03d}_frame{fi+1:02d}.jpg")
                        cv2.imwrite(output_path, img_bgr)
                        
            except Exception as e:
                print(f"Error exporting images from {ep['filename']}: {e}")
        
        print(f"Sample images exported to: {output_dir}")
    
    def create_montage(self, output_path=None, episodes_to_show=6):
        """创建图像蒙太奇（每个 episode 的首尾帧）"""
        if not HAS_CV2:
            print("OpenCV not available, skipping montage")
            return
        
        images = []
        labels = []
        
        for idx, ep in enumerate(self.episodes[:episodes_to_show]):
            try:
                with h5py.File(ep['filepath'], 'r') as f:
                    imgs = f['observations/images']
                    # 首帧和末帧
                    first_frame = imgs[0]
                    last_frame = imgs[-1]
                    images.extend([first_frame, last_frame])
                    labels.extend([f"Ep{idx+1} Start", f"Ep{idx+1} End"])
            except:
                pass
        
        if not images:
            return
        
        # 创建网格
        n_cols = 4
        n_rows = (len(images) + n_cols - 1) // n_cols
        
        h, w = images[0].shape[:2]
        montage = np.zeros((n_rows * h, n_cols * w, 3), dtype=np.uint8)
        
        for i, (img, label) in enumerate(zip(images, labels)):
            row = i // n_cols
            col = i % n_cols
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(img_bgr, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            montage[row*h:(row+1)*h, col*w:(col+1)*w] = img_bgr
        
        if output_path is None:
            output_path = os.path.join(self.data_dir, 'episode_montage.jpg')
        
        cv2.imwrite(output_path, montage)
        print(f"Montage saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='CARM Dataset Analyzer')
    parser.add_argument('--data_dir', type=str, default='./recorded_data',
                        help='Path to recorded data directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for JSON report')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate trajectory visualizations')
    parser.add_argument('--export_images', action='store_true',
                        help='Export sample images')
    parser.add_argument('--montage', action='store_true',
                        help='Create episode montage')
    parser.add_argument('--all', action='store_true',
                        help='Run all analysis and generate all outputs')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = DatasetAnalyzer(args.data_dir)
    
    # 扫描并分析
    analyzer.scan_episodes()
    analyzer.compute_statistics()
    
    # 打印摘要
    analyzer.print_summary()
    
    # 生成报告
    output_path = args.output or os.path.join(args.data_dir, 'dataset_info.json')
    analyzer.generate_report(output_path)
    
    # 可视化
    if args.visualize or args.all:
        analyzer.visualize_trajectories()
    
    # 导出图像
    if args.export_images or args.all:
        analyzer.export_sample_images()
    
    # 蒙太奇
    if args.montage or args.all:
        analyzer.create_montage()


if __name__ == '__main__':
    main()
