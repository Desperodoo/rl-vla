#!/usr/bin/env python3
"""
可视化已标注数据的 Reward 曲线

从已标注的 HDF5 文件中读取 per_frame_reward，绘制：
- 上部：Reward 随时间变化的曲线
- 下部：采样关键帧的图像

使用示例:
    # 可视化单个 episode
    python plot_labeled_episodes.py --file episode_0001.hdf5 --output ./vis/
    
    # 可视化整个目录（批量）
    python plot_labeled_episodes.py --input-dir ./mix_perframe_reward --output ./vis/ --max-episodes 10
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional
import glob

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm


def load_labeled_episode(filepath: str) -> dict:
    """从已标注的 HDF5 文件加载数据"""
    with h5py.File(filepath, 'r') as f:
        data = {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
        }
        
        # 读取图像
        for key in ['observations/images', 'observations/image', 'images', 'image']:
            try:
                parts = key.split('/')
                obj = f
                for part in parts:
                    obj = obj[part]
                data['images'] = obj[:]
                break
            except (KeyError, TypeError):
                continue
        
        # 读取 per_frame_reward
        if 'per_frame_reward' in f:
            data['per_frame_reward'] = f['per_frame_reward'][:]
            data['has_perframe'] = True
        else:
            # 回退到 checkpoint 模式
            data['has_perframe'] = False
            if 'reward_timeline' in f:
                data['checkpoints'] = f['reward_timeline/checkpoints'][:]
                data['scores'] = f['reward_timeline/scores'][:]
        
        # 读取属性
        data['reward'] = f.attrs.get('reward', 0)
        data['max_score'] = f.attrs.get('max_score', 0)
        data['final_score'] = f.attrs.get('final_score', 0)
        data['done_frame'] = f.attrs.get('done_frame', -1)
        data['score_dropped'] = f.attrs.get('score_dropped', False)
        data['total_frames'] = len(data.get('images', []))
        
    return data


def plot_episode_visualization(
    data: dict,
    output_path: str,
    num_display_frames: int = 8,
    figsize: Tuple[int, int] = (16, 10),
):
    """绘制单个 episode 的可视化图"""
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1], hspace=0.3)
    
    total_frames = data['total_frames']
    
    # ============= 上部：Reward 曲线 =============
    ax_curve = fig.add_subplot(gs[0])
    
    if data['has_perframe']:
        # 逐帧 reward
        rewards = data['per_frame_reward']
        frames = np.arange(len(rewards))
        
        # 绘制曲线
        ax_curve.plot(frames, rewards, '-', color='#2196F3', linewidth=1.5, alpha=0.8)
        ax_curve.fill_between(frames, rewards, alpha=0.15, color='#2196F3')
        
        # 标记首次达到 5 分的位置
        if data['done_frame'] > 0:
            ax_curve.axvline(x=data['done_frame'], color='#4CAF50', linestyle='--', 
                           linewidth=2, alpha=0.7, label=f'Done at frame {data["done_frame"]}')
        
        # 如果分数有下降，标记下降区域
        if data['score_dropped'] and data['done_frame'] > 0:
            ax_curve.axvspan(data['done_frame'], total_frames, alpha=0.1, color='#f44336',
                           label='Score dropped region')
    else:
        # Checkpoint 模式
        checkpoints = data['checkpoints']
        scores = data['scores']
        ax_curve.plot(checkpoints, scores, 'o-', color='#2196F3', linewidth=2.5, 
                     markersize=10, markerfacecolor='white', markeredgewidth=2)
        ax_curve.fill_between(checkpoints, scores, alpha=0.15, color='#2196F3')
    
    # 坐标轴设置
    ax_curve.set_xlabel('Frame', fontsize=12, fontweight='bold')
    ax_curve.set_ylabel('Reward Score', fontsize=12, fontweight='bold')
    ax_curve.set_xlim(-5, total_frames + 5)
    ax_curve.set_ylim(0.5, 5.5)
    ax_curve.set_yticks([1, 2, 3, 4, 5])
    
    # 分数参考线
    score_colors = {1: '#f44336', 2: '#ff9800', 3: '#ffc107', 4: '#8bc34a', 5: '#4caf50'}
    for score in range(1, 6):
        ax_curve.axhline(y=score, color=score_colors[score], linestyle=':', alpha=0.3, linewidth=1)
    
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc='lower right', fontsize=9)
    
    # 标题
    title = f'Reward Timeline - {data["filename"]}\n'
    title += f'Total Frames: {total_frames} | Final Reward: {data["reward"]} (max={data["max_score"]}, final={data["final_score"]})'
    if data['score_dropped']:
        title += ' [DROPPED]'
    ax_curve.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    # ============= 下部：关键帧展示 =============
    images = data.get('images', None)
    if images is None or len(images) == 0:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return output_path
    
    # 选择要展示的帧索引（均匀采样）
    if total_frames <= num_display_frames:
        display_indices = list(range(total_frames))
    else:
        display_indices = np.linspace(0, total_frames - 1, num_display_frames, dtype=int).tolist()
    
    # 确保包含 done_frame
    if data['done_frame'] > 0 and data['done_frame'] - 1 not in display_indices:
        # 替换最近的点
        done_idx = data['done_frame'] - 1
        closest_idx = min(range(len(display_indices)), 
                         key=lambda i: abs(display_indices[i] - done_idx))
        display_indices[closest_idx] = done_idx
        display_indices.sort()
    
    # 创建帧展示子图
    n_display = len(display_indices)
    gs_frames = gridspec.GridSpecFromSubplotSpec(1, n_display, subplot_spec=gs[1], wspace=0.08)
    
    for i, idx in enumerate(display_indices):
        ax_frame = fig.add_subplot(gs_frames[i])
        
        frame = images[idx]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        ax_frame.imshow(frame)
        
        # 获取该帧的 reward
        if data['has_perframe']:
            score_at_frame = data['per_frame_reward'][idx]
        else:
            # 从 checkpoint 插值
            score_at_frame = data['scores'][-1]
            for j, cp in enumerate(data['checkpoints']):
                if idx < cp:
                    score_at_frame = data['scores'][j]
                    break
        
        # 设置标题和边框颜色
        if idx == 0:
            title_text = f'Start\nF{idx} | R={score_at_frame:.0f}'
            border_color = '#2196F3'
        elif idx == total_frames - 1:
            title_text = f'End\nF{idx} | R={score_at_frame:.0f}'
            border_color = '#9C27B0'
        elif data['done_frame'] > 0 and idx == data['done_frame'] - 1:
            title_text = f'Done!\nF{idx} | R={score_at_frame:.0f}'
            border_color = '#4CAF50'
        else:
            title_text = f'F{idx}\nR={score_at_frame:.0f}'
            border_color = '#999'
        
        ax_frame.set_title(title_text, fontsize=8, fontweight='bold',
                          color=border_color if border_color != '#999' else 'black')
        ax_frame.axis('off')
        
        for spine in ax_frame.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)
    
    # 帧展示标题
    fig.text(0.5, 0.38, f'Sampled Frames ({num_display_frames} of {total_frames})',
             ha='center', fontsize=10, fontweight='bold')
    
    # 底部信息
    fig.text(0.02, 0.02, f'File: {data["filename"]}', fontsize=8, alpha=0.5)
    fig.text(0.98, 0.02, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
             fontsize=8, alpha=0.5, ha='right')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='可视化已标注数据的 Reward 曲线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 输入（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', '-f', type=str,
                            help='单个 episode HDF5 文件路径')
    input_group.add_argument('--input-dir', '-i', type=str,
                            help='包含 episode_*.hdf5 的目录')
    
    parser.add_argument('--output', '-o', type=str, default='./reward_visualizations',
                       help='输出目录')
    parser.add_argument('--display-frames', '-n', type=int, default=8,
                       help='展示的关键帧数量 (默认: 8)')
    parser.add_argument('--max-episodes', type=int, default=None,
                       help='最多处理的 episode 数量（用于批量模式）')
    parser.add_argument('--filter', type=str, choices=['all', 'success', 'dropped'],
                       default='all', help='过滤条件')
    
    args = parser.parse_args()
    
    # 确定要处理的文件
    if args.file:
        files = [os.path.abspath(args.file)]
    else:
        pattern = os.path.join(args.input_dir, 'episode_*.hdf5')
        files = sorted(glob.glob(pattern))
        if args.max_episodes:
            files = files[:args.max_episodes]
    
    if not files:
        print("Error: 未找到任何 episode 文件")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("Reward Visualization for Labeled Episodes")
    print("=" * 60)
    print(f"Files to process: {len(files)}")
    print(f"Output directory: {args.output}")
    print("-" * 60)
    
    # 处理每个文件
    success_count = 0
    dropped_count = 0
    
    for filepath in tqdm(files, desc="Processing"):
        try:
            data = load_labeled_episode(filepath)
            
            # 过滤
            if args.filter == 'success' and data['reward'] < 5:
                continue
            if args.filter == 'dropped' and not data['score_dropped']:
                continue
            
            # 生成输出路径
            basename = os.path.splitext(data['filename'])[0]
            output_path = os.path.join(args.output, f'{basename}_reward.png')
            
            # 绘制
            plot_episode_visualization(data, output_path, args.display_frames)
            
            if data['reward'] >= 5:
                success_count += 1
            if data['score_dropped']:
                dropped_count += 1
                
        except Exception as e:
            print(f"\nError processing {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Complete!")
    print(f"Processed: {len(files)} episodes")
    print(f"Success (reward >= 5): {success_count}")
    print(f"Score dropped: {dropped_count}")
    print(f"Output: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
