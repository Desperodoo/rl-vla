#!/usr/bin/env python3
"""
Episode Reward 时间线分析脚本

功能：
1. 分析一个 episode 在不同时间点（进度）的 reward 分数
2. 检测任务是否在中途完成后分数又下降（机械臂移动导致目标丢失）
3. 绘制 reward 随时间变化的曲线 + 关键帧图像
"""

import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from PIL import Image
    import h5py
except ImportError as e:
    print(f"Error: 缺少依赖 - {e}")
    print("请运行: pip install matplotlib pillow h5py")
    sys.exit(1)

from rlft.roboreward.config import RoboRewardConfig, SCORE_DESCRIPTIONS
from rlft.roboreward.labeler import RoboRewardLabeler
from rlft.roboreward.dataset_converter import DatasetConverter


def load_frames_from_hdf5(filepath: str) -> Tuple[np.ndarray, int]:
    """加载 HDF5 文件中的所有帧"""
    with h5py.File(filepath, 'r') as f:
        for key in ['observations/images', 'observations/image', 'images', 'image']:
            try:
                parts = key.split('/')
                obj = f
                for part in parts:
                    obj = obj[part]
                images = obj[:]
                return images, len(images)
            except (KeyError, TypeError):
                continue
    raise ValueError(f"无法在 {filepath} 中找到图像数据")


def analyze_reward_timeline(
    labeler: RoboRewardLabeler,
    filepath: str,
    task: str,
    num_checkpoints: int = 10,
    verbose: bool = True
) -> Dict:
    """分析 episode 在不同时间点的 reward 分数
    
    从 episode 开始到不同的截止点，评估 reward 分数变化。
    例如：评估前10%、20%、30%...100%的帧序列。
    
    Args:
        labeler: RoboReward 模型
        filepath: HDF5 文件路径  
        task: 任务描述
        num_checkpoints: 评估的时间点数量
        verbose: 是否打印详情
    
    Returns:
        包含时间线分析结果的字典
    """
    
    all_frames, total_frames = load_frames_from_hdf5(filepath)
    
    # 生成评估的截止帧位置（均匀分布）
    # 例如 10 个 checkpoint: 10%, 20%, ..., 100%
    checkpoint_indices = np.linspace(
        total_frames // num_checkpoints,  # 从至少有一些帧开始
        total_frames,
        num_checkpoints,
        dtype=int
    ).tolist()
    
    # 确保最后一个是总帧数
    checkpoint_indices[-1] = total_frames
    
    results = {
        'checkpoints': [],      # 截止帧索引
        'progress': [],         # 进度百分比
        'scores': [],           # 对应的分数
        'times': [],            # 推理时间
        'all_frames': all_frames,
        'total_frames': total_frames,
        'filepath': filepath,
        'task': task
    }
    
    if verbose:
        print(f"\n文件: {os.path.basename(filepath)}")
        print(f"总帧数: {total_frames}")
        print(f"任务: {task}")
        print(f"评估点数: {num_checkpoints}")
        print("-" * 60)
    
    for end_frame in checkpoint_indices:
        # 取从第1帧到 end_frame 的所有帧
        frames_subset = all_frames[:end_frame]
        
        # 转换为 PIL 图像列表
        pil_frames = [Image.fromarray(f) for f in frames_subset]
        
        progress = (end_frame / total_frames) * 100
        
        start_time = time.time()
        score, raw_output = labeler.score_episode(pil_frames, task, return_raw=True)
        infer_time = time.time() - start_time
        
        results['checkpoints'].append(end_frame)
        results['progress'].append(progress)
        results['scores'].append(score)
        results['times'].append(infer_time)
        
        if verbose:
            bar = "█" * score + "░" * (5 - score)
            print(f"  Frame 1-{end_frame:4d} ({progress:5.1f}%) | Score: {score} {bar} | {infer_time:.2f}s")
    
    # 分析结果
    scores = results['scores']
    max_score = max(scores)
    max_score_idx = scores.index(max_score)
    final_score = scores[-1]
    
    results['analysis'] = {
        'max_score': max_score,
        'max_score_checkpoint': checkpoint_indices[max_score_idx],
        'max_score_progress': results['progress'][max_score_idx],
        'final_score': final_score,
        'score_dropped': max_score > final_score,
        'drop_amount': max_score - final_score if max_score > final_score else 0
    }
    
    if verbose:
        print("-" * 60)
        analysis = results['analysis']
        if analysis['score_dropped']:
            print(f"  *** Score dropped! ***")
            print(f"  Peak score {analysis['max_score']} at {analysis['max_score_progress']:.1f}% ({analysis['max_score_checkpoint']} frames)")
            print(f"  Final score: {analysis['final_score']} (dropped {analysis['drop_amount']})")
        else:
            print(f"  Score stable, final: {analysis['final_score']}")
    
    return results


def create_timeline_visualization(
    results: Dict,
    output_path: str,
    num_display_frames: int = 8,
    figsize: Tuple[int, int] = (18, 14)
):
    """创建时间线分析可视化
    
    上部：reward 随时间（进度）变化的曲线
    下部：关键帧图像（对应曲线上的采样点）
    """
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1], hspace=0.25)
    
    # ============= 上部：Reward 时间线曲线 =============
    ax_curve = fig.add_subplot(gs[0])
    
    checkpoints = results['checkpoints']
    progress = results['progress']
    scores = results['scores']
    total_frames = results['total_frames']
    analysis = results['analysis']
    
    # 绘制曲线
    ax_curve.plot(progress, scores, 'o-', 
                  color='#2196F3', linewidth=2.5, markersize=10,
                  markerfacecolor='white', markeredgewidth=2,
                  label='Reward Score')
    
    # 填充区域
    ax_curve.fill_between(progress, scores, alpha=0.15, color='#2196F3')
    
    # 标注每个点的分数
    for p, s, cp in zip(progress, scores, checkpoints):
        color = '#4CAF50' if s == analysis['max_score'] else '#1565C0'
        ax_curve.annotate(f'{s}', (p, s), 
                         textcoords="offset points", xytext=(0, 12), 
                         ha='center', fontsize=11, fontweight='bold',
                         color=color)
    
    # 标记最高分位置
    if analysis['score_dropped']:
        max_idx = scores.index(analysis['max_score'])
        ax_curve.axvline(x=progress[max_idx], color='#4CAF50', linestyle='--', 
                        linewidth=2, alpha=0.7, label=f'Peak score at {progress[max_idx]:.0f}%')
        
        # 标记下降区域
        ax_curve.axvspan(progress[max_idx], 100, alpha=0.1, color='#f44336', 
                        label='Score dropped')
    
    # 坐标轴设置
    ax_curve.set_xlabel('Episode Progress (%)', fontsize=13, fontweight='bold')
    ax_curve.set_ylabel('Reward Score', fontsize=13, fontweight='bold')
    
    # 标题显示分析结果
    title = f'Reward Timeline - {os.path.basename(results["filepath"])}\n'
    title += f'Task: "{results["task"]}"\n'
    if analysis['score_dropped']:
        title += f'Score dropped: {analysis["max_score"]} -> {analysis["final_score"]} after {analysis["max_score_progress"]:.0f}%'
    else:
        title += f'Final Score: {analysis["final_score"]}'
    
    ax_curve.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # 刻度设置
    ax_curve.set_xticks(np.arange(0, 101, 10))
    ax_curve.set_yticks([1, 2, 3, 4, 5])
    ax_curve.set_ylim(0.5, 5.8)
    ax_curve.set_xlim(-2, 105)
    
    # 分数参考线
    score_colors = {1: '#f44336', 2: '#ff9800', 3: '#ffc107', 4: '#8bc34a', 5: '#4caf50'}
    for score in range(1, 6):
        ax_curve.axhline(y=score, color=score_colors[score], linestyle=':', alpha=0.3, linewidth=1)
    
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc='lower right', fontsize=10)
    
    # 推理时间信息
    total_time = sum(results['times'])
    ax_curve.text(0.02, 0.98, f'Total inference time: {total_time:.1f}s', 
                 transform=ax_curve.transAxes, fontsize=9, alpha=0.6, va='top')
    
    # ============= 下部：关键帧展示 =============
    all_frames = results['all_frames']
    
    # 选择要展示的帧：对应曲线上的评估点
    if len(checkpoints) <= num_display_frames:
        display_indices = [cp - 1 for cp in checkpoints]  # 转为0-indexed
    else:
        # 均匀选择
        step = len(checkpoints) // num_display_frames
        selected_checkpoints = checkpoints[::step][:num_display_frames]
        display_indices = [cp - 1 for cp in selected_checkpoints]
    
    # 确保包含最高分点和最后一帧
    if analysis['score_dropped']:
        peak_idx = analysis['max_score_checkpoint'] - 1
        if peak_idx not in display_indices:
            display_indices.insert(len(display_indices)//2, peak_idx)
            display_indices = display_indices[:num_display_frames]
    
    display_frames = [all_frames[i] for i in display_indices]
    
    # 获取每个展示帧对应的分数（最接近的checkpoint）
    def get_score_at_frame(frame_idx):
        for i, cp in enumerate(checkpoints):
            if frame_idx + 1 <= cp:
                return scores[i]
        return scores[-1]
    
    # 创建帧展示子图
    n_display = len(display_frames)
    gs_frames = gridspec.GridSpecFromSubplotSpec(
        1, n_display, subplot_spec=gs[1], wspace=0.08
    )
    
    for i, (frame, idx) in enumerate(zip(display_frames, display_indices)):
        ax_frame = fig.add_subplot(gs_frames[i])
        
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
        
        ax_frame.imshow(frame)
        
        progress_at_frame = ((idx + 1) / total_frames) * 100
        score_at_frame = get_score_at_frame(idx)
        
        # 设置标题和边框颜色
        if idx == 0:
            title = f'START\nScore: {score_at_frame}'
            border_color = '#4CAF50'
        elif idx == total_frames - 1:
            title = f'END\nScore: {score_at_frame}'
            border_color = '#f44336'
        elif idx + 1 == analysis['max_score_checkpoint']:
            title = f'PEAK ({progress_at_frame:.0f}%)\nScore: {score_at_frame}'
            border_color = '#FFD700'
        else:
            title = f'Frame {idx+1}\n({progress_at_frame:.0f}%) S:{score_at_frame}'
            border_color = '#2196F3' if score_at_frame >= 4 else '#999'
        
        ax_frame.set_title(title, fontsize=9, fontweight='bold', 
                          color=border_color if border_color != '#999' else 'black')
        ax_frame.axis('off')
        
        for spine in ax_frame.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3 if border_color in ['#FFD700', '#f44336', '#4CAF50'] else 1)
    
    # 帧展示标题
    fig.text(0.5, 0.42, f'Key Frames (Total: {total_frames} frames)', 
             ha='center', fontsize=12, fontweight='bold')
    
    # 底部信息
    fig.text(0.02, 0.02, f'File: {os.path.basename(results["filepath"])}', fontsize=9, alpha=0.5)
    fig.text(0.98, 0.02, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
             fontsize=9, alpha=0.5, ha='right')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n图像已保存: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='分析 Episode Reward 时间线（检测分数是否在任务完成后下降）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法：分析单个episode的reward时间线
  python plot_reward_curve.py --file episode_0010.hdf5 --task "pick up the black tape"
  
  # 增加评估点数（更精细的曲线）
  python plot_reward_curve.py --file episode.hdf5 --task "task" --checkpoints 20
  
  # 指定输出路径
  python plot_reward_curve.py --file episode.hdf5 --task "task" --output ./result.png
        """
    )
    
    parser.add_argument('--file', '-f', type=str, required=True,
                       help='Episode HDF5 文件路径')
    parser.add_argument('--task', '-t', type=str, required=True,
                       help='任务描述')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出图像路径（默认: <文件名>_reward_timeline.png）')
    parser.add_argument('--checkpoints', '-c', type=int, default=10,
                       help='评估的时间点数量 (默认: 10)')
    parser.add_argument('--display-frames', '-n', type=int, default=8,
                       help='展示的关键帧数量 (默认: 8)')
    parser.add_argument('--model', '-m', type=str, default='teetone/RoboReward-8B',
                       help='模型路径 (默认: teetone/RoboReward-8B)')
    
    args = parser.parse_args()
    
    # 检查文件
    filepath = os.path.abspath(args.file)
    if not os.path.exists(filepath):
        print(f"Error: 找不到文件 {filepath}")
        sys.exit(1)
    
    # 输出路径
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        output_path = os.path.join(os.path.dirname(filepath), f'{basename}_reward_timeline.png')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("=" * 60)
    print("Episode Reward Timeline Analysis")
    print("=" * 60)
    print("Goal: Detect if score drops after task completion")
    
    # 初始化
    config = RoboRewardConfig(
        model_name_or_path=args.model,
        torch_dtype='bfloat16',
        sample_frames=-1,  # 使用全部帧
        verbose=False,
    )
    
    labeler = RoboRewardLabeler(config)
    
    # 加载模型
    print("\n[1/3] Loading model...")
    labeler.load_model()
    
    # 分析时间线
    print("\n[2/3] Analyzing reward timeline...")
    results = analyze_reward_timeline(
        labeler, filepath, args.task, 
        num_checkpoints=args.checkpoints,
        verbose=True
    )
    
    # 生成可视化
    print("\n[3/3] Generating visualization...")
    create_timeline_visualization(results, output_path, args.display_frames)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    analysis = results['analysis']
    
    if analysis['score_dropped']:
        print(f"\n  *** Score Dropped! ***")
        print(f"  Peak score: {analysis['max_score']} (at {analysis['max_score_progress']:.1f}%)")
        print(f"  Final score: {analysis['final_score']}")
        print(f"  Drop: {analysis['drop_amount']} points")
        print(f"\n  Suggestion: Task may have completed at {analysis['max_score_progress']:.0f}%,")
        print(f"  robot arm movement after completion caused score drop.")
        print(f"  Consider marking this episode as score 5.")
    else:
        print(f"\n  Score stable")
        print(f"  Final score: {analysis['final_score']}")
    
    print(f"\nTimeline:")
    for p, s in zip(results['progress'], results['scores']):
        bar = "█" * s + "░" * (5 - s)
        marker = " <- PEAK" if s == analysis['max_score'] and analysis['score_dropped'] else ""
        print(f"  {p:5.1f}% -> {s} {bar}{marker}")
    
    print(f"\nOutput: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
