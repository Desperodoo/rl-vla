#!/usr/bin/env python3
"""
低分 Episode 查找与可视化脚本

功能：
1. 从 reward_summary.json 中提取低于指定分数的 episode
2. 为每个低分 episode 生成关键帧可视化图像
3. 生成 HTML 索引页面方便浏览
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import matplotlib.pyplot as plt
    from PIL import Image
    import h5py
except ImportError as e:
    print(f"Error: 缺少依赖 - {e}")
    print("请运行: pip install matplotlib pillow h5py")
    sys.exit(1)


def load_frames_from_hdf5(filepath: str) -> np.ndarray:
    """加载 HDF5 文件中的所有帧"""
    with h5py.File(filepath, 'r') as f:
        for key in ['observations/images', 'observations/image', 'images', 'image']:
            try:
                parts = key.split('/')
                obj = f
                for part in parts:
                    obj = obj[part]
                return obj[:]
            except (KeyError, TypeError):
                continue
    raise ValueError(f"无法在 {filepath} 中找到图像数据")


def sample_frames(frames: np.ndarray, num_frames: int = 10, focus_end: bool = False) -> tuple:
    """采样帧用于展示
    
    Args:
        frames: 所有帧
        num_frames: 采样数量
        focus_end: 是否重点关注后半部分（前半稀疏，后半密集）
    
    Returns:
        sampled_frames, indices
    """
    total = len(frames)
    if total <= num_frames:
        return frames, list(range(total))
    
    if focus_end:
        # 前半部分稀疏，后半部分密集
        mid = num_frames // 2
        front_indices = np.linspace(0, int(total * 0.5), mid, dtype=int).tolist()
        back_indices = np.linspace(int(total * 0.5), total - 1, num_frames - mid, dtype=int).tolist()
        indices = sorted(list(set(front_indices + back_indices)))[:num_frames]
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    
    return frames[indices], indices


def create_episode_visualization(
    filepath: str,
    episode_name: str,
    reward: int,
    task: str,
    output_path: str,
    num_frames: int = 10,
    focus_end: bool = False,
    episode_info: dict = None
) -> int:
    """为单个 episode 创建可视化图"""
    
    all_frames = load_frames_from_hdf5(filepath)
    total_frames = len(all_frames)
    display_frames, indices = sample_frames(all_frames, num_frames, focus_end)
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 计算布局
    n_cols = 5
    n_rows = (len(display_frames) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows + 1.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # 分数颜色
    score_colors = {1: '#f44336', 2: '#ff9800', 3: '#ffc107', 4: '#8bc34a', 5: '#4caf50'}
    title_color = score_colors.get(reward, '#9e9e9e')
    
    # 构建标题（包含done信息）
    title = f'Episode: {episode_name}\n'
    title += f'Score: {reward} | Frames: {total_frames}'
    
    # 如果有额外信息（done_frame, score_dropped等）
    if episode_info:
        done_frame = episode_info.get('done_frame', -1)
        score_dropped = episode_info.get('score_dropped', False)
        final_score = episode_info.get('final_score', reward)
        
        if done_frame > 0:
            title += f' | Done at frame {done_frame}'
        if score_dropped:
            title += f' | DROPPED ({reward}->{final_score})'
    
    title += f'\nTask: "{task}"'
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98, color=title_color)
    
    # 显示帧
    for idx, (frame, frame_idx) in enumerate(zip(display_frames, indices)):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        ax.imshow(frame)
        
        progress = (frame_idx / total_frames) * 100
        
        if frame_idx == 0:
            ax.set_title('START', fontsize=10, color='green', fontweight='bold')
        elif frame_idx == total_frames - 1:
            ax.set_title(f'END ({total_frames})', fontsize=10, color='red', fontweight='bold')
        elif focus_end and frame_idx >= total_frames * 0.5:
            ax.set_title(f'F{frame_idx+1} ({progress:.0f}%)', fontsize=9, color='#2196F3', fontweight='bold')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#2196F3')
                spine.set_linewidth(2)
        else:
            ax.set_title(f'F{frame_idx+1} ({progress:.0f}%)', fontsize=9)
        ax.axis('off')
    
    # 隐藏多余子图
    for idx in range(len(display_frames), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return total_frames


def generate_html_index(episodes: list, output_dir: str, task: str):
    """生成 HTML 索引页面"""
    
    html_path = os.path.join(output_dir, 'index.html')
    
    # 统计分数分布
    score_counts = {}
    for ep in episodes:
        score = ep['reward']
        score_counts[score] = score_counts.get(score, 0) + 1
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Low Score Episodes</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1600px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }}
        .task {{ background: #e3f2fd; padding: 10px 15px; border-radius: 5px; margin-bottom: 20px; }}
        .stats {{ display: flex; gap: 20px; margin-bottom: 20px; }}
        .stat-box {{ background: white; padding: 15px 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
        .stat-value {{ font-size: 32px; font-weight: bold; }}
        .score-1 {{ color: #f44336; }} .score-2 {{ color: #ff9800; }} .score-3 {{ color: #ffc107; }} .score-4 {{ color: #8bc34a; }}
        .episode {{ background: white; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }}
        .episode-header {{ padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; }}
        .episode-name {{ font-weight: bold; font-size: 16px; }}
        .episode-score {{ padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; }}
        .bg-score-1 {{ background: #f44336; }} .bg-score-2 {{ background: #ff9800; }} .bg-score-3 {{ background: #ffc107; color: #333; }} .bg-score-4 {{ background: #8bc34a; }}
        .episode-image {{ width: 100%; display: block; }}
        .section-title {{ margin: 30px 0 15px 0; }}
    </style>
</head>
<body>
    <h1>Low Score Episodes Analysis</h1>
    <div class="task"><strong>Task:</strong> {task}</div>
    <div class="stats">
"""
    
    for score in sorted(score_counts.keys()):
        html += f'<div class="stat-box"><div class="stat-value score-{score}">{score_counts[score]}</div><div>Score {score}</div></div>\n'
    
    html += '</div>\n'
    
    current_score = None
    for ep in episodes:
        score, name = ep['reward'], ep['name']
        if score != current_score:
            current_score = score
            html += f'<h2 class="section-title score-{score}">Score {score} Episodes</h2>\n'
        
        img_name = f"score{score}_{name.replace('.hdf5', '.png')}"
        html += f"""<div class="episode">
    <div class="episode-header">
        <span class="episode-name">{name}</span>
        <span class="episode-score bg-score-{score}">Score: {score}</span>
    </div>
    <img class="episode-image" src="{img_name}" alt="{name}">
</div>
"""
    
    html += '</body></html>'
    
    with open(html_path, 'w') as f:
        f.write(html)
    
    return html_path


def main():
    parser = argparse.ArgumentParser(
        description='查找并可视化低分 Episode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查找所有低于5分的episode
  python find_low_scores.py --summary ../../recorded_data/mix_with_reward/reward_summary.json
  
  # 只查找3分及以下的episode
  python find_low_scores.py --summary ./reward_summary.json --max-score 3
  
  # 使用密集后半部分采样（检查任务完成点）
  python find_low_scores.py --summary ./reward_summary.json --focus-end
        """
    )
    
    parser.add_argument('--summary', '-s', type=str, required=True,
                       help='reward_summary.json 路径')
    parser.add_argument('--data-dir', '-d', type=str, default=None,
                       help='原始数据目录（默认从summary推断）')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='输出目录（默认: <数据目录>_low_scores）')
    parser.add_argument('--max-score', type=int, default=4,
                       help='筛选分数 <= 此值的episode (默认: 4)')
    parser.add_argument('--num-frames', '-n', type=int, default=10,
                       help='每个episode展示的帧数 (默认: 10)')
    parser.add_argument('--focus-end', action='store_true',
                       help='重点关注后半部分帧（用于检查任务完成点）')
    
    args = parser.parse_args()
    
    # 解析路径
    summary_path = os.path.abspath(args.summary)
    
    if not os.path.exists(summary_path):
        print(f"Error: 找不到 {summary_path}")
        sys.exit(1)
    
    # 加载 summary
    print(f"加载: {summary_path}")
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # 推断数据目录
    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
    else:
        # 从 summary 中的 input_dir 推断
        input_dir = summary.get('input_dir', '')
        summary_dir = os.path.dirname(summary_path)
        data_dir = os.path.normpath(os.path.join(summary_dir, input_dir)) if input_dir else os.path.dirname(summary_dir)
    
    # 输出目录
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(os.path.dirname(summary_path), 'low_scores_analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取低分 episode
    low_score_episodes = [ep for ep in summary['episodes'] if ep['reward'] <= args.max_score]
    low_score_episodes.sort(key=lambda x: (x['reward'], x['name']))
    
    if not low_score_episodes:
        print(f"没有找到分数 <= {args.max_score} 的 episode")
        return
    
    # 统计
    print(f"\n找到 {len(low_score_episodes)} 个低分 episode (score <= {args.max_score}):")
    score_groups = {}
    for ep in low_score_episodes:
        score = ep['reward']
        score_groups.setdefault(score, []).append(ep['name'])
    
    for score in sorted(score_groups.keys()):
        print(f"  Score {score}: {len(score_groups[score])} 个")
    
    task = summary.get('task_description', 'unknown')
    
    # 生成可视化
    print(f"\n生成可视化到: {output_dir}")
    print("-" * 60)
    
    for i, ep in enumerate(low_score_episodes):
        name, reward = ep['name'], ep['reward']
        filepath = os.path.join(data_dir, name)
        
        if not os.path.exists(filepath):
            print(f"[{i+1}/{len(low_score_episodes)}] 跳过 {name} - 文件不存在")
            continue
        
        output_name = f"score{reward}_{name.replace('.hdf5', '.png')}"
        output_path = os.path.join(output_dir, output_name)
        
        try:
            total = create_episode_visualization(
                filepath, name, reward, task, output_path,
                args.num_frames, args.focus_end, episode_info=ep
            )
            # 显示额外信息
            extra = ""
            if ep.get('done_frame', -1) > 0:
                extra += f" | done@{ep['done_frame']}"
            if ep.get('score_dropped', False):
                extra += f" | dropped->{ep.get('final_score', '?')}"
            print(f"[{i+1}/{len(low_score_episodes)}] Score {reward} | {name} | {total} frames{extra}")
        except Exception as e:
            print(f"[{i+1}/{len(low_score_episodes)}] Error: {name} - {e}")
    
    # 生成 HTML 索引
    html_path = generate_html_index(low_score_episodes, output_dir, task)
    
    print("-" * 60)
    print(f"\n完成！共 {len(low_score_episodes)} 个低分 episode")
    print(f"输出目录: {output_dir}")
    print(f"索引页面: {html_path}")


if __name__ == '__main__':
    main()
