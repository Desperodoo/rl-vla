#!/usr/bin/env python3
"""
RoboReward 批量标注脚本（带 done 检测）

遍历数据集目录中的所有 episode，使用 RoboReward 模型进行评分，
同时检测任务完成点并记录 done 标记。

功能：
1. 对每个 episode 进行时间线分析（分段评分）
2. 检测首次达到5分的时间点作为任务完成点
3. 标记该帧及后续帧的 done=True
4. 保存带有 reward 和 done 标签的数据

使用示例:
    # 基本用法
    python batch_label.py --input-dir ./recorded_data/mix --task "pick up the object"
    
    # 调整时间线检测点数（更精细）
    python batch_label.py --input-dir ./data --task "task" --checkpoints 20
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple
from tqdm import tqdm
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import h5py
from rlft.roboreward.config import RoboRewardConfig, SCORE_DESCRIPTIONS
from rlft.roboreward.labeler import RoboRewardLabeler
from rlft.roboreward.dataset_converter import DatasetConverter, TaskDescriptionManager


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


def analyze_episode_timeline(
    labeler: RoboRewardLabeler,
    all_frames: np.ndarray,
    task: str,
    num_checkpoints: int = 10,
    verbose: bool = False
) -> dict:
    """分析 episode 时间线，检测任务完成点
    
    Args:
        labeler: RoboReward 模型
        all_frames: 所有帧 (numpy array)
        task: 任务描述
        num_checkpoints: 评估的时间点数量
        verbose: 是否打印详情
    
    Returns:
        dict: {
            'checkpoints': [...],     # 评估的截止帧位置
            'scores': [...],          # 对应分数
            'done_frame': int,        # 首次达到5分的帧（-1表示未完成）
            'max_score': int,         # 最高分
            'final_score': int,       # 最终分数
            'done_array': np.array,   # done 标记数组
        }
    """
    total_frames = len(all_frames)
    
    # 生成评估的截止帧位置
    checkpoint_indices = np.linspace(
        max(total_frames // num_checkpoints, 1),
        total_frames,
        num_checkpoints,
        dtype=int
    ).tolist()
    checkpoint_indices[-1] = total_frames
    
    checkpoints = []
    scores = []
    done_frame = -1  # 首次达到5分的帧位置
    
    for end_frame in checkpoint_indices:
        frames_subset = all_frames[:end_frame]
        pil_frames = [Image.fromarray(f) for f in frames_subset]
        
        score, _ = labeler.score_episode(pil_frames, task, return_raw=True)
        
        checkpoints.append(end_frame)
        scores.append(score)
        
        # 记录首次达到5分的位置
        if score == 5 and done_frame == -1:
            done_frame = end_frame
        
        if verbose:
            progress = (end_frame / total_frames) * 100
            bar = "█" * score + "░" * (5 - score)
            print(f"    Frame 1-{end_frame:4d} ({progress:5.1f}%) | Score: {score} {bar}")
    
    # 构建 done 数组
    done_array = np.zeros(total_frames, dtype=bool)
    if done_frame > 0:
        # 从首次达到5分的帧开始，后续所有帧都标记为 done
        # 但需要更精确地确定 done 开始的位置
        # 使用线性插值估计：在上一个checkpoint和当前checkpoint之间
        done_start_idx = done_frame - 1  # 转为0-indexed，保守估计
        done_array[done_start_idx:] = True
    
    max_score = max(scores)
    final_score = scores[-1]
    
    return {
        'checkpoints': checkpoints,
        'scores': scores,
        'done_frame': done_frame,
        'done_start_index': done_frame - 1 if done_frame > 0 else -1,  # 0-indexed
        'max_score': max_score,
        'final_score': final_score,
        'score_dropped': max_score > final_score,
        'done_array': done_array,
        'total_frames': total_frames,
    }


def save_episode_with_reward_and_done(
    src_path: str,
    dst_path: str,
    reward: int,
    done_array: np.ndarray,
    timeline_info: dict,
    raw_output: str = ""
):
    """保存带有 reward 和 done 标签的 episode
    
    Args:
        src_path: 源 HDF5 文件路径
        dst_path: 目标 HDF5 文件路径
        reward: 最终 reward 分数（使用 max_score）
        done_array: done 标记数组
        timeline_info: 时间线分析信息
        raw_output: 原始模型输出
    """
    import shutil
    
    # 复制原始文件
    shutil.copy2(src_path, dst_path)
    
    # 添加 reward 和 done 信息
    with h5py.File(dst_path, 'r+') as f:
        # 添加 reward 属性
        f.attrs['reward'] = reward
        f.attrs['reward_raw_output'] = raw_output
        f.attrs['reward_model'] = 'teetone/RoboReward-8B'
        f.attrs['reward_timestamp'] = datetime.now().isoformat()
        
        # 添加 done 相关信息
        f.attrs['done_frame'] = timeline_info['done_frame']
        f.attrs['done_start_index'] = timeline_info['done_start_index']
        f.attrs['max_score'] = timeline_info['max_score']
        f.attrs['final_score'] = timeline_info['final_score']
        f.attrs['score_dropped'] = timeline_info['score_dropped']
        
        # 保存 done 数组
        if 'done' in f:
            del f['done']
        f.create_dataset('done', data=done_array, dtype=bool)
        
        # 保存时间线检查点信息
        if 'reward_timeline' in f:
            del f['reward_timeline']
        timeline_grp = f.create_group('reward_timeline')
        timeline_grp.create_dataset('checkpoints', data=np.array(timeline_info['checkpoints']))
        timeline_grp.create_dataset('scores', data=np.array(timeline_info['scores']))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RoboReward 批量标注工具（带 done 检测）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python batch_label.py --input-dir ./recorded_data/mix --task "pick up the red cube"
  python batch_label.py --input-dir ./data --task "task" --checkpoints 15
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="输入数据目录（包含 episode_*.hdf5 文件）"
    )
    
    # 任务描述（二选一）
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--task", "-t",
        type=str,
        help="统一的任务描述（用于所有 episodes）"
    )
    task_group.add_argument(
        "--task-file", "-tf",
        type=str,
        help="任务描述 JSON 文件路径"
    )
    
    # 可选参数
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="输出目录（默认为输入目录同级的 <input_name>_with_reward）"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="teetone/RoboReward-8B",
        help="模型名称或路径（默认: teetone/RoboReward-8B）"
    )
    parser.add_argument(
        "--checkpoints", "-c",
        type=int,
        default=10,
        help="每个 episode 的时间线评估点数（默认: 10）"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="模型精度（默认: bfloat16）"
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="禁用 Flash Attention"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅扫描文件，不进行实际推理"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细输出（包括每个checkpoint的分数）"
    )
    parser.add_argument(
        "--save-summary",
        action="store_true",
        default=True,
        help="保存标注摘要 JSON 文件（默认开启）"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 验证输入目录
    if not os.path.isdir(args.input_dir):
        print(f"Error: 输入目录不存在: {args.input_dir}")
        sys.exit(1)
    
    # 设置输出目录
    if args.output_dir is None:
        parent_dir = os.path.dirname(args.input_dir.rstrip('/'))
        input_name = os.path.basename(args.input_dir.rstrip('/'))
        args.output_dir = os.path.join(parent_dir, f"{input_name}_with_reward")
    
    print("=" * 70)
    print("RoboReward Batch Labeling Tool (with Done Detection)")
    print("=" * 70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Model:  {args.model}")
    print(f"Timeline checkpoints: {args.checkpoints}")
    
    # 创建配置
    config = RoboRewardConfig(
        model_name_or_path=args.model,
        torch_dtype=args.dtype,
        use_flash_attention=not args.no_flash_attn,
        sample_frames=-1,  # 使用全部帧
        input_data_dir=args.input_dir,
        output_data_dir=args.output_dir,
        verbose=args.verbose,
    )
    
    # 创建转换器（用于扫描文件）
    converter = DatasetConverter(config=config, sample_frames=-1)
    
    # 创建任务描述管理器
    if args.task:
        task_manager = TaskDescriptionManager(default_task=args.task)
        print(f"Task: {args.task}")
    else:
        task_manager = TaskDescriptionManager(task_file=args.task_file)
        print(f"Task file: {args.task_file}")
    
    # 扫描 episodes
    episode_files = converter.scan_episodes(args.input_dir)
    print(f"Found {len(episode_files)} episodes")
    print("=" * 70)
    
    if len(episode_files) == 0:
        print("Warning: No episode files found")
        sys.exit(0)
    
    # Dry run 模式
    if args.dry_run:
        print("\n[Dry Run] Files to process:")
        for i, f in enumerate(episode_files):
            task = task_manager.get_task(f)
            print(f"  {i+1}. {os.path.basename(f)} -> Task: {task[:50]}...")
        print("\n[Dry Run] Done")
        sys.exit(0)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建标注器
    print("\nLoading model...")
    labeler = RoboRewardLabeler(config=config)
    labeler.load_model()
    
    # 记录结果
    results = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "model": args.model,
        "checkpoints": args.checkpoints,
        "task_description": args.task if args.task else f"from file: {args.task_file}",
        "episodes": [],
        "statistics": {
            "total": len(episode_files),
            "score_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "mean_score": 0.0,
            "completed_count": 0,  # 达到5分的episode数
            "dropped_count": 0,     # 分数下降的episode数
        }
    }
    
    # 批量处理
    print("\nProcessing episodes with timeline analysis...")
    print("(Using max_score as final reward, detecting done frame)")
    print("-" * 70)
    
    scores = []
    completed = 0
    dropped = 0
    
    for filepath in tqdm(episode_files, desc="Processing"):
        episode_name = os.path.basename(filepath)
        
        try:
            # 加载所有帧
            all_frames, total_frames = load_frames_from_hdf5(filepath)
            
            # 获取任务描述
            task = task_manager.get_task(filepath)
            
            if args.verbose:
                print(f"\n  {episode_name} ({total_frames} frames):")
            
            # 时间线分析
            timeline_info = analyze_episode_timeline(
                labeler,
                all_frames,
                task,
                num_checkpoints=args.checkpoints,
                verbose=args.verbose
            )
            
            # 使用 max_score 作为最终 reward（解决分数下降问题）
            reward = timeline_info['max_score']
            
            # 获取最后一次评估的原始输出
            pil_frames = [Image.fromarray(f) for f in all_frames]
            _, raw_output = labeler.score_episode(pil_frames, task, return_raw=True)
            
            # 保存带 reward 和 done 的文件
            dst_path = os.path.join(args.output_dir, episode_name)
            save_episode_with_reward_and_done(
                filepath, 
                dst_path, 
                reward=reward,
                done_array=timeline_info['done_array'],
                timeline_info=timeline_info,
                raw_output=raw_output
            )
            
            # 统计
            if timeline_info['done_frame'] > 0:
                completed += 1
            if timeline_info['score_dropped']:
                dropped += 1
            
            # 记录结果
            episode_result = {
                "name": episode_name,
                "num_frames": int(total_frames),
                "task": task,
                "reward": int(reward),  # max_score
                "final_score": int(timeline_info['final_score']),
                "max_score": int(timeline_info['max_score']),
                "done_frame": int(timeline_info['done_frame']),
                "done_start_index": int(timeline_info['done_start_index']),
                "score_dropped": timeline_info['score_dropped'],
                "timeline_scores": [int(s) for s in timeline_info['scores']],
                "raw_output": raw_output,
            }
            results["episodes"].append(episode_result)
            results["statistics"]["score_distribution"][reward] += 1
            scores.append(reward)
            
            if args.verbose:
                status = ""
                if timeline_info['score_dropped']:
                    status = f" [DROPPED {timeline_info['max_score']}->{timeline_info['final_score']}]"
                elif timeline_info['done_frame'] > 0:
                    status = f" [DONE at frame {timeline_info['done_frame']}]"
                print(f"    -> Reward: {reward}{status}")
        
        except Exception as e:
            print(f"\nError processing {episode_name}: {e}")
            import traceback
            traceback.print_exc()
            results["episodes"].append({
                "name": episode_name,
                "error": str(e),
            })
    
    # 计算统计信息
    if scores:
        results["statistics"]["mean_score"] = sum(scores) / len(scores)
    results["statistics"]["completed_count"] = completed
    results["statistics"]["dropped_count"] = dropped
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("Labeling Complete!")
    print("=" * 70)
    print(f"Processed: {len(scores)}/{len(episode_files)} episodes")
    print(f"Mean score: {results['statistics']['mean_score']:.2f}")
    print(f"Completed (reached score 5): {completed} ({completed/len(scores)*100:.1f}%)")
    print(f"Score dropped after completion: {dropped}")
    
    print("\nScore distribution (using max_score as reward):")
    for score in range(1, 6):
        count = results["statistics"]["score_distribution"][score]
        pct = count / len(scores) * 100 if scores else 0
        bar = "█" * int(pct / 2)
        desc = SCORE_DESCRIPTIONS[score][:25] if score in SCORE_DESCRIPTIONS else ""
        print(f"  {score}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # 保存摘要
    if args.save_summary:
        summary_path = os.path.join(args.output_dir, "reward_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to: {summary_path}")
    
    print(f"\nOutput directory: {args.output_dir}")
    print("\nNote: 'reward' field uses max_score (peak score during episode)")
    print("      'done' array marks frames from first score-5 point onwards")


if __name__ == "__main__":
    main()
