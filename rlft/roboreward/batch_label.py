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


def analyze_episode_perframe(
    labeler: RoboRewardLabeler,
    all_frames: np.ndarray,
    task: str,
    stride: int = 1,
    verbose: bool = False,
    progress_bar: bool = True
) -> dict:
    """逐帧分析 episode，为每一帧（或按 stride 间隔）计算 reward
    
    Args:
        labeler: RoboReward 模型
        all_frames: 所有帧 (numpy array)
        task: 任务描述
        stride: 评估间隔（1 = 逐帧，10 = 每10帧评估一次）
        verbose: 是否打印详情
        progress_bar: 是否显示进度条
    
    Returns:
        dict: {
            'frame_indices': [...],    # 评估的帧索引
            'frame_scores': [...],     # 每帧的分数
            'per_frame_reward': np.array,  # 完整的逐帧 reward 数组（插值填充）
            'done_frame': int,         # 首次达到5分的帧（-1表示未完成）
            'max_score': int,          # 最高分
            'final_score': int,        # 最终分数
            'done_array': np.array,    # done 标记数组
        }
    """
    total_frames = len(all_frames)
    
    # 生成评估的帧索引
    frame_indices = list(range(0, total_frames, stride))
    if frame_indices[-1] != total_frames - 1:
        frame_indices.append(total_frames - 1)  # 确保包含最后一帧
    
    frame_scores = []
    done_frame = -1  # 首次达到5分的帧位置
    
    iterator = tqdm(frame_indices, desc="Per-frame scoring", leave=False) if progress_bar else frame_indices
    
    for end_idx in iterator:
        end_frame = end_idx + 1  # 转为 1-indexed（包含该帧）
        frames_subset = all_frames[:end_frame]
        pil_frames = [Image.fromarray(f) for f in frames_subset]
        
        score, _ = labeler.score_episode(pil_frames, task, return_raw=True)
        frame_scores.append(score)
        
        # 记录首次达到5分的位置
        if score == 5 and done_frame == -1:
            done_frame = end_frame
        
        if verbose:
            progress = (end_frame / total_frames) * 100
            bar = "█" * score + "░" * (5 - score)
            print(f"    Frame 1-{end_frame:4d} ({progress:5.1f}%) | Score: {score} {bar}")
    
    # 构建完整的逐帧 reward 数组（使用前向填充插值）
    per_frame_reward = np.zeros(total_frames, dtype=np.float32)
    
    # 使用阶梯插值：每个评估点的分数向前填充到下一个评估点
    for i, (idx, score) in enumerate(zip(frame_indices, frame_scores)):
        if i < len(frame_indices) - 1:
            next_idx = frame_indices[i + 1]
            per_frame_reward[idx:next_idx] = score
        else:
            per_frame_reward[idx:] = score
    
    # 构建 done 数组
    done_array = np.zeros(total_frames, dtype=bool)
    if done_frame > 0:
        done_start_idx = done_frame - 1  # 转为0-indexed
        done_array[done_start_idx:] = True
    
    max_score = max(frame_scores)
    final_score = frame_scores[-1]
    
    return {
        'frame_indices': frame_indices,
        'frame_scores': frame_scores,
        'per_frame_reward': per_frame_reward,
        'done_frame': done_frame,
        'done_start_index': done_frame - 1 if done_frame > 0 else -1,
        'max_score': max_score,
        'final_score': final_score,
        'score_dropped': max_score > final_score,
        'done_array': done_array,
        'total_frames': total_frames,
        'stride': stride,
        'num_evaluations': len(frame_indices),
    }


def save_episode_with_reward_and_done(
    src_path: str,
    dst_path: str,
    reward: int,
    done_array: np.ndarray,
    timeline_info: dict,
    raw_output: str = "",
    per_frame_reward: Optional[np.ndarray] = None
):
    """保存带有 reward 和 done 标签的 episode
    
    Args:
        src_path: 源 HDF5 文件路径
        dst_path: 目标 HDF5 文件路径
        reward: 最终 reward 分数（使用 max_score）
        done_array: done 标记数组
        timeline_info: 时间线分析信息
        raw_output: 原始模型输出
        per_frame_reward: 逐帧 reward 数组（可选，用于逐帧模式）
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
        
        # 根据模式保存不同的数据
        if 'checkpoints' in timeline_info:
            # checkpoint 模式
            timeline_grp.create_dataset('checkpoints', data=np.array(timeline_info['checkpoints']))
            timeline_grp.create_dataset('scores', data=np.array(timeline_info['scores']))
        elif 'frame_indices' in timeline_info:
            # 逐帧模式
            timeline_grp.create_dataset('frame_indices', data=np.array(timeline_info['frame_indices']))
            timeline_grp.create_dataset('frame_scores', data=np.array(timeline_info['frame_scores']))
            timeline_grp.attrs['stride'] = timeline_info.get('stride', 1)
            timeline_grp.attrs['num_evaluations'] = timeline_info.get('num_evaluations', 0)
        
        # 保存逐帧 reward 数组（如果有）
        if per_frame_reward is not None:
            if 'per_frame_reward' in f:
                del f['per_frame_reward']
            f.create_dataset('per_frame_reward', data=per_frame_reward, dtype=np.float32)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RoboReward 批量标注工具（带 done 检测）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（使用 10 个 checkpoint）
  python batch_label.py --input-dir ./recorded_data/mix --task "pick up the red cube"
  
  # 调整 checkpoint 数量
  python batch_label.py --input-dir ./data --task "task" --checkpoints 15
  
  # 逐帧标注（每帧都评估，非常慢！）
  python batch_label.py --input-dir ./data --task "task" --per-frame
  
  # 逐帧标注但使用 stride（每 5 帧评估一次）
  python batch_label.py --input-dir ./data --task "task" --per-frame --stride 5
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
        help="每个 episode 的时间线评估点数（默认: 10，仅在非逐帧模式下有效）"
    )
    parser.add_argument(
        "--per-frame",
        action="store_true",
        help="启用逐帧标注模式（非常慢！每帧或每 stride 帧评估一次）"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="逐帧模式下的评估间隔（默认: 1，即每帧都评估）"
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
        suffix = "_perframe_reward" if args.per_frame else "_with_reward"
        args.output_dir = os.path.join(parent_dir, f"{input_name}{suffix}")
    
    print("=" * 70)
    print("RoboReward Batch Labeling Tool (with Done Detection)")
    print("=" * 70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Model:  {args.model}")
    
    # 显示标注模式
    if args.per_frame:
        print(f"Mode: Per-frame labeling (stride={args.stride})")
        print("⚠️  WARNING: Per-frame mode is VERY SLOW! Each episode may take several minutes.")
    else:
        print(f"Mode: Checkpoint labeling (checkpoints={args.checkpoints})")
    
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
        if args.per_frame:
            # 估算总时间
            avg_frames = 300  # 假设平均 300 帧
            eval_per_episode = avg_frames // args.stride
            total_evals = eval_per_episode * len(episode_files)
            est_time_minutes = total_evals * 0.5 / 60  # 假设每次评估 0.5 秒
            print(f"\n[Dry Run] Estimated evaluations: ~{total_evals}")
            print(f"[Dry Run] Estimated time: ~{est_time_minutes:.1f} minutes")
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
        "mode": "per_frame" if args.per_frame else "checkpoint",
        "checkpoints": args.checkpoints if not args.per_frame else None,
        "stride": args.stride if args.per_frame else None,
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
            
            # 根据模式选择分析方法
            if args.per_frame:
                # 逐帧模式
                timeline_info = analyze_episode_perframe(
                    labeler,
                    all_frames,
                    task,
                    stride=args.stride,
                    verbose=args.verbose,
                    progress_bar=not args.verbose  # verbose 模式不显示进度条
                )
                per_frame_reward = timeline_info['per_frame_reward']
            else:
                # checkpoint 模式
                timeline_info = analyze_episode_timeline(
                    labeler,
                    all_frames,
                    task,
                    num_checkpoints=args.checkpoints,
                    verbose=args.verbose
                )
                per_frame_reward = None
            
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
                raw_output=raw_output,
                per_frame_reward=per_frame_reward
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
            }
            
            # 根据模式添加不同的时间线信息
            if args.per_frame:
                episode_result["mode"] = "per_frame"
                episode_result["stride"] = args.stride
                episode_result["num_evaluations"] = timeline_info['num_evaluations']
                episode_result["frame_indices"] = timeline_info['frame_indices']
                episode_result["frame_scores"] = [int(s) for s in timeline_info['frame_scores']]
            else:
                episode_result["mode"] = "checkpoint"
                episode_result["timeline_scores"] = [int(s) for s in timeline_info['scores']]
            
            episode_result["raw_output"] = raw_output
            
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
