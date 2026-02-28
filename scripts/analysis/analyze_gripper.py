#!/usr/bin/env python3
"""
Gripper 值分布分析脚本
用于确定离散化阈值

使用方法:
    python scripts/analyze_gripper.py --data_dir ~/rl-vla/recorded_data/mix
"""
import os
import glob
import argparse
import numpy as np
import h5py
from collections import defaultdict


def analyze_gripper_distribution(data_dir: str, num_bins: int = 50):
    """分析 gripper 值的分布"""
    pattern = os.path.join(data_dir, "episode_*.hdf5")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No episode files found in {data_dir}")
        return None
    
    print(f"Found {len(files)} episode files")
    
    # 收集所有 gripper 值
    all_obs_gripper = []  # observations/gripper
    all_action_gripper_6 = []   # action index 6
    all_action_gripper_14 = []  # action index 14
    
    episode_stats = []
    
    for filepath in files:
        try:
            with h5py.File(filepath, 'r') as f:
                ep_name = os.path.basename(filepath)
                
                # 从 observations/gripper 读取
                if 'observations/gripper' in f:
                    gripper = f['observations/gripper'][:]
                    all_obs_gripper.extend(gripper.flatten())
                
                # 从 action 读取 (index 6 和 14)
                if 'action' in f:
                    action = f['action'][:]
                    if action.shape[-1] >= 15:
                        all_action_gripper_6.extend(action[:, 6])
                        all_action_gripper_14.extend(action[:, 14])
                        
                        # 每个 episode 的统计
                        ep_g14 = action[:, 14]
                        episode_stats.append({
                            'name': ep_name,
                            'min': ep_g14.min(),
                            'max': ep_g14.max(),
                            'mean': ep_g14.mean(),
                            'num_close': (ep_g14 < 0.04).sum(),
                            'num_open': (ep_g14 >= 0.04).sum(),
                        })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    # 转换为 numpy 数组
    obs_gripper = np.array(all_obs_gripper) if all_obs_gripper else None
    action_g6 = np.array(all_action_gripper_6) if all_action_gripper_6 else None
    action_g14 = np.array(all_action_gripper_14) if all_action_gripper_14 else None
    
    print("=" * 70)
    print("Gripper Value Distribution Analysis")
    print("=" * 70)
    
    # 主要分析 action index 14（推理时真正使用的 gripper）
    if action_g14 is not None and len(action_g14) > 0:
        gripper = action_g14
        print(f"\n[action[:, 14]] - 主要 gripper 通道 (推理使用)")
        print(f"  Total samples: {len(gripper)}")
        print(f"  Min: {gripper.min():.6f}")
        print(f"  Max: {gripper.max():.6f}")
        print(f"  Mean: {gripper.mean():.6f}")
        print(f"  Std: {gripper.std():.6f}")
        
        # 分位数分析
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\n  Percentiles:")
        for p in percentiles:
            val = np.percentile(gripper, p)
            print(f"    {p:3d}%: {val:.6f}")
        
        # 阈值分析
        thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        print(f"\n  Threshold Analysis (label: close < threshold, open >= threshold):")
        for thresh in thresholds:
            close_count = (gripper < thresh).sum()
            open_count = (gripper >= thresh).sum()
            close_ratio = close_count / len(gripper) * 100
            open_ratio = open_count / len(gripper) * 100
            print(f"    threshold={thresh:.2f}: close={close_ratio:5.1f}% ({close_count:6d}), open={open_ratio:5.1f}% ({open_count:6d})")
        
        # 直方图数据
        hist, bin_edges = np.histogram(gripper, bins=num_bins)
        print(f"\n  Histogram (top 10 bins by frequency):")
        sorted_idx = np.argsort(hist)[::-1][:10]
        for idx in sorted_idx:
            bin_center = (bin_edges[idx] + bin_edges[idx+1]) / 2
            count = hist[idx]
            pct = count / len(gripper) * 100
            bar = '█' * int(pct / 2)
            print(f"    [{bin_edges[idx]:.4f}, {bin_edges[idx+1]:.4f}): {count:6d} ({pct:5.1f}%) {bar}")
        
        # 双峰检测
        print(f"\n  Bimodal Analysis:")
        low_peak_idx = np.argmax(hist[:num_bins//3])
        high_peak_idx = num_bins//3*2 + np.argmax(hist[num_bins//3*2:])
        low_peak_center = (bin_edges[low_peak_idx] + bin_edges[low_peak_idx+1]) / 2
        high_peak_center = (bin_edges[high_peak_idx] + bin_edges[high_peak_idx+1]) / 2
        print(f"    Low peak (close): {low_peak_center:.4f} (count: {hist[low_peak_idx]})")
        print(f"    High peak (open): {high_peak_center:.4f} (count: {hist[high_peak_idx]})")
        
        # 建议阈值
        mid_threshold = (gripper.min() + gripper.max()) / 2
        print(f"\n  Suggested threshold (midpoint): {mid_threshold:.4f}")
        
        # 基于双峰的建议
        suggested = (low_peak_center + high_peak_center) / 2
        print(f"  Suggested threshold (between peaks): {suggested:.4f}")
    
    # 对比 action index 6
    if action_g6 is not None and len(action_g6) > 0:
        print(f"\n[action[:, 6]] - 第一个 gripper 通道 (joint 模式)")
        print(f"  Min: {action_g6.min():.6f}, Max: {action_g6.max():.6f}, Mean: {action_g6.mean():.6f}")
        # 检查两个通道是否一致
        if action_g14 is not None and len(action_g6) == len(action_g14):
            diff = np.abs(action_g6 - action_g14)
            print(f"  Diff with action[:, 14]: max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    # 对比 observations/gripper
    if obs_gripper is not None and len(obs_gripper) > 0:
        print(f"\n[observations/gripper]")
        print(f"  Min: {obs_gripper.min():.6f}, Max: {obs_gripper.max():.6f}, Mean: {obs_gripper.mean():.6f}")
    
    # 每个 episode 的统计
    print(f"\n" + "=" * 70)
    print("Per-Episode Statistics (using action[:, 14])")
    print("=" * 70)
    print(f"{'Episode':<40} {'Min':>8} {'Max':>8} {'Mean':>8} {'Close':>8} {'Open':>8}")
    print("-" * 70)
    for stat in episode_stats[:20]:  # 只显示前 20 个
        print(f"{stat['name']:<40} {stat['min']:>8.4f} {stat['max']:>8.4f} "
              f"{stat['mean']:>8.4f} {stat['num_close']:>8d} {stat['num_open']:>8d}")
    if len(episode_stats) > 20:
        print(f"... and {len(episode_stats) - 20} more episodes")
    
    # 总结建议
    print(f"\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    if action_g14 is not None:
        g = action_g14
        # 使用较保守的阈值（偏向 close 状态）
        recommended = 0.04
        close_ratio = (g < recommended).mean() * 100
        open_ratio = (g >= recommended).mean() * 100
        print(f"  Recommended threshold: {recommended}")
        print(f"  Expected class distribution: close={close_ratio:.1f}%, open={open_ratio:.1f}%")
        print(f"  Open value (gripper open):   0.078 (max observed: {g.max():.4f})")
        print(f"  Close value (gripper close): 0.012 (min observed: {g.min():.4f})")
        
        # 类别不平衡警告
        if close_ratio < 20 or close_ratio > 80:
            print(f"\n  ⚠️  WARNING: Class imbalance detected!")
            print(f"     Consider using class weights in CE loss: weight=[1.0, {open_ratio/close_ratio:.1f}]")
    
    return action_g14


def main():
    parser = argparse.ArgumentParser(description='Analyze gripper value distribution')
    parser.add_argument('--data_dir', type=str, default='~/rl-vla/recorded_data/mix',
                        help='Path to dataset directory')
    parser.add_argument('--num_bins', type=int, default=50,
                        help='Number of histogram bins')
    args = parser.parse_args()
    
    data_dir = os.path.expanduser(args.data_dir)
    analyze_gripper_distribution(data_dir, args.num_bins)


if __name__ == "__main__":
    main()
