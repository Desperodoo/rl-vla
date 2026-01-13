#!/usr/bin/env python3
"""
CARM 数据集图像通道修复脚本

问题描述：
    之前的 image_sync.py 将 ROS RGB8 消息转换为 BGR 后保存到 HDF5，
    但文档和训练脚本都假设是 RGB 格式，导致颜色不正确。

修复方法：
    将 HDF5 中的 BGR 图像转换为 RGB 格式：images[:, :, :, ::-1]

使用方法：
    python fix_dataset_channels.py --data_dir /path/to/recorded_data --backup
    
    参数：
        --data_dir: 数据集目录
        --backup: 是否备份原文件（推荐）
        --dry_run: 只检查不修改
        --verify: 修复后验证
"""

import os
import glob
import shutil
import argparse
from datetime import datetime
import numpy as np
import h5py
from tqdm import tqdm


def check_image_format(images: np.ndarray) -> str:
    """
    尝试推断图像格式（RGB 或 BGR）
    通过检查常见的颜色分布特征
    
    注意：这只是启发式检测，不保证100%准确
    """
    # 取样本图像
    sample = images[len(images)//2]  # 取中间帧
    
    # 计算各通道的均值
    mean_ch0 = sample[:, :, 0].mean()
    mean_ch1 = sample[:, :, 1].mean()
    mean_ch2 = sample[:, :, 2].mean()
    
    # 蓝色物体（如蓝色桌面）在 BGR 格式下 channel 0 (B) 较高
    # 在 RGB 格式下 channel 2 (B) 较高
    # 这只是一个粗略的启发式方法
    
    return {
        'channel_means': [mean_ch0, mean_ch1, mean_ch2],
        'hint': 'Cannot reliably detect RGB/BGR without ground truth'
    }


def fix_episode(filepath: str, backup: bool = True, dry_run: bool = False) -> dict:
    """
    修复单个 episode 的图像通道顺序
    
    Args:
        filepath: HDF5 文件路径
        backup: 是否备份原文件
        dry_run: 只检查不修改
        
    Returns:
        修复结果字典
    """
    result = {
        'filepath': filepath,
        'status': 'unknown',
        'num_frames': 0,
        'image_shape': None,
        'error': None,
    }
    
    try:
        with h5py.File(filepath, 'r') as f:
            if 'observations/images' not in f:
                result['status'] = 'skipped'
                result['error'] = 'No images dataset found'
                return result
            
            images = f['observations/images'][:]
            result['num_frames'] = len(images)
            result['image_shape'] = images.shape
            
            # 检查当前格式
            format_info = check_image_format(images)
            result['format_info'] = format_info
        
        if dry_run:
            result['status'] = 'dry_run'
            return result
        
        # 备份原文件
        if backup:
            backup_path = filepath + '.backup'
            if not os.path.exists(backup_path):
                shutil.copy2(filepath, backup_path)
                result['backup_path'] = backup_path
        
        # 修复：BGR -> RGB (反转最后一个维度)
        with h5py.File(filepath, 'r+') as f:
            images = f['observations/images'][:]
            
            # 反转通道顺序
            images_fixed = images[:, :, :, ::-1].copy()
            
            # 原地更新
            f['observations/images'][...] = images_fixed
            
            # 添加元数据标记
            f.attrs['channel_format'] = 'RGB'
            f.attrs['channel_fixed_at'] = datetime.now().isoformat()
        
        result['status'] = 'fixed'
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    
    return result


def verify_fix(filepath: str) -> dict:
    """验证修复结果"""
    result = {'filepath': filepath, 'verified': False}
    
    try:
        with h5py.File(filepath, 'r') as f:
            if 'channel_format' in f.attrs:
                result['channel_format'] = f.attrs['channel_format']
                result['fixed_at'] = f.attrs.get('channel_fixed_at', 'unknown')
                result['verified'] = f.attrs['channel_format'] == 'RGB'
            else:
                result['channel_format'] = 'unknown'
                result['verified'] = False
    except Exception as e:
        result['error'] = str(e)
    
    return result


def fix_dataset(data_dir: str, backup: bool = True, dry_run: bool = False, 
                verify: bool = True) -> dict:
    """
    修复整个数据集
    
    Args:
        data_dir: 数据集目录
        backup: 是否备份原文件
        dry_run: 只检查不修改
        verify: 修复后验证
        
    Returns:
        修复统计结果
    """
    data_dir = os.path.expanduser(data_dir)
    
    # 查找所有 HDF5 文件
    pattern = os.path.join(data_dir, "episode_*.hdf5")
    files = sorted(glob.glob(pattern))
    
    if len(files) == 0:
        print(f"No episode files found in {data_dir}")
        return {'error': 'No files found'}
    
    print(f"Found {len(files)} episodes in {data_dir}")
    
    if dry_run:
        print("=" * 50)
        print("DRY RUN MODE - No files will be modified")
        print("=" * 50)
    
    if backup and not dry_run:
        print(f"Backup enabled: original files will be saved as *.hdf5.backup")
    
    # 修复每个文件
    results = []
    stats = {
        'total': len(files),
        'fixed': 0,
        'skipped': 0,
        'errors': 0,
        'dry_run': 0,
    }
    
    for filepath in tqdm(files, desc="Fixing episodes"):
        result = fix_episode(filepath, backup=backup, dry_run=dry_run)
        results.append(result)
        
        if result['status'] == 'fixed':
            stats['fixed'] += 1
        elif result['status'] == 'skipped':
            stats['skipped'] += 1
        elif result['status'] == 'error':
            stats['errors'] += 1
            print(f"\nError in {filepath}: {result['error']}")
        elif result['status'] == 'dry_run':
            stats['dry_run'] += 1
    
    print("\n" + "=" * 50)
    print("Fix Summary:")
    print(f"  Total:   {stats['total']}")
    print(f"  Fixed:   {stats['fixed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Errors:  {stats['errors']}")
    if dry_run:
        print(f"  Dry Run: {stats['dry_run']}")
    print("=" * 50)
    
    # 验证
    if verify and not dry_run and stats['fixed'] > 0:
        print("\nVerifying fixes...")
        verified_count = 0
        for filepath in files:
            v_result = verify_fix(filepath)
            if v_result['verified']:
                verified_count += 1
        print(f"Verified: {verified_count}/{stats['fixed']} files")
    
    return {
        'stats': stats,
        'results': results,
    }


def restore_backup(data_dir: str) -> int:
    """
    从备份恢复原始文件
    
    Args:
        data_dir: 数据集目录
        
    Returns:
        恢复的文件数量
    """
    data_dir = os.path.expanduser(data_dir)
    
    pattern = os.path.join(data_dir, "episode_*.hdf5.backup")
    backup_files = sorted(glob.glob(pattern))
    
    if len(backup_files) == 0:
        print("No backup files found")
        return 0
    
    print(f"Found {len(backup_files)} backup files")
    
    restored = 0
    for backup_path in tqdm(backup_files, desc="Restoring"):
        original_path = backup_path.replace('.backup', '')
        try:
            shutil.copy2(backup_path, original_path)
            restored += 1
        except Exception as e:
            print(f"Error restoring {backup_path}: {e}")
    
    print(f"Restored {restored} files")
    return restored


def main():
    parser = argparse.ArgumentParser(
        description='Fix CARM dataset image channel order (BGR -> RGB)'
    )
    parser.add_argument('--data_dir', type=str, default='~/rl-vla/recorded_data',
                       help='Dataset directory')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='Backup original files before fixing')
    parser.add_argument('--no_backup', action='store_true',
                       help='Do not backup (use with caution)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Check files without modifying')
    parser.add_argument('--verify', action='store_true', default=True,
                       help='Verify fixes after completion')
    parser.add_argument('--restore', action='store_true',
                       help='Restore from backup files')
    
    args = parser.parse_args()
    
    # 处理 backup 参数
    backup = args.backup and not args.no_backup
    
    if args.restore:
        restore_backup(args.data_dir)
    else:
        fix_dataset(
            data_dir=args.data_dir,
            backup=backup,
            dry_run=args.dry_run,
            verify=args.verify,
        )


if __name__ == '__main__':
    main()
