#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集图像回放脚本

回放 HDF5 数据集中的图像序列，支持:
- 同时显示腕部和外部相机
- 调整播放速度
- 暂停/继续
- 显示关节位置信息
"""

import os
import sys
import argparse
import time
import numpy as np
import h5py
import cv2
from pathlib import Path


def list_trajectories(h5_path: str) -> list:
    """列出 HDF5 文件中的所有轨迹"""
    trajectories = []
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            if key.startswith('traj_'):
                trajectories.append(key)
    # 按数字排序
    trajectories.sort(key=lambda x: int(x.split('_')[1]))
    return trajectories


def replay_trajectory(
    h5_path: str,
    traj_name: str,
    fps: float = 30.0,
    show_info: bool = True,
    window_scale: float = 1.0,
):
    """
    回放单条轨迹
    
    Args:
        h5_path: HDF5 文件路径
        traj_name: 轨迹名称 (如 'traj_0')
        fps: 播放帧率
        show_info: 是否显示信息叠加
        window_scale: 窗口缩放比例
    """
    print(f"\n{'='*60}")
    print(f"回放轨迹: {traj_name}")
    print(f"{'='*60}")
    
    with h5py.File(h5_path, 'r') as f:
        traj = f[traj_name]
        
        # 读取图像数据
        wrist_rgb = traj['obs/images/wrist/rgb'][:]  # (N, H, W, 3)
        external_rgb = traj['obs/images/external/rgb'][:]  # (N, H, W, 3)
        
        # 读取状态数据
        joint_pos = traj['obs/joint_pos'][:]  # (N, 6)
        gripper_pos = traj['obs/gripper_pos'][:]  # (N, 1)
        actions = traj['actions'][:]  # (N, 7)
        timestamps = traj['obs/timestamps'][:]  # (N,)
        
        n_frames = len(wrist_rgb)
        duration = timestamps[-1] - timestamps[0]
        actual_fps = n_frames / duration
        
        print(f"帧数: {n_frames}")
        print(f"时长: {duration:.2f}s")
        print(f"原始帧率: {actual_fps:.1f} Hz")
        print(f"播放帧率: {fps:.1f} Hz")
        print(f"图像尺寸: {wrist_rgb.shape[1]}x{wrist_rgb.shape[2]}")
        
        print(f"\n控制:")
        print(f"  空格 - 暂停/继续")
        print(f"  'q'  - 退出")
        print(f"  '+'  - 加速")
        print(f"  '-'  - 减速")
        print(f"  左箭头 - 后退 10 帧")
        print(f"  右箭头 - 前进 10 帧")
        
        # 创建窗口
        cv2.namedWindow('Dataset Replay', cv2.WINDOW_NORMAL)
        
        # 播放状态
        frame_idx = 0
        paused = False
        dt = 1.0 / fps
        speed_multiplier = 1.0
        
        while frame_idx < n_frames:
            t_start = time.time()
            
            # 获取当前帧
            wrist_frame = wrist_rgb[frame_idx]
            external_frame = external_rgb[frame_idx]
            curr_joint = joint_pos[frame_idx]
            curr_gripper = gripper_pos[frame_idx, 0]
            curr_action = actions[frame_idx]
            
            # RGB -> BGR for OpenCV
            wrist_bgr = cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR)
            external_bgr = cv2.cvtColor(external_frame, cv2.COLOR_RGB2BGR)
            
            # 缩放
            if window_scale != 1.0:
                h, w = wrist_bgr.shape[:2]
                new_size = (int(w * window_scale), int(h * window_scale))
                wrist_bgr = cv2.resize(wrist_bgr, new_size)
                external_bgr = cv2.resize(external_bgr, new_size)
            
            # 添加信息叠加
            if show_info:
                # 在腕部图像上显示信息
                info_lines = [
                    f"Frame: {frame_idx}/{n_frames-1}",
                    f"Time: {timestamps[frame_idx] - timestamps[0]:.2f}s / {duration:.2f}s",
                    f"Speed: {speed_multiplier:.1f}x",
                    f"Gripper: {curr_gripper:.3f}m",
                ]
                
                y_offset = 20
                for line in info_lines:
                    cv2.putText(wrist_bgr, line, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 18
                
                # 在外部图像上显示关节信息
                joint_lines = [
                    f"Joint 0: {curr_joint[0]:.3f}",
                    f"Joint 1: {curr_joint[1]:.3f}",
                    f"Joint 2: {curr_joint[2]:.3f}",
                    f"Joint 3: {curr_joint[3]:.3f}",
                    f"Joint 4: {curr_joint[4]:.3f}",
                    f"Joint 5: {curr_joint[5]:.3f}",
                ]
                
                y_offset = 20
                for line in joint_lines:
                    cv2.putText(external_bgr, line, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 18
                
                # 标题
                cv2.putText(wrist_bgr, "Wrist Camera", (10, wrist_bgr.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(external_bgr, "External Camera", (10, external_bgr.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 拼接图像
            combined = np.hstack([wrist_bgr, external_bgr])
            
            # 显示暂停状态
            if paused:
                cv2.putText(combined, "PAUSED", 
                           (combined.shape[1]//2 - 50, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            cv2.imshow('Dataset Replay', combined)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n退出回放")
                break
            elif key == ord(' '):
                paused = not paused
                print(f"{'暂停' if paused else '继续'}")
            elif key == ord('+') or key == ord('='):
                speed_multiplier = min(4.0, speed_multiplier + 0.25)
                print(f"速度: {speed_multiplier:.2f}x")
            elif key == ord('-') or key == ord('_'):
                speed_multiplier = max(0.25, speed_multiplier - 0.25)
                print(f"速度: {speed_multiplier:.2f}x")
            elif key == 81 or key == 2:  # 左箭头
                frame_idx = max(0, frame_idx - 10)
                print(f"跳转到帧 {frame_idx}")
            elif key == 83 or key == 3:  # 右箭头
                frame_idx = min(n_frames - 1, frame_idx + 10)
                print(f"跳转到帧 {frame_idx}")
            
            # 更新帧索引
            if not paused:
                frame_idx += 1
            
            # 控制播放速度
            elapsed = time.time() - t_start
            wait_time = (dt / speed_multiplier) - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
        
        cv2.destroyAllWindows()
        print(f"\n轨迹 {traj_name} 回放完成")


def main():
    parser = argparse.ArgumentParser(description="数据集图像回放")
    parser.add_argument("h5_path", nargs='?', 
                       default="/home/lizh/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5",
                       help="HDF5 数据集路径")
    parser.add_argument("-t", "--traj", type=str, default=None,
                       help="指定轨迹名称 (如 traj_0)，不指定则依次播放所有")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="播放帧率 (默认 30)")
    parser.add_argument("--scale", type=float, default=1.5,
                       help="窗口缩放比例 (默认 1.5)")
    parser.add_argument("--no-info", action="store_true",
                       help="不显示信息叠加")
    parser.add_argument("-l", "--list", action="store_true",
                       help="只列出轨迹，不播放")
    
    args = parser.parse_args()
    
    # 检查文件
    if not os.path.exists(args.h5_path):
        print(f"错误: 文件不存在 {args.h5_path}")
        sys.exit(1)
    
    print(f"数据集: {args.h5_path}")
    
    # 列出轨迹
    trajectories = list_trajectories(args.h5_path)
    print(f"找到 {len(trajectories)} 条轨迹")
    
    if args.list:
        for traj in trajectories:
            with h5py.File(args.h5_path, 'r') as f:
                n_frames = len(f[traj]['obs/joint_pos'])
                print(f"  {traj}: {n_frames} 帧")
        return
    
    # 确定要播放的轨迹
    if args.traj:
        if args.traj not in trajectories:
            print(f"错误: 轨迹 {args.traj} 不存在")
            print(f"可用轨迹: {trajectories}")
            sys.exit(1)
        trajs_to_play = [args.traj]
    else:
        trajs_to_play = trajectories
    
    # 播放
    try:
        for traj in trajs_to_play:
            replay_trajectory(
                h5_path=args.h5_path,
                traj_name=traj,
                fps=args.fps,
                show_info=not args.no_info,
                window_scale=args.scale,
            )
            
            # 如果有多条轨迹，询问是否继续
            if len(trajs_to_play) > 1 and traj != trajs_to_play[-1]:
                print("\n按任意键继续下一条轨迹，按 'q' 退出...")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
