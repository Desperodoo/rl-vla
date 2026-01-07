#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense 相机测试

验证 RealSense 相机模块是否正常工作:
1. 检测连接的相机
2. 测试单相机采集 (腕部 D435i)
3. 测试外部相机录制 (D455)
4. 测试多相机同步采集

用法:
    # 检测相机
    python -m consistency_policy.tests.test_camera --detect
    
    # 测试腕部相机
    python -m consistency_policy.tests.test_camera --wrist
    
    # 测试外部相机 (带录制)
    python -m consistency_policy.tests.test_camera --external --record
    
    # 测试双相机
    python -m consistency_policy.tests.test_camera --both

依赖:
    - pyrealsense2
    - opencv-python
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Optional
from multiprocessing.managers import SharedMemoryManager

import cv2

# 添加项目路径
CONSISTENCY_POLICY_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_VLA_PATH = os.path.dirname(CONSISTENCY_POLICY_PATH)
if RL_VLA_PATH not in sys.path:
    sys.path.insert(0, RL_VLA_PATH)

from consistency_policy.realsense_camera import (
    RealSenseCameraProcess,
    RealSenseCameraManager,
    RealSenseCameraConfig,
    CAMERA_CONFIGS,
    REALSENSE_AVAILABLE,
)


def test_detect_cameras():
    """检测连接的 RealSense 相机"""
    print("\n" + "=" * 60)
    print("RealSense 相机检测")
    print("=" * 60)
    
    if not REALSENSE_AVAILABLE:
        print("\n✗ pyrealsense2 未安装")
        print("  安装命令: pip install pyrealsense2")
        return False
    
    cameras = RealSenseCameraManager.detect_cameras()
    
    if not cameras:
        print("\n✗ 未检测到任何 RealSense 相机")
        print("  请检查相机连接")
        return False
    
    print(f"\n✓ 检测到 {len(cameras)} 个相机:")
    for serial, name in cameras:
        print(f"  - {name}")
        print(f"    序列号: {serial}")
        
        # 检查是否为已知配置
        if serial == CAMERA_CONFIGS['wrist'].serial_number:
            print(f"    角色: 腕部相机 (用于推理)")
        elif serial == CAMERA_CONFIGS['external'].serial_number:
            print(f"    角色: 外部相机 (用于录制)")
        else:
            print(f"    角色: 未知")
    
    return True


def test_single_camera(
    camera_name: str = 'wrist',
    duration: float = 10.0,
    show_preview: bool = True,
    record_path: Optional[str] = None,
):
    """
    测试单个相机
    
    Args:
        camera_name: 相机名称 ('wrist' 或 'external')
        duration: 测试时长 (秒)
        show_preview: 是否显示预览窗口
        record_path: 录制路径 (仅 external 相机)
    """
    print("\n" + "=" * 60)
    print(f"单相机测试: {camera_name}")
    print("=" * 60)
    
    if not REALSENSE_AVAILABLE:
        print("\n✗ pyrealsense2 未安装")
        return False
    
    # 获取相机配置
    if camera_name not in CAMERA_CONFIGS:
        print(f"\n✗ 未知相机配置: {camera_name}")
        return False
    
    config = CAMERA_CONFIGS[camera_name]
    print(f"\n相机配置:")
    print(f"  名称: {config.name}")
    print(f"  序列号: {config.serial_number}")
    print(f"  分辨率: {config.resolution}")
    print(f"  帧率: {config.fps} Hz")
    print(f"  深度: {'启用' if config.enable_depth else '禁用'}")
    print(f"  录制: {'启用' if config.enable_recording else '禁用'}")
    
    # 创建共享内存管理器
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    
    try:
        # 创建相机进程
        camera = RealSenseCameraProcess(
            shm_manager=shm_manager,
            config=config,
            verbose=True,
        )
        
        print(f"\n启动相机...")
        camera.start(wait=True)
        print(f"✓ 相机启动成功 (PID: {camera.pid})")
        
        # 开始录制
        if record_path and config.enable_recording:
            camera.start_recording(record_path)
            print(f"✓ 开始录制: {record_path}")
        
        # 采集循环
        print(f"\n开始采集 ({duration}s)...")
        print("  按 'q' 退出, 按 's' 保存当前帧")
        
        start_time = time.time()
        frame_count = 0
        fps_history = []
        last_time = start_time
        
        while time.time() - start_time < duration:
            # 获取帧
            frame_data = camera.get_frame()
            rgb = frame_data['rgb']  # (H, W, 3) RGB
            timestamp = frame_data['timestamp']
            
            frame_count += 1
            
            # 计算 FPS
            current_time = time.time()
            instant_fps = 1.0 / max(current_time - last_time, 0.001)
            fps_history.append(instant_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            last_time = current_time
            
            # 显示预览
            if show_preview:
                # RGB -> BGR for OpenCV
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                # 添加信息
                info_text = [
                    f"Camera: {camera_name}",
                    f"Frame: {frame_count}",
                    f"FPS: {avg_fps:.1f}",
                    f"Time: {current_time - start_time:.1f}s / {duration}s",
                ]
                
                y_offset = 30
                for text in info_text:
                    cv2.putText(bgr, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
                
                cv2.imshow(f'Camera: {camera_name}', bgr)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n  用户退出")
                    break
                elif key == ord('s'):
                    save_path = f"camera_{camera_name}_{frame_count}.png"
                    cv2.imwrite(save_path, bgr)
                    print(f"  保存帧: {save_path}")
            
            # 短暂休眠
            time.sleep(0.01)
        
        # 停止录制
        if record_path and config.enable_recording:
            camera.stop_recording()
            print(f"✓ 停止录制")
        
        # 统计
        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed
        
        print(f"\n测试统计:")
        print(f"  总帧数: {frame_count}")
        print(f"  总时长: {elapsed:.2f}s")
        print(f"  平均帧率: {actual_fps:.1f} FPS")
        print(f"  相机帧计数: {camera.frame_count}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理
        cv2.destroyAllWindows()
        if 'camera' in dir() and camera is not None:
            camera.stop()
        shm_manager.shutdown()
        print("\n✓ 清理完成")


def test_dual_cameras(duration: float = 10.0):
    """
    测试双相机同步采集
    
    Args:
        duration: 测试时长 (秒)
    """
    print("\n" + "=" * 60)
    print("双相机同步测试")
    print("=" * 60)
    
    if not REALSENSE_AVAILABLE:
        print("\n✗ pyrealsense2 未安装")
        return False
    
    # 创建共享内存管理器
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    
    wrist_camera = None
    external_camera = None
    
    try:
        # 创建腕部相机
        print("\n启动腕部相机...")
        wrist_camera = RealSenseCameraProcess(
            shm_manager=shm_manager,
            config=CAMERA_CONFIGS['wrist'],
            verbose=True,
        )
        wrist_camera.start(wait=True)
        print(f"✓ 腕部相机启动成功")
        
        # 创建外部相机
        print("\n启动外部相机...")
        external_camera = RealSenseCameraProcess(
            shm_manager=shm_manager,
            config=CAMERA_CONFIGS['external'],
            verbose=True,
        )
        external_camera.start(wait=True)
        print(f"✓ 外部相机启动成功")
        
        # 采集循环
        print(f"\n开始同步采集 ({duration}s)...")
        print("  按 'q' 退出")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            # 获取两个相机的帧
            wrist_data = wrist_camera.get_frame()
            external_data = external_camera.get_frame()
            
            wrist_rgb = wrist_data['rgb']
            external_rgb = external_data['rgb']
            
            frame_count += 1
            
            # 显示预览 (并排显示)
            wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
            external_bgr = cv2.cvtColor(external_rgb, cv2.COLOR_RGB2BGR)
            
            # 添加标签
            cv2.putText(wrist_bgr, "Wrist (D435i)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(external_bgr, "External (D455)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 拼接
            combined = np.hstack([wrist_bgr, external_bgr])
            
            # 添加同步信息
            time_diff = abs(wrist_data['timestamp'] - external_data['timestamp']) * 1000
            cv2.putText(combined, f"Frame: {frame_count} | Time diff: {time_diff:.1f}ms",
                       (10, combined.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Dual Camera Test', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n  用户退出")
                break
            
            time.sleep(0.01)
        
        # 统计
        elapsed = time.time() - start_time
        print(f"\n测试统计:")
        print(f"  总帧数: {frame_count}")
        print(f"  总时长: {elapsed:.2f}s")
        print(f"  平均帧率: {frame_count / elapsed:.1f} FPS")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理
        cv2.destroyAllWindows()
        if wrist_camera is not None:
            wrist_camera.stop()
        if external_camera is not None:
            external_camera.stop()
        shm_manager.shutdown()
        print("\n✓ 清理完成")


def main():
    parser = argparse.ArgumentParser(description="RealSense 相机测试")
    parser.add_argument("--detect", action="store_true", help="检测连接的相机")
    parser.add_argument("--wrist", action="store_true", help="测试腕部相机 (D435i)")
    parser.add_argument("--external", action="store_true", help="测试外部相机 (D455)")
    parser.add_argument("--both", action="store_true", help="测试双相机同步")
    parser.add_argument("--record", action="store_true", help="启用录制 (仅 external)")
    parser.add_argument("-d", "--duration", type=float, default=10.0, help="测试时长 (秒)")
    parser.add_argument("--no-preview", action="store_true", help="禁用预览窗口")
    
    args = parser.parse_args()
    
    # 默认行为：检测相机
    if not any([args.detect, args.wrist, args.external, args.both]):
        args.detect = True
    
    success = True
    
    if args.detect:
        success = test_detect_cameras() and success
    
    if args.wrist:
        success = test_single_camera(
            camera_name='wrist',
            duration=args.duration,
            show_preview=not args.no_preview,
        ) and success
    
    if args.external:
        record_path = None
        if args.record:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            record_path = f"test_recording_{timestamp}.mp4"
        
        success = test_single_camera(
            camera_name='external',
            duration=args.duration,
            show_preview=not args.no_preview,
            record_path=record_path,
        ) and success
    
    if args.both:
        success = test_dual_cameras(duration=args.duration) and success
    
    print("\n" + "=" * 60)
    if success:
        print("✓ 所有测试通过")
    else:
        print("✗ 部分测试失败")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
