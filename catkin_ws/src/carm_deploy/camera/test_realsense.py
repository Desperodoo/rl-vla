#!/usr/bin/env python3
"""
Intel RealSense D405 相机测试脚本
用于验证相机连接和基本功能
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import sys

def test_camera():
    """测试相机连接和图像获取"""
    
    print("="*60)
    print("Intel RealSense D405 相机测试")
    print("="*60)
    
    # 创建pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 获取设备列表
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("错误: 未检测到RealSense设备!")
        return False
    
    # 打印设备信息
    for dev in devices:
        print(f"\n设备信息:")
        print(f"  名称: {dev.get_info(rs.camera_info.name)}")
        print(f"  序列号: {dev.get_info(rs.camera_info.serial_number)}")
        print(f"  固件版本: {dev.get_info(rs.camera_info.firmware_version)}")
        print(f"  USB类型: {dev.get_info(rs.camera_info.usb_type_descriptor)}")
    
    # 配置流
    serial_number = devices[0].get_info(rs.camera_info.serial_number)
    config.enable_device(serial_number)
    
    # D405支持的流配置
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print("\n启动相机流...")
    
    try:
        # 启动pipeline
        profile = pipeline.start(config)
        
        # 获取深度传感器
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"深度比例: {depth_scale} 米/单位")
        
        # 获取内参
        depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        print(f"\n深度相机内参:")
        print(f"  分辨率: {depth_intrinsics.width}x{depth_intrinsics.height}")
        print(f"  焦距: fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}")
        print(f"  主点: cx={depth_intrinsics.ppx:.2f}, cy={depth_intrinsics.ppy:.2f}")
        
        print(f"\n彩色相机内参:")
        print(f"  分辨率: {color_intrinsics.width}x{color_intrinsics.height}")
        print(f"  焦距: fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}")
        print(f"  主点: cx={color_intrinsics.ppx:.2f}, cy={color_intrinsics.ppy:.2f}")
        
        # 创建对齐对象
        align = rs.align(rs.stream.color)
        
        # 跳过前几帧以稳定自动曝光
        for _ in range(30):
            pipeline.wait_for_frames()
        
        print("\n测试图像获取...")
        
        # 获取对齐后的帧
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            print("错误: 无法获取帧!")
            pipeline.stop()
            return False
        
        # 转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        print(f"  深度图尺寸: {depth_image.shape}")
        print(f"  彩色图尺寸: {color_image.shape}")
        print(f"  深度范围: {depth_image.min()} - {depth_image.max()} (单位)")
        print(f"  深度范围: {depth_image.min() * depth_scale * 1000:.1f} - {depth_image.max() * depth_scale * 1000:.1f} mm")
        
        # 保存测试图像
        cv2.imwrite('/tmp/realsense_color_test.png', color_image)
        
        # 深度图着色
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        cv2.imwrite('/tmp/realsense_depth_test.png', depth_colormap)
        
        print("\n测试图像已保存:")
        print("  彩色图: /tmp/realsense_color_test.png")
        print("  深度图: /tmp/realsense_depth_test.png")
        
        pipeline.stop()
        
        print("\n" + "="*60)
        print("相机测试通过!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"错误: {e}")
        try:
            pipeline.stop()
        except:
            pass
        return False


def test_ros_topics():
    """测试ROS话题 (需要先启动roscore和realsense节点)"""
    
    print("\n" + "="*60)
    print("ROS话题测试")
    print("="*60)
    
    try:
        import rospy
        from sensor_msgs.msg import Image, CameraInfo, PointCloud2
        
        print("\n可用的ROS话题 (启动realsense节点后):")
        print("  /camera/color/image_raw          - 彩色图像")
        print("  /camera/color/camera_info        - 彩色相机内参")
        print("  /camera/depth/image_rect_raw     - 深度图像")
        print("  /camera/depth/camera_info        - 深度相机内参")
        print("  /camera/aligned_depth_to_color/image_raw - 对齐到彩色的深度图")
        print("  /camera/depth/color/points       - 彩色点云")
        print("  /camera/infra1/image_rect_raw    - 红外图像1")
        print("  /camera/infra2/image_rect_raw    - 红外图像2")
        
        return True
    except ImportError:
        print("注意: rospy未导入，请确保ROS环境已source")
        return False


if __name__ == "__main__":
    success = test_camera()
    if success:
        test_ros_topics()
    
    print("\n使用说明:")
    print("-" * 60)
    print("1. 启动ROS相机节点:")
    print("   roslaunch /home/lizh/rl-vla/carm_real/launch/realsense_d405.launch")
    print("")
    print("2. 查看可用话题:")
    print("   rostopic list | grep camera")
    print("")
    print("3. 查看彩色图像:")
    print("   rosrun image_view image_view image:=/camera/color/image_raw")
    print("")
    print("4. 查看深度图像:")
    print("   rosrun image_view image_view image:=/camera/depth/image_rect_raw")
    print("-" * 60)
    
    sys.exit(0 if success else 1)
