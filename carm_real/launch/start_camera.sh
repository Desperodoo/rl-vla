#!/bin/bash
# Intel RealSense D405 相机启动脚本
# 用于在ROS Noetic中启动D405相机

set -e

echo "==========================================="
echo "Intel RealSense D405 相机启动脚本"
echo "==========================================="

# 设置ROS环境
source /opt/ros/noetic/setup.bash
source /home/lizh/rl-vla/catkin_ws/devel/setup.bash

echo ""
echo "相机信息:"
echo "  型号: Intel RealSense D405"
echo "  序列号: 218622279840"
echo "  固件版本: 5.17.0.10"
echo "  USB类型: 3.2"
echo ""

# 检查roscore是否运行
if ! rostopic list &> /dev/null; then
    echo "启动roscore..."
    roscore &
    sleep 3
fi

echo "启动RealSense相机节点..."
echo ""
echo "可用话题:"
echo "  /camera/color/image_raw      - 彩色图像 (RGB)"
echo "  /camera/depth/image_rect_raw - 深度图像"
echo "  /camera/color/camera_info    - 彩色相机内参"
echo "  /camera/depth/camera_info    - 深度相机内参"
echo ""
echo "启动中..."

# 启动相机节点
roslaunch realsense2_camera rs_camera.launch \
    serial_no:=218622279840 \
    enable_color:=true \
    enable_depth:=true \
    align_depth:=true \
    color_width:=640 \
    color_height:=480 \
    color_fps:=30 \
    depth_width:=640 \
    depth_height:=480 \
    depth_fps:=30 \
    enable_pointcloud:=false \
    enable_infra1:=false \
    enable_infra2:=false
