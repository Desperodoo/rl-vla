#!/usr/bin/env python3
"""
LeRobot-Anything-UArm 快速诊断脚本
用法: python quick_check.py
"""

import os
import sys
import subprocess
import socket
import struct
import time

def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check_ros():
    """检查 ROS 环境"""
    print_header("1. ROS 环境检查")
    
    ros_distro = os.environ.get('ROS_DISTRO', None)
    if ros_distro:
        print(f"  ✓ ROS_DISTRO: {ros_distro}")
    else:
        print("  ✗ ROS 未加载！请运行: source /opt/ros/noetic/setup.bash")
        return False
    
    # 检查 roscore
    try:
        result = subprocess.run(['rostopic', 'list'], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            print("  ✓ roscore 正在运行")
            return True
        else:
            print("  ✗ roscore 未运行！请在新终端运行: roscore")
            return False
    except FileNotFoundError:
        print("  ✗ ROS 命令不可用")
        return False
    except subprocess.TimeoutExpired:
        print("  ✗ roscore 未运行（超时）")
        return False

def check_serial_devices():
    """检查串口设备"""
    print_header("2. 串口设备检查 (UArm遥操作臂)")
    
    usb_devices = []
    for dev in ['/dev/uarm_servo', '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']:
        if os.path.exists(dev):
            usb_devices.append(dev)
            print(f"  ✓ 发现设备: {dev}")
    
    if not usb_devices:
        print("  ✗ 未发现串口设备！")
        print("    请检查 UArm 遥操作臂是否连接")
        return False
    
    return True

def check_can_devices():
    """检查 CAN 设备"""
    print_header("3. CAN 设备检查 (ARX5机械臂)")
    
    try:
        result = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        can_found = []
        for line in lines:
            if 'can' in line and ': <' in line:
                can_found.append(line.strip())
        
        if can_found:
            for can in can_found:
                if 'UP' in can:
                    print(f"  ✓ {can.split(':')[1].split(':')[0].strip()} - 已启动")
                else:
                    print(f"  ⚠ {can.split(':')[1].split(':')[0].strip()} - 未启动")
        else:
            print("  ⚠ 未发现 CAN 设备（如果不使用 ARX5，可忽略）")
            return True
            
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        return False
    
    return True

def check_can_traffic():
    """检查 CAN 通信"""
    print_header("4. CAN 通信测试")
    
    for can_if in ['can0', 'can1']:
        try:
            sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            sock.bind((can_if,))
            sock.settimeout(1.0)
            
            msg_count = 0
            start = time.time()
            while time.time() - start < 1:
                try:
                    msg, _ = sock.recvfrom(16)
                    msg_count += 1
                except socket.timeout:
                    break
            sock.close()
            
            if msg_count > 0:
                print(f"  ✓ {can_if}: 收到 {msg_count} 条消息 - 正常")
            else:
                print(f"  ⚠ {can_if}: 无消息（设备可能空闲或未连接）")
        except OSError:
            print(f"  - {can_if}: 接口不存在")

def check_python_deps():
    """检查 Python 依赖"""
    print_header("5. Python 依赖检查")
    
    deps = ['numpy', 'serial', 'rospy']
    
    for dep in deps:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} 未安装")

def check_arx5_sdk():
    """检查 ARX5 SDK"""
    print_header("6. ARX5 SDK 检查")
    
    arx5_path = "/home/lizh/arx5-sdk/python"
    if os.path.exists(arx5_path):
        print(f"  ✓ ARX5 SDK 路径存在: {arx5_path}")
        sys.path.insert(0, arx5_path)
        try:
            import arx5_interface
            print("  ✓ arx5_interface 可导入")
        except ImportError as e:
            print(f"  ✗ arx5_interface 导入失败: {e}")
    else:
        print(f"  ⚠ ARX5 SDK 未找到（如果不使用 ARX5，可忽略）")

def show_quick_start():
    """显示快速启动指南"""
    print_header("快速启动指南")
    
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  Step 1: 环境配置                                        │
  │    source setup_env.sh                                  │
  │                                                         │
  │  Step 2: 启动 ROS (新终端)                               │
  │    roscore                                              │
  │                                                         │
  │  Step 3: 测试 UArm 舵机读取                              │
  │    python src/uarm/scripts/Uarm_teleop/servo_zero.py    │
  │                                                         │
  │  Step 4: 启动 UArm 数据发布                              │
  │    rosrun uarm servo_reader.py                          │
  │                                                         │
  │  Step 5: 启动机械臂控制 (根据你的机械臂选择)              │
  │    # ARX5:                                              │
  │    python src/uarm/scripts/Follower_Arm/ARX/arx_teleop.py│
  │    # xArm:                                              │
  │    rosrun uarm servo2xarm.py                            │
  │    # Dobot:                                             │
  │    rosrun uarm servo2Dobot.py                           │
  └─────────────────────────────────────────────────────────┘
""")

def main():
    print("\n" + "=" * 60)
    print("  LeRobot-Anything-UArm 系统诊断")
    print("=" * 60)
    
    check_ros()
    check_serial_devices()
    check_can_devices()
    check_can_traffic()
    check_python_deps()
    check_arx5_sdk()
    show_quick_start()

if __name__ == "__main__":
    main()
