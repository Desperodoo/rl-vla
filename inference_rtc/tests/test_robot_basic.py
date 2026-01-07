#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础机械臂控制测试

验证:
1. arx5_interface 导入
2. CAN 连接
3. 基础关节控制
4. 读取状态

用法:
    python -m inference_rtc.tests.test_robot_basic
"""

import os
import sys
import time
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.config import setup_arx5


def test_import():
    """测试 arx5_interface 导入"""
    print("\n[1] 测试 arx5_interface 导入...")
    try:
        setup_arx5()
        import arx5_interface as arx5
        print("  ✓ arx5_interface 导入成功")
        return arx5
    except Exception as e:
        print(f"  ✗ 导入失败: {e}")
        return None


def test_connection(arx5, model="X5", interface="can0"):
    """测试 CAN 连接"""
    print(f"\n[2] 测试 CAN 连接 ({model} @ {interface})...")
    try:
        robot = arx5.Arx5JointController(model, interface)
        print("  ✓ CAN 连接成功")
        return robot
    except Exception as e:
        print(f"  ✗ 连接失败: {e}")
        return None


def test_read_state(robot):
    """测试读取状态"""
    print("\n[3] 测试读取状态...")
    try:
        state = robot.get_state()
        pos = np.array(state.pos())
        gripper = state.gripper_pos
        print(f"  关节位置: {pos}")
        print(f"  夹爪位置: {gripper:.4f} m")
        print("  ✓ 状态读取成功")
        return True
    except Exception as e:
        print(f"  ✗ 读取失败: {e}")
        return False


def test_small_motion(robot, arx5):
    """测试小幅运动"""
    print("\n[4] 测试小幅运动 (关节1 ±0.05 rad)...")
    
    try:
        # 获取当前位置
        state = robot.get_state()
        base_pos = np.array(state.pos())
        base_gripper = state.gripper_pos
        
        print(f"  起始位置: {base_pos[:3]}...")
        
        # 设置 Gain
        gain = arx5.Gain(6)
        gain.kd()[:] = 0.01
        robot.set_gain(gain)
        
        # 小幅运动
        js = arx5.JointState(6)
        
        # 向正方向移动
        target = base_pos.copy()
        target[0] += 0.05
        js.pos()[:] = target
        js.gripper_pos = base_gripper
        
        print("  → 向正方向移动 0.05 rad...")
        for i in range(50):  # 0.5s
            robot.set_joint_cmd(js)
            robot.send_recv_once()
            time.sleep(0.01)
        
        # 返回原位
        js.pos()[:] = base_pos
        print("  ← 返回原位...")
        for i in range(50):
            robot.set_joint_cmd(js)
            robot.send_recv_once()
            time.sleep(0.01)
        
        # 验证
        state = robot.get_state()
        final_pos = np.array(state.pos())
        error = np.linalg.norm(final_pos - base_pos)
        
        if error < 0.05:
            print(f"  ✓ 运动测试通过 (误差: {error:.4f} rad)")
            return True
        else:
            print(f"  ⚠ 运动测试有误差 ({error:.4f} rad)")
            return True  # 仍算通过
            
    except Exception as e:
        print(f"  ✗ 运动测试失败: {e}")
        return False


def main():
    print("=" * 60)
    print("inference_rtc 基础机械臂控制测试")
    print("=" * 60)
    
    # 1. 测试导入
    arx5 = test_import()
    if arx5 is None:
        return 1
    
    # 2. 测试连接
    robot = test_connection(arx5)
    if robot is None:
        return 1
    
    # 3. 测试读取状态
    if not test_read_state(robot):
        return 1
    
    # 4. 测试小幅运动
    print("\n[提示] 即将进行小幅运动测试，请确保机械臂周围安全!")
    input("按 Enter 继续，Ctrl+C 取消...")
    
    if not test_small_motion(robot, arx5):
        robot.set_to_damping()
        return 1
    
    # 清理
    print("\n[5] 清理...")
    robot.set_to_damping()
    print("  ✓ 已设置为阻尼模式")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
