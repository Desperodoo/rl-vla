#!/usr/bin/env python3
"""
CARM 机械臂夹爪测试脚本
测试内容：夹爪开合控制、获取夹爪状态
使用前请先运行: source carm_scripts/setup_carm_env.sh

⚠️ 警告：此脚本会控制夹爪运动
"""

import sys
import time
from carm import carm_py

# 机械臂连接配置
ARM_IP = "10.42.0.101"
ARM_PORT = 8090
TIMEOUT = 1

# 夹爪参数
# 夹爪间隔范围：0 - 0.08m (0mm - 80mm)
# 夹爪力矩范围：0 - 20N
GRIPPER_OPEN = 0.08    # 完全张开 (80mm)
GRIPPER_CLOSE = 0.0    # 完全闭合 (0mm)
GRIPPER_HALF = 0.04    # 半开 (40mm)
GRIPPER_FORCE = 10.0   # 夹持力 (10N)

def test_gripper():
    """测试夹爪控制"""
    print("=" * 50)
    print("  CARM 机械臂夹爪测试")
    print("=" * 50)
    print()
    print("夹爪参数说明:")
    print("  - 间隔范围: 0 - 0.08m (0mm - 80mm)")
    print("  - 力矩范围: 0 - 20N")
    print()
    
    # 1. 创建并连接机械臂
    print(f"[1/7] 连接机械臂...")
    try:
        # 构造函数只需要传 IP，会自动连接
        carm = carm_py.CArmSingleCol(ARM_IP)
        time.sleep(1)  # 等待连接稳定
        if carm.is_connected():
            print("      ✓ 连接成功")
        else:
            print("      ! 连接状态返回 False，尝试继续...")
    except Exception as e:
        print(f"      ✗ 连接异常: {e}")
        return False
    
    # 2. 复位机械臂
    print(f"[2/7] 复位机械臂 (set_ready)...")
    try:
        ret = carm.set_ready()
        if ret == 0:
            print("      ✓ 复位成功")
        else:
            print(f"      ! 复位返回码: {ret}")
    except Exception as e:
        print(f"      ✗ 复位异常: {e}")
    
    # 3. 获取夹爪状态
    print(f"[3/7] 获取夹爪当前状态...")
    try:
        gripper_state = carm.get_gripper_state()
        print(f"      夹爪状态: {gripper_state}")
        
        gripper_pos = carm.get_gripper_pos()
        print(f"      夹爪位置: {gripper_pos:.4f}m ({gripper_pos*1000:.1f}mm)")
        
        gripper_vel = carm.get_gripper_vel()
        print(f"      夹爪速度: {gripper_vel:.4f}")
        
        gripper_tau = carm.get_gripper_tau()
        print(f"      夹爪力矩: {gripper_tau:.4f}N")
    except Exception as e:
        print(f"      ✗ 获取夹爪状态失败: {e}")
    
    # 4. 张开夹爪
    print(f"[4/7] 张开夹爪...")
    print(f"      目标位置: {GRIPPER_OPEN}m ({GRIPPER_OPEN*1000}mm)")
    print(f"      力矩: {GRIPPER_FORCE}N")
    try:
        input("      按 Enter 键开始（或 Ctrl+C 取消）...")
        ret = carm.set_gripper(GRIPPER_OPEN, GRIPPER_FORCE)
        if ret == 0:
            print("      ✓ 夹爪张开命令发送成功")
        else:
            print(f"      ! 返回码: {ret}")
        
        time.sleep(1.0)  # 等待夹爪动作完成
        
        gripper_pos = carm.get_gripper_pos()
        print(f"      当前位置: {gripper_pos:.4f}m ({gripper_pos*1000:.1f}mm)")
    except KeyboardInterrupt:
        print("\n      ! 用户取消")
        carm.disconnect()
        return False
    except Exception as e:
        print(f"      ✗ 张开夹爪失败: {e}")
    
    # 5. 半闭合夹爪
    print(f"[5/7] 半闭合夹爪...")
    print(f"      目标位置: {GRIPPER_HALF}m ({GRIPPER_HALF*1000}mm)")
    try:
        input("      按 Enter 键开始（或 Ctrl+C 取消）...")
        ret = carm.set_gripper(GRIPPER_HALF, GRIPPER_FORCE)
        if ret == 0:
            print("      ✓ 夹爪半闭合命令发送成功")
        else:
            print(f"      ! 返回码: {ret}")
        
        time.sleep(1.0)
        
        gripper_pos = carm.get_gripper_pos()
        print(f"      当前位置: {gripper_pos:.4f}m ({gripper_pos*1000:.1f}mm)")
    except KeyboardInterrupt:
        print("\n      ! 用户取消")
        carm.disconnect()
        return False
    except Exception as e:
        print(f"      ✗ 半闭合夹爪失败: {e}")
    
    # 6. 完全闭合夹爪
    print(f"[6/7] 完全闭合夹爪...")
    print(f"      目标位置: {GRIPPER_CLOSE}m ({GRIPPER_CLOSE*1000}mm)")
    try:
        input("      按 Enter 键开始（或 Ctrl+C 取消）...")
        ret = carm.set_gripper(GRIPPER_CLOSE, GRIPPER_FORCE)
        if ret == 0:
            print("      ✓ 夹爪闭合命令发送成功")
        else:
            print(f"      ! 返回码: {ret}")
        
        time.sleep(1.0)
        
        gripper_pos = carm.get_gripper_pos()
        gripper_tau = carm.get_gripper_tau()
        print(f"      当前位置: {gripper_pos:.4f}m ({gripper_pos*1000:.1f}mm)")
        print(f"      当前力矩: {gripper_tau:.4f}N")
    except KeyboardInterrupt:
        print("\n      ! 用户取消")
        carm.disconnect()
        return False
    except Exception as e:
        print(f"      ✗ 闭合夹爪失败: {e}")
    
    # 7. 重新张开夹爪
    print(f"[7/7] 重新张开夹爪...")
    try:
        input("      按 Enter 键开始（或 Ctrl+C 取消）...")
        ret = carm.set_gripper(GRIPPER_OPEN, GRIPPER_FORCE)
        if ret == 0:
            print("      ✓ 夹爪张开命令发送成功")
        else:
            print(f"      ! 返回码: {ret}")
        
        time.sleep(1.0)
        
        gripper_pos = carm.get_gripper_pos()
        print(f"      最终位置: {gripper_pos:.4f}m ({gripper_pos*1000:.1f}mm)")
    except KeyboardInterrupt:
        print("\n      ! 用户取消")
    except Exception as e:
        print(f"      ✗ 张开夹爪失败: {e}")
    
    # 清理
    print()
    print("[清理] 断开连接...")
    try:
        carm.disconnect()
        print("      ✓ 已断开连接")
    except Exception as e:
        print(f"      ! 断开连接异常: {e}")
    
    print()
    print("=" * 50)
    print("  夹爪测试完成！")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_gripper()
    sys.exit(0 if success else 1)
