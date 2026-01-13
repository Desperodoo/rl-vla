#!/usr/bin/env python3
"""
CARM 机械臂运动测试脚本
测试内容：set_ready 复位、关节运动、获取位置
使用前请先运行: source carm_scripts/setup_carm_env.sh

⚠️ 警告：此脚本会控制机械臂运动，请确保：
1. 机械臂周围无障碍物
2. 已做好急停准备
"""

import sys
import time
from carm import carm_py

# 机械臂连接配置
ARM_IP = "10.42.0.101"
ARM_PORT = 8090
TIMEOUT = 1

# 测试运动目标位置 (弧度)
# 全零位置
HOME_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# 测试位置1：各关节小幅度偏移（根据关节限位调整）
# 关节限位: upper=[2.79, 3.14, 0.0, 2.65, 1.57, 2.88], lower=[-2.79, 0.0, -3.14, -2.65, -1.57, -2.88]
# 关节2的限位是 [-3.14, 0.0]，所以只能取负值
TEST_POSITION_1 = [0.1, 0.1, -0.1, 0.1, 0.1, 0.1]
# 测试位置2：回到零位
TEST_POSITION_2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def test_motion():
    """测试机械臂运动控制"""
    print("=" * 50)
    print("  CARM 机械臂运动测试")
    print("=" * 50)
    print()
    print("⚠️  警告：机械臂即将运动，请确保周围安全！")
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
    
    # 2. 设置速度等级（安全起见使用较低速度）
    print(f"[2/7] 设置速度等级...")
    try:
        # 速度等级 0-10，这里设置为 3（较慢）
        ret = carm.set_speed_level(3, 20)
        if ret == 0:
            print("      ✓ 速度等级设置为 3")
        else:
            print(f"      ! 速度等级设置返回码: {ret}")
    except Exception as e:
        print(f"      ✗ 设置速度等级失败: {e}")
    
    # 3. 调用 set_ready 复位机械臂
    print(f"[3/7] 复位机械臂 (set_ready)...")
    try:
        ret = carm.set_ready()
        if ret == 0:
            print("      ✓ 复位成功，机械臂已就绪")
        else:
            print(f"      ✗ 复位失败，返回码: {ret}")
            # 尝试继续
    except Exception as e:
        print(f"      ✗ 复位异常: {e}")
        return False
    
    # 4. 获取当前位置
    print(f"[4/7] 获取当前位置...")
    try:
        joint_pos = carm.get_joint_pos()
        print(f"      当前关节位置 (rad): {[round(p, 4) for p in joint_pos]}")
        
        cart_pose = carm.get_cart_pose()
        print(f"      当前末端位姿: {[round(p, 4) for p in cart_pose]}")
    except Exception as e:
        print(f"      ✗ 获取位置失败: {e}")
    
    # 5. 运动到零位
    print(f"[5/7] 运动到零位...")
    print(f"      目标位置: {HOME_POSITION}")
    try:
        input("      按 Enter 键开始运动（或 Ctrl+C 取消）...")
        ret = carm.move_joint(HOME_POSITION, -1, True)  # desire_time=-1 使用设定速度，is_sync=True 阻塞等待
        if ret == 0:
            print("      ✓ 运动到零位完成")
        else:
            print(f"      ! 运动返回码: {ret}")
    except KeyboardInterrupt:
        print("\n      ! 用户取消")
        carm.emergency_stop()
        carm.disconnect()
        return False
    except Exception as e:
        print(f"      ✗ 运动失败: {e}")
    
    time.sleep(0.5)
    
    # 6. 运动到测试位置
    print(f"[6/7] 运动到测试位置...")
    print(f"      目标位置: {TEST_POSITION_1}")
    try:
        input("      按 Enter 键开始运动（或 Ctrl+C 取消）...")
        ret = carm.move_joint(TEST_POSITION_1, -1, True)
        if ret == 0:
            print("      ✓ 运动到测试位置完成")
            
            # 获取到达后的位置
            joint_pos = carm.get_joint_pos()
            print(f"      到达位置 (rad): {[round(p, 4) for p in joint_pos]}")
            
            cart_pose = carm.get_cart_pose()
            print(f"      末端位姿: {[round(p, 4) for p in cart_pose]}")
        else:
            print(f"      ! 运动返回码: {ret}")
    except KeyboardInterrupt:
        print("\n      ! 用户取消")
        carm.emergency_stop()
        carm.disconnect()
        return False
    except Exception as e:
        print(f"      ✗ 运动失败: {e}")
    
    time.sleep(0.5)
    
    # 7. 运动回零位
    print(f"[7/7] 运动回零位...")
    print(f"      目标位置: {TEST_POSITION_2}")
    try:
        input("      按 Enter 键开始运动（或 Ctrl+C 取消）...")
        ret = carm.move_joint(TEST_POSITION_2, -1, True)
        if ret == 0:
            print("      ✓ 运动回零位完成")
        else:
            print(f"      ! 运动返回码: {ret}")
    except KeyboardInterrupt:
        print("\n      ! 用户取消")
        carm.emergency_stop()
        carm.disconnect()
        return False
    except Exception as e:
        print(f"      ✗ 运动失败: {e}")
    
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
    print("  运动测试完成！")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_motion()
    sys.exit(0 if success else 1)
