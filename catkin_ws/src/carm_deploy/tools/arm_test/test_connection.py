#!/usr/bin/env python3
"""
CARM 机械臂连接测试脚本
测试内容：连接机械臂、获取版本信息、获取配置和状态
使用前请先运行: source carm_scripts/setup_carm_env.sh
"""

import sys
from carm import carm_py

# 机械臂连接配置
ARM_IP = "10.42.0.101"
ARM_PORT = 8090
TIMEOUT = 1

def test_connection():
    """测试机械臂连接"""
    print("=" * 50)
    print("  CARM 机械臂连接测试")
    print("=" * 50)
    print()
    
    # 1. 创建机械臂对象（构造函数会自动连接）
    print(f"[1/4] 创建 CArmSingleCol 对象并连接...")
    print(f"      IP: {ARM_IP}")
    try:
        # 注意：构造函数只需要传 IP，会自动连接
        carm = carm_py.CArmSingleCol(ARM_IP)
        import time
        time.sleep(1)  # 等待连接稳定
        print("      ✓ 对象创建成功")
    except Exception as e:
        print(f"      ✗ 对象创建失败: {e}")
        return False
    
    # 2. 检查连接状态
    print(f"[2/4] 检查连接状态...")
    try:
        connected = carm.is_connected()
        if connected:
            print("      ✓ 连接状态正常")
        else:
            print("      ! 连接状态返回 False，但可能仍可通信")
    except Exception as e:
        print(f"      ✗ 检查状态异常: {e}")
    
    # 3. 获取版本信息
    print(f"[3/4] 获取版本信息...")
    try:
        version = carm.get_version()
        print(f"      ✓ 版本: {version}")
    except Exception as e:
        print(f"      ✗ 获取版本失败: {e}")
    
    # 4. 获取配置和状态
    print(f"[4/4] 获取机械臂配置和状态...")
    try:
        config = carm.get_config()
        print(f"      自由度 (DOF): {config.dof}")
        print(f"      关节上限: {config.limit_upper}")
        print(f"      关节下限: {config.limit_lower}")
        print(f"      关节速度: {config.joint_vel}")
        
        status = carm.get_status()
        print(f"      机械臂名称: {status.arm_name}")
        print(f"      连接状态: {status.arm_is_connected}")
        print(f"      伺服状态: {status.servo_status}")
        print(f"      当前状态: {status.state}")
        print(f"      速度百分比: {status.speed_percentage}")
        print("      ✓ 配置和状态获取成功")
    except Exception as e:
        print(f"      ✗ 获取配置/状态失败: {e}")
    
    # 6. 获取当前关节位置
    print()
    print("[额外] 获取当前关节和末端位置...")
    try:
        joint_pos = carm.get_joint_pos()
        print(f"      关节位置 (rad): {joint_pos}")
        
        cart_pose = carm.get_cart_pose()
        print(f"      末端位姿 (x,y,z,Qx,Qy,Qz,Qw): {cart_pose}")
    except Exception as e:
        print(f"      ✗ 获取位置失败: {e}")
    
    # 7. 断开连接
    print()
    print("[清理] 断开连接...")
    try:
        carm.disconnect()
        print("      ✓ 已断开连接")
    except Exception as e:
        print(f"      ! 断开连接异常: {e}")
    
    print()
    print("=" * 50)
    print("  连接测试完成！")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
