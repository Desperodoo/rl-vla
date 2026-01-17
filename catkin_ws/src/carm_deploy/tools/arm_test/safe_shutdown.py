#!/usr/bin/env python3
"""
CARM 机械臂安全退出脚本
执行下使能并断开连接，使机械臂进入安全的自由状态
使用前请先运行: source carm_scripts/setup_carm_env.sh

⚠️ 注意：下使能后机械臂会失去保持力，请确保机械臂处于安全位置！
"""

import sys
import time
from carm import carm_py

# 机械臂连接配置
ARM_IP = "10.42.0.101"

def safe_shutdown():
    """安全关闭机械臂"""
    print("=" * 50)
    print("  CARM 机械臂安全退出")
    print("=" * 50)
    print()
    print("⚠️  注意：下使能后机械臂将失去保持力！")
    print("    请确保机械臂处于安全位置。")
    print()
    
    # 1. 创建并连接机械臂
    print(f"[1/4] 连接机械臂...")
    try:
        carm = carm_py.CArmSingleCol(ARM_IP)
        time.sleep(1)
        if carm.is_connected():
            print("      ✓ 连接成功")
        else:
            print("      ! 连接状态异常，尝试继续...")
    except Exception as e:
        print(f"      ✗ 连接失败: {e}")
        return False
    
    # 2. 获取当前状态
    print(f"[2/4] 获取当前状态...")
    try:
        status = carm.get_status()
        print(f"      伺服状态: {status.servo_status}")
        print(f"      控制模式: {status.state}")
        
        joint_pos = carm.get_joint_pos()
        print(f"      当前关节位置: {[round(p, 3) for p in joint_pos]}")
    except Exception as e:
        print(f"      ! 获取状态失败: {e}")
    
    # 3. 下使能
    print(f"[3/4] 下使能机械臂...")
    try:
        input("      按 Enter 键确认下使能（或 Ctrl+C 取消）...")
        
        # 先设置为空闲模式
        ret = carm.set_control_mode(0)  # 0 = IDLE
        print(f"      设置空闲模式: {'✓' if ret == 0 else f'返回码 {ret}'}")
        
        time.sleep(0.5)
        
        # 下使能
        ret = carm.set_servo_enable(False)
        print(f"      下使能: {'✓' if ret == 0 else f'返回码 {ret}'}")
        
        time.sleep(0.5)
        
        # 验证状态
        status = carm.get_status()
        if not status.servo_status:
            print("      ✓ 机械臂已下使能，可以自由拖动")
        else:
            print("      ! 伺服状态仍为启用，请检查")
            
    except KeyboardInterrupt:
        print("\n      ! 用户取消，保持当前状态")
    except Exception as e:
        print(f"      ✗ 下使能失败: {e}")
    
    # 4. 断开连接
    print(f"[4/4] 断开连接...")
    try:
        carm.disconnect()
        print("      ✓ 已断开连接")
    except Exception as e:
        print(f"      ! 断开连接异常: {e}")
    
    print()
    print("=" * 50)
    print("  安全退出完成")
    print("=" * 50)
    print()
    print("提示：")
    print("  - 机械臂现在处于自由状态，可以手动拖动")
    print("  - 重新使用前需要调用 set_ready() 复位")
    print()
    return True

if __name__ == "__main__":
    success = safe_shutdown()
    sys.exit(0 if success else 1)
