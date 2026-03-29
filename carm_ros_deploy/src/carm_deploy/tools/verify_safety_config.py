#!/usr/bin/env python3
"""
Safety Config 验证脚本

验证 safety_config.json 的限制是否合理。

测试模式:
    - check: 检查当前位置是否在安全范围内（默认）
    - visual: 开启拖动示教模式，实时显示安全边界状态，可拖动机械臂验证

退出行为:
    - 退出时自动在 PF 模式下回零位，然后下使能

用法:
    python verify_safety_config.py --config /path/to/carm_deploy/safety_config.json
    python verify_safety_config.py --config /path/to/carm_deploy/safety_config.json --test_mode visual
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import Optional, Tuple, List

# CARM SDK
try:
    from carm import carm_py
    print(f"CARM 模块加载成功 (Python {sys.version_info.major}.{sys.version_info.minor})")
except ImportError as e:
    print(f"错误: CARM 模块加载失败 - {e}")
    print("\n请先设置环境:")
    print("  source ~/rl-vla/scripts/setup_carm_env.sh")
    sys.exit(1)

# 添加 carm_deploy 根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
carm_deploy_root = os.path.dirname(script_dir)
sys.path.insert(0, carm_deploy_root)

from core.safety_controller import SafetyController


class SafetyConfigVerifier:
    """安全配置验证器"""
    
    # 控制模式常量
    MODE_IDLE = 0      # 空闲模式
    MODE_POSITION = 1  # 位置模式 (禁用!)
    MODE_MIT = 2       # MIT 模式 (推荐)
    MODE_DRAG = 3      # 拖动示教模式
    MODE_PF = 4        # 力位混合模式
    
    def __init__(self, config_path: str, robot_ip: str = "10.42.0.101"):
        self.config_path = os.path.expanduser(config_path)
        self.robot_ip = robot_ip
        self.arm: Optional[carm_py.CArmSingleCol] = None
        self.safety: Optional[SafetyController] = None
        
        # 加载配置
        self._load_config()
    
    # 机械臂官方关节限位
    CARM_JOINT_UPPER = np.array([2.79, 3.14, 0.0, 2.65, 1.57, 2.88])
    CARM_JOINT_LOWER = np.array([-2.79, 0.0, -3.14, -2.65, -1.57, -2.88])
        
    def _load_config(self):
        """加载安全配置"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # 创建 SafetyController
        self.safety = SafetyController.from_config(self.config_path)
        
        print(f"\n{'='*60}")
        print(f"  加载安全配置: {self.config_path}")
        print(f"{'='*60}")
        
        # 打印配置信息
        jl = self.config['joint_limits']
        wl = self.config['workspace_limits']
        
        print("\n关节限制 (使用官方限位 + 10% 裕度):")
        for i in range(6):
            official_range = self.CARM_JOINT_UPPER[i] - self.CARM_JOINT_LOWER[i]
            expected_min = self.CARM_JOINT_LOWER[i] + 0.1 * official_range
            expected_max = self.CARM_JOINT_UPPER[i] - 0.1 * official_range
            config_min = jl['joint_min'][i]
            config_max = jl['joint_max'][i]
            # 检查是否与预期一致
            is_correct = abs(config_min - expected_min) < 0.01 and abs(config_max - expected_max) < 0.01
            status = "✓" if is_correct else "⚠"
            print(f"  {status} J{i+1}: [{config_min:+.4f}, {config_max:+.4f}] rad  "
                  f"(官方: [{self.CARM_JOINT_LOWER[i]:+.2f}, {self.CARM_JOINT_UPPER[i]:+.2f}])")
        print(f"  Gripper: [{jl['gripper_min']:.4f}, {jl['gripper_max']:.4f}] m")
        
        print("\n工作空间限制 (来自采集数据):")
        print(f"  X: [{wl['x_min']:.4f}, {wl['x_max']:.4f}] m")
        print(f"  Y: [{wl['y_min']:.4f}, {wl['y_max']:.4f}] m")
        print(f"  Z: [{wl['z_min']:.4f}, {wl['z_max']:.4f}] m")
        
        if 'metadata' in self.config:
            meta = self.config['metadata']
            print(f"\n元数据:")
            print(f"  创建时间: {meta.get('created_at', 'N/A')}")
            print(f"  采样点数: {meta.get('sample_count', 'N/A')}")
            print(f"  Margin: {meta.get('margin', 'N/A')}")
        
        print(f"{'='*60}\n")
    
    def connect(self) -> bool:
        """连接机械臂"""
        print(f"连接机械臂: {self.robot_ip}")
        try:
            self.arm = carm_py.CArmSingleCol(self.robot_ip)
            time.sleep(1.0)
            
            # 获取当前状态验证连接
            status = self.arm.get_status()
            if not status.arm_is_connected:
                print("❌ 机械臂未连接")
                return False
            
            print("✓ 机械臂连接成功")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def safe_shutdown(self):
        """安全关闭：PF 模式回零位 → 下使能 → 断开连接"""
        if not self.arm:
            return
        
        print("\n" + "="*60)
        print("  安全关闭流程")
        print("="*60)
        
        try:
            # 1. 复位并进入 PF 模式
            print("\n[1/4] 设置 PF 模式准备回零位...")
            self.arm.set_ready()
            time.sleep(0.5)
            ret = self.arm.set_control_mode(self.MODE_PF)
            print(f"      PF 模式: {'✓' if ret == 0 else f'返回码 {ret}'}")
            time.sleep(0.3)
            
            # 2. 回零位
            print("[2/4] 回零位...")
            home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ret = self.arm.move_joint(home, -1, True)
            print(f"      回零位: {'✓' if ret == 0 else f'返回码 {ret}'}")
            time.sleep(0.5)
            
            # 3. 切换到 IDLE → 下使能
            print("[3/4] 下使能...")
            self.arm.set_control_mode(self.MODE_IDLE)
            time.sleep(0.3)
            ret = self.arm.set_servo_enable(False)
            print(f"      下使能: {'✓' if ret == 0 else f'返回码 {ret}'}")
            time.sleep(0.3)
            
            # 4. 断开连接
            print("[4/4] 断开连接...")
            self.arm.disconnect()
            print("      ✓ 已断开连接")
            
        except Exception as e:
            print(f"\n⚠️  安全关闭过程异常: {e}")
            try:
                self.arm.set_control_mode(self.MODE_IDLE)
                self.arm.disconnect()
            except:
                pass
        
        print("\n" + "="*60)
        print("  安全关闭完成")
        print("="*60)
    
    def get_current_state(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        获取当前状态
        
        Returns:
            joint_pos: 关节位置 [6]
            end_pose: 末端位姿 [x,y,z,qx,qy,qz,qw]
            gripper: 夹爪开度
        """
        # 使用正确的 carm_py API
        joint_pos = np.array(self.arm.get_joint_pos())  # 返回 [j1-j6]
        end_pose = np.array(self.arm.get_cart_pose())   # 返回 [x,y,z,qx,qy,qz,qw]
        gripper = self.arm.get_gripper_pos()            # 返回 float
        
        return joint_pos, end_pose, gripper
    
    def check_current_position(self) -> dict:
        """
        检查当前位置是否在安全范围内
        
        Returns:
            检查结果字典
        """
        joint_pos, end_pose, gripper = self.get_current_state()
        
        jl = self.config['joint_limits']
        wl = self.config['workspace_limits']
        
        results = {
            'in_bounds': True,
            'joint_status': [],
            'workspace_status': [],
            'gripper_status': '',
        }
        
        # 检查关节
        for i in range(6):
            j_min, j_max = jl['joint_min'][i], jl['joint_max'][i]
            j_val = joint_pos[i]
            
            if j_val < j_min:
                status = f"❌ J{i+1}: {j_val:+.4f} < min({j_min:+.4f})"
                results['in_bounds'] = False
            elif j_val > j_max:
                status = f"❌ J{i+1}: {j_val:+.4f} > max({j_max:+.4f})"
                results['in_bounds'] = False
            else:
                # 计算距离边界的百分比
                range_size = j_max - j_min
                margin_to_min = (j_val - j_min) / range_size * 100
                margin_to_max = (j_max - j_val) / range_size * 100
                min_margin = min(margin_to_min, margin_to_max)
                status = f"✓ J{i+1}: {j_val:+.4f} (距边界 {min_margin:.1f}%)"
            
            results['joint_status'].append(status)
        
        # 检查工作空间
        x, y, z = end_pose[0], end_pose[1], end_pose[2]
        
        for axis, val, (v_min, v_max) in [
            ('X', x, (wl['x_min'], wl['x_max'])),
            ('Y', y, (wl['y_min'], wl['y_max'])),
            ('Z', z, (wl['z_min'], wl['z_max'])),
        ]:
            if val < v_min:
                status = f"❌ {axis}: {val:.4f} < min({v_min:.4f})"
                results['in_bounds'] = False
            elif val > v_max:
                status = f"❌ {axis}: {val:.4f} > max({v_max:.4f})"
                results['in_bounds'] = False
            else:
                range_size = v_max - v_min
                margin_to_min = (val - v_min) / range_size * 100
                margin_to_max = (v_max - val) / range_size * 100
                min_margin = min(margin_to_min, margin_to_max)
                status = f"✓ {axis}: {val:.4f} (距边界 {min_margin:.1f}%)"
            
            results['workspace_status'].append(status)
        
        # 检查夹爪
        g_min, g_max = jl['gripper_min'], jl['gripper_max']
        if gripper < g_min:
            results['gripper_status'] = f"❌ Gripper: {gripper:.4f} < min({g_min:.4f})"
            results['in_bounds'] = False
        elif gripper > g_max:
            results['gripper_status'] = f"❌ Gripper: {gripper:.4f} > max({g_max:.4f})"
            results['in_bounds'] = False
        else:
            results['gripper_status'] = f"✓ Gripper: {gripper:.4f}"
        
        return results
    
    def print_status(self):
        """打印当前状态"""
        joint_pos, end_pose, gripper = self.get_current_state()
        
        print("\n当前机械臂状态:")
        print("-" * 40)
        print("关节角度 (rad):")
        for i, j in enumerate(joint_pos):
            print(f"  J{i+1}: {j:+.4f}")
        
        print("\n末端位姿:")
        print(f"  位置: ({end_pose[0]:.4f}, {end_pose[1]:.4f}, {end_pose[2]:.4f})")
        print(f"  四元数: ({end_pose[3]:.4f}, {end_pose[4]:.4f}, {end_pose[5]:.4f}, {end_pose[6]:.4f})")
        print(f"  夹爪: {gripper:.4f}")
        print("-" * 40)
    
    def verify_visual(self):
        """可视化验证模式 - 开启拖动示教，实时显示安全边界状态"""
        print("\n" + "="*60)
        print("  可视化验证模式 (拖动示教)")
        print("  拖动机械臂验证安全边界是否合理")
        print("  按 Ctrl+C 退出")
        print("="*60)
        
        # 开启拖动示教模式
        print("\n设置拖动示教模式 (DRAG, mode=3)...")
        self.arm.set_ready()
        time.sleep(0.3)
        ret = self.arm.set_control_mode(self.MODE_DRAG)
        if ret != 0:
            print(f"⚠️  set_control_mode 返回: {ret}")
        else:
            print("✓ 拖动示教已开启，可以手动拖动机械臂")
        time.sleep(0.3)
        
        try:
            while True:
                # 获取并检查状态
                results = self.check_current_position()
                
                # 清屏效果
                print("\033[2J\033[H")  # 清屏并移动光标到左上角
                
                print("="*60)
                print("  Safety Config 实时验证 (拖动示教模式)")
                print("="*60)
                
                # 打印关节状态
                print("\n关节状态:")
                for status in results['joint_status']:
                    print(f"  {status}")
                
                # 打印工作空间状态
                print("\n工作空间状态:")
                for status in results['workspace_status']:
                    print(f"  {status}")
                
                # 打印夹爪状态
                print(f"\n{results['gripper_status']}")
                
                # 总体状态
                if results['in_bounds']:
                    print("\n✅ 当前位置在安全范围内")
                else:
                    print("\n⚠️  当前位置超出安全范围!")
                
                print("\n按 Ctrl+C 退出 (自动回零位并下使能)...")
                time.sleep(0.2)
                
        except KeyboardInterrupt:
            print("\n\n验证结束")
    



def main():
    parser = argparse.ArgumentParser(description='Safety Config 验证工具')
    parser.add_argument('--config', '-c', type=str, default='',
                        help='安全配置文件路径 (default: carm_deploy/safety_config.json)')
    parser.add_argument('--robot_ip', type=str, default='10.42.0.101',
                        help='机械臂 IP 地址')
    parser.add_argument('--test_mode', type=str, choices=['visual', 'check'],
                        default='visual',
                        help='测试模式: visual=拖动示教+实时验证, check=仅检查当前位置')
    
    args = parser.parse_args()

    if not args.config:
        carm_deploy_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.config = os.path.join(carm_deploy_root, 'safety_config.json')
    
    # 创建验证器
    verifier = SafetyConfigVerifier(args.config, args.robot_ip)
    
    # 连接机械臂
    if not verifier.connect():
        sys.exit(1)
    
    try:
        if args.test_mode == 'check':
            # 仅检查当前位置
            verifier.print_status()
            results = verifier.check_current_position()
            
            print("\n安全范围检查结果:")
            print("-" * 40)
            
            for status in results['joint_status']:
                print(f"  {status}")
            
            for status in results['workspace_status']:
                print(f"  {status}")
            
            print(f"  {results['gripper_status']}")
            
            print("-" * 40)
            if results['in_bounds']:
                print("✅ 当前位置在安全范围内")
            else:
                print("⚠️  当前位置超出安全范围!")
                
        elif args.test_mode == 'visual':
            verifier.verify_visual()
            
    finally:
        verifier.safe_shutdown()


if __name__ == '__main__':
    main()
