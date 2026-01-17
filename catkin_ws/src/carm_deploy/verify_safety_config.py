#!/usr/bin/env python3
"""
Safety Config 验证脚本

验证 safety_config.json 的限制是否合理，通过在真机上执行边界测试。

安全说明:
    - 使用 MIT 模式 (mode=2) 进行控制，该模式为力矩模式，更安全
    - 严禁使用 Position 模式 (mode=1)
    - 测试时会缓慢移动到边界附近，不会超出限制

用法:
    python verify_safety_config.py --config ~/rl-vla/safety_config.json
    python verify_safety_config.py --config ~/rl-vla/safety_config.json --test_mode visual  # 仅可视化
    python verify_safety_config.py --config ~/rl-vla/safety_config.json --test_mode boundary  # 边界测试
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

from safety_controller import SafetyController


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
    
    def disconnect(self):
        """断开连接"""
        if self.arm:
            # 设置回空闲模式
            self.arm.set_control_mode(self.MODE_IDLE)
            print("机械臂已设置为空闲模式")
    
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
        """可视化验证模式 - 仅显示状态，不执行动作"""
        print("\n" + "="*60)
        print("  可视化验证模式")
        print("  按 Ctrl+C 退出")
        print("="*60)
        
        try:
            while True:
                # 获取并检查状态
                results = self.check_current_position()
                
                # 清屏效果
                print("\033[2J\033[H")  # 清屏并移动光标到左上角
                
                print("="*60)
                print("  Safety Config 实时验证")
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
                
                print("\n按 Ctrl+C 退出...")
                time.sleep(0.2)
                
        except KeyboardInterrupt:
            print("\n\n验证结束")
    
    def verify_boundary_test(self, speed: float = 2.0):
        """
        边界测试模式 - 缓慢移动测试边界
        
        使用 MIT 模式 (mode=2) 进行安全控制
        """
        print("\n" + "="*60)
        print("  边界测试模式 (MIT 模式)")
        print("="*60)
        
        # 设置 MIT 模式 (绝对禁止使用 Position 模式!)
        print("\n设置控制模式: MIT (mode=2)")
        self.arm.set_ready()
        ret = self.arm.set_control_mode(self.MODE_MIT)  # MIT 力矩模式
        if ret != 0:
            print(f"⚠️  set_control_mode 返回: {ret}")
        
        # 获取当前状态作为起点
        joint_pos, end_pose, gripper = self.get_current_state()
        
        print("\n当前关节位置:")
        for i, j in enumerate(joint_pos):
            jl = self.config['joint_limits']
            j_min, j_max = jl['joint_min'][i], jl['joint_max'][i]
            print(f"  J{i+1}: {j:+.4f}  范围: [{j_min:+.4f}, {j_max:+.4f}]")
        
        # 测试选项
        print("\n可用的测试:")
        print("  1. 测试当前位置是否在安全范围内")
        print("  2. 移动到工作空间中心")
        print("  3. 沿 X 轴测试边界")
        print("  4. 沿 Y 轴测试边界")
        print("  5. 沿 Z 轴测试边界")
        print("  6. 测试夹爪范围")
        print("  q. 退出")
        
        while True:
            try:
                choice = input("\n选择测试 (1-6, q): ").strip().lower()
                
                if choice == 'q':
                    break
                elif choice == '1':
                    self._test_current_position()
                elif choice == '2':
                    self._move_to_center(speed)
                elif choice == '3':
                    self._test_axis('x', speed)
                elif choice == '4':
                    self._test_axis('y', speed)
                elif choice == '5':
                    self._test_axis('z', speed)
                elif choice == '6':
                    self._test_gripper(speed)
                else:
                    print("无效选择")
                    
            except KeyboardInterrupt:
                print("\n\n测试中断")
                break
        
        # 恢复空闲模式
        self.arm.set_control_mode(self.MODE_IDLE)
        print("\n已恢复空闲模式")
    
    def _test_current_position(self):
        """测试当前位置"""
        results = self.check_current_position()
        
        print("\n关节状态:")
        for status in results['joint_status']:
            print(f"  {status}")
        
        print("\n工作空间状态:")
        for status in results['workspace_status']:
            print(f"  {status}")
        
        print(f"\n{results['gripper_status']}")
        
        if results['in_bounds']:
            print("\n✅ 当前位置在安全范围内")
        else:
            print("\n⚠️  当前位置超出安全范围!")
    
    def _move_to_center(self, speed: float):
        """移动到工作空间中心"""
        wl = self.config['workspace_limits']
        
        center_x = (wl['x_min'] + wl['x_max']) / 2
        center_y = (wl['y_min'] + wl['y_max']) / 2
        center_z = (wl['z_min'] + wl['z_max']) / 2
        
        print(f"\n移动到工作空间中心: ({center_x:.4f}, {center_y:.4f}, {center_z:.4f})")
        
        # 获取当前位姿的四元数
        _, end_pose, _ = self.get_current_state()
        qx, qy, qz, qw = end_pose[3], end_pose[4], end_pose[5], end_pose[6]
        
        # 移动到中心位置（保持当前姿态）
        print("⚠️  即将移动，请确保路径无障碍物")
        confirm = input("确认移动? (y/n): ").strip().lower()
        
        if confirm == 'y':
            self.arm.move_p_with_speed(
                center_x, center_y, center_z,
                qx, qy, qz, qw,
                speed
            )
            print("移动完成")
            self._test_current_position()
        else:
            print("已取消")
    
    def _test_axis(self, axis: str, speed: float):
        """沿指定轴测试边界"""
        wl = self.config['workspace_limits']
        
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        v_min = wl[f'{axis}_min']
        v_max = wl[f'{axis}_max']
        
        print(f"\n{axis.upper()} 轴范围: [{v_min:.4f}, {v_max:.4f}]")
        
        _, end_pose, _ = self.get_current_state()
        current_val = end_pose[axis_idx]
        
        print(f"当前 {axis.upper()} 值: {current_val:.4f}")
        
        # 计算测试点（距离边界 10%）
        range_size = v_max - v_min
        test_min = v_min + 0.1 * range_size
        test_max = v_max - 0.1 * range_size
        
        print(f"测试点 (距边界 10%): [{test_min:.4f}, {test_max:.4f}]")
        
        print("\n选择:")
        print("  1. 移动到最小边界附近")
        print("  2. 移动到最大边界附近")
        print("  3. 返回")
        
        choice = input("选择: ").strip()
        
        if choice in ['1', '2']:
            target_val = test_min if choice == '1' else test_max
            target_pose = list(end_pose[:3])
            target_pose[axis_idx] = target_val
            
            print(f"⚠️  即将移动到 {axis.upper()}={target_val:.4f}")
            confirm = input("确认移动? (y/n): ").strip().lower()
            
            if confirm == 'y':
                self.arm.move_p_with_speed(
                    target_pose[0], target_pose[1], target_pose[2],
                    end_pose[3], end_pose[4], end_pose[5], end_pose[6],
                    speed
                )
                print("移动完成")
                self._test_current_position()
            else:
                print("已取消")
    
    def _test_gripper(self, speed: float):
        """测试夹爪范围"""
        jl = self.config['joint_limits']
        g_min = jl['gripper_min']
        g_max = jl['gripper_max']
        
        _, _, gripper = self.get_current_state()
        
        print(f"\n夹爪范围: [{g_min:.4f}, {g_max:.4f}]")
        print(f"当前夹爪: {gripper:.4f}")
        
        print("\n选择:")
        print("  1. 关闭夹爪 (到最小值)")
        print("  2. 打开夹爪 (到最大值)")
        print("  3. 返回")
        
        choice = input("选择: ").strip()
        
        if choice == '1':
            target = g_min + 0.005  # 留一点余量
            self.arm.set_gripper(target)
            print(f"夹爪设置为: {target:.4f}")
        elif choice == '2':
            target = g_max - 0.005
            self.arm.set_gripper(target)
            print(f"夹爪设置为: {target:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Safety Config 验证工具')
    parser.add_argument('--config', '-c', type=str, default='~/rl-vla/safety_config.json',
                        help='安全配置文件路径')
    parser.add_argument('--robot_ip', type=str, default='10.42.0.101',
                        help='机械臂 IP 地址')
    parser.add_argument('--test_mode', type=str, choices=['visual', 'boundary', 'check'],
                        default='check',
                        help='测试模式: visual=实时可视化, boundary=边界测试, check=仅检查当前位置')
    
    args = parser.parse_args()
    
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
            
        elif args.test_mode == 'boundary':
            verifier.verify_boundary_test()
            
    finally:
        verifier.disconnect()


if __name__ == '__main__':
    main()
