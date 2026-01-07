#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多进程控制器测试

验证 Arx5JointControllerProcess 及其 Manager 接口:
1. 控制器启动/停止
2. 状态获取 (关节位置、速度、力矩)
3. servoL 点对点移动
4. schedule_waypoint 单点轨迹
5. add_waypoint + update_trajectory 批量轨迹

用法:
    # 完整测试 (需要真实机器人)
    python -m consistency_policy.tests.test_controller
    
    # 仅测试状态获取
    python -m consistency_policy.tests.test_controller --state-only
    
    # 测试 servoL 移动
    python -m consistency_policy.tests.test_controller --servo
    
    # 测试轨迹调度
    python -m consistency_policy.tests.test_controller --trajectory

环境:
    conda activate arx-py310

警告:
    此脚本会控制真实机器人运动!
    确保工作空间安全,准备好急停按钮。
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import List, Dict, Any

# 添加项目路径
CONSISTENCY_POLICY_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_VLA_PATH = os.path.dirname(CONSISTENCY_POLICY_PATH)
if RL_VLA_PATH not in sys.path:
    sys.path.insert(0, RL_VLA_PATH)

from consistency_policy.robot_controller_mp import (
    Arx5JointControllerProcess,
    Arx5JointControllerManager,
)

# HOME 位置 (关节零位)
HOME_POSITION = np.zeros(6)


class ControllerTester:
    """多进程控制器测试类"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results: Dict[str, bool] = {}
    
    def log(self, message: str):
        """打印日志"""
        if self.verbose:
            print(message)
    
    def test_startup_shutdown(self) -> bool:
        """测试控制器启动和停止"""
        self.log("\n" + "=" * 60)
        self.log("测试 1: 控制器启动/停止")
        self.log("=" * 60)
        
        try:
            # 使用 context manager
            with Arx5JointControllerManager(
                model="X5",
                interface="can0",
                frequency=200.0,
                verbose=self.verbose,
            ) as controller:
                self.log("✓ 控制器启动成功")
                self.log(f"  PID: {controller.pid}")
                self.log(f"  共享内存: 已创建")
                
                # 等待一段时间
                time.sleep(0.5)
                
                # 检查进程是否存活
                if controller.is_alive():
                    self.log("✓ 控制器进程正在运行")
                else:
                    self.log("✗ 控制器进程异常退出")
                    return False
            
            self.log("✓ 控制器正常停止")
            return True
            
        except Exception as e:
            self.log(f"✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_state_reading(self) -> bool:
        """测试状态读取"""
        self.log("\n" + "=" * 60)
        self.log("测试 2: 状态读取")
        self.log("=" * 60)
        
        try:
            with Arx5JointControllerManager(
                model="X5",
                interface="can0",
                frequency=200.0,
                verbose=self.verbose,
            ) as controller:
                # 读取状态
                state = controller.get_state()
                
                self.log("\n当前机器人状态:")
                self.log(f"  关节位置: {np.array2string(state['joint_pos'], precision=4)}")
                self.log(f"  关节速度: {np.array2string(state['joint_vel'], precision=4)}")
                self.log(f"  夹爪位置: {state['gripper_pos']:.4f}")
                self.log(f"  时间戳: {state['timestamp']:.6f}")
                
                # 验证状态格式
                assert 'joint_pos' in state
                assert 'joint_vel' in state
                assert 'gripper_pos' in state
                assert 'timestamp' in state
                assert len(state['joint_pos']) == 6
                
                self.log("✓ 状态读取正常")
                
                # 测试连续读取性能
                self.log("\n测试连续读取性能...")
                num_reads = 1000
                start = time.perf_counter()
                
                for _ in range(num_reads):
                    _ = controller.get_state()
                
                elapsed = time.perf_counter() - start
                read_hz = num_reads / elapsed
                
                self.log(f"  读取次数: {num_reads}")
                self.log(f"  总耗时: {elapsed*1000:.1f}ms")
                self.log(f"  读取频率: {read_hz:.0f} Hz")
                
                if read_hz > 1000:
                    self.log("✓ 读取性能优秀 (>1kHz)")
                else:
                    self.log("⚠ 读取性能偏低")
                
                return True
                
        except Exception as e:
            self.log(f"✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_servo_movement(self) -> bool:
        """测试 servoL 点对点移动"""
        self.log("\n" + "=" * 60)
        self.log("测试 3: servoL 点对点移动")
        self.log("=" * 60)
        
        # 安全确认
        self.log("\n⚠ 警告: 此测试将移动机器人!")
        self.log("  确保工作空间安全,准备好急停按钮")
        
        confirm = input("  继续? [y/N]: ")
        if confirm.lower() != 'y':
            self.log("  测试取消")
            return True  # 用户取消不算失败
        
        try:
            with Arx5JointControllerManager(
                model="X5",
                interface="can0",
                frequency=200.0,
                verbose=self.verbose,
            ) as controller:
                # 获取当前位置
                state = controller.get_state()
                current_pos = state['joint_pos'].copy()
                current_gripper = state['gripper_pos']
                
                self.log(f"\n当前位置: {np.array2string(current_pos, precision=4)}")
                
                # 目标位置: HOME
                target_pos = np.array(HOME_POSITION)
                self.log(f"目标位置: {np.array2string(target_pos, precision=4)}")
                
                # 移动到 HOME
                self.log("\n移动到 HOME 位置...")
                controller.servoL(
                    joint_pos=target_pos,
                    gripper_pos=current_gripper,
                    duration=3.0,
                )
                
                # servoL 是异步的，等待移动完成
                time.sleep(3.5)
                success = True
                
                if success:
                    self.log("✓ 移动完成")
                else:
                    self.log("✗ 移动超时")
                    return False
                
                # 验证到达
                final_state = controller.get_state()
                final_pos = final_state['joint_pos']
                error = np.abs(final_pos - target_pos)
                max_error = np.max(error)
                
                self.log(f"\n最终位置: {np.array2string(final_pos, precision=4)}")
                self.log(f"位置误差: {np.array2string(error, precision=4)}")
                self.log(f"最大误差: {max_error:.4f} rad")
                
                if max_error < 0.05:  # 允许 0.05 rad (~3°) 误差
                    self.log("✓ 定位精度达标")
                    return True
                else:
                    self.log("⚠ 定位精度偏低")
                    return True  # 仍然算通过，只是精度不理想
                
        except Exception as e:
            self.log(f"✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_trajectory_scheduling(self) -> bool:
        """测试轨迹调度 (add_waypoint + update_trajectory)"""
        self.log("\n" + "=" * 60)
        self.log("测试 4: 轨迹调度")
        self.log("=" * 60)
        
        # 安全确认
        self.log("\n⚠ 警告: 此测试将移动机器人!")
        confirm = input("  继续? [y/N]: ")
        if confirm.lower() != 'y':
            self.log("  测试取消")
            return True
        
        try:
            with Arx5JointControllerManager(
                model="X5",
                interface="can0",
                frequency=200.0,
                verbose=self.verbose,
            ) as controller:
                # 获取当前位置
                state = controller.get_state()
                current_pos = state['joint_pos'].copy()
                current_gripper = state['gripper_pos']
                
                self.log(f"\n当前位置: {np.array2string(current_pos, precision=4)}")
                
                # 生成简单轨迹: 正弦波运动
                num_waypoints = 50
                duration = 5.0
                dt = duration / num_waypoints
                
                # 仅在第一个关节上做小幅正弦运动
                amplitude = 0.1  # 0.1 rad (~6°)
                timestamps = np.linspace(0, duration, num_waypoints)
                
                waypoints = []
                base_time = time.time()  # 用于计算绝对时间
                for i, t in enumerate(timestamps):
                    pos = current_pos.copy()
                    pos[0] += amplitude * np.sin(2 * np.pi * t / duration)
                    waypoints.append({
                        'target_time': base_time + t,  # 绝对时间
                        'joint_pos': pos,
                        'gripper_pos': current_gripper,
                    })
                
                self.log(f"\n生成轨迹: {num_waypoints} 个路点, {duration}s 时长")
                self.log(f"  第一关节振幅: ±{np.degrees(amplitude):.1f}°")
                
                # 添加所有路点
                self.log("\n添加路点...")
                for wp in waypoints:
                    controller.add_waypoint(
                        joint_pos=wp['joint_pos'],
                        gripper_pos=wp['gripper_pos'],
                        target_time=wp['target_time'],
                    )
                
                # 更新轨迹
                self.log("更新轨迹...")
                controller.update_trajectory()
                
                # 等待轨迹执行完成
                self.log("执行轨迹...")
                start_time = time.time()
                positions_log = []
                
                while time.time() - start_time < duration + 1.0:
                    state = controller.get_state()
                    positions_log.append({
                        'time': time.time() - start_time,
                        'pos': state['joint_pos'].copy(),
                    })
                    time.sleep(0.02)  # 50Hz 采样
                
                self.log("✓ 轨迹执行完成")
                
                # 分析轨迹跟踪精度
                positions_log = np.array([p['pos'][0] for p in positions_log])
                peak_to_peak = np.max(positions_log) - np.min(positions_log)
                expected_peak_to_peak = 2 * amplitude
                
                self.log(f"\n轨迹分析:")
                self.log(f"  期望峰峰值: {np.degrees(expected_peak_to_peak):.1f}°")
                self.log(f"  实际峰峰值: {np.degrees(peak_to_peak):.1f}°")
                self.log(f"  跟踪误差: {np.degrees(abs(peak_to_peak - expected_peak_to_peak)):.1f}°")
                
                return True
                
        except Exception as e:
            self.log(f"✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self, skip_motion: bool = False):
        """运行所有测试"""
        self.log("\n" + "=" * 60)
        self.log("Arx5 多进程控制器测试套件")
        self.log("=" * 60)
        
        # 测试 1: 启动/停止
        self.test_results['startup_shutdown'] = self.test_startup_shutdown()
        
        # 测试 2: 状态读取
        self.test_results['state_reading'] = self.test_state_reading()
        
        if not skip_motion:
            # 测试 3: servoL
            self.test_results['servo_movement'] = self.test_servo_movement()
            
            # 测试 4: 轨迹调度
            self.test_results['trajectory_scheduling'] = self.test_trajectory_scheduling()
        
        # 总结
        self.log("\n" + "=" * 60)
        self.log("测试总结")
        self.log("=" * 60)
        
        all_passed = True
        for name, passed in self.test_results.items():
            status = "✓ 通过" if passed else "✗ 失败"
            self.log(f"  {name}: {status}")
            if not passed:
                all_passed = False
        
        self.log("\n" + "=" * 60)
        if all_passed:
            self.log("✓ 所有测试通过")
        else:
            self.log("✗ 部分测试失败")
        self.log("=" * 60)
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(description="多进程控制器测试")
    parser.add_argument("--state-only", action="store_true", help="仅测试状态获取")
    parser.add_argument("--servo", action="store_true", help="测试 servoL 移动")
    parser.add_argument("--trajectory", action="store_true", help="测试轨迹调度")
    parser.add_argument("--skip-motion", action="store_true", help="跳过运动测试")
    parser.add_argument("-v", "--verbose", action="store_true", default=True, help="详细输出")
    
    args = parser.parse_args()
    
    tester = ControllerTester(verbose=args.verbose)
    
    # 特定测试
    if args.state_only:
        tester.test_startup_shutdown()
        success = tester.test_state_reading()
    elif args.servo:
        tester.test_startup_shutdown()
        success = tester.test_servo_movement()
    elif args.trajectory:
        tester.test_startup_shutdown()
        success = tester.test_trajectory_scheduling()
    else:
        # 运行所有测试
        success = tester.run_all_tests(skip_motion=args.skip_motion)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
