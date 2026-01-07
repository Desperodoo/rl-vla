#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂回放测试

验证控制链路完整性（需要真机）:
1. 从 demo 读取 actions 序列
2. 使用 arx5_interface 按时间顺序回放
3. 支持速度缩放（默认 50% 速度）
4. 比较实际状态 vs 期望状态

用法:
    python -m inference_rtc.tests.test_robot_replay \
        --demo ~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5 \
        --speed 0.5

安全提示:
    - 首次运行请使用 --speed 0.3 慢速执行
    - 确保机械臂周围无障碍物
    - 随时准备按 Ctrl+C 停止
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.config import setup_arx5, setup_rlft


def easeInOutQuad(t: float) -> float:
    """平滑插值函数"""
    t *= 2
    if t < 1:
        return t * t / 2
    else:
        t -= 1
        return -(t * (t - 2) - 1) / 2


class RobotReplayTest:
    """机械臂回放测试器"""
    
    def __init__(
        self,
        demo_path: str,
        model: str = "X5",
        interface: str = "can0",
        speed_scale: float = 0.5,
        verbose: bool = True,
    ):
        self.demo_path = os.path.expanduser(demo_path)
        self.model = model
        self.interface = interface
        self.speed_scale = speed_scale
        self.verbose = verbose
        
        self.robot = None
        self.actions = None
        self.obs = None
        
    def load_demo(self, traj_idx: int = 0):
        """加载 demo 数据"""
        print(f"\n[1] 加载 demo 数据: {self.demo_path}")
        
        setup_rlft()
        from diffusion_policy.utils import load_traj_hdf5
        
        raw_data = load_traj_hdf5(self.demo_path, num_traj=traj_idx + 1)
        traj_key = f"traj_{traj_idx}"
        traj = raw_data[traj_key]
        
        self.actions = traj['actions']  # (T, 7): 6 关节 + 1 夹爪
        self.obs = traj['obs']
        
        self.T = len(self.actions)
        print(f"  轨迹长度: {self.T} 帧")
        print(f"  动作维度: {self.actions.shape}")
        print(f"  关节范围: [{self.actions[:, :6].min():.3f}, {self.actions[:, :6].max():.3f}]")
        print(f"  夹爪范围: [{self.actions[:, 6].min():.3f}, {self.actions[:, 6].max():.3f}]")
        
        # 获取初始状态
        if 'joint_pos' in self.obs:
            initial_pos = self.obs['joint_pos'][0]
            print(f"  初始关节位置: {initial_pos}")
        
        return self.actions
    
    def connect_robot(self):
        """连接机械臂"""
        print(f"\n[2] 连接机械臂 ({self.model} @ {self.interface})...")
        
        setup_arx5()
        import arx5_interface as arx5
        
        self.arx5 = arx5
        
        # 创建控制器
        robot_config = arx5.RobotConfigFactory.get_instance().get_config(self.model)
        controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", robot_config.joint_dof
        )
        
        self.robot = arx5.Arx5JointController(robot_config, controller_config, self.interface)
        self.robot.set_log_level(arx5.LogLevel.INFO)
        
        # ========== 关键: 启用后台通信 ==========
        self.robot.enable_background_send_recv()
        print(f"  ✓ 已启用后台通信")
        
        # ========== 关键: 复位到 home 位置 (会自动设置 kp/kd 增益) ==========
        print(f"  正在复位到 home 位置...")
        self.robot.reset_to_home()
        print(f"  ✓ 已复位到 home")
        
        # ========== 启用重力补偿 ==========
        urdf_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "..", "..", "arx5-sdk", "models", "arx5.urdf"
        )
        if os.path.exists(urdf_path):
            self.robot.enable_gravity_compensation(urdf_path)
            print(f"  ✓ 已启用重力补偿")
        else:
            print(f"  ⚠ URDF 文件不存在: {urdf_path}")
        
        # 保存 controller_config 用于后续
        self.controller_config = controller_config
        
        # 获取当前状态
        state = self.robot.get_state()
        current_pos = np.array(state.pos())
        print(f"  当前关节位置: {current_pos}")
        print(f"  当前夹爪位置: {state.gripper_pos:.4f}")
        
        print(f"  ✓ 机械臂连接成功")
        return True
    
    def move_to_start(self, duration: float = 3.0):
        """平滑移动到起始位置"""
        print(f"\n[3] 移动到起始位置...")
        
        # 获取起始动作（第一帧）
        start_action = self.actions[0]
        target_joints = start_action[:6]
        target_gripper = float(start_action[6])
        
        # 获取当前位置
        state = self.robot.get_state()
        current_joints = np.array(state.pos())
        current_gripper = state.gripper_pos
        
        print(f"  目标位置: {target_joints}")
        print(f"  当前位置: {current_joints}")
        
        # 计算步数 (使用 controller_dt)
        dt = self.controller_config.controller_dt  # 使用控制器配置的 dt
        steps = int(duration / dt)
        
        # 平滑移动
        print(f"  平滑移动中 ({duration}s)...")
        for i in range(steps):
            t = easeInOutQuad(float(i) / steps)
            
            cmd = self.arx5.JointState(6)
            cmd.pos()[:] = current_joints + t * (target_joints - current_joints)
            cmd.gripper_pos = current_gripper + t * (target_gripper - current_gripper)
            
            self.robot.set_joint_cmd(cmd)
            # 后台模式下不需要 send_recv_once(), 只需要 sleep
            time.sleep(dt)
        
        # 验证
        state = self.robot.get_state()
        final_pos = np.array(state.pos())
        error = np.linalg.norm(final_pos - target_joints)
        print(f"  到达位置: {final_pos}")
        print(f"  位置误差: {error:.4f} rad")
        
        if error < 0.05:
            print(f"  ✓ 已到达起始位置")
        else:
            print(f"  ⚠ 位置误差较大")
    
    def replay(self, start_frame: int = 0, end_frame: int = None):
        """回放轨迹"""
        if end_frame is None:
            end_frame = self.T
        
        replay_frames = end_frame - start_frame
        original_duration = replay_frames / 30.0  # 30Hz 采集
        actual_duration = original_duration / self.speed_scale
        
        print(f"\n[4] 开始回放轨迹...")
        print(f"  帧范围: [{start_frame}, {end_frame})")
        print(f"  原始时长: {original_duration:.2f}s")
        print(f"  速度缩放: {self.speed_scale:.0%}")
        print(f"  实际时长: {actual_duration:.2f}s")
        
        input("\n按 Enter 开始回放，Ctrl+C 取消...")
        
        # 记录状态
        recorded_states = []
        recorded_times = []
        
        dt = 1.0 / 30.0 / self.speed_scale  # 调整后的时间间隔
        controller_dt = self.controller_config.controller_dt
        
        try:
            start_time = time.time()
            
            for frame_idx in range(start_frame, end_frame):
                loop_start = time.time()
                
                # 获取动作
                action = self.actions[frame_idx]
                target_joints = action[:6]
                target_gripper = float(action[6])
                
                # 发送命令
                cmd = self.arx5.JointState(6)
                cmd.pos()[:] = target_joints
                cmd.gripper_pos = target_gripper
                
                self.robot.set_joint_cmd(cmd)
                # 后台模式下不需要 send_recv_once()
                
                # 记录状态
                state = self.robot.get_state()
                recorded_states.append({
                    'target': action.copy(),
                    'actual_joints': np.array(state.pos()),
                    'actual_gripper': state.gripper_pos,
                })
                recorded_times.append(time.time() - start_time)
                
                # 打印进度
                if self.verbose and (frame_idx - start_frame) % 30 == 0:
                    progress = (frame_idx - start_frame) / replay_frames * 100
                    elapsed = time.time() - start_time
                    print(f"  进度: {progress:.1f}% | 已用时: {elapsed:.1f}s | "
                          f"帧: {frame_idx}/{end_frame}")
                
                # 频率控制
                elapsed = time.time() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            total_time = time.time() - start_time
            print(f"\n  ✓ 回放完成! 总时长: {total_time:.2f}s")
            
        except KeyboardInterrupt:
            print(f"\n  ⚠ 回放被中断")
        
        return recorded_states
    
    def analyze_results(self, recorded_states: list):
        """分析回放结果"""
        print(f"\n[5] 分析回放结果...")
        
        if not recorded_states:
            print("  无记录数据")
            return
        
        # 计算跟踪误差
        joint_errors = []
        gripper_errors = []
        
        for state in recorded_states:
            target = state['target']
            actual_joints = state['actual_joints']
            actual_gripper = state['actual_gripper']
            
            joint_error = np.linalg.norm(actual_joints - target[:6])
            gripper_error = abs(actual_gripper - target[6])
            
            joint_errors.append(joint_error)
            gripper_errors.append(gripper_error)
        
        joint_errors = np.array(joint_errors)
        gripper_errors = np.array(gripper_errors)
        
        print(f"  关节跟踪误差:")
        print(f"    Mean: {np.mean(joint_errors):.4f} rad")
        print(f"    Max:  {np.max(joint_errors):.4f} rad")
        print(f"    Std:  {np.std(joint_errors):.4f} rad")
        
        print(f"  夹爪跟踪误差:")
        print(f"    Mean: {np.mean(gripper_errors):.4f} m")
        print(f"    Max:  {np.max(gripper_errors):.4f} m")
        
        return {
            'joint_error_mean': np.mean(joint_errors),
            'joint_error_max': np.max(joint_errors),
            'gripper_error_mean': np.mean(gripper_errors),
        }
    
    def cleanup(self):
        """清理"""
        print(f"\n[6] 清理...")
        if self.robot is not None:
            print("  设置阻尼模式...")
            self.robot.set_to_damping()
            print("  ✓ 清理完成")


def main():
    parser = argparse.ArgumentParser(description="机械臂回放测试")
    parser.add_argument("-d", "--demo",
                       default="~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5",
                       help="Demo 数据路径")
    parser.add_argument("--traj-idx", type=int, default=0, help="轨迹索引")
    parser.add_argument("-m", "--model", default="X5", help="机械臂型号")
    parser.add_argument("-i", "--interface", default="can0", help="CAN 接口")
    parser.add_argument("-s", "--speed", type=float, default=0.5,
                       help="速度缩放 (0.1-1.0, 默认 0.5)")
    parser.add_argument("--start", type=int, default=0, help="起始帧")
    parser.add_argument("--end", type=int, default=None, help="结束帧")
    parser.add_argument("--skip-move", action="store_true", help="跳过移动到起始位置")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 限制速度范围
    args.speed = np.clip(args.speed, 0.1, 1.0)
    
    print("=" * 60)
    print("机械臂回放测试")
    print("=" * 60)
    print(f"\n⚠ 安全提示:")
    print(f"  - 速度: {args.speed:.0%}")
    print(f"  - 确保机械臂周围无障碍物")
    print(f"  - 随时准备按 Ctrl+C 停止")
    
    # 创建测试器
    tester = RobotReplayTest(
        demo_path=args.demo,
        model=args.model,
        interface=args.interface,
        speed_scale=args.speed,
        verbose=args.verbose,
    )
    
    try:
        # 加载 demo
        tester.load_demo(traj_idx=args.traj_idx)
        
        # 连接机械臂
        tester.connect_robot()
        
        # 移动到起始位置
        if not args.skip_move:
            tester.move_to_start(duration=3.0)
        
        # 回放
        recorded_states = tester.replay(
            start_frame=args.start,
            end_frame=args.end,
        )
        
        # 分析结果
        metrics = tester.analyze_results(recorded_states)
        
    except KeyboardInterrupt:
        print("\n\n⚠ 测试被中断")
    
    except Exception as e:
        print(f"\n\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        tester.cleanup()
    
    print("\n" + "=" * 60)
    print("测试结束")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
