#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂回放测试 (多进程版本)

验证控制链路完整性（需要真机）:
1. 从 demo 读取 actions 序列
2. 使用 Arx5JointControllerProcess (200Hz) 回放
3. 使用 schedule_waypoint() + update_trajectory() 模式（与实际部署一致）
4. 支持速度缩放（默认 50% 速度）
5. 比较实际状态 vs 期望状态

用法:
    python -m consistency_policy.tests.test_robot_replay \
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
from typing import Optional, List, Dict

# 添加项目路径
CONSISTENCY_POLICY_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_VLA_PATH = os.path.dirname(CONSISTENCY_POLICY_PATH)
if RL_VLA_PATH not in sys.path:
    sys.path.insert(0, RL_VLA_PATH)

from consistency_policy.config import setup_rlft
from consistency_policy.robot_controller_mp import Arx5JointControllerManager


class RobotReplayTest:
    """机械臂回放测试器 (多进程版本)"""
    
    def __init__(
        self,
        demo_path: str,
        model: str = "X5",
        interface: str = "can0",
        control_frequency: float = 200.0,
        speed_scale: float = 0.5,
        verbose: bool = True,
    ):
        self.demo_path = os.path.expanduser(demo_path)
        self.model = model
        self.interface = interface
        self.control_frequency = control_frequency
        self.speed_scale = speed_scale
        self.verbose = verbose
        
        self.controller_manager: Optional[Arx5JointControllerManager] = None
        self.controller = None  # Arx5JointControllerProcess
        self.actions: Optional[np.ndarray] = None
        self.obs: Optional[dict] = None
        self.T = 0
        
    def load_demo(self, traj_idx: int = 0) -> np.ndarray:
        """
        加载 demo 数据
        
        Args:
            traj_idx: 轨迹索引
            
        Returns:
            actions: (T, action_dim) 动作序列
        """
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
    
    def connect_robot(self) -> bool:
        """
        连接机械臂 (多进程控制器)
        
        Returns:
            bool: 连接是否成功
        """
        print(f"\n[2] 连接机械臂 (多进程模式, {self.control_frequency}Hz)...")
        
        self.controller_manager = Arx5JointControllerManager(
            model=self.model,
            interface=self.interface,
            frequency=self.control_frequency,
            verbose=self.verbose,
        )
        self.controller_manager.start()
        self.controller = self.controller_manager.controller
        
        # 获取初始状态
        state = self.controller.get_state()
        print(f"  当前关节位置: {state['joint_pos']}")
        print(f"  当前夹爪位置: {state['gripper_pos']:.4f}")
        print(f"  ✓ 机械臂连接成功 (PID: {self.controller.pid})")
        
        return True
    
    def move_to_start(self, duration: float = 3.0) -> bool:
        """
        平滑移动到起始位置
        
        使用 servoL 实现平滑移动到第一帧位置
        
        Args:
            duration: 移动时间 s
            
        Returns:
            bool: 是否成功到达
        """
        print(f"\n[3] 移动到起始位置...")
        
        # 获取起始动作（第一帧）
        start_action = self.actions[0]
        target_joints = start_action[:6]
        target_gripper = float(start_action[6])
        
        print(f"  目标关节: {target_joints}")
        print(f"  目标夹爪: {target_gripper:.4f}")
        print(f"  移动时长: {duration:.1f}s")
        
        # 使用 servoL 平滑移动
        self.controller.servoL(
            joint_pos=target_joints,
            gripper_pos=target_gripper,
            duration=duration,
        )
        
        # 等待移动完成
        time.sleep(duration + 0.5)
        
        # 验证位置
        state = self.controller.get_state()
        error = np.linalg.norm(state['joint_pos'] - target_joints)
        print(f"  到达位置: {state['joint_pos']}")
        print(f"  位置误差: {error:.4f} rad")
        
        return error < 0.1
    
    def replay(
        self, 
        start_frame: int = 0, 
        end_frame: Optional[int] = None,
        batch_size: int = 16,  # 每批航点数量 (类似 action_horizon)
    ) -> List[Dict]:
        """
        回放轨迹
        
        使用 add_waypoint() + update_trajectory() 模式，与实际部署一致。
        
        Args:
            start_frame: 起始帧
            end_frame: 结束帧 (None 表示到末尾)
            batch_size: 每批航点数量
            
        Returns:
            recorded_states: 记录的状态列表
        """
        if end_frame is None:
            end_frame = self.T
        
        replay_frames = end_frame - start_frame
        original_duration = replay_frames / 30.0  # 30Hz 采集
        actual_duration = original_duration / self.speed_scale
        
        print(f"\n[4] 开始回放轨迹...")
        print(f"  帧范围: [{start_frame}, {end_frame})")
        print(f"  原始时长: {original_duration:.2f}s (30Hz)")
        print(f"  速度缩放: {self.speed_scale:.0%}")
        print(f"  实际时长: {actual_duration:.2f}s")
        print(f"  批量大小: {batch_size} (类似 action_horizon)")
        print(f"  使用模式: add_waypoint() + update_trajectory()")
        
        input("\n按 Enter 开始回放，Ctrl+C 取消...")
        
        # 记录状态
        recorded_states = []
        
        # 计算时间间隔 (原始 30Hz 采集，根据速度缩放调整)
        dt = (1.0 / 30.0) / self.speed_scale
        
        try:
            start_time = time.time()
            current_frame = start_frame
            
            while current_frame < end_frame:
                batch_start_time = time.time()
                
                # 确定本批次的帧范围
                batch_end = min(current_frame + batch_size, end_frame)
                batch_frames = batch_end - current_frame
                
                # 添加航点到缓冲区
                for i, frame_idx in enumerate(range(current_frame, batch_end)):
                    action = self.actions[frame_idx]
                    target_joints = action[:6]
                    target_gripper = float(action[6])
                    
                    # 计算目标时间 (相对于当前时间)
                    target_time = batch_start_time + (i + 1) * dt
                    
                    self.controller.add_waypoint(
                        joint_pos=target_joints,
                        gripper_pos=target_gripper,
                        target_time=target_time,
                    )
                
                # 触发轨迹更新
                self.controller.update_trajectory()
                
                # 等待本批次执行 (稍微多等一点确保完成)
                batch_duration = batch_frames * dt
                time.sleep(batch_duration * 0.9)  # 90% 时间后开始采样状态
                
                # 记录当前状态
                state = self.controller.get_state()
                recorded_states.append({
                    'batch_start_frame': current_frame,
                    'batch_end_frame': batch_end,
                    'target': self.actions[batch_end - 1].copy(),  # 最后一帧的目标
                    'actual_joints': state['joint_pos'].copy(),
                    'actual_gripper': state['gripper_pos'],
                    'timestamp': state['timestamp'],
                })
                
                # 等待剩余时间
                time.sleep(batch_duration * 0.15)
                
                # 打印进度
                if self.verbose:
                    progress = (batch_end - start_frame) / replay_frames * 100
                    elapsed = time.time() - start_time
                    print(f"  进度: {progress:.1f}% | 已用时: {elapsed:.1f}s | "
                          f"帧: {batch_end}/{end_frame}")
                
                current_frame = batch_end
            
            total_time = time.time() - start_time
            print(f"\n  ✓ 回放完成! 总时长: {total_time:.2f}s")
            
        except KeyboardInterrupt:
            print(f"\n  ⚠ 回放被中断")
        
        return recorded_states
    
    def analyze_results(self, recorded_states: List[Dict]) -> Dict[str, float]:
        """
        分析回放结果
        
        Args:
            recorded_states: 记录的状态列表
            
        Returns:
            metrics: 评估指标
        """
        print(f"\n[5] 分析回放结果...")
        
        if not recorded_states:
            print("  无记录数据")
            return {}
        
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
        
        metrics = {
            'joint_error_mean': float(np.mean(joint_errors)),
            'joint_error_max': float(np.max(joint_errors)),
            'joint_error_std': float(np.std(joint_errors)),
            'gripper_error_mean': float(np.mean(gripper_errors)),
            'gripper_error_max': float(np.max(gripper_errors)),
            'num_batches': len(recorded_states),
        }
        
        print(f"  采样批次数: {metrics['num_batches']}")
        print(f"  关节跟踪误差:")
        print(f"    Mean: {metrics['joint_error_mean']:.4f} rad")
        print(f"    Max:  {metrics['joint_error_max']:.4f} rad")
        print(f"    Std:  {metrics['joint_error_std']:.4f} rad")
        print(f"  夹爪跟踪误差:")
        print(f"    Mean: {metrics['gripper_error_mean']:.4f} m")
        print(f"    Max:  {metrics['gripper_error_max']:.4f} m")
        
        return metrics
    
    def cleanup(self):
        """清理资源"""
        print(f"\n[6] 清理...")
        if self.controller_manager is not None:
            self.controller_manager.stop()
            self.controller_manager = None
            self.controller = None
        print("  ✓ 清理完成")


def main():
    parser = argparse.ArgumentParser(description="机械臂回放测试 (多进程版本)")
    parser.add_argument("-d", "--demo",
                       default="~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5",
                       help="Demo 数据路径")
    parser.add_argument("--traj-idx", type=int, default=0, help="轨迹索引")
    parser.add_argument("-m", "--model", default="X5", help="机械臂型号")
    parser.add_argument("-i", "--interface", default="can0", help="CAN 接口")
    parser.add_argument("-f", "--frequency", type=float, default=200.0,
                       help="控制器频率 Hz (默认 200)")
    parser.add_argument("-s", "--speed", type=float, default=0.5,
                       help="速度缩放 (0.1-1.0, 默认 0.5)")
    parser.add_argument("-b", "--batch-size", type=int, default=16,
                       help="每批航点数量 (类似 action_horizon, 默认 16)")
    parser.add_argument("--start", type=int, default=0, help="起始帧")
    parser.add_argument("--end", type=int, default=None, help="结束帧")
    parser.add_argument("--skip-move", action="store_true", help="跳过移动到起始位置")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 限制速度范围
    args.speed = np.clip(args.speed, 0.1, 1.0)
    
    print("=" * 60)
    print("机械臂回放测试 (多进程版本)")
    print("=" * 60)
    print(f"\n⚠ 安全提示:")
    print(f"  - 速度: {args.speed:.0%}")
    print(f"  - 控制频率: {args.frequency} Hz")
    print(f"  - 确保机械臂周围无障碍物")
    print(f"  - 随时准备按 Ctrl+C 停止")
    
    # 创建测试器
    tester = RobotReplayTest(
        demo_path=args.demo,
        model=args.model,
        interface=args.interface,
        control_frequency=args.frequency,
        speed_scale=args.speed,
        verbose=args.verbose or True,
    )
    
    metrics = {}
    
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
            batch_size=args.batch_size,
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
    if metrics and metrics.get('joint_error_mean', float('inf')) < 0.1:
        print("✓ 测试通过! 平均关节误差 < 0.1 rad")
    elif metrics:
        print("⚠ 测试完成，但误差较大，请检查")
    else:
        print("测试未完成")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
