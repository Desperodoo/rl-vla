#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARX5 控制进程

作为"安全核心"独立运行，独占机器人控制权。
即使推理进程崩溃，控制进程也能安全停止机器人。

特性:
- 500Hz 高频控制循环
- SDK 后台通信模式 (enable_background_send_recv)
- 独立于推理进程的安全监控
- 命令超时检测
- 优雅的暂停/停止处理

用法:
    # 启动控制节点 (需先启动)
    python -m inference.control_node --model X5 --interface can0
    
    # 模拟模式 (无真实硬件)
    python -m inference.control_node --dry-run
"""

import os
import sys
import time
import signal
import argparse
import numpy as np
from typing import Optional
from dataclasses import dataclass
from enum import Enum

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.config import setup_arx5
from inference.shared_state import SharedState, ControlFlags, SharedMemoryLayout
from inference.trajectory_logger import TrajectoryLogger


@dataclass
class ControlConfig:
    """控制配置"""
    model: str = "X5"
    interface: str = "can0"
    urdf_path: str = None
    
    # 控制参数
    control_freq: float = 500.0
    # 重要：与原版推理/遥操作保持一致，使用同步通信模式
    # 后台通信模式可能导致命令执行时序不同
    use_background_send_recv: bool = False
    
    # 安全参数
    command_timeout: float = 0.2       # 命令超时 (秒)，超时后进入阻尼模式
    watchdog_timeout: float = 1.0      # 看门狗超时，推理进程无响应则停止
    max_joint_delta: float = 0.1       # 每步最大关节变化 (rad) - 与原版一致
    max_gripper_delta: float = 0.02    # 每步最大夹爪变化
    
    # EMA 滤波 (在控制循环中应用)
    enable_filter: bool = True
    filter_alpha: float = 0.3          # EMA 系数 (越低越平滑)
    
    # 关节限位
    joint_limits_low: tuple = (-3.14, -3.14, -3.14, -3.14, -3.14, -3.14)
    joint_limits_high: tuple = (3.14, 3.14, 3.14, 3.14, 3.14, 3.14)
    gripper_limits: tuple = (0.0, 0.08)
    
    # Gain 设置
    kd_scale: float = 0.01  # 默认 kd 缩放
    
    # 重力补偿 - 默认禁用以与原版推理/遥操作保持一致
    # 遥操作采集数据时没有启用重力补偿
    enable_gravity_compensation: bool = False
    
    # 夹爪初始状态
    # 训练数据中初始夹爪位置约为 0.074m (张开)，而 reset_to_home() 会将夹爪设为 0 (闭合)
    # 需要在 reset_to_home() 后将夹爪移到与训练数据一致的位置
    initial_gripper_pos: float = 0.074  # 与训练数据平均初始值一致


class ActionFilter:
    """EMA 低通滤波器 - 在 500Hz 控制循环中应用"""
    
    def __init__(self, action_dim: int = 7, alpha: float = 0.3):
        self.action_dim = action_dim
        self.alpha = alpha
        self.prev_action: Optional[np.ndarray] = None
        
    def reset(self, initial_action: Optional[np.ndarray] = None):
        self.prev_action = initial_action.copy() if initial_action is not None else None
        
    def filter(self, action: np.ndarray) -> np.ndarray:
        """应用 EMA 滤波: filtered = alpha * current + (1-alpha) * previous"""
        if self.prev_action is None:
            self.prev_action = action.copy()
            return action
        
        filtered = self.alpha * action + (1 - self.alpha) * self.prev_action
        self.prev_action = filtered.copy()
        return filtered


class ControlNode:
    """
    ARX5 控制节点
    
    作为独立进程运行，负责:
    1. 与机器人硬件通信 (500Hz)
    2. 执行动作序列
    3. 安全监控和紧急停止
    4. 与推理进程通过共享内存通信
    """
    
    SHM_NAME = "arx5_control"
    
    def __init__(self, config: ControlConfig, dry_run: bool = False, verbose: bool = True):
        self.config = config
        self.dry_run = dry_run
        self.verbose = verbose
        
        # 硬件
        self.robot_ctrl = None
        self.robot_cfg = None
        self.ctrl_cfg = None
        self.controller_dt = 1.0 / config.control_freq
        
        # 共享内存
        self.shared_state: Optional[SharedState] = None
        
        # EMA 滤波器 (关键: 在控制循环中应用，而不是推理时)
        self.action_filter = ActionFilter(
            action_dim=7,
            alpha=config.filter_alpha if config.enable_filter else 1.0
        )
        
        # 状态
        self._running = False
        self._current_target: Optional[np.ndarray] = None
        self._last_action_time: float = 0.0
        self._step_count: int = 0
        
        # 轨迹记录器 (用于对比分析)
        self.trajectory_logger: Optional[TrajectoryLogger] = None
        self._log_trajectory: bool = False
        self._trajectory_output_path: str = "trajectory_multi_process_control.npz"
        
        # 用于记录的临时变量
        self._current_chunk_id: int = 0
        self._current_buffer_idx: int = 0
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理"""
        print(f"\n[控制节点] 收到信号 {signum}，安全退出...")
        self._running = False
    
    def enable_trajectory_logging(self, output_path: str = "trajectory_multi_process_control.npz"):
        """启用轨迹记录功能"""
        self._log_trajectory = True
        self._trajectory_output_path = output_path
        self.trajectory_logger = TrajectoryLogger()
        print(f"[控制节点] 轨迹记录已启用，输出: {output_path}")
    
    def setup(self):
        """初始化"""
        print("\n" + "=" * 60)
        print("ARX5 控制节点")
        print("=" * 60)
        
        # 1. 创建共享内存
        print("\n[1/3] 创建共享内存...")
        self.shared_state = SharedState.create(self.SHM_NAME)
        print(f"  共享内存名称: {self.SHM_NAME}")
        print(f"  大小: {SharedMemoryLayout.TOTAL_SIZE} bytes")
        
        # 2. 初始化机器人
        if not self.dry_run:
            print("\n[2/3] 初始化机器人...")
            self._setup_robot()
        else:
            print("\n[2/3] 模拟模式 - 跳过机器人初始化")
            self.controller_dt = 1.0 / self.config.control_freq
        
        # 3. 复位到 Home 并设置夹爪初始位置
        if not self.dry_run and self.robot_ctrl is not None:
            print("\n[3/3] 复位到 Home 位置...")
            self.robot_ctrl.reset_to_home()
            time.sleep(0.5)
            
            # 重要：reset_to_home() 将夹爪设为 0 (闭合)，但训练数据中夹爪是张开的
            # 需要将夹爪移动到与训练数据一致的初始位置
            print(f"  将夹爪移动到初始位置 ({self.config.initial_gripper_pos:.3f}m)...")
            self._move_gripper_to_position(self.config.initial_gripper_pos)
            time.sleep(0.2)
            
            # 读取初始状态
            state = self.robot_ctrl.get_state()
            self._current_target = np.concatenate([
                np.array(state.pos()),
                [state.gripper_pos]
            ])
            
            # 写入初始状态到共享内存
            self.shared_state.write_robot_state(
                joint_pos=np.array(state.pos()),
                gripper_pos=state.gripper_pos,
                joint_vel=np.array(state.vel()),
            )
            self.shared_state.write_target_pose(self._current_target)
            
            print(f"  初始关节位置: {self._current_target[:6]}")
            print(f"  初始夹爪位置: {self._current_target[6]:.3f}m (目标: {self.config.initial_gripper_pos:.3f}m)")
        else:
            self._current_target = np.zeros(7, dtype=np.float64)
            self._current_target[6] = self.config.initial_gripper_pos
            self.shared_state.write_robot_state(
                joint_pos=np.zeros(6),
                gripper_pos=self.config.initial_gripper_pos,
            )
        
        print("\n✓ 控制节点初始化完成")
        print(f"  控制频率: {1.0/self.controller_dt:.1f} Hz")
        print(f"  后台通信: {self.config.use_background_send_recv}")
        print(f"  命令超时: {self.config.command_timeout}s")
    
    def _setup_robot(self):
        """初始化机器人硬件"""
        setup_arx5()
        import arx5_interface as arx5
        
        self.robot_cfg = arx5.RobotConfigFactory.get_instance().get_config(self.config.model)
        self.ctrl_cfg = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", self.robot_cfg.joint_dof
        )
        
        self.robot_ctrl = arx5.Arx5JointController(
            self.robot_cfg, self.ctrl_cfg, self.config.interface
        )
        
        # 设置 PID 增益
        gain = arx5.Gain(self.robot_cfg.joint_dof)
        gain.kd()[:] = self.config.kd_scale
        self.robot_ctrl.set_gain(gain)
        
        # 启用后台通信
        if self.config.use_background_send_recv:
            self.robot_ctrl.enable_background_send_recv()
            print("  已启用后台通信模式")
        
        # 重力补偿 - 默认禁用以与原版保持一致
        if self.config.enable_gravity_compensation and self.config.urdf_path:
            self.robot_ctrl.enable_gravity_compensation(self.config.urdf_path)
            print(f"  已启用重力补偿: {self.config.urdf_path}")
        else:
            print("  重力补偿: 禁用 (与原版推理一致)")
        
        self.controller_dt = float(self.ctrl_cfg.controller_dt)
        
        print(f"  机器人型号: {self.config.model}")
        print(f"  自由度: {self.robot_cfg.joint_dof}")
        print(f"  控制周期: {self.controller_dt*1000:.1f}ms")
        print(f"  通信模式: {'后台异步' if self.config.use_background_send_recv else '同步 send_recv_once'}")
    
    def _move_gripper_to_position(self, target_gripper_pos: float, duration: float = 0.5):
        """
        平滑地将夹爪移动到目标位置
        
        这是必要的，因为 reset_to_home() 会将夹爪设为 0 (闭合)，
        但训练数据中夹爪初始状态是张开的。
        
        Args:
            target_gripper_pos: 目标夹爪位置 (m)
            duration: 移动持续时间 (秒)
        """
        if self.robot_ctrl is None:
            return
        
        setup_arx5()
        import arx5_interface as arx5
        
        # 获取当前状态
        state = self.robot_ctrl.get_state()
        init_gripper = state.gripper_pos
        init_joints = np.array(state.pos())
        
        # 计算步数
        step_num = int(duration / self.controller_dt)
        if step_num < 1:
            step_num = 1
        
        # 平滑插值移动夹爪
        for i in range(step_num + 1):
            alpha = float(i) / float(step_num)
            current_gripper = init_gripper + alpha * (target_gripper_pos - init_gripper)
            
            js = arx5.JointState(self.robot_cfg.joint_dof)
            js.pos()[:] = init_joints  # 保持关节位置不变
            js.gripper_pos = current_gripper
            
            self.robot_ctrl.set_joint_cmd(js)
            
            # 与原版一致：使用同步 send_recv_once
            if not self.config.use_background_send_recv:
                self.robot_ctrl.send_recv_once()
            
            time.sleep(self.controller_dt)
    
    def _apply_initial_pose(self):
        """
        应用从数据集加载的初始位姿
        
        从共享内存读取初始位姿，平滑地将机器人移动到目标位置。
        应用后清除初始位姿标记。
        """
        initial_pose = self.shared_state.read_initial_pose()
        if initial_pose is None:
            return
        
        print("\n[控制] 应用初始位姿 (从数据集加载)...")
        print(f"  目标关节: {initial_pose[:3]}...")
        print(f"  目标夹爪: {initial_pose[6]:.4f}m")
        
        if self.robot_ctrl is None:
            # 模拟模式
            self._current_target = initial_pose.copy()
            self.shared_state.write_target_pose(initial_pose)
            self.shared_state.clear_initial_pose()
            print("  (模拟模式) 已设置初始位姿")
            return
        
        setup_arx5()
        import arx5_interface as arx5
        
        # 获取当前状态
        state = self.robot_ctrl.get_state()
        init_joints = np.array(state.pos())
        init_gripper = state.gripper_pos
        
        target_joints = initial_pose[:6]
        target_gripper = initial_pose[6]
        
        # 平滑移动 (2 秒)
        duration = 2.0
        step_num = int(duration / self.controller_dt)
        if step_num < 1:
            step_num = 1
        
        print(f"  移动到初始位姿 (约 {duration} 秒)...")
        
        for i in range(step_num + 1):
            alpha = float(i) / float(step_num)
            # 使用 smoothstep 插值让运动更平滑
            alpha = alpha * alpha * (3 - 2 * alpha)  # smoothstep
            
            current_joints = init_joints + alpha * (target_joints - init_joints)
            current_gripper = init_gripper + alpha * (target_gripper - init_gripper)
            
            js = arx5.JointState(self.robot_cfg.joint_dof)
            js.pos()[:] = current_joints
            js.gripper_pos = current_gripper
            
            self.robot_ctrl.set_joint_cmd(js)
            
            if not self.config.use_background_send_recv:
                self.robot_ctrl.send_recv_once()
            
            time.sleep(self.controller_dt)
        
        # 更新内部状态
        self._current_target = initial_pose.copy()
        self.shared_state.write_target_pose(initial_pose)
        
        # 重置 EMA 滤波器以新位置开始
        if self.config.enable_filter and self.action_filter is not None:
            self.action_filter.reset(initial_pose)
        
        # 清除标记
        self.shared_state.clear_initial_pose()
        
        print("  初始位姿已应用!")
        
        # 读取并验证最终状态
        state = self.robot_ctrl.get_state()
        final_joints = np.array(state.pos())
        final_gripper = state.gripper_pos
        joint_error = np.abs(final_joints - target_joints).max()
        gripper_error = abs(final_gripper - target_gripper)
        print(f"  关节误差: {joint_error:.4f} rad")
        print(f"  夹爪误差: {gripper_error:.4f} m")
    
    def run(self):
        """主控制循环"""
        print("\n" + "=" * 60)
        print("控制循环启动")
        print("=" * 60)
        print("  等待推理进程连接...")
        print("  按 Ctrl+C 安全退出")
        
        self._running = True
        self._last_action_time = time.time()
        loop_count = 0
        
        try:
            while self._running:
                loop_start = time.time()
                
                # 处理请求
                self._process_requests()
                
                # 根据状态执行
                current_state = self.shared_state.get_state()
                
                if current_state == ControlFlags.RUNNING:
                    self._execute_action()
                elif current_state == ControlFlags.PAUSED:
                    self._hold_position()
                elif current_state == ControlFlags.RESETTING:
                    self._do_reset()
                elif current_state == ControlFlags.EMERGENCY_STOP:
                    self._do_emergency_stop()
                    break
                elif current_state == ControlFlags.DAMPING:
                    # 已在阻尼模式，只需更新状态
                    pass
                else:  # IDLE
                    self._hold_position()
                
                # 更新机器人状态到共享内存
                self._update_robot_state()
                
                # 看门狗检查
                if current_state == ControlFlags.RUNNING:
                    self._check_watchdog()
                
                # 频率控制
                loop_count += 1
                elapsed = time.time() - loop_start
                sleep_time = self.controller_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # 统计
                if loop_count % 5000 == 0 and self.verbose:
                    actual_freq = 1.0 / (time.time() - loop_start) if elapsed > 0 else 0
                    print(f"[控制] 步数: {self._step_count}, 频率: {actual_freq:.1f}Hz")
        
        except Exception as e:
            print(f"\n[控制节点] 异常: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._cleanup()
    
    def _process_requests(self):
        """处理来自推理进程的请求"""
        flags = self.shared_state.get_flags()
        
        # 检查是否有初始位姿需要应用 (优先处理)
        if self.shared_state.has_initial_pose():
            self._apply_initial_pose()
        
        # START 请求
        if flags & ControlFlags.REQUEST_START:
            if self.verbose:
                print("\n[控制] 收到 START 请求")
            self._last_action_time = time.time()
            self.shared_state.set_state(ControlFlags.RUNNING)
            self.shared_state.ack_start()
        
        # PAUSE 请求
        if flags & ControlFlags.REQUEST_PAUSE:
            if self.verbose:
                print("\n[控制] 收到 PAUSE 请求")
            # 保存当前目标位置
            if self._current_target is not None:
                self.shared_state.write_target_pose(self._current_target)
            self.shared_state.set_state(ControlFlags.PAUSED)
            self.shared_state.ack_pause()
        
        # RESET 请求
        if flags & ControlFlags.REQUEST_RESET:
            if self.verbose:
                print("\n[控制] 收到 RESET 请求")
            self.shared_state.set_state(ControlFlags.RESETTING)
            self.shared_state.ack_reset()
        
        # STOP 请求
        if flags & ControlFlags.REQUEST_STOP:
            if self.verbose:
                print("\n[控制] 收到 STOP 请求")
            self.shared_state.set_state(ControlFlags.EMERGENCY_STOP)
            self.shared_state.ack_stop()
    
    def _execute_action(self):
        """执行动作"""
        # 从共享内存获取下一个动作
        action, is_last = self.shared_state.get_next_action()
        
        if action is not None:
            # 保存原始动作用于记录
            raw_action = action.copy()
            
            # 获取缓冲区状态用于记录
            buf_idx, buf_size = self.shared_state.get_action_buffer_status()
            self._current_buffer_idx = buf_idx
            
            # 检测动作跳变 (抖动诊断)
            if self._current_target is not None:
                raw_delta = action - self._current_target
                max_joint_delta = np.abs(raw_delta[:6]).max()
                gripper_delta = np.abs(raw_delta[6])
                
                # 如果跳变太大，记录警告
                if max_joint_delta > 0.05 or gripper_delta > 0.01:
                    print(f"[控制] ⚠ 动作跳变! Step {self._step_count}:")
                    print(f"  关节最大跳变: {max_joint_delta:.4f} rad")
                    print(f"  夹爪跳变: {gripper_delta:.4f} m")
                    print(f"  当前目标: {self._current_target[:3]}")
                    print(f"  新动作: {action[:3]}")
            
            # 1. 先应用 EMA 滤波 (关键: 与原版一致，在 500Hz 循环中滤波)
            if self.config.enable_filter:
                filtered_action = self.action_filter.filter(action)
            else:
                filtered_action = action.copy()
            
            # 2. 再应用安全限制
            safe_action = self._apply_safety_limits(filtered_action)
            
            # Debug: 每 100 步打印一次
            if self.verbose and self._step_count % 100 == 0:
                print(f"[控制] Step {self._step_count}: pos={safe_action[:3]}, "
                      f"gripper={safe_action[6]:.4f}, buf={buf_idx}/{buf_size}")
            
            # 发送命令
            self._send_joint_command(safe_action)
            
            # 轨迹记录
            if self._log_trajectory and self.trajectory_logger is not None:
                # 获取当前机器人状态
                robot_state = np.zeros(7)
                if self.robot_ctrl is not None:
                    state = self.robot_ctrl.get_state()
                    robot_state = np.concatenate([
                        np.array(state.pos()),
                        [state.gripper_pos]
                    ])
                
                self.trajectory_logger.log_control_step(
                    step=self._step_count,
                    buffer_idx=self._current_buffer_idx,
                    raw_action=raw_action,
                    filtered_action=filtered_action,
                    executed_action=safe_action,
                    robot_state=robot_state,
                )
            
            # 更新状态
            self._current_target = safe_action.copy()
            self._last_action_time = time.time()
            self._step_count += 1
            
            # 关键：定期更新 target_pose 到共享内存，供推理进程读取
            # 用于新 chunk 的平滑过渡
            if self._step_count % 10 == 0:  # 每 10 步更新一次 (50Hz)
                self.shared_state.write_target_pose(self._current_target)
        else:
            # 动作缓冲区空，保持当前位置
            self._hold_position()
    
    def _hold_position(self):
        """保持当前位置"""
        if self._current_target is not None:
            self._send_joint_command(self._current_target)
    
    def _do_reset(self):
        """执行复位"""
        if self.robot_ctrl is not None:
            print("[控制] 执行复位...")
            self.robot_ctrl.reset_to_home()
            time.sleep(0.5)
            
            # 重要：reset_to_home() 将夹爪设为 0 (闭合)，但训练数据中夹爪是张开的
            # 需要将夹爪移动到与训练数据一致的初始位置
            print(f"[控制] 将夹爪移动到初始位置 ({self.config.initial_gripper_pos:.3f}m)...")
            self._move_gripper_to_position(self.config.initial_gripper_pos)
            time.sleep(0.2)
            
            # 更新当前目标
            state = self.robot_ctrl.get_state()
            self._current_target = np.concatenate([
                np.array(state.pos()),
                [state.gripper_pos]
            ])
        else:
            self._current_target = np.zeros(7, dtype=np.float64)
            self._current_target[6] = self.config.initial_gripper_pos
        
        # 重置滤波器状态
        self.action_filter.reset(self._current_target)
        
        self.shared_state.set_state(ControlFlags.IDLE)
        print("[控制] 复位完成，进入 IDLE 状态")
    
    def _do_emergency_stop(self):
        """紧急停止"""
        print("[控制] 执行紧急停止...")
        
        if self.robot_ctrl is not None:
            # 先保持当前位置
            self._hold_position()
            time.sleep(0.1)
            
            # 进入阻尼模式
            self.robot_ctrl.set_to_damping()
            print("[控制] 已进入阻尼模式")
        
        self.shared_state.set_state(ControlFlags.DAMPING)
    
    def _send_joint_command(self, action: np.ndarray):
        """发送关节命令"""
        if self.robot_ctrl is None:
            return
        
        setup_arx5()
        import arx5_interface as arx5
        
        js = arx5.JointState(self.robot_cfg.joint_dof)
        js.pos()[:] = action[:6]
        js.gripper_pos = float(action[6])
        
        self.robot_ctrl.set_joint_cmd(js)
        
        # 关键：与原版推理一致，使用同步 send_recv_once()
        # 这确保命令立即执行，而不是由后台线程异步处理
        if not self.config.use_background_send_recv:
            self.robot_ctrl.send_recv_once()
    
    def _apply_safety_limits(self, action: np.ndarray) -> np.ndarray:
        """应用安全限制"""
        safe_action = action.copy()
        
        if self._current_target is not None:
            # 限制关节变化速率
            joint_delta = safe_action[:6] - self._current_target[:6]
            joint_delta = np.clip(joint_delta, 
                                 -self.config.max_joint_delta,
                                 self.config.max_joint_delta)
            safe_action[:6] = self._current_target[:6] + joint_delta
            
            # 限制夹爪变化速率
            gripper_delta = safe_action[6] - self._current_target[6]
            gripper_delta = np.clip(gripper_delta,
                                   -self.config.max_gripper_delta,
                                   self.config.max_gripper_delta)
            safe_action[6] = self._current_target[6] + gripper_delta
        
        # 关节限位
        safe_action[:6] = np.clip(
            safe_action[:6],
            self.config.joint_limits_low,
            self.config.joint_limits_high
        )
        
        # 夹爪限位
        safe_action[6] = np.clip(
            safe_action[6],
            self.config.gripper_limits[0],
            self.config.gripper_limits[1]
        )
        
        return safe_action
    
    def _update_robot_state(self):
        """更新机器人状态到共享内存"""
        if self.robot_ctrl is not None:
            state = self.robot_ctrl.get_state()
            self.shared_state.write_robot_state(
                joint_pos=np.array(state.pos()),
                gripper_pos=state.gripper_pos,
                joint_vel=np.array(state.vel()),
            )
        
        self.shared_state.update_control_timestamp()
    
    def _check_watchdog(self):
        """看门狗检查"""
        # 检查推理进程是否活跃
        inference_ts = self.shared_state.get_inference_timestamp()
        if inference_ts > 0:
            elapsed = time.time() - inference_ts
            if elapsed > self.config.watchdog_timeout:
                print(f"\n[警告] 推理进程无响应 ({elapsed:.1f}s)，进入暂停状态")
                self.shared_state.set_state(ControlFlags.PAUSED)
                self.shared_state.set_error(ControlFlags.ERROR_TIMEOUT)
        
        # 检查命令超时
        action_elapsed = time.time() - self._last_action_time
        if action_elapsed > self.config.command_timeout:
            # 动作缓冲区可能空了，检查一下
            if self.shared_state.is_action_buffer_empty():
                # 正常情况，等待新动作
                pass
            else:
                # 异常：有动作但超时了
                print(f"\n[警告] 命令执行超时 ({action_elapsed:.1f}s)")
    
    def _cleanup(self):
        """清理资源"""
        print("\n[控制节点] 清理资源...")
        
        # 保存轨迹记录
        if self._log_trajectory and self.trajectory_logger is not None:
            self.trajectory_logger.stop_logging()
            self.trajectory_logger.save(self._trajectory_output_path)
            print(f"  轨迹已保存到: {self._trajectory_output_path}")
        
        # 确保机器人安全
        if self.robot_ctrl is not None:
            try:
                # 先回到 home 位置（包括闭合夹爪）
                print("  机器人正在回到 home 位置...")
                self.robot_ctrl.reset_to_home()
                time.sleep(0.5)
                print("  ✓ 已回到 home 位置，夹爪已闭合")
            except Exception as e:
                print(f"  警告: 无法回到 home 位置: {e}")
                # 失败时至少进入阻尼模式
                try:
                    self.robot_ctrl.set_to_damping()
                    print("  已进入阻尼模式")
                except:
                    pass
        
        # 关闭共享内存
        if self.shared_state is not None:
            self.shared_state.close()
            print("  共享内存已关闭")
        
        print("[控制节点] 退出")


def main():
    parser = argparse.ArgumentParser(
        description="ARX5 控制节点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
控制模式说明:
    默认使用同步通信模式 (send_recv_once)，与原版推理/遥操作保持一致。
    默认禁用重力补偿，与数据采集时的配置一致。
    
示例:
    # 默认配置 (与原版一致)
    python -m inference.control_node --model X5 --interface can0
    
    # 启用后台通信 (可能有时序差异)
    python -m inference.control_node --background
    
    # 启用重力补偿
    python -m inference.control_node --gravity-comp
"""
    )
    
    parser.add_argument("--model", "-m", type=str, default="X5",
                       help="机器人型号 (X5/L5)")
    parser.add_argument("--interface", "-i", type=str, default="can0",
                       help="CAN 接口名称")
    parser.add_argument("--urdf", type=str, default=None,
                       help="URDF 文件路径 (用于重力补偿)")
    parser.add_argument("--dry-run", action="store_true",
                       help="模拟模式，不连接真实硬件")
    parser.add_argument("--background", action="store_true",
                       help="启用后台通信模式 (默认禁用，与原版一致)")
    parser.add_argument("--gravity-comp", action="store_true",
                       help="启用重力补偿 (默认禁用，与原版一致)")
    parser.add_argument("--freq", type=float, default=500.0,
                       help="控制频率 (Hz)")
    parser.add_argument("--no-filter", action="store_true",
                       help="禁用 EMA 滤波")
    parser.add_argument("--filter-alpha", type=float, default=0.3,
                       help="EMA 滤波系数 (越低越平滑)")
    parser.add_argument("--gripper-init", type=float, default=0.074,
                       help="夹爪初始位置 (m)，默认 0.074 与训练数据一致")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="减少输出")
    
    # 轨迹记录 (用于对比分析)
    parser.add_argument("--log-trajectory", action="store_true",
                       help="启用轨迹记录 (用于对比分析)")
    parser.add_argument("--trajectory-output", type=str, default="trajectory_multi_process_control.npz",
                       help="轨迹记录输出文件 (默认: trajectory_multi_process_control.npz)")
    
    args = parser.parse_args()
    
    # 设置 URDF 路径
    urdf_path = args.urdf
    if urdf_path is None:
        # 默认 URDF 路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        urdf_path = os.path.join(project_root, "arx5-sdk", "models", "arx5.urdf")
        if not os.path.exists(urdf_path):
            urdf_path = None
    
    config = ControlConfig(
        model=args.model,
        interface=args.interface,
        urdf_path=urdf_path,
        control_freq=args.freq,
        use_background_send_recv=args.background,  # 默认 False
        enable_gravity_compensation=args.gravity_comp,  # 默认 False
        enable_filter=not args.no_filter,
        filter_alpha=args.filter_alpha,
        initial_gripper_pos=args.gripper_init,
    )
    
    node = ControlNode(
        config=config,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )
    
    try:
        node.setup()
        
        # 启用轨迹记录
        if args.log_trajectory:
            node.enable_trajectory_logging(args.trajectory_output)
            if node.trajectory_logger:
                node.trajectory_logger.start_logging()
        
        node.run()
    except KeyboardInterrupt:
        print("\n[控制节点] 被用户中断")
    except Exception as e:
        print(f"\n[控制节点] 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
