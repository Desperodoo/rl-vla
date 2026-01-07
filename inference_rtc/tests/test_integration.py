#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成测试: 使用 demo 数据验证完整控制链路

双进程架构:
  FakeInferenceProcess (10Hz) → 共享内存 → ServoProcess (500Hz) → 机器人

用法:
    # 模拟模式 (无真机, 验证通信和时序)
    python -m inference_rtc.tests.test_integration \
        -d ~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5 \
        --dry-run --speed 1.0
    
    # 真机模式 (低速)
    python -m inference_rtc.tests.test_integration \
        -d ~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5 \
        --speed 0.3 -v
"""

import os
import sys
import time
import signal
import argparse
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import deque

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.config import setup_arx5


# ============================================================
# 配置
# ============================================================

@dataclass
class IntegrationTestConfig:
    """集成测试配置"""
    # Demo 数据
    demo_path: str = ""
    
    # 时序参数 (与部署一致)
    policy_rate: float = 10.0       # Hz, 策略推理频率
    record_freq: float = 30.0       # Hz, demo 数据采集频率
    servo_rate: float = 500.0       # Hz, 伺服控制频率
    
    act_horizon: int = 8            # 每次输出的关键帧数
    frame_step: int = 3             # 帧索引前进步长 (≈100ms @ 30Hz)
    
    dof: int = 7                    # 6 关节 + 1 夹爪
    
    # 速度缩放
    speed_scale: float = 1.0        # 速度缩放因子 (0.1 ~ 1.0)
    
    # 机器人
    model: str = "X5"
    interface: str = "can0"
    
    # 安全参数
    ema_alpha: float = 0.3
    max_joint_delta: float = 0.1    # rad/step @ 500Hz
    max_gripper_delta: float = 0.001  # m/step
    
    # 模式
    dry_run: bool = False
    verbose: bool = False
    
    # 共享内存
    shm_name: str = "rtc_keyframes_test"
    
    @property
    def dt_key(self) -> float:
        """关键帧时间间隔 (考虑速度缩放)"""
        return 1.0 / self.record_freq / self.speed_scale
    
    @property
    def policy_dt(self) -> float:
        """策略推理间隔 (考虑速度缩放)"""
        return 1.0 / self.policy_rate / self.speed_scale
    
    @property
    def servo_dt(self) -> float:
        """伺服控制间隔"""
        return 1.0 / self.servo_rate


# ============================================================
# 数据加载
# ============================================================

def load_demo_actions(demo_path: str) -> np.ndarray:
    """加载 demo 数据中的 actions"""
    demo_path = os.path.expanduser(demo_path)
    
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), "..", "..", "rlft", "diffusion_policy"
    ))
    from diffusion_policy.utils import load_traj_hdf5
    
    print(f"[DataLoader] 加载 demo: {demo_path}")
    
    raw_data = load_traj_hdf5(demo_path, num_traj=1)
    traj = raw_data["traj_0"]
    actions = traj['actions']  # (T, 7)
    
    print(f"  轨迹长度: {len(actions)} 帧")
    print(f"  动作维度: {actions.shape}")
    print(f"  关节范围: [{actions[:, :6].min():.3f}, {actions[:, :6].max():.3f}]")
    print(f"  夹爪范围: [{actions[:, 6].min():.3f}, {actions[:, 6].max():.3f}]")
    
    return actions.astype(np.float64)


# ============================================================
# FakeInferenceProcess
# ============================================================

def fake_inference_process(
    config: IntegrationTestConfig,
    actions: np.ndarray,
    stop_event: mp.Event,
    ready_event: mp.Event,
    stats_queue: mp.Queue,
):
    """
    模拟推理进程
    
    以 10Hz 从 demo 读取 actions，写入共享内存
    """
    import struct
    
    print("\n" + "=" * 60)
    print("[FakeInference] 启动")
    print("=" * 60)
    
    T = len(actions)
    H = config.act_horizon
    dof = config.dof
    
    # 创建共享内存
    from inference_rtc.shared.shm_protocol import ShmProtocol, get_monotonic_time
    
    total_size = ShmProtocol.calc_total_size(H, dof)
    
    # 删除已存在的
    try:
        old_shm = shared_memory.SharedMemory(name=config.shm_name)
        old_shm.close()
        old_shm.unlink()
    except FileNotFoundError:
        pass
    
    shm = shared_memory.SharedMemory(name=config.shm_name, create=True, size=total_size)
    buf = np.ndarray((total_size,), dtype=np.uint8, buffer=shm.buf)
    
    # 初始化 header
    struct.pack_into('<Q', buf, ShmProtocol.OFFSET_VERSION, 0)
    struct.pack_into('<d', buf, ShmProtocol.OFFSET_T_WRITE, 0.0)
    struct.pack_into('<i', buf, ShmProtocol.OFFSET_DOF, dof)
    struct.pack_into('<i', buf, ShmProtocol.OFFSET_H, H)
    struct.pack_into('<d', buf, ShmProtocol.OFFSET_DT_KEY, config.dt_key)
    
    print(f"[FakeInference] 共享内存已创建: {config.shm_name}")
    print(f"  H={H}, dof={dof}, dt_key={config.dt_key*1000:.1f}ms")
    print(f"  policy_dt={config.policy_dt*1000:.1f}ms (10Hz × speed={config.speed_scale})")
    print(f"  frame_step={config.frame_step}")
    
    # 通知 servo 进程可以连接
    ready_event.set()
    
    # 等待 servo 准备好
    print("[FakeInference] 等待 Servo 进程准备...")
    time.sleep(2.0)
    
    # 主循环
    version = 0
    frame_idx = 0
    total_writes = 0
    write_times = []
    
    print(f"\n[FakeInference] 开始写入关键帧 (frame_step={config.frame_step})...")
    
    try:
        while not stop_event.is_set() and frame_idx < T:
            loop_start = time.time()
            
            # 提取 8 帧关键帧
            end_idx = min(frame_idx + H, T)
            q_key = actions[frame_idx:end_idx].copy()
            
            # 填充不足的帧
            while q_key.shape[0] < H:
                q_key = np.vstack([q_key, q_key[-1:]])
            
            # 写入共享内存
            # 1. 写入 payload
            q_key_bytes = q_key.astype(np.float64).tobytes()
            buf[ShmProtocol.OFFSET_Q_KEY:ShmProtocol.OFFSET_Q_KEY + len(q_key_bytes)] = \
                np.frombuffer(q_key_bytes, dtype=np.uint8)
            
            # 2. 写入时间戳
            t_write = get_monotonic_time()
            struct.pack_into('<d', buf, ShmProtocol.OFFSET_T_WRITE, t_write)
            
            # 3. 递增版本号
            version += 1
            struct.pack_into('<Q', buf, ShmProtocol.OFFSET_VERSION, version)
            
            total_writes += 1
            write_times.append(t_write)
            
            if config.verbose:
                print(f"[FakeInference] v={version}, frame=[{frame_idx}, {end_idx}), "
                      f"t_write={t_write:.3f}")
            elif total_writes % 10 == 0:
                progress = frame_idx / T * 100
                print(f"[FakeInference] 进度: {progress:.1f}% | v={version} | frame={frame_idx}/{T}")
            
            # 帧索引前进
            frame_idx += config.frame_step
            
            # 频率控制 (10Hz, 考虑速度缩放)
            elapsed = time.time() - loop_start
            sleep_time = config.policy_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except Exception as e:
        print(f"[FakeInference] 错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 统计
        stats = {
            'total_writes': total_writes,
            'write_times': write_times,
            'final_frame_idx': frame_idx,
        }
        stats_queue.put(stats)
        
        print(f"\n[FakeInference] 完成")
        print(f"  总写入次数: {total_writes}")
        print(f"  最终帧索引: {frame_idx}/{T}")
        
        # 等待一段时间让 servo 处理完剩余数据
        print("[FakeInference] 等待 servo 处理剩余数据...")
        time.sleep(1.0)
        
        # 设置停止标志
        stop_event.set()
        
        # 等待 servo 退出后再清理共享内存
        time.sleep(2.0)
        
        # 清理共享内存
        shm.close()
        try:
            shm.unlink()
        except:
            pass


# ============================================================
# ServoProcess
# ============================================================

def servo_process(
    config: IntegrationTestConfig,
    actions: np.ndarray,
    stop_event: mp.Event,
    ready_event: mp.Event,
    stats_queue: mp.Queue,
):
    """
    伺服进程
    
    500Hz 读取共享内存 → 插值 → 安全限制 → 控制机器人
    """
    import struct
    from scipy.interpolate import CubicSpline
    
    print("\n" + "=" * 60)
    print("[Servo] 启动")
    print("=" * 60)
    
    # 等待 FakeInference 创建共享内存
    print("[Servo] 等待共享内存...")
    ready_event.wait(timeout=30.0)
    time.sleep(0.5)
    
    # 连接共享内存
    try:
        shm = shared_memory.SharedMemory(name=config.shm_name)
        buf = np.ndarray((shm.size,), dtype=np.uint8, buffer=shm.buf)
        print(f"[Servo] 连接共享内存成功: {config.shm_name}")
    except Exception as e:
        print(f"[Servo] 连接共享内存失败: {e}")
        stop_event.set()
        return
    
    # 协议常量
    OFFSET_VERSION = 0
    OFFSET_T_WRITE = 8
    OFFSET_DOF = 16
    OFFSET_H = 20
    OFFSET_DT_KEY = 24
    OFFSET_Q_KEY = 32
    
    # 初始化机器人
    robot = None
    arx5 = None
    controller_config = None
    
    if not config.dry_run:
        try:
            setup_arx5()
            import arx5_interface as arx5_module
            arx5 = arx5_module
            
            robot_config = arx5.RobotConfigFactory.get_instance().get_config(config.model)
            controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
                "joint_controller", robot_config.joint_dof
            )
            
            robot = arx5.Arx5JointController(robot_config, controller_config, config.interface)
            robot.set_log_level(arx5.LogLevel.INFO)
            robot.enable_background_send_recv()
            robot.reset_to_home()
            
            # 启用重力补偿
            urdf_path = os.path.join(
                os.path.dirname(__file__), 
                "..", "..", "arx5-sdk", "models", "arx5.urdf"
            )
            if os.path.exists(urdf_path):
                robot.enable_gravity_compensation(urdf_path)
            
            print(f"[Servo] 机器人初始化成功: {config.model} @ {config.interface}")
            
        except Exception as e:
            print(f"[Servo] 机器人初始化失败: {e}")
            import traceback
            traceback.print_exc()
            stop_event.set()
            shm.close()
            return
    else:
        print("[Servo] 模拟模式 - 不连接真机")
    
    # 获取初始位置
    current_pos = np.zeros(config.dof)
    if robot is not None:
        state = robot.get_state()
        current_pos[:6] = np.array(state.pos())
        current_pos[6] = state.gripper_pos
    
    # 平滑移动到第一帧位置
    if robot is not None and len(actions) > 0:
        print("[Servo] 平滑移动到起始位置...")
        target_pos = actions[0].copy()
        
        dt = controller_config.controller_dt
        duration = 3.0
        steps = int(duration / dt)
        
        for i in range(steps):
            t = i / steps
            # ease-in-out
            t = t * t * (3 - 2 * t)
            
            cmd = arx5.JointState(6)
            cmd.pos()[:] = current_pos[:6] + t * (target_pos[:6] - current_pos[:6])
            cmd.gripper_pos = current_pos[6] + t * (target_pos[6] - current_pos[6])
            
            robot.set_joint_cmd(cmd)
            time.sleep(dt)
        
        # 更新当前位置
        state = robot.get_state()
        current_pos[:6] = np.array(state.pos())
        current_pos[6] = state.gripper_pos
        print(f"[Servo] 已到达起始位置")
    
    # 插值器状态
    splines = []
    interp_valid = False
    interp_t0 = 0.0
    interp_t_end = 0.0
    
    # EMA 滤波器
    ema_state = current_pos.copy()
    ema_initialized = True
    
    # 统计
    last_version = 0
    loop_count = 0
    update_count = 0
    read_latencies = []
    tracking_errors = []
    
    # 用于跟踪误差计算的 ground truth 索引
    gt_frame_idx = 0
    gt_time_offset = None
    
    last_print = time.time()
    
    print(f"\n[Servo] 500Hz 控制循环启动...")
    
    try:
        while not stop_event.is_set():
            loop_start = time.time()
            
            # 1. 读取共享内存
            version = struct.unpack_from('<Q', buf, OFFSET_VERSION)[0]
            
            if version != last_version and version != 0:
                # 有新数据
                t_write = struct.unpack_from('<d', buf, OFFSET_T_WRITE)[0]
                dof = struct.unpack_from('<i', buf, OFFSET_DOF)[0]
                H = struct.unpack_from('<i', buf, OFFSET_H)[0]
                dt_key = struct.unpack_from('<d', buf, OFFSET_DT_KEY)[0]
                
                # 读取关键帧
                payload_size = H * dof * 8
                q_key_bytes = bytes(buf[OFFSET_Q_KEY:OFFSET_Q_KEY + payload_size])
                q_key = np.frombuffer(q_key_bytes, dtype=np.float64).reshape(H, dof).copy()
                
                # 验证版本一致性
                version2 = struct.unpack_from('<Q', buf, OFFSET_VERSION)[0]
                if version == version2:
                    # 重建插值器
                    t = np.arange(H) * dt_key + t_write
                    splines = []
                    for j in range(dof):
                        cs = CubicSpline(t, q_key[:, j], bc_type='clamped')
                        splines.append(cs)
                    
                    interp_valid = True
                    interp_t0 = t_write
                    interp_t_end = t_write + (H - 1) * dt_key
                    
                    last_version = version
                    update_count += 1
                    
                    # 记录读取延迟
                    t_now = time.clock_gettime(time.CLOCK_MONOTONIC)
                    latency = (t_now - t_write) * 1000  # ms
                    read_latencies.append(latency)
                    
                    # 记录 ground truth 时间偏移 (第一次更新时)
                    if gt_time_offset is None:
                        gt_time_offset = t_write
                    
                    if config.verbose:
                        print(f"[Servo] v={version}, latency={latency:.1f}ms, "
                              f"t_write={t_write:.3f}, t_end={interp_t_end:.3f}")
            
            # 2. 插值采样
            t_now = time.clock_gettime(time.CLOCK_MONOTONIC)
            
            if interp_valid:
                t_query = np.clip(t_now, interp_t0, interp_t_end)
                target = np.zeros(config.dof)
                for j, cs in enumerate(splines):
                    target[j] = cs(t_query)
                
                # 3. 安全限制
                # 位置限制
                joint_pos_min = np.array([-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0.0])
                joint_pos_max = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 0.08])
                safe_target = np.clip(target, joint_pos_min, joint_pos_max)
                
                # 速率限制
                for i in range(6):
                    delta = safe_target[i] - current_pos[i]
                    delta = np.clip(delta, -config.max_joint_delta, config.max_joint_delta)
                    safe_target[i] = current_pos[i] + delta
                
                gripper_delta = safe_target[6] - current_pos[6]
                gripper_delta = np.clip(gripper_delta, -config.max_gripper_delta, config.max_gripper_delta)
                safe_target[6] = current_pos[6] + gripper_delta
                
                # EMA 滤波
                ema_state = config.ema_alpha * safe_target + (1 - config.ema_alpha) * ema_state
                safe_target = ema_state.copy()
                
                # 4. 发送命令
                if robot is not None:
                    cmd = arx5.JointState(6)
                    cmd.pos()[:] = safe_target[:6]
                    cmd.gripper_pos = float(safe_target[6])
                    robot.set_joint_cmd(cmd)
                    
                    # 读取实际状态
                    state = robot.get_state()
                    current_pos[:6] = np.array(state.pos())
                    current_pos[6] = state.gripper_pos
                else:
                    current_pos = safe_target.copy()
                
                # 计算跟踪误差 (与 ground truth 对比)
                if gt_time_offset is not None:
                    gt_time = (t_now - gt_time_offset) * config.speed_scale
                    gt_idx = int(gt_time * config.record_freq)
                    if 0 <= gt_idx < len(actions):
                        gt_action = actions[gt_idx]
                        error = np.linalg.norm(current_pos[:6] - gt_action[:6])
                        tracking_errors.append({
                            'time': t_now - gt_time_offset,
                            'gt_idx': gt_idx,
                            'error': error,
                            'target': safe_target.copy(),
                            'actual': current_pos.copy(),
                            'gt': gt_action.copy(),
                        })
            
            loop_count += 1
            
            # 定期打印
            if time.time() - last_print >= 1.0:
                remaining = max(0, interp_t_end - t_now) if interp_valid else 0
                avg_latency = np.mean(read_latencies[-100:]) if read_latencies else 0
                avg_error = np.mean([e['error'] for e in tracking_errors[-500:]]) if tracking_errors else 0
                
                print(f"[Servo] loops={loop_count}, updates={update_count}, "
                      f"remaining={remaining:.2f}s, latency={avg_latency:.1f}ms, "
                      f"error={avg_error:.4f}rad")
                last_print = time.time()
            
            # 5. 频率控制
            elapsed = time.time() - loop_start
            sleep_time = config.servo_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except Exception as e:
        print(f"[Servo] 错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n[Servo] 控制循环结束")
        print(f"  总循环数: {loop_count}")
        print(f"  轨迹更新数: {update_count}")
        
        # 复位到 home
        if robot is not None:
            print("[Servo] 复位到 home...")
            try:
                robot.reset_to_home()
                print("[Servo] 已复位到 home")
            except Exception as e:
                print(f"[Servo] 复位失败: {e}")
            
            print("[Servo] 设置阻尼模式...")
            robot.set_to_damping()
        
        # 统计
        stats = {
            'loop_count': loop_count,
            'update_count': update_count,
            'read_latencies': read_latencies,
            'tracking_errors': tracking_errors,
        }
        stats_queue.put(stats)
        
        # 清理
        shm.close()


# ============================================================
# 主进程
# ============================================================

def print_stats(inference_stats: dict, servo_stats: dict, config: IntegrationTestConfig):
    """打印统计结果"""
    print("\n" + "=" * 60)
    print("集成测试统计")
    print("=" * 60)
    
    # 推理进程统计
    print("\n[FakeInference 统计]")
    print(f"  总写入次数: {inference_stats.get('total_writes', 0)}")
    print(f"  最终帧索引: {inference_stats.get('final_frame_idx', 0)}")
    
    # 伺服进程统计
    print("\n[Servo 统计]")
    print(f"  总循环数: {servo_stats.get('loop_count', 0)}")
    print(f"  轨迹更新数: {servo_stats.get('update_count', 0)}")
    
    # 读取延迟
    latencies = servo_stats.get('read_latencies', [])
    if latencies:
        print(f"\n  读取延迟:")
        print(f"    Mean: {np.mean(latencies):.2f} ms")
        print(f"    Std:  {np.std(latencies):.2f} ms")
        print(f"    Max:  {np.max(latencies):.2f} ms")
        print(f"    Min:  {np.min(latencies):.2f} ms")
    
    # 跟踪误差
    errors = servo_stats.get('tracking_errors', [])
    if errors:
        joint_errors = [e['error'] for e in errors]
        print(f"\n  跟踪误差 (vs ground truth):")
        print(f"    Mean: {np.mean(joint_errors):.4f} rad")
        print(f"    Std:  {np.std(joint_errors):.4f} rad")
        print(f"    Max:  {np.max(joint_errors):.4f} rad")
        
        # 分析是否有跳变
        if len(joint_errors) > 1:
            error_diff = np.abs(np.diff(joint_errors))
            max_jump = np.max(error_diff)
            print(f"    最大跳变: {max_jump:.4f} rad")
    
    # 频率分析
    write_times = inference_stats.get('write_times', [])
    if len(write_times) > 1:
        write_intervals = np.diff(write_times) * 1000  # ms
        print(f"\n  写入频率分析:")
        print(f"    目标间隔: {config.policy_dt*1000:.1f} ms")
        print(f"    实际间隔: {np.mean(write_intervals):.1f} ± {np.std(write_intervals):.1f} ms")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="集成测试: 完整控制链路验证")
    parser.add_argument("-d", "--demo", required=True, help="Demo 数据路径 (HDF5)")
    parser.add_argument("--speed", type=float, default=0.5, help="速度缩放 (0.1~1.0, 默认 0.5)")
    parser.add_argument("-m", "--model", default="X5", help="机械臂型号")
    parser.add_argument("-i", "--interface", default="can0", help="CAN 接口")
    parser.add_argument("--dry-run", action="store_true", help="模拟模式 (不连接真机)")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 验证速度参数
    if args.dry_run:
        # dry-run 模式允许更高速度用于快速测试
        if not 0.1 <= args.speed <= 5.0:
            print(f"[错误] dry-run 模式速度必须在 0.1~5.0 之间，当前: {args.speed}")
            return 1
    else:
        # 真机模式限制速度
        if not 0.1 <= args.speed <= 1.0:
            print(f"[错误] 真机模式速度必须在 0.1~1.0 之间，当前: {args.speed}")
            return 1
    
    config = IntegrationTestConfig(
        demo_path=args.demo,
        speed_scale=args.speed,
        model=args.model,
        interface=args.interface,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    
    print("\n" + "=" * 60)
    print("集成测试: 完整控制链路验证")
    print("=" * 60)
    
    print(f"\n配置:")
    print(f"  Demo: {config.demo_path}")
    print(f"  速度缩放: {config.speed_scale:.0%}")
    print(f"  推理频率: {config.policy_rate} Hz (dt={config.policy_dt*1000:.1f}ms)")
    print(f"  伺服频率: {config.servo_rate} Hz (dt={config.servo_dt*1000:.1f}ms)")
    print(f"  关键帧间隔: {config.dt_key*1000:.1f}ms (考虑速度缩放)")
    print(f"  帧步长: {config.frame_step} (≈{config.frame_step/30*1000:.1f}ms @ 30Hz)")
    print(f"  模式: {'模拟' if config.dry_run else '真机'}")
    
    if not config.dry_run:
        print(f"\n⚠ 安全提示:")
        print(f"  - 速度: {config.speed_scale:.0%}")
        print(f"  - 确保机械臂周围无障碍物")
        print(f"  - 随时准备按 Ctrl+C 停止")
        input("\n按 Enter 继续，Ctrl+C 取消...")
    
    # 加载数据
    try:
        actions = load_demo_actions(config.demo_path)
    except Exception as e:
        print(f"[错误] 加载数据失败: {e}")
        return 1
    
    # 创建进程间通信
    stop_event = mp.Event()
    ready_event = mp.Event()
    inference_stats_queue = mp.Queue()
    servo_stats_queue = mp.Queue()
    
    # 创建进程
    inference_proc = mp.Process(
        target=fake_inference_process,
        args=(config, actions, stop_event, ready_event, inference_stats_queue),
        name="FakeInference"
    )
    
    servo_proc = mp.Process(
        target=servo_process,
        args=(config, actions, stop_event, ready_event, servo_stats_queue),
        name="Servo"
    )
    
    # 信号处理
    def signal_handler(sig, frame):
        print(f"\n[Main] 收到信号 {sig}, 停止所有进程...")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动进程
    print("\n[Main] 启动进程...")
    inference_proc.start()
    servo_proc.start()
    
    # 等待进程完成
    try:
        inference_proc.join()
        servo_proc.join(timeout=10)
        
        if servo_proc.is_alive():
            print("[Main] Servo 进程超时，强制终止...")
            servo_proc.terminate()
            servo_proc.join(timeout=5)
    
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C 中断，停止进程...")
        stop_event.set()
        inference_proc.join(timeout=5)
        servo_proc.join(timeout=10)
    
    # 收集统计
    inference_stats = {}
    servo_stats = {}
    
    try:
        inference_stats = inference_stats_queue.get_nowait()
    except:
        pass
    
    try:
        servo_stats = servo_stats_queue.get_nowait()
    except:
        pass
    
    # 打印统计
    print_stats(inference_stats, servo_stats, config)
    
    print("\n[Main] 测试完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
