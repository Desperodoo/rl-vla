#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整调试数据记录器

同时记录 inference 和 servo 的所有数据用于分析：
1. Inference 端: 输入图像、输入状态、输出动作
2. Servo 端: 关键帧、插值输出、实际机器人状态

运行方式:
    # 终端 1: 启动带数据记录的 servo
    python -m inference_rtc.tests.test_full_debug --servo --duration 10
    
    # 终端 2: 启动带数据记录的 inference
    python -m inference_rtc.tests.test_full_debug --inference -c <checkpoint> --duration 10
    
    # 数据分析和与 demo 对比
    python -m inference_rtc.tests.test_full_debug --analyze \
        --inference-data debug_inference_*.npz \
        --servo-data debug_servo_*.npz \
        --demo ~/.arx_demos/processed/pick_cube/20251218_235920/trajectory.h5
"""

import os
import sys
import time
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import deque

import cv2

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================
# 数据记录器
# ============================================================

class InferenceDataRecorder:
    """Inference 端数据记录器"""
    
    def __init__(self, max_frames: int = 1000):
        self.max_frames = max_frames
        
        # 输入数据
        self.input_images: List[np.ndarray] = []      # [H, W, 3] uint8
        self.input_states: List[np.ndarray] = []      # [state_dim]
        self.input_robot_pos: List[np.ndarray] = []   # [7] 从共享内存读取的位置
        self.input_robot_vel: List[np.ndarray] = []   # [7] 从共享内存读取的速度
        
        # 输出数据
        self.output_keyframes: List[np.ndarray] = []  # [H, dof]
        self.output_versions: List[int] = []
        
        # 时间戳
        self.timestamps: List[float] = []
        self.inference_times: List[float] = []        # 推理耗时
        
    def record(self, 
               image: np.ndarray,
               state: np.ndarray,
               robot_pos: np.ndarray,
               robot_vel: np.ndarray,
               keyframes: np.ndarray,
               version: int,
               inference_time: float):
        """记录一次推理"""
        if len(self.timestamps) >= self.max_frames:
            return
        
        self.timestamps.append(time.time())
        self.input_images.append(image.copy())
        self.input_states.append(state.copy())
        self.input_robot_pos.append(robot_pos.copy())
        self.input_robot_vel.append(robot_vel.copy())
        self.output_keyframes.append(keyframes.copy())
        self.output_versions.append(version)
        self.inference_times.append(inference_time)
    
    def save(self, filename: str = None) -> str:
        """保存数据"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"debug_inference_{timestamp}.npz"
        
        # 转换为 numpy 数组
        data = {
            'timestamps': np.array(self.timestamps),
            'input_images': np.array(self.input_images) if self.input_images else np.array([]),
            'input_states': np.array(self.input_states) if self.input_states else np.array([]),
            'input_robot_pos': np.array(self.input_robot_pos) if self.input_robot_pos else np.array([]),
            'input_robot_vel': np.array(self.input_robot_vel) if self.input_robot_vel else np.array([]),
            'output_keyframes': np.array(self.output_keyframes) if self.output_keyframes else np.array([]),
            'output_versions': np.array(self.output_versions),
            'inference_times': np.array(self.inference_times),
        }
        
        np.savez_compressed(filename, **data)
        print(f"[InferenceRecorder] 保存 {len(self.timestamps)} 帧到 {filename}")
        return filename


class ServoDataRecorder:
    """Servo 端数据记录器"""
    
    def __init__(self, max_frames: int = 10000):
        self.max_frames = max_frames
        
        # 关键帧数据
        self.keyframe_versions: List[int] = []
        self.keyframe_t_writes: List[float] = []
        self.keyframe_data: List[np.ndarray] = []     # [H, dof]
        
        # 插值输出
        self.interp_outputs: List[np.ndarray] = []    # [dof]
        self.interp_t_queries: List[float] = []
        
        # 安全限制后
        self.safe_outputs: List[np.ndarray] = []      # [dof]
        
        # 实际机器人状态
        self.actual_pos: List[np.ndarray] = []        # [dof]
        self.actual_vel: List[np.ndarray] = []        # [dof]
        
        # 时间戳
        self.timestamps: List[float] = []
        self.loop_times: List[float] = []             # 循环耗时
        
    def record_keyframe(self, version: int, t_write: float, keyframes: np.ndarray):
        """记录关键帧"""
        self.keyframe_versions.append(version)
        self.keyframe_t_writes.append(t_write)
        self.keyframe_data.append(keyframes.copy())
    
    def record_loop(self,
                    interp_output: np.ndarray,
                    t_query: float,
                    safe_output: np.ndarray,
                    actual_pos: np.ndarray,
                    actual_vel: np.ndarray,
                    loop_time: float):
        """记录控制循环"""
        if len(self.timestamps) >= self.max_frames:
            return
        
        self.timestamps.append(time.time())
        self.interp_outputs.append(interp_output.copy() if interp_output is not None else np.zeros(7))
        self.interp_t_queries.append(t_query)
        self.safe_outputs.append(safe_output.copy() if safe_output is not None else np.zeros(7))
        self.actual_pos.append(actual_pos.copy())
        self.actual_vel.append(actual_vel.copy())
        self.loop_times.append(loop_time)
    
    def save(self, filename: str = None) -> str:
        """保存数据"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"debug_servo_{timestamp}.npz"
        
        data = {
            'timestamps': np.array(self.timestamps),
            'keyframe_versions': np.array(self.keyframe_versions),
            'keyframe_t_writes': np.array(self.keyframe_t_writes),
            'keyframe_data': np.array(self.keyframe_data) if self.keyframe_data else np.array([]),
            'interp_outputs': np.array(self.interp_outputs) if self.interp_outputs else np.array([]),
            'interp_t_queries': np.array(self.interp_t_queries),
            'safe_outputs': np.array(self.safe_outputs) if self.safe_outputs else np.array([]),
            'actual_pos': np.array(self.actual_pos) if self.actual_pos else np.array([]),
            'actual_vel': np.array(self.actual_vel) if self.actual_vel else np.array([]),
            'loop_times': np.array(self.loop_times),
        }
        
        np.savez_compressed(filename, **data)
        print(f"[ServoRecorder] 保存 {len(self.timestamps)} 帧到 {filename}")
        return filename


# ============================================================
# 带记录的 Inference
# ============================================================

def run_inference_with_recording(
    checkpoint_path: str,
    duration: float = 10.0,
    output_dir: str = ".",
):
    """运行带数据记录的 inference"""
    from inference.config import setup_arx5, setup_rlft
    from inference.shared_state import SharedState, ControlFlags
    from inference.camera_manager import CameraManager, DEFAULT_CAMERA_CONFIGS
    from inference_rtc.shared.shm_protocol import ShmKeyframeWriter
    from inference_rtc.python.policy_runner import PolicyRunner
    from inference_rtc.python.config import InferenceConfig
    
    print("=" * 60)
    print("带数据记录的 Inference")
    print("=" * 60)
    
    config = InferenceConfig()
    recorder = InferenceDataRecorder(max_frames=int(duration * config.policy_rate * 2))
    
    # 初始化组件
    print("\n[1] 加载策略...")
    policy_runner = PolicyRunner(checkpoint_path, config)
    policy_runner.load_model()
    policy_runner.warmup()
    
    print("\n[2] 创建关键帧共享内存...")
    shm_writer = ShmKeyframeWriter.create(
        name=config.shm_name,
        H=config.act_horizon,
        dof=config.dof,
        dt_key=config.dt_key,
    )
    
    print("\n[3] 连接控制共享内存 (等待 servo 写入状态)...")
    robot_state = None
    wait_start = time.time()
    wait_timeout = 60.0
    
    while time.time() - wait_start < wait_timeout:
        try:
            robot_state = SharedState.connect(config.control_shm_name, timeout=1.0)
            # 尝试读取状态，确认数据有效
            pos, vel, _ = robot_state.read_robot_state()
            pos_norm = np.linalg.norm(pos[:6])
            if pos_norm > 0.01:  # 状态非零
                print(f"  已连接, 初始状态范数: {pos_norm:.4f}")
                print(f"  初始位置: {pos}")
                break
            else:
                print(f"  已连接但状态为零 (norm={pos_norm:.4f})，等待 servo 写入...")
                robot_state = None
                time.sleep(1.0)
        except Exception as e:
            elapsed = time.time() - wait_start
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                print(f"  等待中... {elapsed:.0f}s ({e})")
            time.sleep(0.5)
    
    if robot_state is None:
        print("  [警告] 无法连接或状态为零，使用零状态 (可能导致抖动!)")
    
    print("\n[4] 初始化相机...")
    try:
        camera_manager = CameraManager(DEFAULT_CAMERA_CONFIGS)
        camera_manager.initialize(auto_assign=True)
        camera_manager.start()
        print("  相机已启动")
    except Exception as e:
        print(f"  [警告] 相机初始化失败: {e}")
        camera_manager = None
    
    print(f"\n[5] 开始推理循环 ({duration}s)...")
    
    policy_dt = 1.0 / config.policy_rate
    start_time = time.time()
    last_inference_time = 0.0
    inference_count = 0
    
    try:
        while time.time() - start_time < duration:
            loop_start = time.time()
            
            # 获取相机帧
            if camera_manager:
                try:
                    frames = camera_manager.get_frames()
                    if "wrist" in frames and frames["wrist"] is not None:
                        rgb = frames["wrist"].rgb.copy()
                    else:
                        rgb = np.zeros((config.image_size[0], config.image_size[1], 3), dtype=np.uint8)
                except:
                    rgb = np.zeros((config.image_size[0], config.image_size[1], 3), dtype=np.uint8)
            else:
                rgb = np.zeros((config.image_size[0], config.image_size[1], 3), dtype=np.uint8)
            
            # 获取机器人状态
            if robot_state:
                try:
                    pos, vel, _ = robot_state.read_robot_state()
                except:
                    pos, vel = np.zeros(7), np.zeros(7)
            else:
                pos, vel = np.zeros(7), np.zeros(7)
            
            # 构建状态
            state = np.zeros(config.state_dim, dtype=np.float32)
            state[:6] = pos[:6]
            state[6:12] = vel[:6]
            state[12] = pos[6]
            
            # 添加观测
            rgb_resized = cv2.resize(rgb, config.image_size[::-1])
            policy_runner.add_observation(rgb_resized, state)
            
            # 推理
            if time.time() - last_inference_time >= policy_dt:
                infer_start = time.time()
                keyframes = policy_runner.predict_keyframes()
                infer_time = time.time() - infer_start
                
                version = shm_writer.write_keyframes(keyframes)
                last_inference_time = time.time()
                inference_count += 1
                
                # 记录
                recorder.record(
                    image=rgb_resized,
                    state=state,
                    robot_pos=pos,
                    robot_vel=vel,
                    keyframes=keyframes,
                    version=version,
                    inference_time=infer_time,
                )
                
                if inference_count % 10 == 0:
                    print(f"  推理 #{inference_count}, v={version}")
            
            # 频率控制
            elapsed = time.time() - loop_start
            sleep_time = (1.0 / 30.0) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time * 0.8)
    
    except KeyboardInterrupt:
        print("\n  中断")
    
    finally:
        print("\n[6] 清理...")
        if camera_manager:
            camera_manager.stop()
        shm_writer.close()
    
    # 保存数据
    output_path = os.path.join(output_dir, f"debug_inference_{time.strftime('%Y%m%d_%H%M%S')}.npz")
    recorder.save(output_path)
    
    return output_path


# ============================================================
# 带记录的 Servo
# ============================================================

def run_servo_with_recording(
    duration: float = 10.0,
    output_dir: str = ".",
    dry_run: bool = False,
    wait_timeout: float = 60.0,
    demo_file: str = None,
    demo_frame: int = 50,
    move_time: float = 3.0,
    safe_exit: bool = True,
):
    """运行带数据记录的 servo
    
    Args:
        duration: 运行时长
        output_dir: 输出目录
        dry_run: 模拟模式
        wait_timeout: 等待共享内存超时
        demo_file: Demo 文件路径，如果提供则先移动到 demo 的指定帧
        demo_frame: Demo 中的帧索引
        move_time: 移动到目标位置的时间
        safe_exit: 是否在退出时安全返回 home
    """
    from inference.config import setup_arx5
    from inference.shared_state import SharedState
    from scipy.interpolate import CubicSpline
    import struct
    from multiprocessing import shared_memory
    import signal
    
    print("=" * 60)
    print("带数据记录的 Servo (安全模式)")
    print("=" * 60)
    
    recorder = ServoDataRecorder(max_frames=int(duration * 500 * 1.5))
    
    # 全局变量用于安全退出
    robot = None
    arx5 = None
    control_shm = None
    shm = None
    current_pos = np.zeros(7)
    running = True
    
    def safe_shutdown(signum=None, frame=None):
        """安全关闭处理"""
        nonlocal running
        running = False
        print("\n\n⚠ 收到中断信号，正在安全退出...")
    
    # 注册信号处理
    signal.signal(signal.SIGINT, safe_shutdown)
    signal.signal(signal.SIGTERM, safe_shutdown)
    
    try:
        # 创建控制共享内存
        print("\n[1] 创建控制共享内存...")
        try:
            control_shm = SharedState.create("arx5_control")
            print("  已创建")
        except Exception as e:
            print(f"  [警告] 创建失败: {e}")
            control_shm = None
        
        # 等待并连接关键帧共享内存
        print(f"\n[2] 等待关键帧共享内存 (超时 {wait_timeout}s)...")
        print("    请在另一个终端启动 inference...")
        
        buf = None
        wait_start = time.time()
        
        while time.time() - wait_start < wait_timeout and running:
            try:
                shm = shared_memory.SharedMemory(name="rtc_keyframes")
                buf = np.ndarray((shm.size,), dtype=np.uint8, buffer=shm.buf)
                print(f"  已连接, size={shm.size}")
                break
            except FileNotFoundError:
                time.sleep(0.5)
                elapsed = time.time() - wait_start
                if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                    print(f"    等待中... {elapsed:.0f}s")
        
        if shm is None or not running:
            print("  [错误] 超时或中断，未找到关键帧共享内存")
            if control_shm:
                control_shm.close()
            return None
        
        # 初始化机械臂
        current_vel = np.zeros(7)
        target_demo_pos = None
        
        if not dry_run:
            print("\n[3] 初始化机械臂...")
            setup_arx5()
            import arx5_interface as arx5_module
            arx5 = arx5_module
            
            robot_config = arx5.RobotConfigFactory.get_instance().get_config("X5")
            controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
                "joint_controller", robot_config.joint_dof
            )
            
            robot = arx5.Arx5JointController(robot_config, controller_config, "can0")
            robot.set_log_level(arx5.LogLevel.INFO)
            robot.enable_background_send_recv()
            
            print("  复位到 home...")
            robot.reset_to_home()
            
            state = robot.get_state()
            current_pos[:6] = np.array(state.pos())
            current_pos[6] = state.gripper_pos
            current_vel[:6] = np.array(state.vel())
            
            print(f"  Home 位置: {current_pos}")
            
            # 如果提供了 demo 文件，先移动到 demo 位置
            if demo_file and running:
                print(f"\n[3.5] 加载 Demo 帧 {demo_frame}...")
                target_demo_pos = load_demo_frame(demo_file, demo_frame)
                
                if target_demo_pos is not None:
                    print(f"  Demo 帧位置: {target_demo_pos}")
                    current_pos = safe_move_to_position(
                        robot, arx5, current_pos, target_demo_pos, move_time
                    )
                    current_vel[:6] = np.array(robot.get_state().vel())
                else:
                    print("  [警告] 无法加载 demo 帧，使用 home 位置")
            
            # 写入初始状态到共享内存
            if control_shm and running:
                control_shm.write_robot_state(
                    joint_pos=current_pos[:6], 
                    gripper_pos=float(current_pos[6]),
                    joint_vel=current_vel[:6], 
                    gripper_vel=float(current_vel[6])
                )
                print(f"  已写入初始状态到共享内存")
        else:
            print("\n[3] 模拟模式 (注意: 不会写入真实状态)")
        
        if not running:
            raise KeyboardInterrupt("用户中断")
        
        # 插值器状态
        splines = []
        t0 = 0.0
        t_end = 0.0
        valid = False
        last_version = 0
    
        def read_keyframes():
            nonlocal last_version
            header = bytes(buf[:32])
            version = struct.unpack_from('<Q', header, 0)[0]
            
            if version == last_version or version == 0:
                return None
            
            t_write = struct.unpack_from('<d', header, 8)[0]
            dof = struct.unpack_from('<i', header, 16)[0]
            H = struct.unpack_from('<i', header, 20)[0]
            dt_key = struct.unpack_from('<d', header, 24)[0]
            
            payload_size = H * dof * 8
            q_key = np.frombuffer(bytes(buf[32:32+payload_size]), dtype=np.float64).reshape(H, dof).copy()
            
            last_version = version
            return {'version': version, 't_write': t_write, 'H': H, 'dof': dof, 'dt_key': dt_key, 'q_key': q_key}
    
        def build_interpolator(q_key, dt_key, t_start):
            nonlocal splines, t0, t_end, valid
            H, dof = q_key.shape
            t0 = t_start
            t_end = t_start + (H - 1) * dt_key
            t = np.arange(H) * dt_key + t_start
            splines = [CubicSpline(t, q_key[:, j], bc_type='clamped') for j in range(dof)]
            valid = True
    
        def eval_interpolator(t_query):
            if not valid:
                return current_pos.copy()
            t_clipped = np.clip(t_query, t0, t_end)
            return np.array([cs(t_clipped) for cs in splines])
    
        def apply_safety(target, current, alpha=0.3, max_delta=0.1):
            filtered = alpha * target + (1 - alpha) * current
            delta = filtered - current
            delta_norm = np.linalg.norm(delta[:6])
            if delta_norm > max_delta:
                delta[:6] = delta[:6] / delta_norm * max_delta
            return current + delta
    
        print(f"\n[4] 开始控制循环 ({duration}s)...")
        print("    ⚠ 按 Ctrl+C 可安全退出并返回 Home 位置")
        
        start_time = time.time()
        loop_count = 0
        update_count = 0
        servo_dt = 1.0 / 500.0
        
        while time.time() - start_time < duration and running:
            loop_start = time.time()
            t_now = time.clock_gettime(time.CLOCK_MONOTONIC)
            
            # 读取关键帧
            data = read_keyframes()
            if data is not None:
                recorder.record_keyframe(data['version'], data['t_write'], data['q_key'])
                build_interpolator(data['q_key'], data['dt_key'], data['t_write'])
                update_count += 1
            
            # 插值
            interp_out = eval_interpolator(t_now) if valid else current_pos.copy()
            
            # 安全限制
            safe_out = apply_safety(interp_out, current_pos)
            
            # 发送命令
            if robot is not None:
                js = arx5.JointState(6)
                js.pos()[:] = safe_out[:6]
                js.gripper_pos = float(safe_out[6])
                robot.set_joint_cmd(js)
                
                state = robot.get_state()
                current_pos[:6] = np.array(state.pos())
                current_pos[6] = state.gripper_pos
                current_vel[:6] = np.array(state.vel())
                
                # 更新控制共享内存
                if control_shm:
                    try:
                        control_shm.write_robot_state(
                            joint_pos=current_pos[:6],
                            gripper_pos=float(current_pos[6]),
                            joint_vel=current_vel[:6],
                            gripper_vel=float(current_vel[6])
                        )
                    except:
                        pass
            
            # 记录
            loop_time = time.time() - loop_start
            recorder.record_loop(
                interp_output=interp_out,
                t_query=t_now,
                safe_output=safe_out,
                actual_pos=current_pos.copy(),
                actual_vel=current_vel.copy(),
                loop_time=loop_time,
            )
            
            loop_count += 1
            if loop_count % 500 == 0:
                elapsed = time.time() - start_time
                print(f"  loops={loop_count}, updates={update_count}, elapsed={elapsed:.1f}s")
            
            # 频率控制
            elapsed = time.time() - loop_start
            sleep_time = servo_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n  收到中断信号")
    except Exception as e:
        print(f"\n  [错误] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n[5] 安全清理...")
        
        # 安全返回 home（如果启用且机器人有效）
        if safe_exit and robot is not None and arx5 is not None:
            try:
                safe_return_to_home(robot, arx5, current_pos, move_time=2.0)
            except Exception as e:
                print(f"  [警告] 返回 home 失败: {e}")
        
        # 设置阻尼模式
        if robot is not None:
            try:
                robot.set_to_damping()
                print("  已设置阻尼模式")
            except Exception as e:
                print(f"  [警告] 设置阻尼模式失败: {e}")
        
        # 关闭共享内存
        if shm is not None:
            try:
                shm.close()
            except:
                pass
        
        if control_shm is not None:
            try:
                control_shm.close()
            except:
                pass
        
        print("  清理完成")
    
    # 保存数据
    output_path = os.path.join(output_dir, f"debug_servo_{time.strftime('%Y%m%d_%H%M%S')}.npz")
    recorder.save(output_path)
    
    return output_path


# ============================================================
# 数据分析
# ============================================================

def analyze_data(
    inference_file: str = None,
    servo_file: str = None,
    demo_file: str = None,
    save_plots: bool = True,
):
    """分析调试数据"""
    print("=" * 60)
    print("调试数据分析")
    print("=" * 60)
    
    # 加载数据
    inference_data = None
    servo_data = None
    demo_data = None
    
    if inference_file and os.path.exists(inference_file):
        inference_data = np.load(inference_file)
        print(f"\n[Inference 数据] {inference_file}")
        for key in inference_data.files:
            shape = inference_data[key].shape
            print(f"  {key}: {shape}")
    
    if servo_file and os.path.exists(servo_file):
        servo_data = np.load(servo_file)
        print(f"\n[Servo 数据] {servo_file}")
        for key in servo_data.files:
            shape = servo_data[key].shape
            print(f"  {key}: {shape}")
    
    if demo_file:
        demo_file = os.path.expanduser(demo_file)
        if os.path.exists(demo_file):
            sys.path.insert(0, os.path.expanduser("~/rl-vla"))
            from inference.config import setup_rlft
            setup_rlft()
            from diffusion_policy.utils import load_traj_hdf5
            
            raw = load_traj_hdf5(demo_file, num_traj=1)
            demo_data = {
                'actions': raw['traj_0']['actions'],
                'obs': raw['traj_0']['obs'],
            }
            print(f"\n[Demo 数据] {demo_file}")
            print(f"  actions: {demo_data['actions'].shape}")
            if 'joint_pos' in demo_data['obs']:
                print(f"  joint_pos: {demo_data['obs']['joint_pos'].shape}")
    
    # ========== 分析 ==========
    
    print("\n" + "=" * 60)
    print("分析结果")
    print("=" * 60)
    
    # 1. 动作范围对比
    if inference_data is not None and 'output_keyframes' in inference_data.files:
        keyframes = inference_data['output_keyframes']
        if len(keyframes) > 0:
            all_actions = keyframes.reshape(-1, keyframes.shape[-1])
            print(f"\n[推理输出动作]")
            print(f"  形状: {keyframes.shape} (推理次数 x 关键帧数 x 动作维度)")
            print(f"  关节范围: [{all_actions[:, :6].min():.4f}, {all_actions[:, :6].max():.4f}]")
            print(f"  关节均值: {all_actions[:, :6].mean(axis=0)}")
            print(f"  关节标准差: {all_actions[:, :6].std(axis=0)}")
            print(f"  夹爪范围: [{all_actions[:, 6].min():.4f}, {all_actions[:, 6].max():.4f}]")
            
            # 相邻关键帧差异
            if len(keyframes) > 1:
                kf_diffs = np.diff(keyframes, axis=0)
                kf_diff_norms = np.linalg.norm(kf_diffs[:, 0, :6], axis=1)  # 第一帧的差异
                print(f"\n  相邻推理输出差异 (第一帧):")
                print(f"    Mean: {kf_diff_norms.mean():.6f}")
                print(f"    Max:  {kf_diff_norms.max():.6f}")
                print(f"    Std:  {kf_diff_norms.std():.6f}")
    
    # 2. 输入状态分析
    if inference_data is not None and 'input_robot_pos' in inference_data.files:
        robot_pos = inference_data['input_robot_pos']
        if len(robot_pos) > 0:
            print(f"\n[输入机器人状态]")
            print(f"  位置范围: [{robot_pos[:, :6].min():.4f}, {robot_pos[:, :6].max():.4f}]")
            print(f"  位置均值: {robot_pos[:, :6].mean(axis=0)}")
            
            # 检查是否为零
            pos_norm = np.linalg.norm(robot_pos[:, :6].mean(axis=0))
            if pos_norm < 0.01:
                print(f"  ⚠ 警告: 输入状态接近零! 可能没有正确读取机器人状态")
    
    # 3. Demo 数据分析
    if demo_data is not None:
        demo_actions = demo_data['actions']
        print(f"\n[Demo 动作]")
        print(f"  形状: {demo_actions.shape}")
        print(f"  关节范围: [{demo_actions[:, :6].min():.4f}, {demo_actions[:, :6].max():.4f}]")
        print(f"  关节均值: {demo_actions[:, :6].mean(axis=0)}")
        print(f"  关节标准差: {demo_actions[:, :6].std(axis=0)}")
        
        # 相邻帧差异
        action_diffs = np.diff(demo_actions, axis=0)
        action_diff_norms = np.linalg.norm(action_diffs[:, :6], axis=1)
        print(f"\n  相邻帧差异:")
        print(f"    Mean: {action_diff_norms.mean():.6f}")
        print(f"    Max:  {action_diff_norms.max():.6f}")
        print(f"    Std:  {action_diff_norms.std():.6f}")
        
        if 'joint_pos' in demo_data['obs']:
            demo_pos = demo_data['obs']['joint_pos']
            print(f"\n[Demo 状态]")
            print(f"  位置范围: [{demo_pos[:, :6].min():.4f}, {demo_pos[:, :6].max():.4f}]")
            print(f"  位置均值: {demo_pos[:, :6].mean(axis=0)}")
    
    # 4. Servo 数据分析
    if servo_data is not None:
        print(f"\n[Servo 数据分析]")
        
        if 'actual_pos' in servo_data.files and len(servo_data['actual_pos']) > 0:
            actual_pos = servo_data['actual_pos']
            print(f"\n  实际位置:")
            print(f"    范围: [{actual_pos[:, :6].min():.4f}, {actual_pos[:, :6].max():.4f}]")
            print(f"    均值: {actual_pos[:, :6].mean(axis=0)}")
            
            # 位置变化
            pos_diffs = np.diff(actual_pos, axis=0)
            pos_diff_norms = np.linalg.norm(pos_diffs[:, :6], axis=1)
            print(f"\n    相邻帧变化:")
            print(f"      Mean: {pos_diff_norms.mean():.6f}")
            print(f"      Max:  {pos_diff_norms.max():.6f}")
            print(f"      P99:  {np.percentile(pos_diff_norms, 99):.6f}")
            
            # 抖动检测
            high_change_count = np.sum(pos_diff_norms > 0.01)  # 阈值 0.01 rad
            print(f"\n    抖动检测 (变化 > 0.01 rad):")
            print(f"      次数: {high_change_count} / {len(pos_diff_norms)}")
            print(f"      比例: {high_change_count / len(pos_diff_norms) * 100:.1f}%")
        
        if 'interp_outputs' in servo_data.files and len(servo_data['interp_outputs']) > 0:
            interp = servo_data['interp_outputs']
            safe = servo_data['safe_outputs']
            
            # 插值 vs 安全限制
            interp_safe_diff = np.linalg.norm(interp[:, :6] - safe[:, :6], axis=1)
            print(f"\n  插值 vs 安全限制修正:")
            print(f"    Mean: {interp_safe_diff.mean():.6f}")
            print(f"    Max:  {interp_safe_diff.max():.6f}")
        
        if 'keyframe_data' in servo_data.files and len(servo_data['keyframe_data']) > 0:
            keyframes = servo_data['keyframe_data']
            print(f"\n  收到的关键帧:")
            print(f"    数量: {len(keyframes)}")
            
            # 关键帧范围
            all_kf = keyframes.reshape(-1, keyframes.shape[-1])
            print(f"    关节范围: [{all_kf[:, :6].min():.4f}, {all_kf[:, :6].max():.4f}]")
    
    # 5. 对比分析
    if inference_data is not None and demo_data is not None:
        print(f"\n[对比分析]")
        
        infer_actions = inference_data['output_keyframes']
        if len(infer_actions) > 0:
            infer_mean = infer_actions.reshape(-1, infer_actions.shape[-1])[:, :6].mean(axis=0)
            demo_mean = demo_data['actions'][:, :6].mean(axis=0)
            
            print(f"\n  均值对比:")
            print(f"    推理: {infer_mean}")
            print(f"    Demo: {demo_mean}")
            print(f"    差异: {infer_mean - demo_mean}")
            
            mean_diff_norm = np.linalg.norm(infer_mean - demo_mean)
            print(f"    差异范数: {mean_diff_norm:.4f}")
            
            if mean_diff_norm > 0.5:
                print(f"\n  ⚠ 警告: 推理输出与 Demo 均值差异较大!")
                print(f"     可能原因:")
                print(f"     1. 输入状态不正确 (检查 input_robot_pos)")
                print(f"     2. 图像输入问题")
                print(f"     3. 策略模型问题")
    
    return {
        'inference_data': inference_data,
        'servo_data': servo_data,
        'demo_data': demo_data,
    }


def move_to_demo_start(demo_file: str, move_time: float = 3.0):
    """将机器人移动到 Demo 的起始位置"""
    from inference.config import setup_arx5, setup_rlft
    
    print("=" * 60)
    print("移动到 Demo 起始位置")
    print("=" * 60)
    
    # 加载 Demo 数据
    print("\n[1] 加载 Demo 数据...")
    demo_file = os.path.expanduser(demo_file)
    
    setup_rlft()
    from diffusion_policy.utils import load_traj_hdf5
    
    raw = load_traj_hdf5(demo_file, num_traj=1)
    demo_obs = raw['traj_0']['obs']
    
    if 'joint_pos' in demo_obs:
        start_pos = demo_obs['joint_pos'][0]  # 第一帧位置
    else:
        print("  [错误] Demo 数据中没有 joint_pos")
        return False
    
    # Demo 的 joint_pos 只有 6 个关节
    target_pos = np.zeros(7)
    target_pos[:6] = start_pos[:6]
    target_pos[6] = 0.0  # 夹爪
    
    print(f"  Demo 起始位置: {target_pos}")
    
    # 初始化机械臂
    print("\n[2] 初始化机械臂...")
    setup_arx5()
    import arx5_interface as arx5
    
    robot_config = arx5.RobotConfigFactory.get_instance().get_config("X5")
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    
    robot = arx5.Arx5JointController(robot_config, controller_config, "can0")
    robot.set_log_level(arx5.LogLevel.INFO)
    robot.enable_background_send_recv()
    
    # 先复位到 home
    print("\n[3] 复位到 home...")
    robot.reset_to_home()
    
    # 获取当前位置
    state = robot.get_state()
    current_pos = np.zeros(7)
    current_pos[:6] = np.array(state.pos())
    current_pos[6] = state.gripper_pos
    
    print(f"  当前位置: {current_pos}")
    print(f"  目标位置: {target_pos}")
    print(f"  差异: {np.linalg.norm(target_pos[:6] - current_pos[:6]):.4f} rad")
    
    # 平滑移动到目标位置
    print(f"\n[4] 平滑移动到目标 ({move_time}s)...")
    
    move_dt = 0.002  # 500Hz
    steps = int(move_time / move_dt)
    
    for i in range(steps):
        t = (i + 1) / steps  # 0 到 1
        # 使用 smooth step 插值
        t_smooth = t * t * (3 - 2 * t)
        
        interp_pos = current_pos + t_smooth * (target_pos - current_pos)
        
        js = arx5.JointState(6)
        js.pos()[:] = interp_pos[:6]
        js.gripper_pos = float(interp_pos[6])
        robot.set_joint_cmd(js)
        
        time.sleep(move_dt)
        
        if i % 500 == 0:
            print(f"  进度: {t*100:.0f}%")
    
    # 验证最终位置
    state = robot.get_state()
    final_pos = np.zeros(7)
    final_pos[:6] = np.array(state.pos())
    final_pos[6] = state.gripper_pos
    
    error = np.linalg.norm(final_pos[:6] - target_pos[:6])
    print(f"\n[5] 移动完成")
    print(f"  最终位置: {final_pos}")
    print(f"  位置误差: {error:.4f} rad")
    
    if error > 0.1:
        print(f"  ⚠ 警告: 位置误差较大!")
    else:
        print(f"  ✓ 位置正确")
    
    # 保持位置
    print("\n  机器人将保持在当前位置。按 Ctrl+C 退出并设置阻尼模式...")
    
    try:
        while True:
            js = arx5.JointState(6)
            js.pos()[:] = target_pos[:6]
            js.gripper_pos = float(target_pos[6])
            robot.set_joint_cmd(js)
            time.sleep(0.002)
    except KeyboardInterrupt:
        print("\n  退出...")
    finally:
        robot.set_to_damping()
    
    return True


def load_demo_frame(demo_file: str, frame_idx: int = 50) -> Optional[np.ndarray]:
    """加载 demo 的某一帧位置"""
    from inference.config import setup_rlft
    import h5py
    
    demo_file = os.path.expanduser(demo_file)
    
    setup_rlft()
    
    with h5py.File(demo_file, 'r') as f:
        if 'traj_0/obs/joint_pos' in f:
            jp = f['traj_0/obs/joint_pos'][:]
            # 确保 frame_idx 在有效范围内
            frame_idx = min(frame_idx, len(jp) - 1)
            target = np.zeros(7)
            target[:6] = jp[frame_idx]
            
            # 获取夹爪
            if 'traj_0/obs/gripper_pos' in f:
                gp = f['traj_0/obs/gripper_pos'][:]
                target[6] = gp[frame_idx, 0] if gp.ndim > 1 else gp[frame_idx]
            
            return target
    return None


def safe_move_to_position(robot, arx5, current_pos: np.ndarray, target_pos: np.ndarray, 
                          move_time: float = 3.0) -> np.ndarray:
    """安全地将机器人移动到目标位置，返回最终位置"""
    print(f"\n  当前位置: {current_pos[:6]}")
    print(f"  目标位置: {target_pos[:6]}")
    diff = np.linalg.norm(target_pos[:6] - current_pos[:6])
    print(f"  距离: {diff:.4f} rad")
    
    if diff < 0.01:
        print("  已在目标位置附近，跳过移动")
        return current_pos.copy()
    
    move_dt = 0.002  # 500Hz
    steps = int(move_time / move_dt)
    
    print(f"  开始移动 ({move_time}s)...")
    
    for i in range(steps):
        t = (i + 1) / steps
        # Smooth step
        t_smooth = t * t * (3 - 2 * t)
        
        interp_pos = current_pos + t_smooth * (target_pos - current_pos)
        
        js = arx5.JointState(6)
        js.pos()[:] = interp_pos[:6]
        js.gripper_pos = float(interp_pos[6])
        robot.set_joint_cmd(js)
        
        time.sleep(move_dt)
        
        if (i + 1) % 500 == 0:
            print(f"    进度: {t*100:.0f}%")
    
    # 获取最终位置
    state = robot.get_state()
    final_pos = np.zeros(7)
    final_pos[:6] = np.array(state.pos())
    final_pos[6] = state.gripper_pos
    
    error = np.linalg.norm(final_pos[:6] - target_pos[:6])
    print(f"  移动完成, 误差: {error:.4f} rad")
    
    return final_pos


def safe_return_to_home(robot, arx5, current_pos: np.ndarray, move_time: float = 2.0):
    """安全地返回 home 位置"""
    print("\n  安全返回 Home 位置...")
    
    # Home 位置（接近零位）
    home_pos = np.zeros(7)
    
    move_dt = 0.002
    steps = int(move_time / move_dt)
    
    for i in range(steps):
        t = (i + 1) / steps
        t_smooth = t * t * (3 - 2 * t)
        
        interp_pos = current_pos + t_smooth * (home_pos - current_pos)
        
        js = arx5.JointState(6)
        js.pos()[:] = interp_pos[:6]
        js.gripper_pos = float(interp_pos[6])
        robot.set_joint_cmd(js)
        
        time.sleep(move_dt)
        
        if (i + 1) % 500 == 0:
            print(f"    返回进度: {t*100:.0f}%")
    
    print("  已返回 Home 位置")


def main():
    parser = argparse.ArgumentParser(description="完整调试数据记录和分析")
    
    # 模式选择
    parser.add_argument("--servo", action="store_true", help="运行 servo 数据记录")
    parser.add_argument("--inference", action="store_true", help="运行 inference 数据记录")
    parser.add_argument("--analyze", action="store_true", help="分析已有数据")
    parser.add_argument("--move-to-start", action="store_true", help="移动到 Demo 起始位置")
    
    # 公共参数
    parser.add_argument("-d", "--duration", type=float, default=10.0, help="运行时长 (秒)")
    parser.add_argument("-o", "--output-dir", default=".", help="输出目录")
    parser.add_argument("--dry-run", action="store_true", help="模拟模式 (servo)")
    parser.add_argument("--wait-timeout", type=float, default=60.0, help="等待共享内存超时 (秒)")
    parser.add_argument("--move-time", type=float, default=3.0, help="移动到起始位置的时间 (秒)")
    
    # Inference 参数
    parser.add_argument("-c", "--checkpoint", type=str, help="策略 checkpoint 路径")
    
    # 分析参数
    parser.add_argument("--inference-data", type=str, help="Inference 数据文件")
    parser.add_argument("--servo-data", type=str, help="Servo 数据文件")
    parser.add_argument("--demo", type=str, help="Demo 数据文件 (用于分析或定位)")
    parser.add_argument("--demo-frame", type=int, default=50, help="Demo 帧索引 (默认: 50)")
    parser.add_argument("--safe-exit", action="store_true", help="启用安全退出 (Ctrl+C 返回 Home)")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_data(
            inference_file=args.inference_data,
            servo_file=args.servo_data,
            demo_file=args.demo,
        )
    elif args.move_to_start:
        if not args.demo:
            print("错误: --move-to-start 需要 --demo 参数")
            return 1
        move_to_demo_start(args.demo, args.move_time)
    elif args.servo:
        run_servo_with_recording(
            duration=args.duration,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            wait_timeout=args.wait_timeout,
            demo_file=args.demo,
            demo_frame=args.demo_frame,
            move_time=args.move_time,
            safe_exit=args.safe_exit,
        )
    elif args.inference:
        if not args.checkpoint:
            print("错误: inference 模式需要 --checkpoint 参数")
            return 1
        run_inference_with_recording(
            checkpoint_path=args.checkpoint,
            duration=args.duration,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
