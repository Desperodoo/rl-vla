#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARX5 关节空间多进程控制器

基于 umi-arx 的多进程架构设计，使用 SharedMemoryQueue 和 SharedMemoryRingBuffer
实现高频控制 (200Hz+)。

特点:
- 控制循环在独立进程中运行 (200Hz)
- SharedMemoryQueue 传递控制命令 (FIFO)
- SharedMemoryRingBuffer 传递机器人状态 (FILO)  
- JointTrajectoryInterpolator 实现平滑轨迹插值
- 支持 SERVOL, SCHEDULE_WAYPOINT, ADD_WAYPOINT, UPDATE_TRAJECTORY 等命令

架构:
    主进程 (Policy ~10Hz)              控制进程 (200Hz)
    ┌─────────────────┐               ┌──────────────────────┐
    │  schedule_waypoint()            │  while True:         │
    │       │                         │    cmd = queue.get() │
    │       ▼                         │    interp.update()   │
    │  SharedMemoryQueue ───────────> │    joints = interp(t)│
    │                                 │    robot.set_cmd()   │
    │  get_state() <───────────────── │    state -> buffer   │
    │       ▲                         │                      │
    │  SharedMemoryRingBuffer         │                      │
    └─────────────────┘               └──────────────────────┘
"""

import os
import sys
import time
import enum
import ctypes
import multiprocessing as mp
from multiprocessing import Value
from multiprocessing.managers import SharedMemoryManager
from typing import Optional, Dict, Any
import numpy as np
import numpy.typing as npt

# 添加项目路径
CONSISTENCY_POLICY_PATH = os.path.dirname(os.path.abspath(__file__))
RL_VLA_PATH = os.path.dirname(CONSISTENCY_POLICY_PATH)
UMI_ARX_PATH = os.path.join(RL_VLA_PATH, 'umi-arx')

if RL_VLA_PATH not in sys.path:
    sys.path.insert(0, RL_VLA_PATH)
if UMI_ARX_PATH not in sys.path:
    sys.path.insert(0, UMI_ARX_PATH)

from consistency_policy.config import setup_arx5, RL_VLA_CONFIG
from consistency_policy.joint_trajectory_interpolator import JointTrajectoryInterpolator
from shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


def precise_wait(target_time: float, time_func=time.monotonic, min_sleep: float = 0.0001):
    """精确等待到目标时间"""
    while True:
        remaining = target_time - time_func()
        if remaining <= 0:
            return
        if remaining > min_sleep:
            time.sleep(min_sleep)


# ===================== 命令枚举 =====================

class Command(enum.Enum):
    """控制命令类型"""
    STOP = 0
    SERVOL = 1              # 立即移动到目标位置 (覆盖后续轨迹)
    SCHEDULE_WAYPOINT = 2   # 调度一个航点 (排队执行)
    RESET_TO_HOME = 3       # 复位到 home 位置
    ADD_WAYPOINT = 4        # 添加航点到缓冲区
    UPDATE_TRAJECTORY = 5   # 使用缓冲区更新轨迹
    SET_DAMPING = 6         # 设置为阻尼模式


# ===================== 多进程控制器 =====================

class Arx5JointControllerProcess(mp.Process):
    """
    ARX5 关节空间多进程控制器
    
    控制循环在独立进程中运行，通过共享内存与主进程通信。
    """
    
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        model: str = "X5",
        interface: str = "can0",
        urdf_path: Optional[str] = None,
        frequency: float = 200.0,  # 控制频率
        get_max_k: Optional[int] = None,  # RingBuffer 最大查询数
        enable_gravity_compensation: bool = True,
        verbose: bool = False,
        launch_timeout: float = 30.0,
    ):
        """
        Args:
            shm_manager: 共享内存管理器 (必须已 start)
            model: 机器人型号
            interface: CAN 接口
            urdf_path: URDF 文件路径
            frequency: 控制频率 (Hz)
            get_max_k: 状态 RingBuffer 最大查询帧数
            enable_gravity_compensation: 是否启用重力补偿
            verbose: 是否打印详细日志
            launch_timeout: 启动超时时间
        """
        super().__init__(name="Arx5JointController")
        
        self.model = model
        self.interface = interface
        self.urdf_path = urdf_path or str(RL_VLA_CONFIG.get_model_path('arx5.urdf'))
        self.frequency = frequency
        self.enable_gravity_compensation_flag = enable_gravity_compensation
        self.verbose = verbose
        self.launch_timeout = launch_timeout
        
        # 控制命令队列
        example_cmd = {
            'cmd': Command.SERVOL.value,
            'target_joints': np.zeros(7, dtype=np.float64),  # 6 关节 + 1 夹爪
            'duration': 0.0,
            'target_time': 0.0,
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example_cmd,
            buffer_size=256,
        )
        
        # 机器人状态 RingBuffer
        if get_max_k is None:
            get_max_k = int(frequency * 5)
        
        example_state = {
            'joint_pos': np.zeros(6, dtype=np.float64),
            'joint_vel': np.zeros(6, dtype=np.float64),
            'gripper_pos': 0.0,
            'timestamp': 0.0,  # 系统时间 (time.time())
            'monotonic': 0.0,  # 单调时间 (time.monotonic())
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_state,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )
        
        # 进程控制
        self.ready_event = mp.Event()
        self.reset_success = Value(ctypes.c_bool, False)
        
        # 在子进程中初始化的变量
        self.robot = None
        self.arx5 = None
    
    # ========= 启动/停止方法 =========
    
    def start(self, wait: bool = True):
        """启动控制器进程"""
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[Arx5Controller] 控制器进程已启动 (PID: {self.pid})")
    
    def stop(self, wait: bool = True):
        """停止控制器进程"""
        message = {'cmd': Command.STOP.value}
        self.input_queue.put(message)
        if wait:
            self.stop_wait()
        if self.verbose:
            print(f"[Arx5Controller] 控制器进程已停止")
    
    def start_wait(self):
        """等待控制器就绪"""
        print(f"[Arx5Controller] 等待控制器就绪...")
        self.ready_event.wait(self.launch_timeout)
        if not self.ready_event.is_set():
            raise TimeoutError(f"[Arx5Controller] 启动超时 ({self.launch_timeout}s)")
        assert self.is_alive(), "控制器进程未运行"
    
    def stop_wait(self):
        """等待进程结束"""
        self.join()
    
    @property
    def is_ready(self) -> bool:
        return self.ready_event.is_set()
    
    # ========= 上下文管理 =========
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    # ========= 状态获取 API =========
    
    def get_state(self, k: Optional[int] = None, out=None) -> Dict[str, np.ndarray]:
        """
        获取机器人状态
        
        Args:
            k: 获取最近 k 帧状态，None 表示只获取最新状态
            out: 预分配的输出缓冲区
        
        Returns:
            状态字典，包含 'joint_pos', 'joint_vel', 'gripper_pos', 'timestamp'
        """
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)
    
    def get_all_state(self) -> Dict[str, np.ndarray]:
        """获取所有可用状态"""
        return self.ring_buffer.get_all()
    
    # ========= 控制命令 API =========
    
    def servoL(
        self,
        joint_pos: npt.NDArray[np.float64],
        gripper_pos: float,
        duration: float = 0.1,
    ):
        """
        立即移动到目标位置 (覆盖后续轨迹)
        
        Args:
            joint_pos: (6,) 目标关节位置 rad
            gripper_pos: 目标夹爪位置
            duration: 期望到达时间
        """
        assert self.is_alive(), "控制器未运行"
        assert duration >= (1 / self.frequency), f"duration 必须 >= {1/self.frequency}"
        
        joint_pos = np.array(joint_pos, dtype=np.float64)
        assert joint_pos.shape == (6,), f"关节位置维度错误: {joint_pos.shape}"
        
        # 合并关节和夹爪
        target_joints = np.concatenate([joint_pos, [gripper_pos]])
        
        message = {
            'cmd': Command.SERVOL.value,
            'target_joints': target_joints,
            'duration': float(duration),
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(
        self,
        joint_pos: npt.NDArray[np.float64],
        gripper_pos: float,
        target_time: float,
    ):
        """
        调度一个航点 (排队执行)
        
        Args:
            joint_pos: (6,) 目标关节位置 rad  
            gripper_pos: 目标夹爪位置
            target_time: 期望到达的系统时间 (time.time())
        """
        assert self.is_alive(), "控制器未运行"
        
        joint_pos = np.array(joint_pos, dtype=np.float64)
        assert joint_pos.shape == (6,), f"关节位置维度错误: {joint_pos.shape}"
        
        target_joints = np.concatenate([joint_pos, [gripper_pos]])
        
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_joints': target_joints,
            'target_time': float(target_time),
        }
        self.input_queue.put(message)
    
    def add_waypoint(
        self,
        joint_pos: npt.NDArray[np.float64],
        gripper_pos: float,
        target_time: float,
    ):
        """
        添加航点到缓冲区 (不立即执行)
        
        需要调用 update_trajectory() 来应用缓冲的航点。
        
        Args:
            joint_pos: (6,) 目标关节位置 rad
            gripper_pos: 目标夹爪位置
            target_time: 期望到达的系统时间 (time.time())
        """
        assert self.is_alive(), "控制器未运行"
        
        joint_pos = np.array(joint_pos, dtype=np.float64)
        assert joint_pos.shape == (6,), f"关节位置维度错误: {joint_pos.shape}"
        
        target_joints = np.concatenate([joint_pos, [gripper_pos]])
        
        message = {
            'cmd': Command.ADD_WAYPOINT.value,
            'target_joints': target_joints,
            'target_time': float(target_time),
        }
        self.input_queue.put(message)
    
    def update_trajectory(self):
        """
        使用缓冲的航点更新轨迹
        
        调用此方法后，控制器将使用 add_waypoint 添加的航点
        创建新的插值轨迹并执行。
        """
        assert self.is_alive(), "控制器未运行"
        
        message = {'cmd': Command.UPDATE_TRAJECTORY.value}
        self.input_queue.put(message)
    
    def reset_to_home(self, wait: bool = True):
        """
        复位到 home 位置
        
        Args:
            wait: 是否等待复位完成
        """
        assert self.is_alive(), "控制器未运行"
        
        self.reset_success.value = False
        message = {'cmd': Command.RESET_TO_HOME.value}
        self.input_queue.put(message)
        
        if wait:
            while not self.reset_success.value:
                print("[Arx5Controller] 等待复位完成...")
                time.sleep(0.1)
    
    def set_to_damping(self):
        """设置为阻尼模式"""
        if self.is_alive():
            message = {'cmd': Command.SET_DAMPING.value}
            self.input_queue.put(message)
    
    # ========= 控制循环 (在子进程中运行) =========
    
    def run(self):
        """控制器主循环"""
        try:
            self._run_control_loop()
        except Exception as e:
            print(f"[Arx5Controller] 控制循环异常: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _run_control_loop(self):
        """控制循环实现"""
        # 初始化机械臂
        print(f"[Arx5Controller] 初始化机械臂 ({self.model} @ {self.interface})...")
        setup_arx5()
        import arx5_interface as arx5
        self.arx5 = arx5
        
        # 创建配置
        robot_config = arx5.RobotConfigFactory.get_instance().get_config(self.model)
        controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", robot_config.joint_dof
        )
        
        # 创建控制器
        self.robot = arx5.Arx5JointController(
            robot_config, controller_config, self.interface
        )
        self.robot.set_log_level(arx5.LogLevel.INFO)
        
        # 启用后台通信
        self.robot.enable_background_send_recv()
        print(f"[Arx5Controller] 已启用后台通信")
        
        # 复位到 home 位置
        print(f"[Arx5Controller] 复位到 home 位置...")
        self.robot.reset_to_home()
        time.sleep(1)
        
        # 启用重力补偿
        if self.enable_gravity_compensation_flag and os.path.exists(self.urdf_path):
            self.robot.enable_gravity_compensation(self.urdf_path)
            print(f"[Arx5Controller] 已启用重力补偿")
        
        # 获取初始状态
        state = self.robot.get_state()
        curr_joints = np.concatenate([
            np.array(state.pos()),
            [state.gripper_pos]
        ])  # (7,)
        
        # 初始化插值器
        curr_t = time.monotonic()
        joint_interp = JointTrajectoryInterpolator(
            times=np.array([curr_t]),
            joints=np.array([curr_joints]),
        )
        
        # 航点缓冲区
        waypoint_buffer = []
        last_waypoint_time = curr_t
        
        # 控制参数
        dt = 1.0 / self.frequency
        t_start = time.monotonic()
        iter_idx = 0
        keep_running = True
        
        try:
            while keep_running:
                t_now = time.monotonic()
                
                # 插值计算当前命令
                joints_cmd = joint_interp(t_now)  # (7,)
                joint_pos_cmd = joints_cmd[:6]
                gripper_cmd = joints_cmd[6]
                
                # 发送命令到机械臂
                cmd = self.arx5.JointState(6)
                cmd.pos()[:] = joint_pos_cmd
                cmd.gripper_pos = gripper_cmd
                self.robot.set_joint_cmd(cmd)
                
                # 获取状态
                state = self.robot.get_state()
                state_dict = {
                    'joint_pos': np.array(state.pos()),
                    'joint_vel': np.array(state.vel()),
                    'gripper_pos': float(state.gripper_pos),
                    'timestamp': time.time(),
                    'monotonic': time.monotonic(),
                }
                self.ring_buffer.put(state_dict)
                
                # 处理命令
                try:
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    commands = {}
                    n_cmd = 0
                
                # 执行命令
                for i in range(n_cmd):
                    command = {key: value[i] for key, value in commands.items()}
                    cmd_type = command['cmd']
                    
                    if cmd_type == Command.STOP.value:
                        keep_running = False
                        break
                    
                    elif cmd_type == Command.SERVOL.value:
                        target_joints = command['target_joints']
                        duration = float(command['duration'])
                        
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        
                        joint_interp = joint_interp.drive_to_waypoint(
                            joints=target_joints,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        
                        if self.verbose:
                            print(f"[Arx5Controller] SERVOL: 目标 {target_joints[:3]}... 时长 {duration:.3f}s")
                    
                    elif cmd_type == Command.SCHEDULE_WAYPOINT.value:
                        target_joints = command['target_joints']
                        target_time = float(command['target_time'])
                        
                        # 将系统时间转换为单调时间
                        target_monotonic = time.monotonic() - time.time() + target_time
                        
                        joint_interp = joint_interp.schedule_waypoint(
                            joints=target_joints,
                            time=target_monotonic,
                            curr_time=t_now,
                            last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = target_monotonic
                        
                        if self.verbose:
                            print(f"[Arx5Controller] SCHEDULE: 目标 {target_joints[:3]}...")
                    
                    elif cmd_type == Command.ADD_WAYPOINT.value:
                        if len(waypoint_buffer) > 0:
                            last_wp_time = waypoint_buffer[-1]['target_time']
                            if command['target_time'] <= last_wp_time:
                                if self.verbose:
                                    print(f"[Arx5Controller] 跳过过期航点")
                                continue
                        waypoint_buffer.append(command)
                    
                    elif cmd_type == Command.UPDATE_TRAJECTORY.value:
                        if len(waypoint_buffer) == 0:
                            if self.verbose:
                                print(f"[Arx5Controller] 无新航点，跳过更新")
                            continue
                        
                        t_cmd_recv = time.time()  # 使用 time.time() 保持一致
                        
                        # 从缓冲区创建新轨迹
                        input_joints = np.array([
                            wp['target_joints'] for wp in waypoint_buffer
                        ])
                        input_times = np.array([
                            wp['target_time'] for wp in waypoint_buffer
                        ])
                        
                        # 转换为单调时间
                        input_times_mono = input_times - input_times[0] + t_now
                        
                        # 计算时间差值用于日志
                        time_diff_ms = (t_cmd_recv - input_times[0]) * 1000
                        chunk_duration_ms = (input_times[-1] - input_times[0]) * 1000
                        
                        # 平滑过渡: 开始部分混合当前轨迹
                        smoothing_time = 0.3  # 300ms 平滑过渡
                        smoothened_joints = input_joints.copy()
                        
                        for j in range(len(input_times_mono)):
                            t_wp = input_times_mono[j]
                            if t_wp < t_now:
                                smoothened_joints[j] = joint_interp(t_wp)
                            elif t_now <= t_wp < t_now + smoothing_time:
                                alpha = (t_wp - t_now) / smoothing_time
                                smoothened_joints[j] = (
                                    (1 - alpha) * joint_interp(t_wp) + 
                                    alpha * input_joints[j]
                                )
                        
                        # 创建新插值器
                        joint_interp = JointTrajectoryInterpolator(
                            times=input_times_mono,
                            joints=smoothened_joints,
                        )
                        
                        # 清空缓冲区
                        waypoint_buffer = []
                        
                        if self.verbose:
                            print(f"[Arx5Controller] 轨迹更新: {len(input_times)} 个航点, "
                                  f"覆盖 {chunk_duration_ms:.1f}ms, "
                                  f"time_diff={time_diff_ms:.1f}ms")
                    
                    elif cmd_type == Command.RESET_TO_HOME.value:
                        self.robot.reset_to_home()
                        state = self.robot.get_state()
                        
                        # 更新 RingBuffer
                        self.ring_buffer.clear()
                        state_dict = {
                            'joint_pos': np.array(state.pos()),
                            'joint_vel': np.array(state.vel()),
                            'gripper_pos': float(state.gripper_pos),
                            'timestamp': time.time(),
                            'monotonic': time.monotonic(),
                        }
                        self.ring_buffer.put(state_dict)
                        
                        # 重新初始化插值器
                        curr_joints = np.concatenate([
                            np.array(state.pos()),
                            [state.gripper_pos]
                        ])
                        curr_t = time.monotonic()
                        joint_interp = JointTrajectoryInterpolator(
                            times=np.array([curr_t]),
                            joints=np.array([curr_joints]),
                        )
                        last_waypoint_time = curr_t
                        
                        self.reset_success.value = True
                        
                        if self.verbose:
                            print(f"[Arx5Controller] 已复位到 home")
                    
                    elif cmd_type == Command.SET_DAMPING.value:
                        self.robot.set_to_damping()
                        if self.verbose:
                            print(f"[Arx5Controller] 已设置为阻尼模式")
                    
                    else:
                        print(f"[Arx5Controller] 未知命令: {cmd_type}")
                
                # 维持控制频率
                t_wait_until = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_until)
                
                # 第一次循环成功，标记就绪
                if iter_idx == 0:
                    self.ready_event.set()
                    print(f"[Arx5Controller] 控制器就绪，频率 {self.frequency}Hz")
                
                iter_idx += 1
        
        finally:
            # 清理
            print("[Arx5Controller] 设置阻尼模式...")
            if self.robot is not None:
                self.robot.set_to_damping()
            self.ready_event.set()  # 确保主进程不会阻塞
            print("[Arx5Controller] 控制循环结束")


# ===================== 便捷封装类 =====================

class Arx5JointControllerManager:
    """
    ARX5 关节控制器管理器
    
    封装了 SharedMemoryManager 的生命周期管理。
    """
    
    def __init__(
        self,
        model: str = "X5",
        interface: str = "can0",
        frequency: float = 200.0,
        verbose: bool = False,
    ):
        self.model = model
        self.interface = interface
        self.frequency = frequency
        self.verbose = verbose
        
        self.shm_manager: Optional[SharedMemoryManager] = None
        self.controller: Optional[Arx5JointControllerProcess] = None
    
    def start(self):
        """启动控制器"""
        # 创建共享内存管理器
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        
        # 创建控制器进程
        self.controller = Arx5JointControllerProcess(
            shm_manager=self.shm_manager,
            model=self.model,
            interface=self.interface,
            frequency=self.frequency,
            verbose=self.verbose,
        )
        self.controller.start(wait=True)
    
    def stop(self):
        """停止控制器"""
        if self.controller is not None:
            self.controller.stop(wait=True)
            self.controller = None
        
        if self.shm_manager is not None:
            self.shm_manager.shutdown()
            self.shm_manager = None
    
    def __enter__(self):
        self.start()
        return self.controller
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ===================== 测试代码 =====================

if __name__ == "__main__":
    print("=" * 60)
    print("ARX5 多进程关节控制器测试")
    print("=" * 60)
    
    with Arx5JointControllerManager(
        model="X5",
        interface="can0",
        frequency=200.0,
        verbose=True,
    ) as controller:
        # 获取状态
        state = controller.get_state()
        print(f"\n当前状态:")
        print(f"  关节位置: {state['joint_pos']}")
        print(f"  夹爪位置: {state['gripper_pos']:.4f}")
        
        # 测试 servoL
        input("\n按 Enter 测试 servoL (小幅移动)...")
        target = state['joint_pos'].copy()
        target[0] += 0.1  # 第一个关节移动 0.1 rad
        controller.servoL(target, state['gripper_pos'], duration=1.0)
        time.sleep(1.5)
        
        # 测试移回原位
        input("\n按 Enter 移回原位...")
        controller.servoL(state['joint_pos'], state['gripper_pos'], duration=1.0)
        time.sleep(1.5)
        
        # 获取最终状态
        final_state = controller.get_state()
        print(f"\n最终状态:")
        print(f"  关节位置: {final_state['joint_pos']}")
    
    print("\n测试完成!")
