#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTC 伺服进程 (Python 版本)

Phase A 实现: 500Hz 控制循环
- 从共享内存读取关键帧
- 三次样条插值
- 安全限制 (EMA + 速率限制)
- 发送到 arx5 机械臂

注意: 这是 Python 版本，用于快速验证。
      生产环境应使用 C++ 版本以获得更好的实时性。
"""

import os
import sys
import time
import signal
import argparse
import threading
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.interpolate import CubicSpline

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference.config import setup_arx5


@dataclass
class ServoConfig:
    """伺服配置"""
    # 机器人
    model: str = "X5"
    interface: str = "can0"
    
    # 控制参数
    servo_rate: float = 500.0   # Hz
    dof: int = 7                # 6 关节 + 1 夹爪
    
    # 安全参数
    ema_alpha: float = 0.3
    max_joint_delta: float = 0.1     # rad/step
    max_gripper_delta: float = 0.001  # m/step
    
    # 关节限制
    joint_pos_min: tuple = (-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0.0)
    joint_pos_max: tuple = (3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 0.08)
    
    # 模式
    dry_run: bool = False
    verbose: bool = False


class ShmKeyframeReader:
    """共享内存关键帧读取器 (Python 版本)"""
    
    SHM_NAME = "rtc_keyframes"
    
    def __init__(self, name: str = None):
        from multiprocessing import shared_memory
        import struct
        
        self._name = name or self.SHM_NAME
        self._shm = None
        self._buf = None
        self._last_version = 0
        
        # 协议常量
        self.OFFSET_VERSION = 0
        self.OFFSET_T_WRITE = 8
        self.OFFSET_DOF = 16
        self.OFFSET_H = 20
        self.OFFSET_DT_KEY = 24
        self.OFFSET_Q_KEY = 32
    
    def connect(self, timeout: float = 30.0) -> bool:
        """连接共享内存"""
        from multiprocessing import shared_memory
        
        start = time.time()
        while time.time() - start < timeout:
            try:
                self._shm = shared_memory.SharedMemory(name=self._name)
                self._buf = np.ndarray((self._shm.size,), dtype=np.uint8, buffer=self._shm.buf)
                print(f"[ShmReader] 连接成功: {self._name}, size={self._shm.size} bytes")
                return True
            except FileNotFoundError:
                time.sleep(0.1)
        
        print(f"[ShmReader] 连接超时: {self._name}")
        return False
    
    def read(self) -> Optional[dict]:
        """读取关键帧 (返回 None 如果没有新数据)"""
        if self._buf is None:
            return None
        
        import struct
        
        # 读取 version1 - 使用 bytes() 确保正确读取
        version1 = struct.unpack_from('<Q', bytes(self._buf[:8]), 0)[0]
        
        if version1 == self._last_version or version1 == 0:
            return None
        
        # 读取 header - 使用 bytes() 确保正确读取
        header_bytes = bytes(self._buf[:32])
        t_write = struct.unpack_from('<d', header_bytes, self.OFFSET_T_WRITE)[0]
        dof = struct.unpack_from('<i', header_bytes, self.OFFSET_DOF)[0]
        H = struct.unpack_from('<i', header_bytes, self.OFFSET_H)[0]
        dt_key = struct.unpack_from('<d', header_bytes, self.OFFSET_DT_KEY)[0]
        
        # 读取 payload
        payload_size = H * dof * 8
        q_key_bytes = bytes(self._buf[self.OFFSET_Q_KEY:self.OFFSET_Q_KEY + payload_size])
        q_key = np.frombuffer(q_key_bytes, dtype=np.float64).reshape(H, dof).copy()
        
        # 读取 version2
        version2 = struct.unpack_from('<Q', bytes(self._buf[:8]), 0)[0]
        
        if version1 != version2:
            return None  # 写入过程中被读取
        
        self._last_version = version1
        
        return {
            'version': version1,
            't_write': t_write,
            'dof': dof,
            'H': H,
            'dt_key': dt_key,
            'q_key': q_key,
        }
    
    def disconnect(self):
        """断开连接"""
        if self._shm is not None:
            self._shm.close()
            self._shm = None
            self._buf = None


class CubicInterpolator:
    """三次样条插值器"""
    
    def __init__(self):
        self._splines = []
        self._t0 = 0.0
        self._t_end = 0.0
        self._dof = 0
        self._valid = False
    
    def build(self, q_key: np.ndarray, dt_key: float, t0: float):
        """从关键帧构建插值器"""
        H, dof = q_key.shape
        if H < 2:
            return
        
        self._dof = dof
        self._t0 = t0
        self._t_end = t0 + (H - 1) * dt_key
        
        # 时间数组
        t = np.arange(H) * dt_key + t0
        
        # 为每个关节构建样条
        self._splines = []
        for j in range(dof):
            cs = CubicSpline(t, q_key[:, j], bc_type='clamped')
            self._splines.append(cs)
        
        self._valid = True
    
    def eval(self, t_query: float) -> np.ndarray:
        """采样"""
        if not self._valid:
            return np.zeros(self._dof)
        
        t_query = np.clip(t_query, self._t0, self._t_end)
        
        q = np.zeros(self._dof)
        for j, cs in enumerate(self._splines):
            q[j] = cs(t_query)
        
        return q
    
    @property
    def valid(self) -> bool:
        return self._valid
    
    @property
    def t_end(self) -> float:
        return self._t_end
    
    def remaining_time(self, t_now: float) -> float:
        if not self._valid:
            return 0.0
        return max(0.0, self._t_end - t_now)


class EMAFilter:
    """EMA 滤波器"""
    
    def __init__(self, alpha: float = 0.3, dof: int = 7):
        self._alpha = alpha
        self._dof = dof
        self._state = np.zeros(dof)
        self._initialized = False
    
    def reset(self, initial: np.ndarray = None):
        if initial is not None:
            self._state = initial.copy()
            self._initialized = True
        else:
            self._initialized = False
    
    def filter(self, x: np.ndarray) -> np.ndarray:
        if not self._initialized:
            self._state = x.copy()
            self._initialized = True
            return self._state.copy()
        
        self._state = self._alpha * x + (1 - self._alpha) * self._state
        return self._state.copy()


class SafetyLimiter:
    """安全限制器"""
    
    def __init__(self, config: ServoConfig):
        self._config = config
        self._ema = EMAFilter(alpha=config.ema_alpha, dof=config.dof)
    
    def reset(self, initial: np.ndarray = None):
        self._ema.reset(initial)
    
    def apply(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        """应用安全限制"""
        safe = target.copy()
        
        # 1. 位置限制
        for i in range(len(safe)):
            safe[i] = np.clip(safe[i], 
                             self._config.joint_pos_min[i],
                             self._config.joint_pos_max[i])
        
        # 2. 速率限制
        if current is not None:
            # 关节 (0-5)
            for i in range(6):
                delta = safe[i] - current[i]
                delta = np.clip(delta, 
                               -self._config.max_joint_delta,
                               self._config.max_joint_delta)
                safe[i] = current[i] + delta
            
            # 夹爪 (6)
            gripper_delta = safe[6] - current[6]
            gripper_delta = np.clip(gripper_delta,
                                   -self._config.max_gripper_delta,
                                   self._config.max_gripper_delta)
            safe[6] = current[6] + gripper_delta
        
        # 3. EMA 滤波
        safe = self._ema.filter(safe)
        
        return safe


class PythonServo:
    """Python 伺服控制器"""
    
    def __init__(self, config: ServoConfig):
        self._config = config
        self._shm_reader = ShmKeyframeReader()
        self._interpolator = CubicInterpolator()
        self._safety = SafetyLimiter(config)
        
        self._robot = None
        self._arx5 = None  # arx5 模块引用
        self._current_pos = np.zeros(config.dof)
        self._current_vel = np.zeros(config.dof)
        
        # 控制状态共享内存 (供 inference 读取机器人状态)
        self._control_shm = None
        
        self._running = False
    
    def initialize(self) -> bool:
        """初始化"""
        print("\n" + "=" * 60)
        print("RTC 伺服控制器 (Python 版本)")
        print("=" * 60)
        
        # 0. 创建控制状态共享内存 (供 inference 读取机器人状态)
        print("\n[0] 创建控制状态共享内存...")
        try:
            from inference.shared_state import SharedState
            self._control_shm = SharedState.create("arx5_control")
            print("  ✓ 控制共享内存已创建")
        except Exception as e:
            print(f"  ⚠ 创建控制共享内存失败: {e}")
            self._control_shm = None
        
        # 1. 连接关键帧共享内存
        print("\n[1] 连接关键帧共享内存...")
        if not self._shm_reader.connect(timeout=30.0):
            print("[错误] 无法连接共享内存")
            return False
        
        # 2. 初始化机械臂
        print("\n[2] 初始化机械臂...")
        if not self._config.dry_run:
            try:
                setup_arx5()
                import arx5_interface as arx5
                self._arx5 = arx5
                
                # ========== 关键: 使用正确的配置工厂 ==========
                robot_config = arx5.RobotConfigFactory.get_instance().get_config(self._config.model)
                controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
                    "joint_controller", robot_config.joint_dof
                )
                
                self._robot = arx5.Arx5JointController(
                    robot_config, controller_config, self._config.interface
                )
                self._robot.set_log_level(arx5.LogLevel.INFO)
                
                print(f"  模型: {self._config.model}")
                print(f"  接口: {self._config.interface}")
                
                # ========== 关键: 启用后台通信 ==========
                self._robot.enable_background_send_recv()
                print(f"  ✓ 已启用后台通信")
                
                # ========== 关键: 复位到 home 位置 (设置 PD 增益) ==========
                print(f"  正在复位到 home 位置...")
                self._robot.reset_to_home()
                print(f"  ✓ 已复位到 home")
                
                # ========== 可选: 启用重力补偿 ==========
                urdf_path = os.path.join(
                    os.path.dirname(__file__), 
                    "..", "..", "arx5-sdk", "models", "arx5.urdf"
                )
                if os.path.exists(urdf_path):
                    self._robot.enable_gravity_compensation(urdf_path)
                    print(f"  ✓ 已启用重力补偿")
                
                # 获取初始状态
                state = self._robot.get_state()
                self._current_pos[:6] = np.array(state.pos())
                self._current_pos[6] = state.gripper_pos
                self._current_vel[:6] = np.array(state.vel())
                
                print(f"  初始位置: {self._current_pos}")
                
                # ========== 关键: 立即写入初始状态到共享内存 ==========
                if self._control_shm:
                    self._control_shm.write_robot_state(
                        joint_pos=self._current_pos[:6],
                        gripper_pos=float(self._current_pos[6]),
                        joint_vel=self._current_vel[:6],
                        gripper_vel=float(self._current_vel[6])
                    )
                    print(f"  ✓ 已写入初始状态到共享内存")
                
            except Exception as e:
                print(f"[错误] 初始化机械臂失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("  [模拟模式] 不连接真机")
        
        # 初始化安全限制器
        self._safety.reset(self._current_pos)
        
        print("\n✓ 伺服控制器初始化完成")
        return True
    
    def run(self):
        """主循环"""
        print("\n[Servo] 500Hz 控制循环启动...")
        
        self._running = True
        servo_dt = 1.0 / self._config.servo_rate
        
        loop_count = 0
        update_count = 0
        last_print = time.time()
        
        try:
            while self._running:
                loop_start = time.time()
                
                # 1. 读取关键帧
                data = self._shm_reader.read()
                if data is not None:
                    t_write = data['t_write']
                    self._interpolator.build(data['q_key'], data['dt_key'], t_write)
                    update_count += 1
                    
                    if self._config.verbose:
                        print(f"[Servo] 收到关键帧 v={data['version']}")
                
                # 2. 采样轨迹
                t_now = time.clock_gettime(time.CLOCK_MONOTONIC)
                if self._interpolator.valid:
                    target = self._interpolator.eval(t_now)
                    
                    # 3. 安全限制
                    safe_target = self._safety.apply(target, self._current_pos)
                    
                    # 4. 发送命令
                    self._send_command(safe_target)
                    self._current_pos = safe_target.copy()
                else:
                    # 保持当前位置
                    self._hold_position()
                
                loop_count += 1
                
                # 定期打印
                if time.time() - last_print >= 1.0:
                    remaining = self._interpolator.remaining_time(t_now)
                    print(f"[Servo] loops={loop_count}, updates={update_count}, "
                          f"remaining={remaining:.3f}s")
                    last_print = time.time()
                
                # 5. 频率控制
                elapsed = time.time() - loop_start
                sleep_time = servo_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n[Servo] Ctrl+C 中断")
        
        finally:
            print(f"\n[Servo] 控制循环结束")
            print(f"  总循环数: {loop_count}")
            print(f"  轨迹更新数: {update_count}")
    
    def _send_command(self, target: np.ndarray):
        """发送关节命令"""
        if self._config.dry_run or self._robot is None:
            return
        
        js = self._arx5.JointState(6)  # 6 关节
        js.pos()[:] = target[:6]
        js.gripper_pos = float(target[6])
        
        self._robot.set_joint_cmd(js)
        # 后台模式下不需要 send_recv_once()
        
        # 更新当前状态
        state = self._robot.get_state()
        self._current_pos[:6] = np.array(state.pos())
        self._current_pos[6] = state.gripper_pos
        self._current_vel[:6] = np.array(state.vel())
        
        # 更新控制共享内存 (供 inference 读取)
        if self._control_shm is not None:
            try:
                self._control_shm.write_robot_state(
                    joint_pos=self._current_pos[:6],
                    gripper_pos=float(self._current_pos[6]),
                    joint_vel=self._current_vel[:6],
                    gripper_vel=float(self._current_vel[6])
                )
            except:
                pass  # 忽略写入错误
    
    def _hold_position(self):
        """保持当前位置"""
        self._send_command(self._current_pos)
    
    def shutdown(self):
        """关闭"""
        self._running = False
        
        if self._robot is not None and not self._config.dry_run:
            print("  进入阻尼模式...")
            self._robot.set_to_damping()
        
        self._shm_reader.disconnect()
        
        # 关闭控制共享内存
        if self._control_shm is not None:
            try:
                self._control_shm.close()
            except:
                pass


# 全局变量用于信号处理
_servo_instance: Optional[PythonServo] = None

def signal_handler(sig, frame):
    global _servo_instance
    print(f"\n[Servo] 收到信号 {sig}")
    if _servo_instance:
        _servo_instance._running = False


def main():
    global _servo_instance
    
    parser = argparse.ArgumentParser(description="RTC 伺服控制器 (Python)")
    parser.add_argument("-m", "--model", default="X5", help="机械臂型号")
    parser.add_argument("-i", "--interface", default="can0", help="CAN 接口")
    parser.add_argument("-d", "--dry-run", action="store_true", help="模拟模式")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    config = ServoConfig(
        model=args.model,
        interface=args.interface,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建并运行
    _servo_instance = PythonServo(config)
    
    if not _servo_instance.initialize():
        return 1
    
    _servo_instance.run()
    _servo_instance.shutdown()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
