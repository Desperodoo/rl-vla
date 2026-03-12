"""
手柄遥操核心控制逻辑
从 surreal_server.py 迁移并适配 Flask 后端
"""
import os
import sys
import threading
import asyncio
import time
import logging
import subprocess
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional, List

# ==========================================
# 0. 常量定义
# ==========================================
# 超时时间
CONTROLLER_THREAD_TIMEOUT = 2.0
SDK_LOOP_TIMEOUT = 3.0
SDK_DISCONNECT_TIMEOUT = 5.0
CLEANUP_THREAD_TIMEOUT = 2.0
WORKER_CHECK_INTERVAL = 1.0

# 其他常量
CLEANUP_WORKER_NAME = "JoystickCleanupWorker"

# ==========================================
# 1. 路径配置和依赖导入
# ==========================================
def setup_import_paths():
    """配置手柄遥操模块的导入路径"""
    try:
        # 1. 添加 dependencies 目录（生产环境：surreal 和 utils 已打包在此）
        # auto_start.sh 已经将 dependencies/ 添加到 PYTHONPATH，这里显式添加确保可用
        dependencies_path = '/var/www/backend/carm_backend/dependencies'
        if os.path.exists(dependencies_path) and dependencies_path not in sys.path:
            sys.path.insert(0, dependencies_path)
            logging.debug(f"[Joystick] Added dependencies path: {dependencies_path}")
        
        # 2. 配置 carm_surreal 路径（开发环境兼容）
        carm_surreal_path = os.environ.get('CARM_SURREAL_PATH', '/home/cvte/carm_surreal')
        if os.path.exists(carm_surreal_path):
            for subdir in ['surreal', 'utils', '']:
                path = os.path.join(carm_surreal_path, subdir) if subdir else carm_surreal_path
                if os.path.exists(path) and path not in sys.path:
                    sys.path.insert(0, path)
                    logging.debug(f"[Joystick] Added CARM_SURREAL_PATH: {path}")
        
        # 3. 配置 carm SDK 路径（必需，用于导入 carm 模块）
        sdk_dirs = [
            '/usr/local/bin/carm_ws_ros2/arm_control_sdk',
            '/usr/local/bin/arm_control_sdk',
        ]
        
        for sdk_dir in sdk_dirs:
            if not os.path.exists(sdk_dir):
                continue
                
            python_dir = os.path.join(sdk_dir, 'python')
            lib1 = os.path.join(sdk_dir, 'lib')
            lib2 = os.path.join(sdk_dir, 'poco', 'lib')
            
            # 添加 Python 模块路径
            if python_dir not in sys.path:
                sys.path.insert(0, python_dir)
            
            # 设置 LD_LIBRARY_PATH（如果尚未设置，auto_start.sh 的 setup.bash 也会设置）
            ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            for lib_path in [lib1, lib2]:
                if os.path.exists(lib_path) and lib_path not in ld_path:
                    ld_path = f"{ld_path}:{lib_path}" if ld_path else lib_path
            if ld_path != os.environ.get('LD_LIBRARY_PATH', ''):
                os.environ['LD_LIBRARY_PATH'] = ld_path
            
            # 设置 SDK 目录环境变量
            os.environ['arm_control_sdk_DIR'] = sdk_dir
            logging.debug(f"[Joystick] Configured SDK: {sdk_dir}")
            break
            
    except Exception as e:
        logging.error(f"[Joystick] Error setting up paths: {e}")

# 调用路径配置（必须在导入依赖之前）
setup_import_paths()

# 导入标准库和第三方库（numpy、scipy 从系统镜像中导入）
# 注意：numpy 和 scipy 已经安装在系统镜像中，不需要从 deb 包导入
try:
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    logging.info(f"[Joystick] ✓ Successfully imported numpy and scipy (numpy version: {np.__version__})")
except ImportError as e:
    logging.error(f"[Joystick] ✗ Failed to import numpy/scipy: {e}")
    logging.error(f"[Joystick] Please ensure numpy and scipy are installed in the system image")
    np = None
    R = None
except Exception as e:
    logging.error(f"[Joystick] ✗ Error importing numpy/scipy: {e}")
    np = None
    R = None

# 导入依赖
DEPENDENCIES_AVAILABLE = False
BLEDevice = None

try:
    from surreal.surreal_sdk import SurrealSdk, BLEDevice
    from surreal.surreal_data_model import PoseData, ButtonStatus, CalibrationMode
    from utils.pose_diff import apply_transform_to_pose, homogeneous_diff_with_scale
    from utils.rotation_utils import surreal_to_arm_projection
    from carm import carm_py
    DEPENDENCIES_AVAILABLE = True
    logging.info("[Joystick] ✓ All dependencies imported successfully")
except ImportError as e:
    logging.error(f"[Joystick] ✗ Import failed: {e}")
    logging.error(f"[Joystick] Please check if all dependencies are installed and paths are correct")


def pose_to_dict(pose: Any) -> Dict[str, Any]:
    """将 PoseData 转换为可序列化的字典"""
    if pose is None:
        return None

    def vector_from_attr(obj, base_name):
        value = getattr(obj, base_name, None)
        if isinstance(value, (list, tuple)):
            return [float(value[0]) if len(value) > 0 else 0.0,
                    float(value[1]) if len(value) > 1 else 0.0,
                    float(value[2]) if len(value) > 2 else 0.0]
        comps = []
        for suffix in ('x', 'y', 'z'):
            attr_name = f'{base_name}_{suffix}'
            comps.append(float(getattr(obj, attr_name, 0.0)))
        return comps

    return {
        "x": getattr(pose, 'x', 0.0),
        "y": getattr(pose, 'y', 0.0),
        "z": getattr(pose, 'z', 0.0),
        "qx": getattr(pose, 'qx', 0.0),
        "qy": getattr(pose, 'qy', 0.0),
        "qz": getattr(pose, 'qz', 0.0),
        "qw": getattr(pose, 'qw', 1.0),
        "timestamp": getattr(pose, 'timestamp', None),
        "confidence": getattr(pose, 'confidence', 1.0),
        "linear_velocity": vector_from_attr(pose, 'linear_velocity'),
        "angular_velocity": vector_from_attr(pose, 'angular_velocity'),
        "acceleration": vector_from_attr(pose, 'acceleration'),
    }


def button_to_dict(button: Any) -> Dict[str, Any]:
    """将 ButtonStatus 转换为可序列化的字典"""
    if button is None:
        return None
    return {
        "trigger": getattr(button, 'trigger', 0),
        "grip": getattr(button, 'grip', 0),
        "btn_xa": getattr(button, 'btn_xa', False),
        "btn_yb": getattr(button, 'btn_yb', False),
        "btn_menu": getattr(button, 'btn_menu', False),
        "btn_system": getattr(button, 'btn_system', False),
        "joystick_x": getattr(button, 'joystick_x', getattr(button, 'thumbstick_x', 0.0)),
        "joystick_y": getattr(button, 'joystick_y', getattr(button, 'thumbstick_y', 0.0)),
        "joystick_z": getattr(button, 'joystick_z', getattr(button, 'thumbstick_z', 0.0)),
        "menu_capture": getattr(button, 'menu_capture', getattr(button, 'btn_menu', False)),
        "timestamp": getattr(button, 'timestamp', None),
    }

# ==========================================
# 2. 状态管理类
# ==========================================
class BackgroundBluetoothAgent:
    """后台运行的蓝牙 Agent，用于处理配对请求"""
    def __init__(self):
        self.process = None
        self.running = False

    def start(self):
        if self.running:
            # 检查进程是否真的还在运行
            if self.process:
                try:
                    poll_result = self.process.poll()
                    if poll_result is None:
                        # 进程还在运行
                        return
                    # 进程已结束，清理状态
                    logging.warning(f"[Joystick] Agent process {self.process.pid} already terminated, cleaning up")
                except Exception as e:
                    logging.warning(f"[Joystick] Error checking agent process status: {e}")
                finally:
                    # 统一清理状态
                    self.process = None
                    self.running = False
            else:
                # 状态不一致，重置
                logging.warning("[Joystick] Agent marked as running but process is None, resetting state")
                self.running = False
                return
            
        try:
            # 启动 bluetoothctl
            # -a NoInputNoOutput 实际上不是 bluetoothctl 的参数，而是要通过内部命令设置
            # 但用户已经修改了 agent.c，所以 default-agent 就会注册为 NoInputNoOutput
            self.process = subprocess.Popen(
                ['bluetoothctl'],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=0  # 无缓冲，立即写入
            )
            
            # 发送配置命令
            commands = [
                "power on",
                "agent on",
                "default-agent", 
                # "scan on" # 不开启扫描，以免干扰
            ]
            
            for cmd in commands:
                if self.process and self.process.stdin:
                    self.process.stdin.write(f"{cmd}\n")
            
            if self.process and self.process.stdin:
                self.process.stdin.flush()
                
            self.running = True
            logging.info("[Joystick] Background bluetoothctl agent started")
            
        except Exception as e:
            logging.error(f"[Joystick] Failed to start background agent: {e}")

    def stop(self):
        if not self.process:
            self.running = False
            return
            
        process_pid = self.process.pid
        try:
            # 先尝试优雅终止
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
                logging.info(f"[Joystick] Process {process_pid} terminated gracefully")
            except subprocess.TimeoutExpired:
                # 如果优雅终止失败，强制杀死
                logging.warning(f"[Joystick] Process {process_pid} did not terminate, killing...")
                self.process.kill()
                try:
                    self.process.wait(timeout=1)
                    logging.info(f"[Joystick] Process {process_pid} killed successfully")
                except subprocess.TimeoutExpired:
                    logging.error(f"[Joystick] Process {process_pid} still alive after kill, may become zombie")
        except ProcessLookupError:
            # 进程已经不存在了
            logging.debug(f"[Joystick] Process {process_pid} already terminated")
        except Exception as e:
            logging.error(f"[Joystick] Error stopping bluetoothctl process {process_pid}: {e}")
            # 确保进程被清理
            if self.process:
                try:
                    self.process.kill()
                    self.process.wait(timeout=1)
                except (ProcessLookupError, subprocess.TimeoutExpired, Exception) as e2:
                    logging.debug(f"[Joystick] Final cleanup attempt for {process_pid}: {e2}")
        finally:
            # 确保引用被清理
            self.process = None
            self.running = False
            logging.info("[Joystick] Background bluetoothctl agent stopped")

class JoystickState:
    """全局状态存储（简化版：参考 PyQt 实现，减少锁使用）"""
    def __init__(self):
        # 连接状态相关
        self.ble_connected = False
        self.arm_initialized = False
        self.scan_results = []
        self.calibration = 1  # CalibrationMode.X_LEFT
        self.scale = 0.4
        self.status = "idle"  # idle/scanning/connecting/connected/initializing_arm/arm_ready
        self.log_messages = []
        self.connected_address = None  # 当前连接的手柄地址
        self.connected_name = None  # 当前连接的手柄名称
        self.last_error = None  # 存储最后一次操作的错误信息
        self.disconnect_reason = None  # 断开连接原因
        
        self.pose_data = None  # PoseData 对象（
        self.button_data = None  # ButtonStatus 对象（
        self.data_sequence = 0
        self.last_update_ts = 0.0
        self.battery_level = None  # 手柄电量 (0-100)
        self.rssi = None  # RSSI 信号强度 (dBm)
        self.gripper_pose = None  # 计算后的夹爪位置 (0.0 - gripper_max_value)
        self.gripper_tau = None  # 计算后的夹爪力矩
        
        # 只保留一个锁用于状态管理（低频操作）
        self.status_lock = threading.Lock()

# ==========================================
# 3. CommandController 类
# ==========================================
class CommandController:
    """控制算法核心类"""
    def __init__(self, scale=0.4, gripper_max_value=0.08, joystick_controller=None):
        self.isStop = False
        self.end_arm_pose = [0, 0, 0, 0, 0, 0, 1]
        self.gripper_max_value = gripper_max_value
        self.scale = scale
        
        # 保存JoystickController引用用于缓存操作
        self.joystick_controller = joystick_controller
        
        # 默认初始化位置（保持现有硬编码值作为fallback）
        self.default_init_joint = [-0.000192,1.56713,-0.956551,0.018883,0.907339,-0.004005]
        self.init_joint_ = self.default_init_joint.copy()
        
        self.zero_joint = [0,0,0,0,0,0]
        
        self.init_end_vr_pose = None
        self.init_end_arm_pose = None
        self.target_end_arm_pose = None
        self.end_gripper_pose = None
        self.end_gripper_tau = None
        self.zero_quat = [0, 0, 0, 1]
        
        self.last_aButton = False
        self.last_bButton = False
        self.need_sync_feedback = False
        self.carm_ = None
        self.pose_update_callback = None  # 回调函数，用于更新gripper_pose和gripper_tau 到 state
        self._init_arm_event = None  # 用于等待初始化完成
        self._stop_arm_event = None  # 用于等待停止运动完成

    def _with_temp_arm_connection(self, ip: str, operation):
        """使用临时机械臂连接执行操作

        Args:
            ip: 机械臂 IP 地址
            operation: 操作函数，接收 carm 对象作为参数

        Returns:
            操作函数的返回值
        """
        temp_carm = None
        try:
            temp_carm = carm_py.CArmSingleCol(ip)
            if temp_carm is None:
                raise RuntimeError(f"Failed to create CArmSingleCol instance for {ip}")

            time.sleep(0.2)
            if not temp_carm.is_connected():
                raise ConnectionError(f"Failed to connect to {ip}")

            return operation(temp_carm)
        
        finally:
            if temp_carm and temp_carm.is_connected():
                try:
                    temp_carm.disconnect()
                    logging.debug(f"[CommandController] Temporary connection to {ip} disconnected")
                except Exception as e:
                    logging.warning(f"[CommandController] Error disconnecting temporary connection: {e}")

    def confirmInitPosition(self, ip: str) -> bool:
        """确认初始化位置：读取当前机械臂位置并赋值给 init_joint_"""
        try:
            def read_position(carm):
                temp_joint = carm.get_joint_pos()
                if temp_joint is None or len(temp_joint) != 6:
                    logging.error(f"[CommandController] Invalid joint position data: {temp_joint}")
                    return False
                
                self.init_joint_ = list(temp_joint)
                
                # 保存到缓存
                if self.joystick_controller:
                    self.joystick_controller.set_cached_init_position(self.init_joint_)
                
                logging.warning(f"[CommandController] Init position confirmed and cached: {self.init_joint_}")
                return True
            
            return self._with_temp_arm_connection(ip, read_position)
        
        except ConnectionError as e:
            logging.error(f"[CommandController] {e}")
            return False
        except Exception as e:
            logging.error(f"[CommandController] Error confirming init position: {e}")
            logging.exception(e)
            return False
        
    def getInitPosition(self) -> dict:
        """获取初始化位置"""
        logging.warning(f"[CommandController] Getting init position: {self.init_joint_}")
        if self.init_joint_ is None:
            return {"init_joint": None}
        return {"init_joint": self.init_joint_}
    
    def load_cached_init_position(self, ip: str) -> bool:
        """从缓存加载初始化位置"""
        try:
            if self.joystick_controller:
                cached_position = self.joystick_controller.get_cached_init_position()
                if cached_position and len(cached_position) == 6:
                    # 额外验证缓存数据的有效性
                    if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in cached_position):
                        self.init_joint_ = cached_position
                        logging.info(f"[CommandController] Loaded cached init position: {cached_position}")
                        return True
                    else:
                        logging.warning(f"[CommandController] Cached position contains invalid data types: {cached_position}")
                else:
                    logging.info(f"[CommandController] No valid cached position found")
            else:
                logging.warning("[CommandController] No joystick_controller reference available for cache loading")
        except Exception as e:
            logging.error(f"[CommandController] Error loading cached init position: {e}")
        
        # 使用默认位置（fallback）
        try:
            self.init_joint_ = self.default_init_joint.copy()
            logging.info(f"[CommandController] Using default init position: {self.init_joint_}")
        except Exception as e:
            logging.error(f"[CommandController] Error setting default init position: {e}")
            # 最后的fallback，硬编码默认值
            self.init_joint_ = [-0.000192,1.56713,-0.956551,0.018883,0.907339,-0.004005]
        
        return False

    def getCurrentJoints(self, ip: str) -> dict:
        """获取当前关节值"""
        try:
            def read_joints(carm):
                temp_joint = carm.get_joint_pos()
                if temp_joint is None or len(temp_joint) != 6:
                    logging.error(f"[CommandController] Invalid joint position data: {temp_joint}")
                    return {"current_joints": None, "error": "Invalid joint position data"}
                
                logging.warning(f"[CommandController] Getting current joints: {temp_joint}")
                return {"current_joints": temp_joint}
            
            return self._with_temp_arm_connection(ip, read_joints)
        
        except ConnectionError as e:
            error_msg = f"[CommandController] {e}"
            logging.error(error_msg)
            return {"current_joints": None, "error": error_msg}
        except Exception as e:
            error_msg = f"[CommandController] Error getting current joints: {e}"
            logging.exception(e)
            return {"current_joints": None, "error": error_msg}

    def _release_completion_callback(self, callback_name: str):
        """释放完成回调（通用方法，避免代码重复）"""
        try:
            if self.carm_:
                self.carm_.release_completion_cbk(callback_name)
        except Exception as e:
            logging.warning(f"[CommandController] Error releasing {callback_name} completion callback: {e}")
    
    def _wait_for_arm_movement(self, event: threading.Event, callback_name: str, max_wait_time: float = 50.0):
        """等待机械臂运动完成的通用方法"""
        poll_interval = 0.1
        elapsed_time = 0.0
        last_log_time = 0.0
        
        while not event.wait(timeout=poll_interval):
            elapsed_time += poll_interval
            if elapsed_time >= max_wait_time:
                logging.error(f"[CommandController] {callback_name} completion timeout after {elapsed_time}s")
                self._release_completion_callback(callback_name)
                raise TimeoutError(f"Arm movement completion timeout after {elapsed_time}s")
            
            # 每 1 秒打印一次等待状态
            if elapsed_time - last_log_time >= 1.0:
                logging.debug(f"[CommandController] Waiting for {callback_name} completion... ({elapsed_time:.1f}s)")
                last_log_time = elapsed_time
        
        logging.info(f"[CommandController] {callback_name} movement completed after {elapsed_time:.1f}s")
    
    def _completion_callback_wrapper(self, callback_name: str, event: threading.Event, action):
        """完成回调的通用包装器
        
        Args:
            callback_name: 回调名称
            event: 事件对象
            action: 要执行的操作函数
        """
        try:
            action()
        finally:
            # 释放完成回调
            self._release_completion_callback(callback_name)
            # 通知等待的线程
            if event:
                logging.info(f"[CommandController] Setting {callback_name} event to signal completion")
                event.set()
            else:
                logging.warning(f"[CommandController] {callback_name} event is None, cannot signal completion")
    
    def _init_arm_completion_callback(self, key):
        """初始化机械臂运动完成回调"""
        def action():
            logging.info("[Arm] Moving to safe position completed")
            logging.info("[CommandController] InitArm completion callback triggered")
            logging.info("[Arm] Registering callbacks...")
            self.carm_.register_error_cbk("onCarmError", lambda c, m: logging.error(f"[Arm Error] {c}: {m}"))
            self.carm_.register_pose_cbk(lambda t, p: self.update_pose(p))
            logging.info("[Arm] Initialization completed successfully")
        
        self._completion_callback_wrapper("InitArm", self._init_arm_event, action)
    
    def _stop_arm_completion_callback(self, key):
        """停止机械臂运动完成回调"""
        def action():
            logging.info("[Arm] Moving to zero position completed")
            logging.info("[CommandController] StopArm completion callback triggered")
            logging.info("[Arm] Stop arm movement completed successfully")
        
        self._completion_callback_wrapper("StopArm", self._stop_arm_event, action)

    def initArm(self, ip) -> bool:
        """初始化机械臂连接"""
        if not DEPENDENCIES_AVAILABLE:
            logging.error("[CommandController] Dependencies not available - check carm_api_py, surreal, utils modules")
            return False
            
        try:
            logging.info(f"[Arm] Attempting to connect to {ip}...")

            # 创建 API 包装器
            if self.carm_ is None:
                try:
                    self.carm_ = carm_py.CArmSingleCol(ip)
                except Exception as e:
                    logging.error(f"[Arm] Failed to create CArmSingleCol instance: {e}")
                    self.carm_ = None
                    return False

            # 验证实例是否成功创建
            if self.carm_ is None:
                logging.error("[Arm] CArmSingleCol instance is None after creation attempt")
                return False

            time.sleep(0.2)  # 增加等待时间

            # 检查连接状态
            try:
                if not self.carm_.is_connected():
                    error_msg = f"[Arm] Failed to connect to {ip} - check IP address and network"
                    logging.error(error_msg)
                    return False
            except Exception as e:
                logging.error(f"[Arm] Error checking connection status: {e}")
                return False

            logging.info("[Arm] Connected. Enabling servo...")
            self.carm_.set_ready()
            time.sleep(0.2)

            logging.info("[Arm] Setting control mode...")
            self.carm_.set_control_mode(4)
            time.sleep(0.2)

            logging.info("[Arm] Moving to safe position...")
            # 使用完成回调替代 sleep，等待运动完成
            self._init_arm_event = threading.Event()
            try:
                # 注册运动完成回调
                logging.info("[CommandController] Registering InitArm completion callback...")
                self.carm_.register_completion_cbk("InitArm", self._init_arm_completion_callback)
                # 开始运动
                logging.info(f"[CommandController] Sending move_joint command to position: {self.init_joint_}")
                self.carm_.move_joint(self.init_joint_, 5, False)
                # 等待运动完成
                self._wait_for_arm_movement(self._init_arm_event, "InitArm")
            except Exception as e:
                logging.error(f"[CommandController] Error during initArm movement: {e}")
                # 发生错误时也要释放回调
                self._release_completion_callback("InitArm")
                raise
            finally:
                # 清理事件对象
                self._init_arm_event = None
            return True
        except Exception as e:
            error_msg = f"[Arm] Initialization failed: {e}"
            logging.error(error_msg)
            logging.exception(e)
            return False

    def update_pose(self, pos):
        """更新机械臂位置"""
        self.end_arm_pose = pos

    def _reset_pose_state(self, end_vr_pose):
        """重置姿态状态（提取重复代码）"""
        self.init_end_vr_pose = end_vr_pose.copy()
        self.init_end_arm_pose = self.end_arm_pose[:7]
        self.target_end_arm_pose = None
        # 重置时也更新 state（清空 gripper_pose 和 gripper_tau）
        if self.pose_update_callback:
            try:
                self.pose_update_callback(None, None)
            except Exception as e:
                logging.warning(f"[CommandController] Error resetting gripper state: {e}")

    def surreal_callback(self, data):
        """手柄数据回调"""
        if not DEPENDENCIES_AVAILABLE:
            return
        
        try:
            poseData: PoseData = data['poseData']
            buttonStatus: ButtonStatus = data['buttonStatus']
            
            # 数据解析
            value_trigger = buttonStatus.trigger
            value_squeeze = buttonStatus.grip
            flag_squeeze = value_squeeze > 100
            flag_aButton = buttonStatus.btn_xa
            flag_bButton = buttonStatus.btn_yb

            # 坐标转换
            pose = [poseData.x, poseData.y, poseData.z]
            quat = [poseData.qx, poseData.qy, poseData.qz, poseData.qw]
            end_vr_pose = np.eye(4)
            end_vr_pose[:3, 3] = pose
            end_vr_pose[:3, :3] = R.from_quat(quat).as_matrix()
            end_vr_pose = surreal_to_arm_projection(end_vr_pose, data['calibration'])
            
            # 控制算法计算
            self.compute_control(end_vr_pose, flag_squeeze, flag_aButton, flag_bButton, value_trigger)
        except Exception as e:
            logging.exception(e)
            logging.error(f"[CommandController] Callback error: {e}")

    def compute_control(self, end_vr_pose, flag_squeeze, flag_aButton, flag_bButton, value_trigger):
        """控制算法计算"""
        # A键回零
        if flag_aButton:
            if not flag_squeeze and not self.last_aButton:
                logging.info("[Control] Reset to home")
                if self.carm_:
                    self.carm_.move_joint(self.init_joint_, -1, False)
                self.last_aButton = True
                self.need_sync_feedback = True
            else:
                logging.debug("请松开控制按键，等待回零点完成")
        else:
            if self.last_aButton:
                self._reset_pose_state(end_vr_pose)
            self.last_aButton = False

        # B键姿态锁
        if flag_bButton:
            if not self.last_bButton:
                self._reset_pose_state(end_vr_pose)
                self.last_bButton = True
                logging.info("[Control] Rotation locked")
        else:
            if self.last_bButton:
                self._reset_pose_state(end_vr_pose)
            self.last_bButton = False

        # 初始化/同步逻辑
        if self.init_end_arm_pose is None or self.need_sync_feedback:
            if flag_squeeze:
                self._reset_pose_state(end_vr_pose)
                self.need_sync_feedback = False
                logging.info("[Control] Pose state initialized on squeeze button press")
            else:
                # 松开按钮时不做任何操作，等待按下按钮进行初始化
                return

        # 离合逻辑: 松开不控制，并重置初始位置
        if not flag_squeeze:
            # 重置目标姿态（停止控制）
            self.target_end_arm_pose = None
            # 重置初始位置，避免再次按下时"补偿"松开期间的位移
            self._reset_pose_state(end_vr_pose)
            return

        # 增量计算
        delta_pos, delta_quat = homogeneous_diff_with_scale(self.init_end_vr_pose, end_vr_pose, self.scale)
        if self.last_bButton:
            delta_quat = self.zero_quat

        # 积分
        self.target_end_arm_pose = apply_transform_to_pose(self.init_end_arm_pose, delta_pos, delta_quat)

        # 夹爪控制
        self.end_gripper_pose = max(0.0, min((1.0 - value_trigger/127.0) * self.gripper_max_value, self.gripper_max_value))
        self.end_gripper_tau = 10
        
        # 通过回调更新 state 中的 gripper_pose和gripper_tau
        if self.pose_update_callback:
            try:
                self.pose_update_callback(self.end_gripper_pose, self.end_gripper_tau)
            except Exception as e:
                logging.warning(f"[CommandController] Error updating gripper_pose and gripper_tau in state: {e}")

    def loop(self):
        """控制循环（50Hz）"""
        logging.info("[Control] Loop started")
        while not self.isStop:
            try:
                if self.joystick_controller and self.joystick_controller.state.pose_data and self.joystick_controller.state.button_data:
                    # 读取最新数据
                    pose_data = self.joystick_controller.state.pose_data
                    button_data = self.joystick_controller.state.button_data
                    calibration = self.joystick_controller.state.calibration
                    
                    # 调用控制算法
                    self.surreal_callback({
                        "poseData": pose_data,
                        "buttonStatus": button_data,
                        "calibration": calibration
                    })
                
                # 发送控制指令到机械臂
                if self.target_end_arm_pose is not None and self.carm_ and self.carm_.is_connected():
                    self.carm_.track_pose(self.target_end_arm_pose[:7])
                    
                    if self.end_gripper_pose is not None:
                        self.carm_.set_gripper(self.end_gripper_pose, self.end_gripper_tau)
                
                time.sleep(0.02)  # 50Hz
            except Exception as e:
                logging.exception(e)
                logging.error(f"[Control] Loop error: {e}")
                time.sleep(0.02)
    
    def get_teleop_state(self) -> dict:
        """返回当前遥操作状态（供数据采集端使用）

        Returns:
            dict with target_pose, gripper_pose, gripper_tau, scale, active
        """
        return {
            'target_pose': list(self.target_end_arm_pose) if self.target_end_arm_pose is not None else None,
            'gripper_pose': self.end_gripper_pose,
            'gripper_tau': self.end_gripper_tau,
            'scale': self.scale,
            'active': self.target_end_arm_pose is not None,
        }

    def stop(self):
        """停止控制循环并回到零点位置（等待运动完成）"""
        self.isStop = True
        try:
            if self.carm_ and self.carm_.is_connected():
                logging.info("[CommandController] Moving to zero position...")
                # 使用完成回调等待运动完成
                self._stop_arm_event = threading.Event()
                logging.info("[CommandController] Created _stop_arm_event, waiting for arm movement completion...")
                try:
                    # 注册运动完成回调
                    self.carm_.register_completion_cbk("StopArm", self._stop_arm_completion_callback)
                    logging.info("[CommandController] Registered StopArm completion callback")
                    # 开始运动
                    self.carm_.move_joint(self.zero_joint, 5, False)
                    logging.info("[CommandController] Sent move_joint command, waiting for completion...")
                    # 使用通用方法等待运动完成
                    self._wait_for_arm_movement(self._stop_arm_event, "StopArm")
                    logging.info("[CommandController] Stop command completed, arm reached zero position")
                except Exception as e:
                    logging.error(f"[CommandController] Error during stop movement: {e}")
                    # 发生错误时也要释放回调
                    self._release_completion_callback("StopArm")
                    raise
                finally:
                    # 清理事件对象
                    self._stop_arm_event = None
            else:
                logging.warning("[CommandController] Arm not connected, cannot move to zero position")
        except Exception as e:
            logging.exception(e)
            logging.error(f"[CommandController] Error moving to zero position: {e}")

# ==========================================
# 4. JoystickController 管理类
# ==========================================
class JoystickController:
    """手柄遥操控制器管理类"""
    def __init__(self, scale=0.4, gripper_max_value=0.08):
        if not DEPENDENCIES_AVAILABLE:
            logging.warning("[JoystickController] Dependencies not available")
            self.available = False
            return

        self.available = True
        self.scale = scale
        self.gripper_max_value = gripper_max_value

        self.sdk = None
        self.controller = None
        self.controller_thread = None
        self.sdk_loop = None
        self.sdk_loop_thread = None

        # 启动后台 Agent
        self.agent = BackgroundBluetoothAgent()
        self.agent.start()

        self.state = JoystickState()
        self.state.scale = scale

        self._initialized = False
        self._lock = threading.Lock()

        # 初始化位置内存缓存
        self.cached_init_position: Optional[List[float]] = None

        # 单线程清理模式：使用常驻线程处理所有清理请求
        self._cleanup_event = threading.Event()  # 触发清理的事件
        self._cleanup_thread = None  # 常驻清理线程
        self._cleanup_lock = threading.Lock()  # 保护清理状态
        self._cleanup_in_progress = False  # 标记清理是否正在进行
        self._cleanup_shutdown = False  # 标记是否正在关闭

        # 启动清理线程
        self._start_cleanup_thread()

    def _safe_execute(self, operation: callable, error_msg: str, log_level: str = "error") -> bool:
        """安全执行操作，自动处理异常和日志记录

        Args:
            operation: 要执行的操作函数
            error_msg: 错误信息
            log_level: 日志级别 ("debug", "info", "warning", "error")

        Returns:
            bool: 操作是否成功
        """
        try:
            operation()
            return True
        except Exception as e:
            log_func = getattr(logging, log_level, logging.error)
            log_func(f"{error_msg}: {e}")
            if log_level == "error":
                logging.exception(e)
            return False

    def _stop_controller_thread(self, timeout: float = CONTROLLER_THREAD_TIMEOUT) -> bool:
        """停止控制器线程（辅助方法，避免代码重复）
        
        Args:
            timeout: 等待线程停止的超时时间
            
        Returns:
            bool: 线程是否成功停止
        """
        if not self.controller_thread:
            return True
        
        if not self.controller_thread.is_alive():
            self.controller_thread = None
            return True
        
        # 停止控制器
        if self.controller:
            self.controller.stop()
        
        # 等待线程停止
        self.controller_thread.join(timeout=timeout)
        if self.controller_thread.is_alive():
            logging.error("[JoystickController] Controller thread did not stop within timeout")
            return False
        
        self.controller_thread = None
        return True

    def _ensure_controller_thread_stopped(self) -> bool:
        """确保控制器线程已停止（辅助方法，用于init_arm前的检查）
        
        Returns:
            bool: 如果线程已停止或不存在则返回True，否则返回False
        """
        if not self.controller_thread:
            return True
        
        if not self.controller_thread.is_alive():
            self.controller_thread = None
            return True
        
        return False

    def _start_cleanup_thread(self):
        """启动常驻清理线程"""
        with self._cleanup_lock:
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                return
            if self._cleanup_shutdown:
                logging.warning("[JoystickController] Cleanup system is shutting down, cannot start new thread")
                return

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name=CLEANUP_WORKER_NAME
        )
        self._cleanup_thread.start()
        logging.info("[JoystickController] Cleanup worker thread started")

    def _cleanup_worker(self):
        """常驻清理工作线程，处理所有清理请求"""
        while not self._cleanup_shutdown:
            try:
                # 等待清理请求或关闭信号
                if self._cleanup_event.wait(timeout=WORKER_CHECK_INTERVAL):  # 定期检查shutdown标志
                    self._cleanup_event.clear()

                    # 执行清理
                    with self._cleanup_lock:
                        self._cleanup_in_progress = True

                    try:
                        logging.info("[CleanupWorker] Starting cleanup operation")
                        self.stop_session()
                        logging.info("[CleanupWorker] Cleanup operation completed")
                    except Exception as e:
                        logging.error(f"[CleanupWorker] Cleanup operation failed: {e}")
                        logging.exception(e)
                    finally:
                        with self._cleanup_lock:
                            self._cleanup_in_progress = False
                # 如果是timeout，继续循环检查shutdown标志

            except Exception as e:
                logging.error(f"[CleanupWorker] Worker error: {e}")
                # 短暂延迟，避免异常循环
                time.sleep(0.1)

        logging.info("[CleanupWorker] Cleanup worker thread shutting down")

    def _stop_controller(self) -> bool:
        """停止控制器和控制线程"""
        if not self.controller:
            return True

        try:
            logging.info("[JoystickController] Stopping control loop and moving to zero position")
            
            # 先停止控制线程（会调用controller.stop()移动到零位）
            # 必须在断开连接之前执行，因为stop()需要连接才能移动机械臂
            if not self._stop_controller_thread():
                return False
            logging.info("[JoystickController] Controller thread stopped successfully")

            # 断开机械臂连接（在移动到零位之后）
            if self.controller and self.controller.carm_:
                try:
                    if self.controller.carm_.is_connected():
                        self.controller.carm_.disconnect()
                        logging.info("[JoystickController] Arm disconnected")
                    else:
                        logging.debug("[JoystickController] Arm connection already closed")
                except Exception as e:
                    logging.error(f"[JoystickController] Error disconnecting arm: {e}")
                    # 即使断开失败，也继续清理controller引用

            # 清理controller引用
            self.controller = None
            return True

        except Exception as e:
            logging.exception(e)
            return False

    def _stop_sdk(self) -> bool:
        """停止SDK和事件循环"""
        if not self.sdk or not self.sdk_loop:
            return True

        try:
            # 断开BLE连接
            future = asyncio.run_coroutine_threadsafe(
                self.sdk.disconnect(),
                self.sdk_loop
            )
            future.result(timeout=SDK_DISCONNECT_TIMEOUT)
            logging.info("[JoystickController] SDK disconnected")
            return True

        except FuturesTimeoutError:
            logging.warning("[JoystickController] SDK disconnect timed out, using fallback")
            self._disconnect_now_trusted_devices()
            return False
        except Exception as e:
            logging.error(f"[JoystickController] Error disconnecting SDK: {e}")
            self._disconnect_now_trusted_devices()
            return False

    def _stop_sdk_loop(self) -> bool:
        """停止SDK事件循环线程"""
        if not self.sdk_loop_thread:
            return True

        # 先停止 SDK 中的所有任务（如果 SDK 还存在）
        if self.sdk:
            try:
                self.sdk.stop()  # 这会停止所有后台任务并调用 loop.stop()
            except Exception as e:
                logging.debug(f"[JoystickController] Error calling sdk.stop(): {e}")

        # 停止事件循环
        if self.sdk_loop and self.sdk_loop.is_running():
            self.sdk_loop.call_soon_threadsafe(self.sdk_loop.stop)

        # 等待线程结束
        if self.sdk_loop_thread.is_alive():
            self.sdk_loop_thread.join(timeout=SDK_LOOP_TIMEOUT)
            if self.sdk_loop_thread.is_alive():
                logging.warning("[JoystickController] SDK loop thread did not stop within timeout - this indicates a thread leak")
                # 清理引用以便后续可以创建新线程
                # 注意：daemon 线程在主进程退出时会自动终止，但在长期运行的服务中，这会导致线程泄漏
                self.sdk_loop_thread = None
                self.sdk_loop = None
                return False

        # 清理引用
        self.sdk_loop_thread = None
        self.sdk_loop = None
        logging.info("[JoystickController] SDK loop thread stopped successfully")
        return True

    def _cleanup_resources(self) -> bool:
        """清理其他资源"""
        success = True

        # 停止蓝牙代理
        if self.agent:
            success &= self._safe_execute(
                lambda: self.agent.stop(),
                "[JoystickController] Error stopping background agent"
            )
            if success:
                logging.info("[JoystickController] Background bluetooth agent stopped")

        # 断开信任设备
        success &= self._safe_execute(
            self._disconnect_now_trusted_devices,
            "[JoystickController] Error disconnecting trusted devices",
            "debug"  # 这个不是严重错误
        )

        return success

    def _reset_state(self) -> None:
        """重置状态"""
        with self.state.status_lock:
            self.state.ble_connected = False
            self.state.arm_initialized = False
            self.state.status = "idle"
            self.state.connected_address = None
            self.state.connected_name = None
            self.state.pose_data = None
            self.state.button_data = None
            self.state.battery_level = None
            self.state.rssi = None
            self.state.gripper_pose = None
            self.state.gripper_tau = None
            self.state.data_sequence += 1

    def shutdown_cleanup_thread(self):
        """停止清理线程"""
        # 检查当前线程是否是清理线程本身
        current_thread = threading.current_thread()
        is_cleanup_thread = (self._cleanup_thread and 
                            current_thread.ident == self._cleanup_thread.ident)
        
        with self._cleanup_lock:
            self._cleanup_shutdown = True

        # 设置事件以唤醒等待的线程
        self._cleanup_event.set()

        # 如果当前线程就是清理线程本身，不能 join 自己
        if is_cleanup_thread:
            logging.info("[JoystickController] Cleanup thread shutting down itself, skipping join()")
            return

        # 等待线程退出（只有非清理线程才能执行到这里）
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=CLEANUP_THREAD_TIMEOUT)
            if self._cleanup_thread.is_alive():
                logging.warning("[JoystickController] Cleanup thread did not stop within timeout")
            else:
                logging.info("[JoystickController] Cleanup thread stopped successfully")

        self._cleanup_thread = None
        logging.info("[JoystickController] Cleanup thread shutdown completed")

    def request_stop_session(self) -> None:
        """请求停止会话（线程安全，非阻塞）"""
        with self._cleanup_lock:
            if self._cleanup_in_progress or self._cleanup_shutdown:
                logging.warning("[JoystickController] Cleanup already in progress or shutting down, ignoring new request")
                return

        logging.info("[JoystickController] Requesting cleanup operation")
        self._cleanup_event.set()

    def initialize(self):
        """初始化 SDK（延迟初始化）"""
        if not self.available:
            return False
            
        if self._initialized:
            return True
            
        try:
            with self._lock:
                if self._initialized:
                    return True
                    
                self.sdk = SurrealSdk()
                self.sdk.set_data_callback(self.on_pose)
                self.sdk.set_button_callback(self.on_button)
                self.sdk.set_battery_callback(self.on_battery)
                self.sdk.set_rssi_callback(self.on_rssi)
                # 设置断开连接回调，使用线程安全的包装函数
                # 注意：断开连接回调可能在事件循环线程中被调用，需要线程安全
                def safe_stop_session():
                    """线程安全的 stop_session 包装函数，避免在初始化过程中误触发"""
                    try:
                        # 检查是否正在初始化机械臂，如果是则忽略断开连接
                        with self.state.status_lock:
                            is_initializing = (
                                self.state.status == "initializing_arm"
                                or (self.controller and self.controller.carm_ and not self.state.arm_initialized)
                            )
                        
                        if is_initializing:
                            logging.warning("[JoystickController] Ignoring disconnection callback during arm initialization")
                            return
                        
                        # 记录断开原因
                        logging.warning("[Disconnect] 🔌 BLE disconnection detected - Reason: Joystick initiated disconnect (battery low, power off, or signal lost)")
                        
                        # 使用单线程清理模式：通过Event触发常驻清理线程
                        self.request_stop_session()
                    except Exception as e:
                        logging.error(f"[JoystickController] Error in safe_stop_session: {e}")
                
                self.sdk.set_disconnected_callback(safe_stop_session)
                
                # 确保 agent 运行（初始化 SDK 时需要 agent 处理配对请求）
                self._ensure_agent_running()
                
                # 启动 SDK 事件循环
                self.start_sdk_loop()

                self._initialized = True
                logging.info("[JoystickController] Initialized successfully")
                return True
        except Exception as e:
            logging.exception(e)
            logging.error(f"[JoystickController] Initialization failed: {e}")
            return False
    
    def _ensure_agent_running(self):
        """确保蓝牙 Agent 正在运行（辅助方法，避免代码重复）"""
        if not self.agent:
            logging.warning("[JoystickController] Agent not initialized, creating new one")
            self.agent = BackgroundBluetoothAgent()
        
        if not self.agent.running:
            logging.info("[JoystickController] Agent not running, starting...")
            self.agent.start()
        elif self.agent.process:
            # 检查进程是否真的还在运行
            try:
                poll_result = self.agent.process.poll()
                if poll_result is not None:
                    # 进程已结束，重新启动
                    logging.warning("[JoystickController] Agent process terminated, restarting...")
                    self.agent.running = False
                    self.agent.start()
            except Exception as e:
                logging.warning(f"[JoystickController] Error checking agent process: {e}")
                # 如果检查失败，尝试重新启动
                self.agent.running = False
                self.agent.start()

    def _ensure_sdk_loop_running(self):
        """确保 SDK 事件循环正在运行（辅助方法，避免代码重复）"""
        if not self.sdk_loop or not self.sdk_loop.is_running():
            logging.warning("[JoystickController] SDK loop not running, restarting...")
            self.start_sdk_loop()
            time.sleep(0.2)  # 等待事件循环启动
            # 重新设置 SDK 的事件循环引用
            if self.sdk and self.sdk_loop:
                self.sdk.loop = self.sdk_loop

    def start_sdk_loop(self):
        """启动 SDK 事件循环（独立线程）"""
        # 如果已有线程在运行，先尝试停止它
        if self.sdk_loop_thread and self.sdk_loop_thread.is_alive():
            logging.warning("[JoystickController] SDK loop thread already running, stopping it first")
            if not self._stop_sdk_loop():
                logging.error("[JoystickController] Failed to stop existing SDK loop thread, cannot create new one! This indicates a thread leak.")
                return
        
        # 清理已停止的线程引用（确保引用被清理）
        if self.sdk_loop_thread and not self.sdk_loop_thread.is_alive():
            logging.info("[JoystickController] Cleaning up old SDK loop thread")
            self.sdk_loop_thread = None
            self.sdk_loop = None
        
        def run_loop():
            try:
                self.sdk_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.sdk_loop)
                if self.sdk:
                    self.sdk.loop = self.sdk_loop
                self.sdk_loop.run_forever()
            except Exception as e:
                logging.exception(e)
                logging.error(f"[JoystickController] SDK loop error: {e}")
        
        self.sdk_loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.sdk_loop_thread.start()
        time.sleep(0.1)  # 等待循环启动
        logging.info("[JoystickController] SDK loop started")

    def on_pose(self, data):
        """位置数据回调"""
        # 直接存储原始对象
        self.state.pose_data = data
        self.state.last_update_ts = time.time()
        self.state.data_sequence += 1
    
    def on_button(self, data):
        """按钮数据回调"""
        # 直接存储原始对象
        self.state.button_data = data
        self.state.last_update_ts = time.time()
        self.state.data_sequence += 1

    def on_battery(self, battery_level: int):
        """电量数据回调"""
        self.state.battery_level = battery_level
        self.state.data_sequence += 1

    def on_rssi(self, rssi_level: int):
        """RSSI 数据回调"""
        self.state.rssi = rssi_level
        self.state.data_sequence += 1

    def _update_gripper_state(self, gripper_pose, gripper_tau):
        """更新 gripper_pose 和 gripper_tau 到 state（由 CommandController 调用）"""
        self.state.gripper_pose = float(gripper_pose) if gripper_pose is not None else None
        self.state.gripper_tau = float(gripper_tau) if gripper_tau is not None else None
        self.state.data_sequence += 1

    def _disconnect_now_trusted_devices(self):
        """强制取消信任并移除当前连接的蓝牙设备，防止僵尸连接和缓存影响"""
        try:
            target_address = None

            # 1) 优先使用状态机中记录的连接地址（最准确）
            with self.state.status_lock:
                if self.state.connected_address:
                    target_address = self.state.connected_address

            # 2) 如无记录，再从 bluetoothctl devices Trusted/Paired 中兜底获取
            if not target_address:
                for option in ["Trusted", "Paired"]:
                    try:
                        cmd = f"bluetoothctl devices {option}"
                        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
                        for line in output.split("\n"):
                            line = line.strip()
                            if not line.startswith("Device "):
                                continue
                            parts = line.split()
                            if len(parts) >= 2:
                                # 只取第一个匹配的设备作为兜底目标
                                target_address = parts[1]
                                logging.info(
                                    f"[JoystickController] Fallback found {option} device to untrust/remove: {target_address}"
                                )
                                break
                        if target_address:
                            break
                    except Exception as e:
                        logging.warning(
                            f"[JoystickController] Failed to list bluetoothctl devices {option}: {e}"
                        )

            # 3) 对目标地址执行 untrust + remove（如果有）
            if target_address:
                logging.info(f"[JoystickController] Try to untrust and remove device {target_address}")
                cmd = f"bluetoothctl untrust {target_address}"
                subprocess.run(
                    cmd,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                cmd = f"bluetoothctl remove {target_address}"
                subprocess.run(
                    cmd,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logging.info(
                    f"[JoystickController] Force untrusted and removed current connected device {target_address}"
                )
            else:
                logging.info("[JoystickController] No target device found to untrust/remove")
        except Exception as e:
            logging.error(f"[JoystickController] Failed to clean up bluetooth devices: {e}")

    def scan_devices(self) -> list:
        """扫描 BLE 设备"""
        if not self.available or not self._initialized:
            if not self.initialize():
                with self.state.status_lock:
                    self.state.last_error = "手柄遥操功能不可用"
                return []
        
        # 确保 agent 和 SDK 事件循环都在运行
        self._ensure_agent_running()
        self._ensure_sdk_loop_running()
        
        try:
            with self.state.status_lock:
                self.state.status = "scanning"
                self.state.last_error = None
            
            future = asyncio.run_coroutine_threadsafe(
                self.sdk.scan_devices(),
                self.sdk_loop
            )
            scanned_devices = future.result(timeout=10)
            
            device_list = [{"name": d.name, "address": d.address} for d in scanned_devices]
            
            with self.state.status_lock:
                self.state.scan_results = device_list
                self.state.status = "idle"
                self.state.last_error = None
            
            logging.info(f"[JoystickController] Scan completed, found {len(device_list)} devices")
            return device_list
            
        except (asyncio.TimeoutError, FuturesTimeoutError):
            error_msg = "扫描超时，请检查蓝牙是否开启"
            logging.warning(f"[JoystickController] {error_msg}")
            with self.state.status_lock:
                self.state.status = "idle"
                self.state.last_error = error_msg
            return []
        except Exception as e:
            error_msg = f"扫描设备失败: {str(e)}"
            logging.exception(e)
            with self.state.status_lock:
                self.state.status = "idle"
                self.state.last_error = error_msg
            return []
    
    def connect_device(self, address: str, name: str = "Unknown") -> bool:
        """连接 BLE 设备
        
        Args:
            address: 设备地址
            name: 设备名称（可选，默认为 "Unknown"）
        """
        if not self.available or not self._initialized:
            if not self.initialize():
                with self.state.status_lock:
                    self.state.last_error = "手柄遥操功能不可用"
                return False
        
        # 确保 agent 和 SDK 事件循环都在运行（连接前必须确保 agent 运行）
        self._ensure_agent_running()
        self._ensure_sdk_loop_running()
        
        if BLEDevice is None:
            error_msg = "BLEDevice 模块不可用"
            logging.error(f"[JoystickController] {error_msg}")
            with self.state.status_lock:
                self.state.last_error = error_msg
            return False
        
        try:
            logging.info(f"[JoystickController] Connecting to device {name} ({address})...")
            
            with self.state.status_lock:
                self.state.status = "connecting"
            
            device = BLEDevice(name=name, address=address)
            
            future = asyncio.run_coroutine_threadsafe(
                self.sdk.connect_device(device),
                self.sdk_loop
            )
            success = future.result(timeout=15)
            
            # 更新状态（包含设备名称）
            with self.state.status_lock:
                self.state.ble_connected = success
                self.state.status = "connected" if success else "idle"
                self.state.last_error = None if success else "连接失败"
                if success:
                    self.state.connected_address = address
                    self.state.connected_name = name
            
            if success:
                logging.info(f"[JoystickController] Device {name} ({address}) connected successfully")
            
            return success
            
        except asyncio.TimeoutError:
            error_msg = f"连接超时，请检查设备 {address} 是否在范围内"
            logging.error(f"[JoystickController] {error_msg}")
            with self.state.status_lock:
                self.state.ble_connected = False
                self.state.status = "idle"
                self.state.last_error = error_msg
            return False
            
        except Exception as e:
            error_msg = f"连接设备失败: {str(e)}"
            logging.exception(e)
            with self.state.status_lock:
                self.state.ble_connected = False
                self.state.status = "idle"
                self.state.last_error = error_msg
            return False
    
    def get_last_error(self) -> str:
        """获取最后一次操作的错误信息"""
        with self.state.status_lock:
            return self.state.last_error

    def init_arm(self, ip: str) -> bool:
        """初始化机械臂"""
        if not self.available:
            logging.warning("[JoystickController] Controller not available")
            return False
        
        try:
            # 设置状态为 initializing_arm
            with self.state.status_lock:
                self.state.status = "initializing_arm"
            
            # 如果 controller 已存在且已连接，直接返回成功
            if self.controller and self.controller.carm_ and self.controller.carm_.is_connected():
                if self.controller_thread and self.controller_thread.is_alive():
                    logging.info("[JoystickController] Arm already initialized and connected")
                    with self.state.status_lock:
                        self.state.arm_initialized = True
                        self.state.status = "arm_ready"
                    return True
            
            # 如果 controller 存在但连接断开，先停止旧的 loop 线程
            if self.controller and self.controller_thread and self.controller_thread.is_alive():
                logging.info("[JoystickController] Stopping old control loop...")
                self.controller.stop()
                self.controller_thread.join(timeout=2.0)
                if self.controller_thread.is_alive():
                    logging.error("[JoystickController] Old controller thread did not stop within timeout, cannot create new one")
                    with self.state.status_lock:
                        self.state.arm_initialized = False
                        self.state.status = "connected" if self.state.ble_connected else "idle"
                    return False
                if self.controller.carm_:
                    self.controller.carm_ = None
            
            # 创建或重用 controller
            if not self.controller:
                logging.info("[JoystickController] Creating new CommandController")
                self.controller = CommandController(self.scale, self.gripper_max_value, self)
                # 设置回调，用于更新 gripper_pose 和 gripper_tau 到 state
                self.controller.pose_update_callback = self._update_gripper_state
            
            # 初始化机械臂（同步调用）
            logging.info(f"[JoystickController] Attempting to initialize arm at {ip}...")
            # 在初始化前尝试加载缓存的初始化位置
            self.controller.load_cached_init_position(ip)
            success = self.controller.initArm(ip)
            
            if success:
                # 启动控制循环（确保没有旧线程在运行）
                if not self._ensure_controller_thread_stopped():
                    logging.error("[JoystickController] Old controller thread still alive, cannot create new one! This indicates a thread leak.")
                    with self.state.status_lock:
                        self.state.arm_initialized = False
                        self.state.status = "connected" if self.state.ble_connected else "idle"
                    return False
                
                self.controller.isStop = False
                self.controller_thread = threading.Thread(
                    target=self.controller.loop,
                    daemon=True
                )
                self.controller_thread.start()
                logging.warning("[JoystickController] Control loop thread started")
                
                with self.state.status_lock:
                    self.state.arm_initialized = True
                    self.state.status = "arm_ready"
                
                logging.warning(f"[JoystickController] Arm initialized successfully at {ip}")
                return True
            else:
                error_msg = f"[JoystickController] Failed to initialize arm at {ip}"
                logging.error(error_msg)
                # 检查连接状态
                if self.controller and self.controller.carm_:
                    connected = self.controller.carm_.is_connected()
                    logging.debug(f"[JoystickController] Arm connection status: {connected}")
                with self.state.status_lock:
                    self.state.arm_initialized = False
                    self.state.status = "connected" if self.state.ble_connected else "idle"
                return False
        except Exception as e:
            error_msg = f"[JoystickController] Exception during arm initialization: {str(e)}"
            logging.error(error_msg)
            logging.exception(e)
            with self.state.status_lock:
                self.state.arm_initialized = False
                self.state.status = "connected" if self.state.ble_connected else "idle"
            return False
    
    def set_calibration(self, value: int) -> bool:
        """设置标定模式"""
        try:
            calibration = CalibrationMode(value)
            with self.state.status_lock:
                self.state.calibration = value
            return True
        except Exception as e:
            logging.error(f"[JoystickController] Invalid calibration value: {e}")
            return False
    
    def set_scale(self, value: float) -> bool:
        """设置灵敏度"""
        try:
            with self.state.status_lock:
                self.state.scale = value
            if self.controller:
                self.controller.scale = value
            return True
        except Exception as e:
            logging.exception(e)
            return False
    
    def get_status(self) -> dict:
        """获取当前状态"""
        with self.state.status_lock:
            return {
                "ble_connected": self.state.ble_connected,
                "arm_initialized": self.state.arm_initialized,
                "calibration": self.state.calibration,
                "scale": self.state.scale,
                "status": self.state.status,
                "battery_level": self.state.battery_level,
                "rssi": self.state.rssi
            }
    
    def get_teleop_state(self) -> dict:
        """获取当前遥操作目标位姿（供数据采集端使用）"""
        if self.controller:
            return self.controller.get_teleop_state()
        return {
            'target_pose': None,
            'gripper_pose': None,
            'gripper_tau': None,
            'scale': None,
            'active': False,
        }

    def confirm_init_position(self, ip: str) -> bool:
        """确认初始化位置"""
        if not self.available:
            logging.warning("[JoystickController] Controller not available")
            return False
        
        try:
            # 如果 controller 不存在，创建一个（但不初始化机械臂）
            if not self.controller:
                logging.info("[JoystickController] Creating CommandController for init position confirmation")
                self.controller = CommandController(self.scale, self.gripper_max_value, self)
            
            # 调用 CommandController 的方法确认初始化位置
            success = self.controller.confirmInitPosition(ip)
            if success:
                logging.warning(f"[JoystickController] Init position confirmed successfully")
            else:
                logging.error(f"[JoystickController] Failed to confirm init position")
            return success
        except Exception as e:
            error_msg = f"[JoystickController] Exception during confirm init position: {str(e)}"
            logging.error(error_msg)
            logging.exception(e)
            return False
    
    def get_init_position(self) -> dict:
        """获取初始化位置"""
        if not self.available:
            return {
                "init_joint": None,
                "error": "Controller not available"
            }
        
        try:
            if not self.controller:
                logging.info("[JoystickController] Creating CommandController for getting init position")
                self.controller = CommandController(self.scale, self.gripper_max_value, self)
            
            return self.controller.getInitPosition()
        except Exception as e:
            error_msg = f"[JoystickController] Exception getting init position: {str(e)}"
            logging.exception(e)
            return {
                "init_joint": None,
                "error": error_msg
            }
    
    def get_current_joints(self, ip: str) -> dict:
        """获取当前关节值"""
        if not self.available:
            return {
                "current_joints": None,
                "error": "Controller not available"
            }
        
        try:
            if not self.controller:
                logging.info("[JoystickController] Creating CommandController for getting current joints")
                self.controller = CommandController(self.scale, self.gripper_max_value, self)
            return self.controller.getCurrentJoints(ip)
        except Exception as e:
            error_msg = f"[JoystickController] Exception getting current joints: {str(e)}"
            logging.exception(e)
            return {
                "current_joints": None,
                "error": error_msg
            }
    
    def stop_session(self) -> bool:
        """停止遥操会话：停止控制循环，回到零点位置，并断开 BLE"""
        if not self.available:
            return False

        logging.info("[JoystickController] Beginning session cleanup")

        # 按顺序停止各个组件
        success = True
        success &= self._stop_controller()
        success &= self._stop_sdk()
        success &= self._stop_sdk_loop()
        success &= self._cleanup_resources()

        # 重置状态
        self._reset_state()

        logging.info("[JoystickController] Session cleanup completed" + (" with errors" if not success else " successfully"))
        return success
    
    def set_cached_init_position(self, position: List[float]) -> None:
        """设置缓存的初始化位置"""
        # 验证位置数据
        if not isinstance(position, (list, tuple)):
            logging.error(f"[JoystickController] Position must be a list or tuple, got {type(position)}")
            return
        
        if len(position) != 6:
            logging.error(f"[JoystickController] Position must have exactly 6 elements, got {len(position)}")
            return
        
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in position):
            logging.error(f"[JoystickController] All position elements must be numbers, got types: {[type(x) for x in position]}")
            return
        
        try:
            with self.state.status_lock:
                self.cached_init_position = list(position)  # 确保是list类型
                logging.info(f"[JoystickController] Cached init position: {position}")
        except Exception as e:
            logging.error(f"[JoystickController] Error caching init position: {e}")
    
    def get_cached_init_position(self) -> Optional[List[float]]:
        """获取缓存的初始化位置"""
        try:
            with self.state.status_lock:
                if (self.cached_init_position and 
                    len(self.cached_init_position) == 6 and
                    all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in self.cached_init_position)):
                    logging.info(f"[JoystickController] Retrieved cached init position: {self.cached_init_position}")
                    return self.cached_init_position.copy()
                elif self.cached_init_position:
                    logging.warning(f"[JoystickController] Cached position validation failed. Position: {self.cached_init_position}")
                return None
        except Exception as e:
            logging.error(f"[JoystickController] Error retrieving cached init position: {e}")
            return None
    

# ==========================================
# 5. 全局单例
# ==========================================
joystick_controller = None

def get_joystick_controller():
    """获取控制器实例（单例）"""
    global joystick_controller
    if joystick_controller is None:
        joystick_controller = JoystickController()
    return joystick_controller
