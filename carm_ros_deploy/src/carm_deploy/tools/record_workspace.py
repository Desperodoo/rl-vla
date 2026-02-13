#!/usr/bin/env python3
"""
CARM 安全操作空间记录工具

通过将机械臂设置为拖动模式（示教模式），用户手动拖动机械臂覆盖工作空间，
脚本实时记录边界点，自动计算安全限位并保存到 JSON 配置文件。

使用方法:
    1. 运行脚本: python record_workspace.py --output /path/to/carm_deploy/safety_config.json
    2. 机械臂将进入拖动模式，可以手动拖动
    3. 拖动机械臂覆盖整个工作空间
    4. 按 's' 保存配置，按 'r' 重置，按 'q' 退出

输出配置文件可用于 inference_ros.py:
    rosrun carm_deploy inference_ros.py --safety_config /path/to/carm_deploy/safety_config.json
"""

import argparse
import os
import sys
import time
import threading
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple

# CARM SDK
from carm import carm_py


@dataclass
class WorkspaceBounds:
    """工作空间边界（末端位置）"""
    x_min: float = float('inf')
    x_max: float = float('-inf')
    y_min: float = float('inf')
    y_max: float = float('-inf')
    z_min: float = float('inf')
    z_max: float = float('-inf')
    
    def update(self, x: float, y: float, z: float):
        """更新边界"""
        self.x_min = min(self.x_min, x)
        self.x_max = max(self.x_max, x)
        self.y_min = min(self.y_min, y)
        self.y_max = max(self.y_max, y)
        self.z_min = min(self.z_min, z)
        self.z_max = max(self.z_max, z)
    
    def is_valid(self) -> bool:
        """检查是否有有效数据"""
        return (self.x_min != float('inf') and 
                self.x_max != float('-inf'))
    
    def apply_margin(self, margin: float = 0.05) -> 'WorkspaceBounds':
        """应用安全裕度（收缩边界，与关节限位保持一致）
        
        注意：收缩边界 = 采集范围向内缩小 margin%
        - x_min, y_min, z_min 增大（远离边界）
        - x_max, y_max, z_max 减小（远离边界）
        
        这样确保机械臂不会超出采集时的安全范围。
        """
        if not self.is_valid():
            return self
        
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        z_range = self.z_max - self.z_min
        
        return WorkspaceBounds(
            x_min=self.x_min + margin * x_range,  # 收缩：min 增大
            x_max=self.x_max - margin * x_range,  # 收缩：max 减小
            y_min=self.y_min + margin * y_range,  # 收缩：min 增大
            y_max=self.y_max - margin * y_range,  # 收缩：max 减小
            z_min=self.z_min + margin * z_range,  # 收缩：min 增大（远离桌面）
            z_max=self.z_max - margin * z_range,  # 收缩：max 减小
        )
    
    def __str__(self):
        if not self.is_valid():
            return "WorkspaceBounds(no data)"
        return (f"X: [{self.x_min:.3f}, {self.x_max:.3f}] "
                f"Y: [{self.y_min:.3f}, {self.y_max:.3f}] "
                f"Z: [{self.z_min:.3f}, {self.z_max:.3f}]")


@dataclass
class JointBounds:
    """关节角度边界"""
    joint_min: np.ndarray = field(default_factory=lambda: np.full(6, float('inf')))
    joint_max: np.ndarray = field(default_factory=lambda: np.full(6, float('-inf')))
    gripper_min: float = float('inf')
    gripper_max: float = float('-inf')
    
    def update(self, joints: List[float], gripper: float):
        """更新边界"""
        for i in range(min(6, len(joints))):
            self.joint_min[i] = min(self.joint_min[i], joints[i])
            self.joint_max[i] = max(self.joint_max[i], joints[i])
        self.gripper_min = min(self.gripper_min, gripper)
        self.gripper_max = max(self.gripper_max, gripper)
    
    def is_valid(self) -> bool:
        """检查是否有有效数据"""
        return self.joint_min[0] != float('inf')
    
    def apply_margin(self, margin: float = 0.05) -> 'JointBounds':
        """应用安全裕度（扩展边界）"""
        if not self.is_valid():
            return self
        
        joint_range = self.joint_max - self.joint_min
        gripper_range = self.gripper_max - self.gripper_min
        
        new_bounds = JointBounds()
        new_bounds.joint_min = self.joint_min - margin * joint_range
        new_bounds.joint_max = self.joint_max + margin * joint_range
        new_bounds.gripper_min = max(0.0, self.gripper_min - margin * gripper_range)
        new_bounds.gripper_max = min(0.08, self.gripper_max + margin * gripper_range)
        
        return new_bounds
    
    def __str__(self):
        if not self.is_valid():
            return "JointBounds(no data)"
        lines = []
        for i in range(6):
            lines.append(f"  J{i+1}: [{self.joint_min[i]:.3f}, {self.joint_max[i]:.3f}]")
        lines.append(f"  Gripper: [{self.gripper_min:.3f}, {self.gripper_max:.3f}]")
        return "JointBounds:\n" + "\n".join(lines)


class WorkspaceRecorder:
    """
    工作空间记录器
    
    通过拖动模式记录机械臂的安全操作空间
    """
    
    def __init__(self, robot_ip: str = '10.42.0.101', margin: float = 0.05):
        """
        初始化记录器
        
        Args:
            robot_ip: 机械臂 IP 地址
            margin: 安全裕度（默认 5%）
        """
        self.robot_ip = robot_ip
        self.margin = margin
        
        # 边界记录
        self.workspace_bounds = WorkspaceBounds()
        self.joint_bounds = JointBounds()
        
        # 采样点数量
        self.sample_count = 0
        
        # 当前状态
        self.current_pose = None  # [x, y, z, qx, qy, qz, qw]
        self.current_joints = None  # [j1-j6]
        self.current_gripper = None
        
        # 控制
        self.running = True
        self.recording = False
        
        # 连接机械臂
        print(f"Connecting to robot at {robot_ip}...")
        self.arm = carm_py.CArmSingleCol(robot_ip)
        time.sleep(1.0)
        
        # 初始化
        print("Setting robot ready...")
        self.arm.set_ready()
        time.sleep(0.5)
        
        # 打印状态
        self._print_status()
    
    def _print_status(self):
        """打印机械臂状态"""
        status = self.arm.get_status()
        print("=" * 50)
        print(f"Robot: {status.arm_name}")
        print(f"Connected: {status.arm_is_connected}")
        print(f"DOF: {status.arm_dof}")
        print(f"State: {status.state}")
        print("=" * 50)
    
    def enable_drag_mode(self):
        """
        启用拖动模式
        
        注意: 这里使用 DRAG 模式 (mode=3) 而不是 PF 模式 (mode=4)，
        因为需要手动拖动机械臂记录工作空间边界。
        控制模式说明:
            - mode=0: IDLE (空闲)
            - mode=1: POSITION (禁用! 不安全)
            - mode=2: MIT
            - mode=3: DRAG (拖动示教，本脚本专用)
            - mode=4: PF (力位混合，推理时使用)
        """
        print("\n>>> Enabling DRAG mode (mode=3)...")
        ret = self.arm.set_control_mode(3)  # DRAG 模式用于示教
        if ret != 0:
            print(f"Warning: set_control_mode returned {ret}")
        time.sleep(0.5)
        print(">>> Robot is now in DRAG mode. You can manually move the arm.")
        self.recording = True
    
    def disable_drag_mode(self):
        """禁用拖动模式，恢复空闲模式"""
        print("\n>>> Disabling DRAG mode, returning to IDLE...")
        self.recording = False
        self.arm.set_control_mode(0)
        time.sleep(0.5)
    
    def safe_shutdown(self):
        """安全关闭：PF 模式回零位 → 下使能 → 断开连接"""
        print("\n" + "=" * 60)
        print("  安全关闭流程")
        print("=" * 60)
        
        try:
            # 1. 复位并进入 PF 模式
            print("\n[1/4] 设置 PF 模式准备回零位...")
            self.arm.set_ready()
            time.sleep(0.5)
            ret = self.arm.set_control_mode(4)  # PF 模式
            print(f"      PF 模式: {'✓' if ret == 0 else f'返回码 {ret}'}")
            time.sleep(0.3)
            
            # 2. 回零位
            print("[2/4] 回零位...")
            home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ret = self.arm.move_joint(home, -1, True)
            print(f"      回零位: {'✓' if ret == 0 else f'返回码 {ret}'}")
            time.sleep(0.5)
            
            # 3. 切换到 IDLE → 下使能
            print("[3/4] 下使能...")
            self.arm.set_control_mode(0)  # IDLE
            time.sleep(0.3)
            ret = self.arm.set_servo_enable(False)
            print(f"      下使能: {'✓' if ret == 0 else f'返回码 {ret}'}")
            time.sleep(0.3)
            
            # 4. 断开连接
            print("[4/4] 断开连接...")
            self.arm.disconnect()
            print("      ✓ 已断开连接")
            
        except Exception as e:
            print(f"\n⚠️  安全关闭过程异常: {e}")
            try:
                self.arm.set_control_mode(0)
                self.arm.disconnect()
            except:
                pass
        
        print("\n" + "=" * 60)
        print("  安全关闭完成")
        print("=" * 60)
    
    def update_state(self):
        """更新当前状态"""
        try:
            self.current_pose = self.arm.get_cart_pose()  # [x,y,z,qx,qy,qz,qw]
            self.current_joints = self.arm.get_joint_pos()  # [j1-j6]
            self.current_gripper = self.arm.get_gripper_pos()
            
            # 如果正在记录，更新边界
            if self.recording and self.current_pose:
                self.workspace_bounds.update(
                    self.current_pose[0],
                    self.current_pose[1],
                    self.current_pose[2]
                )
                self.joint_bounds.update(
                    self.current_joints,
                    self.current_gripper
                )
                self.sample_count += 1
                
        except Exception as e:
            print(f"Error updating state: {e}")
    
    def reset_bounds(self):
        """重置边界记录"""
        self.workspace_bounds = WorkspaceBounds()
        self.joint_bounds = JointBounds()
        self.sample_count = 0
        print("\n>>> Bounds reset!")
    
    def get_config_dict(self) -> dict:
        """
        生成配置字典
        
        注意:
            - 关节限制: 使用机械臂官方限位 + 10% 裕度 (不使用采集数据)
            - 工作空间: 使用采集数据 + margin 裕度 (严格遵守)
        """
        # 工作空间使用采集数据 + margin
        ws = self.workspace_bounds.apply_margin(self.margin)
        
        # 关节限制使用机械臂官方限位 + 10% 裕度
        # 官方限位: upper=[2.79, 3.14, 0.0, 2.65, 1.57, 2.88]
        #          lower=[-2.79, 0.0, -3.14, -2.65, -1.57, -2.88]
        # 注意: 对于边界为0的关节（J2下限、J3上限），不收缩该边界，确保零位在安全范围内
        CARM_JOINT_UPPER = np.array([2.79, 3.14, 0.0, 2.65, 1.57, 2.88])
        CARM_JOINT_LOWER = np.array([-2.79, 0.0, -3.14, -2.65, -1.57, -2.88])
        joint_range = CARM_JOINT_UPPER - CARM_JOINT_LOWER
        joint_margin = 0.10  # 10% 裕度
        
        safe_joint_min = np.zeros(6)
        safe_joint_max = np.zeros(6)
        for i in range(6):
            # 下限: 如果是0就保持0，否则向内收缩
            if abs(CARM_JOINT_LOWER[i]) < 0.01:
                safe_joint_min[i] = CARM_JOINT_LOWER[i]
            else:
                safe_joint_min[i] = CARM_JOINT_LOWER[i] + joint_margin * joint_range[i]
            # 上限: 如果是0就保持0，否则向内收缩
            if abs(CARM_JOINT_UPPER[i]) < 0.01:
                safe_joint_max[i] = CARM_JOINT_UPPER[i]
            else:
                safe_joint_max[i] = CARM_JOINT_UPPER[i] - joint_margin * joint_range[i]
        
        # 夹爪使用官方限位 + 10% 裕度 (与关节保持一致)
        # 官方范围: 0 ~ 0.08m
        CARM_GRIPPER_MIN = 0.0
        CARM_GRIPPER_MAX = 0.08
        gripper_margin = 0.10  # 10% 裕度
        gripper_range = CARM_GRIPPER_MAX - CARM_GRIPPER_MIN
        safe_gripper_min = CARM_GRIPPER_MIN + gripper_margin * gripper_range  # 0.008
        safe_gripper_max = CARM_GRIPPER_MAX - gripper_margin * gripper_range  # 0.072
        
        return {
            'joint_limits': {
                'joint_min': safe_joint_min.tolist(),
                'joint_max': safe_joint_max.tolist(),
                'gripper_min': safe_gripper_min,
                'gripper_max': safe_gripper_max,
            },
            'workspace_limits': {
                'x_min': ws.x_min,
                'x_max': ws.x_max,
                'y_min': ws.y_min,
                'y_max': ws.y_max,
                'z_min': ws.z_min,
                'z_max': ws.z_max,
            },
            'safety_params': {
                'max_joint_delta': 0.1,
                'max_gripper_delta': 0.02,
                'max_position_delta': 0.02,
                'max_rotation_delta': 0.1,
                'filter_alpha': 0.3,
            },
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'robot_ip': self.robot_ip,
                'sample_count': self.sample_count,
                'margin': self.margin,
            }
        }
    
    def save_config(self, output_path: str):
        """保存配置到 JSON 文件"""
        if not self.workspace_bounds.is_valid():
            print("Error: No valid data recorded. Move the arm first!")
            return False
        
        config = self.get_config_dict()
        
        output_path = os.path.expanduser(output_path)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n>>> Configuration saved to: {output_path}")
        print(f">>> Samples recorded: {self.sample_count}")
        print(f">>> Margin applied: {self.margin * 100:.0f}%")
        return True
    
    def print_current_bounds(self):
        """打印当前边界"""
        print("\n" + "=" * 50)
        print("CURRENT RECORDED BOUNDS")
        print("=" * 50)
        
        if not self.workspace_bounds.is_valid():
            print("No data recorded yet. Move the arm!")
        else:
            print(f"\n[Raw bounds] (samples: {self.sample_count})")
            print(f"Workspace: {self.workspace_bounds}")
            print(f"\n{self.joint_bounds}")
            
            # 带 margin 的边界
            ws_margin = self.workspace_bounds.apply_margin(self.margin)
            jb_margin = self.joint_bounds.apply_margin(self.margin)
            print(f"\n[With {self.margin*100:.0f}% margin]")
            print(f"Workspace: {ws_margin}")
        
        print("=" * 50)
    
    def run_interactive(self, output_path: str):
        """运行交互式记录"""
        print("\n" + "=" * 60)
        print("WORKSPACE RECORDER - Interactive Mode")
        print("=" * 60)
        print("\nControls:")
        print("  ENTER - Toggle recording on/off (enables/disables drag mode)")
        print("  p     - Print current bounds")
        print("  r     - Reset bounds")
        print("  s     - Save configuration and continue")
        print("  q     - Save and quit")
        print("  Ctrl+C - Quit without saving")
        print("\n" + "=" * 60)
        
        # 启动状态更新线程
        update_thread = threading.Thread(target=self._update_loop, daemon=True)
        update_thread.start()
        
        # 启动显示线程
        display_thread = threading.Thread(target=self._display_loop, daemon=True)
        display_thread.start()
        
        print("\n>>> Press ENTER to start recording (enable drag mode)...")
        
        try:
            import termios
            import tty
            
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                
                while self.running:
                    c = sys.stdin.read(1)
                    
                    if c == '\n' or c == '\r':  # Enter
                        if not self.recording:
                            self.enable_drag_mode()
                        else:
                            self.disable_drag_mode()
                            print(">>> Recording paused. Press ENTER to resume.")
                    
                    elif c == 'p':
                        self.print_current_bounds()
                    
                    elif c == 'r':
                        self.reset_bounds()
                    
                    elif c == 's':
                        self.save_config(output_path)
                    
                    elif c == 'q':
                        if self.workspace_bounds.is_valid():
                            self.save_config(output_path)
                        self.running = False
                        break
                    
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                
        except (KeyboardInterrupt, EOFError):
            print("\n>>> Interrupted by user")
        
        finally:
            self.running = False
            if self.recording:
                self.recording = False
            print("\n>>> Workspace recorder stopped.")
            self.safe_shutdown()
    
    def _update_loop(self):
        """状态更新循环"""
        while self.running:
            self.update_state()
            time.sleep(0.01)  # 100 Hz
    
    def _display_loop(self):
        """显示更新循环"""
        while self.running:
            if self.current_pose and self.recording:
                # 清行并打印当前状态
                status = "RECORDING" if self.recording else "PAUSED"
                pose_str = f"XYZ: [{self.current_pose[0]:.3f}, {self.current_pose[1]:.3f}, {self.current_pose[2]:.3f}]"
                bounds_str = f"X:[{self.workspace_bounds.x_min:.2f},{self.workspace_bounds.x_max:.2f}] "
                bounds_str += f"Y:[{self.workspace_bounds.y_min:.2f},{self.workspace_bounds.y_max:.2f}] "
                bounds_str += f"Z:[{self.workspace_bounds.z_min:.2f},{self.workspace_bounds.z_max:.2f}]"
                
                # 使用回车符覆盖当前行
                print(f"\r[{status}] {pose_str} | Bounds: {bounds_str} | Samples: {self.sample_count}    ", end='', flush=True)
            
            time.sleep(0.1)  # 10 Hz 显示更新


def parse_args():
    parser = argparse.ArgumentParser(description='CARM Workspace Recorder')
    parser.add_argument('--robot_ip', type=str, default='10.42.0.101',
                        help='Robot IP address')
    parser.add_argument('--output', '-o', type=str, default='',
                        help='Output configuration file path (default: carm_deploy/safety_config.json)')
    parser.add_argument('--margin', type=float, default=0.05,
                        help='Safety margin (0.05 = 5%%)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output:
        output_path = os.path.expanduser(args.output)
    else:
        carm_deploy_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(carm_deploy_root, 'safety_config.json')
    
    print("=" * 60)
    print("CARM Workspace Recorder")
    print("=" * 60)
    print(f"Robot IP: {args.robot_ip}")
    print(f"Output file: {output_path}")
    print(f"Safety margin: {args.margin * 100:.0f}%")
    print("=" * 60)
    
    try:
        recorder = WorkspaceRecorder(
            robot_ip=args.robot_ip,
            margin=args.margin
        )
        recorder.run_interactive(output_path)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
