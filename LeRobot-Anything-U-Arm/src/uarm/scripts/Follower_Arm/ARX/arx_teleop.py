#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import serial
import serial.tools.list_ports
import time
import re
import numpy as np
import os
import sys
import subprocess

# ====== ARX5 Interface Path Configuration ======
# 使用 rl-vla 统一配置模块
try:
    from inference.config import setup_arx5
    setup_arx5()
except ImportError:
    # 兼容旧方式: 通过环境变量或默认路径
    ARX5_SDK_PATH = os.environ.get('ARX5_SDK_PATH', os.path.expanduser('~/rl-vla/arx5-sdk'))
    ROOT_DIR = os.path.join(ARX5_SDK_PATH, 'python')
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import arx5_interface as arx5  # You should install arx5 api first, following https://github.com/real-stanford/arx5-sdk/tree/main/python/examples


# ====== Serial Port Detection ======
def find_servo_port():
    """Auto-detect the master arm servo controller serial port.
    
    Excludes known non-servo devices:
    - CANable (for ARX5 CAN bus)
    - WLAN adapters
    - RealSense cameras
    
    Returns the first suitable USB serial port or None.
    """
    exclude_keywords = ['canable', 'wlan', 'realsense', '802.11', 'wifi']
    
    # Try using udevadm for better device identification
    try:
        # 优先检查 uarm_servo（当前已知的 CH341 串口）
        for port in ['/dev/uarm_servo', '/dev/ttyUSB0', '/dev/ttyUSB1']:
            if not os.path.exists(port):
                continue
                
            # Get device info
            result = subprocess.run(
                ['udevadm', 'info', '--name=' + port, '--attribute-walk'],
                capture_output=True, text=True, timeout=2
            )
            info = result.stdout.lower()
            
            # Skip if matches exclude keywords
            if any(keyword in info for keyword in exclude_keywords):
                continue
                
            # Check if it looks like a USB serial device (CH341, CP210x, FTDI)
            if 'usb serial' in info or 'ch340' in info or 'ch341' in info or 'cp210' in info or 'ftdi' in info or '1a86' in info:
                print(f"[AutoDetect] Found potential servo controller: {port}")
                return port
                
    except Exception as e:
        print(f"[AutoDetect] udevadm detection failed: {e}")
    
    # Fallback: use pyserial's list_ports
    print("[AutoDetect] Using pyserial port detection...")
    for port in serial.tools.list_ports.comports():
        # 安全处理 None 值
        description = port.description or ""
        manufacturer = port.manufacturer or ""
        port_lower = (description + " " + manufacturer).lower()
        
        if any(keyword in port_lower for keyword in exclude_keywords):
            continue
        if 'usb' in port_lower or port.device.startswith('/dev/ttyUSB'):
            print(f"[AutoDetect] Found potential servo controller: {port.device}")
            return port.device
    
    return None


# ====== Master Arm Serial Reading Class ======
# ====== Master Arm Serial Reading Class ======
class ServoReader:
    def __init__(self, port=None, baudrate=115200):
        # Auto-detect port if not specified
        if port is None:
            port = find_servo_port()
            if port is None:
                raise RuntimeError(
                    "Could not find servo controller serial port!\n"
                    "Please connect your master arm servo controller and try again.\n"
                    "Or specify the port manually: ServoReader(port='/dev/ttyUSBx')"
                )
        
        self.SERIAL_PORT = port
        self.BAUDRATE = baudrate
        
        try:
            self.ser = serial.Serial(self.SERIAL_PORT, self.BAUDRATE, timeout=0.1)
            print(f"[ServoReader] ✓ Serial port {self.SERIAL_PORT} opened successfully")
        except serial.SerialException as e:
            raise RuntimeError(
                f"Failed to open {self.SERIAL_PORT}: {e}\n"
                f"Try: sudo chmod 666 {self.SERIAL_PORT}"
            )

        self.zero_angles = [0.0] * 7
        self.current_angles = [0.0] * 7
        self.lock = threading.Lock()

        self._init_servos()

    def send_command(self, cmd):
        self.ser.write(cmd.encode('ascii'))
        time.sleep(0.008)
        return self.ser.read_all().decode('ascii', errors='ignore')

    def pwm_to_angle(self, response_str, pwm_min=500, pwm_max=2500, angle_range=270):
        match = re.search(r'P(\d{4})', response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        pwm_span = pwm_max - pwm_min
        angle = (pwm_val - pwm_min) / pwm_span * angle_range
        return angle

    def _init_servos(self):
        self.send_command('#000PVER!')
        time.sleep(0.01)
        for i in range(7):
            self.send_command("#000PCSK!")
            self.send_command(f'#{i:03d}PULK!')
            response = self.send_command(f'#{i:03d}PRAD!')
            angle = self.pwm_to_angle(response.strip())
            self.zero_angles[i] = angle if angle is not None else 0.0
        print("[ServoReader] Servo initial angle calibration completed")

    def read_loop(self, hz=100):
        dt = 1.0 / hz
        while True:
            new_angles = [0.0] * 7
            for i in range(7):
                response = self.send_command(f'#{i:03d}PRAD!')
                angle = self.pwm_to_angle(response.strip())
                if angle is not None:
                    new_angles[i] = angle - self.zero_angles[i]
            with self.lock:
                self.current_angles = new_angles
            time.sleep(dt)

    def get_angles(self):
        """Get the most recently read angles (thread-safe)"""
        with self.lock:
            return list(self.current_angles)


# ====== Slave Arm Control Class (with velocity limiting) ======
class ArxTeleop:
    def __init__(self, model="X5", interface="can0"):
        # Initialize robot with simplified API
        self.ctrl = arx5.Arx5JointController(model, interface)
        
        self.robot_cfg = self.ctrl.get_robot_config()
        self.ctrl_cfg = self.ctrl.get_controller_config()
        self.dof = self.robot_cfg.joint_dof  # e.g. 6

        # Optional: Set log level
        # self.ctrl.set_log_level(arx5.LogLevel.DEBUG)

        # PID gains
        gain = arx5.Gain(self.dof)
        gain.kd()[:] = 0.01
        self.ctrl.set_gain(gain)

        self.ctrl.reset_to_home()

        # Master-slave mapping parameters (assuming servo 0-5 corresponds to joints 0-5)
        self.scale = [1.0] * self.dof
        self.offset_rad = [0.0] * self.dof

        # Gripper channel
        self.gripper_index = 6
        self.gripper_min_deg = -10.0
        self.gripper_max_deg = 30

        # === Velocity limiting parameters (can be adjusted per joint) ===
        # Maximum joint velocity (rad/s), default uniform upper limit (about 69 deg/s)
        self.max_joint_vel = np.array([1.2] * self.dof, dtype=np.float64)
        self.max_joint_vel[3:] = np.array([2] * (self.dof - 3), dtype=np.float64)
        # Maximum gripper "normalized velocity" (/s), maximum change per second in 0~1 range
        self.max_gripper_vel = 2.0

        # Optional: velocity feedforward switch and ratio (refer to your example)
        self.use_vel_feedforward = False
        self.vel_ff_gain = 0.3

        # Velocity limiting state (target sent in previous cycle)
        self._inited_cmd = False
        self._last_cmd_pos = np.zeros(self.dof, dtype=np.float64)
        self._last_cmd_grip = 0.0

    def _deg_to_rad_mapped(self, master_angles_deg):
        """Master arm angle (degrees) -> Slave arm joint angle (radians)"""
        joints_rad = np.zeros(self.dof, dtype=np.float64)
        
        # Map first 6 servos to joints (servo 6 is gripper, handled separately)
        for j in range(min(self.dof, 6)):
            joints_rad[j] = np.deg2rad(master_angles_deg[j]) * self.scale[j] + self.offset_rad[j]
        
        # Apply transformations based on simulation's working configuration
        # Joint 2 needs to be inverted
        joints_rad[2] = -joints_rad[2]
        
        # Swap and invert joints 4 and 5
        if self.dof >= 6:
            joints_rad[4], joints_rad[5] = -joints_rad[5], -joints_rad[4]

        return joints_rad

    def _map_gripper(self, master_angles_deg):
        """Map master arm gripper angle to ARX5 gripper position
        
        Uses same physical range as test_joint_control.py: 0 to 0.08m
        """
        # Convert degrees to radians
        grip_deg = master_angles_deg[self.gripper_index]
        grip_rad = np.deg2rad(grip_deg)
        
        # Use same mapping as simulation: angle_to_gripper()
        # For ARX5: gripper range is 0 to 0.08m (verified by test_joint_control.py)
        angle_range = 0.15 * np.pi  # Default servo range (~4.71 rad)
        ratio = max(0, 1 + (grip_rad / angle_range))
        
        # ARX5 gripper physical range: 0 to 0.08m (same as test_joint_control.py)
        gripper_pos = 0.0 + (0.08 - 0.0) * ratio
        
        return float(np.clip(gripper_pos, 0.0, 0.08))

    def send_cmd(self, master_angles_deg):
        # Desired pose
        desired_pos = self._deg_to_rad_mapped(master_angles_deg)
        desired_grip = self._map_gripper(master_angles_deg)

        dt = float(self.ctrl_cfg.controller_dt)

        # Initialize: first frame directly align
        if not self._inited_cmd:
            self._last_cmd_pos[:] = desired_pos
            self._last_cmd_grip = desired_grip
            self._inited_cmd = True
            print(f"[ArxTeleop] Initial joint positions (rad): {desired_pos}")
            print(f"[ArxTeleop] DOF: {self.dof}")

        # === Joint velocity limiting: limit maximum step per cycle ===
        max_step = self.max_joint_vel * dt                     # Maximum allowed displacement per cycle
        delta = desired_pos - self._last_cmd_pos
        delta_clipped = np.clip(delta, -max_step, max_step)
        cmd_pos = self._last_cmd_pos + delta_clipped

        # === Gripper velocity limiting ===
        grip_delta = desired_grip - self._last_cmd_grip
        grip_step = self.max_gripper_vel * dt
        grip_cmd = self._last_cmd_grip + float(np.clip(grip_delta, -grip_step, grip_step))

        # Organize command
        js = arx5.JointState(self.dof)

        js.pos()[:] = cmd_pos
        js.gripper_pos = grip_cmd

        # Optional: velocity feedforward (based on displacement after velocity limiting)
        if self.use_vel_feedforward and hasattr(js, "vel"):
            safe_vel = delta_clipped / max(dt, 1e-6)
            js.vel()[:] = self.vel_ff_gain * safe_vel

        # Send command
        self.ctrl.set_joint_cmd(js)
        
        # Without background thread, must explicitly send/receive
        self.ctrl.send_recv_once()

        # Update state
        self._last_cmd_pos[:] = cmd_pos
        self._last_cmd_grip = grip_cmd


# ====== Main Program ======
if __name__ == "__main__":
    servo_reader = None
    teleop = None
    
    try:
        print("\n" + "="*60)
        print("ARX5 Teleoperation System")
        print("="*60)
        
        # Create master arm reader (auto-detect port)
        print("\n[1/2] Initializing master arm serial connection...")
        servo_reader = ServoReader(port=None, baudrate=115200)  # Auto-detect
        
        # Create slave arm controller (with velocity limiting)
        print("\n[2/2] Initializing ARX5 slave arm (CAN)...")
        teleop = ArxTeleop(model="X5", interface="can0")

        # Start reading thread
        t_reader = threading.Thread(target=servo_reader.read_loop, kwargs={"hz": 100}, daemon=True)
        t_reader.start()

        print("\n" + "="*60)
        print("✓ Teleoperation started successfully!")
        print("  Master arm: {} @ {} baud".format(servo_reader.SERIAL_PORT, servo_reader.BAUDRATE))
        print("  Slave arm:  ARX5 via can0")
        print("\nPress Ctrl+C to exit.")
        print("="*60 + "\n")
        
        dt = teleop.ctrl_cfg.controller_dt
        
        loop_count = 0
        # Main control loop - just read latest angles and send commands
        while True:
            master_angles = servo_reader.get_angles()
            
            # Print debug info every 50 loops (~0.1s)
            if loop_count % 50 == 0:
                print(f"\n[Main] Master angles (deg): {[f'{a:.2f}' for a in master_angles]}")
                # Get current joint state
                state = teleop.ctrl.get_state()
                print(f"[Main] ARX joint pos (rad): {[f'{p:.3f}' for p in state.pos()]}")
            
            teleop.send_cmd(master_angles)
            time.sleep(dt)
            loop_count += 1

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user, robot arm returning to home...")
        if teleop:
            teleop.ctrl.reset_to_home()
        print("[Main] ✓ Exited cleanly.")
    except Exception as e:
        print(f"\n[Main] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        if teleop:
            try:
                teleop.ctrl.reset_to_home()
            except:
                pass
        print("[Main] Exited with error.")
