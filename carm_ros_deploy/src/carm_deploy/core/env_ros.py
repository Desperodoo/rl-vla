#!/usr/bin/env python3
"""
CARM 机械臂 ROS 环境封装
基于 carm_real/env_api.py 重构，将 svar 通信替换为 ROS1 原生通信
"""

import threading
import rospy
import cv2
import numpy as np
import time
import requests
from scipy.spatial.transform import Rotation as R
from typing import Optional

# CARM SDK
from carm import carm_py

# ROS 图像同步
import sys
import os
# 添加父目录到路径以访问 utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_sync import ImageSynchronizer
from core.camera_config import DEFAULT_SYNC_SLOP, normalize_camera_topics


class RealEnvironment:
    """
    真实机械臂环境封装
    提供机械臂状态获取、控制和图像观测
    """
    
    def __init__(self, config):
        """
        初始化环境
        
        Args:
            config: 配置字典，包含以下字段：
                - robot_ip: 机械臂 IP 地址
                - robot_mode: 控制模式 (0-idle 空闲模式 1-position 点位控制模式, 2-MIT 力矩模式， 3-drag 拖动模式，4-PF 力位混合模式)
                - robot_tau: 夹爪力矩
                - arm_init_pose: 初始末端位姿 [x, y, z, qx, qy, qz, qw]
                - arm_init_gripper: 初始夹爪开度
                - camera_topics: 相机话题列表
                - sync_slop: 图像同步容差
                - vis: 是否可视化图像
        """
        self.config = config
        
        # 默认参数
        self.robot_ip = config.get('robot_ip', '10.42.0.101')
        self.robot_mode = config.get('robot_mode', 2)  # 默认 MIT 模式
        self.tau = config.get('robot_tau', 10.0)
        
        # 安全检查：禁止使用 Position 模式 (mode=1)
        if self.robot_mode == 1:
            rospy.logwarn("=" * 60)
            rospy.logwarn("⚠️  安全警告: Position 模式 (mode=1) 已被禁用!")
            rospy.logwarn("   自动切换到 MIT 模式 (mode=2)")
            rospy.logwarn("=" * 60)
            self.robot_mode = 2  # 强制使用 MIT 模式
        
        # 从实际机械臂读取的初始位姿 (2026-01-13)
        self.arm_init_pose = config.get('arm_init_pose', [0.2475, 0.0014, 0.3251, 0.9996, -0.0034, 0.0255, -0.0074])
        self.arm_init_gripper = config.get('arm_init_gripper', 0.078)
        self.camera_topics = normalize_camera_topics(config.get('camera_topics'))
        self.sync_slop = float(config.get('sync_slop', DEFAULT_SYNC_SLOP))
        self.vis = config.get('vis', False)
        self.passive_mode = config.get('passive_mode', False)  # 被动模式：不设置控制模式
        self.skip_init_confirm = config.get('skip_init_confirm', False)  # 跳过初始化确认
        self.return_to_zero = config.get('return_to_zero', True)  # 退出时回到零位(关节角度全为0)
        
        # 速度语义拆分为：
        # - init_speed: 初始化/回位等大动作的速度
        # - normal_speed_level: 运行阶段与退出恢复后的默认速度
        self.init_speed = float(config.get('init_speed', 2.0))
        self.normal_speed_level = float(config.get('normal_speed_level', 10.0))
        for key, value in (
            ('init_speed', self.init_speed),
            ('normal_speed_level', self.normal_speed_level),
        ):
            if value < 0.0:
                rospy.logwarn(f"⚠️  {key} 从 {value} 提升到 0.0")
            if value > 10.0:
                rospy.logwarn(f"⚠️  {key} 从 {value} 限制到 10.0")
        self.init_speed = min(max(self.init_speed, 0.0), 10.0)
        self.normal_speed_level = min(max(self.normal_speed_level, 0.0), 10.0)
        
        # backend 请求显式禁用环境代理，避免 10.42.x.x 内网地址被 http_proxy/https_proxy 误转发
        self._teleop_http = requests.Session()
        self._teleop_http.trust_env = False

        # 初始化机械臂
        self.arm = carm_py.CArmSingleCol(self.robot_ip)
        time.sleep(1.0)  # 等待连接稳定
        
        # 设置机械臂（被动模式下跳过，避免干扰手柄遥操作）
        if not self.passive_mode:
            self.arm.set_ready()
            self.arm.set_control_mode(self.robot_mode)
        else:
            rospy.loginfo("Passive mode: skipping control mode setup")
        
        # 打印状态
        self._print_arm_status()
        
        # 初始化位置（被动模式下跳过）
        if not self.passive_mode:
            # 用户确认（除非跳过）
            if not self.skip_init_confirm:
                self._wait_for_init_confirmation()
            self.init_status()
        
        # 状态变量
        self.end_state = None      # 末端位姿 + 夹爪 [x,y,z,qx,qy,qz,qw,gripper]
        self.joint_state = None    # 关节位置 + 夹爪 [j1-j6, gripper]
        self.joint_vel = None      # 关节速度
        self.gripper_vel = None    # 夹爪速度
        self.joint_cmd = None      # 规划关节位置
        
        self.state_lock = threading.Lock()
        self.freq = 200  # 状态更新频率
        
        # 启动状态更新线程
        self.running = True
        self.status_thread = threading.Thread(target=self._arm_status_thread, daemon=True)
        self.status_thread.start()
        
        self.plan_thread = threading.Thread(target=self._arm_plan_thread, daemon=True)
        self.plan_thread.start()
        
        # 初始化图像同步器
        rospy.loginfo(f"Setting up image synchronizer for topics: {self.camera_topics}")
        self.image_sync = ImageSynchronizer(
            camera_topics=self.camera_topics,
            sync_slop=self.sync_slop,
            queue_size=10,
            target_width=config.get('image_width'),
            target_height=config.get('image_height')
        )
        
        rospy.loginfo("RealEnvironment initialized successfully")
    
    def _print_arm_status(self):
        """打印机械臂状态"""
        arm_status = self.arm.get_status()
        print('----------------------------')
        print("arm_index: ", arm_status.arm_index)
        print("arm_name: ", arm_status.arm_name)
        print("arm_is_connected: ", arm_status.arm_is_connected)
        print("arm_dof: ", arm_status.arm_dof)
        print("servo_status: ", arm_status.servo_status)
        print("state: ", arm_status.state)
        print("speed_percentage: ", arm_status.speed_percentage)
        print("on_debug_mode: ", arm_status.on_debug_mode)
        print('arm_version: ', self.arm.get_version())
        print("gripper_tau: ", self.tau)
        print('----------------------------')
    
    def _wait_for_init_confirmation(self):
        """
        等待用户确认后再初始化机械臂位置
        这是一个安全机制，防止机械臂意外移动
        """
        rospy.logwarn("=" * 60)
        rospy.logwarn("ROBOT ARM WILL MOVE TO INITIAL POSITION!")
        rospy.logwarn(f"Target pose: {self.arm_init_pose}")
        rospy.logwarn(f"Target gripper: {self.arm_init_gripper}")
        rospy.logwarn("=" * 60)
        rospy.logwarn("Please ensure:")
        rospy.logwarn("  1. The workspace is clear")
        rospy.logwarn("  2. E-stop is ready")
        rospy.logwarn("  3. No obstacles in the robot's path")
        rospy.logwarn("=" * 60)
        
        try:
            user_input = input("\n>>> Press ENTER to confirm and initialize, or Ctrl+C to abort: ")
            rospy.loginfo("User confirmed, proceeding with initialization...")
        except (KeyboardInterrupt, EOFError):
            rospy.logwarn("\nInitialization aborted by user")
            raise KeyboardInterrupt("User aborted initialization")
    
    def init_status(self):
        """初始化机械臂位置"""
        rospy.loginfo(f"Initializing arm position (speed level: {self.init_speed})...")
        
        # 设置较慢的初始化速度 (0-10, 默认 3.0 较慢)
        self.arm.set_speed_level(self.init_speed)
        
        self.arm.set_gripper(self.arm_init_gripper, self.tau)
        self.arm.move_pose(self.arm_init_pose)
        time.sleep(0.5)
        
        # 恢复运行期默认速度，对齐 teleop 正常控制
        self.arm.set_speed_level(self.normal_speed_level)
        rospy.loginfo(f"Arm position initialized, speed restored to {self.normal_speed_level}")
    
    def _arm_status_thread(self):
        """机械臂状态更新线程"""
        rate = 1.0 / self.freq
        while self.running and not rospy.is_shutdown():
            try:
                self._update_arm_state()
            except Exception as e:
                rospy.logwarn(f"Error updating arm state: {e}")
            time.sleep(rate)
    
    def _arm_plan_thread(self):
        """机械臂规划状态更新线程"""
        rate = 1.0 / self.freq
        while self.running and not rospy.is_shutdown():
            try:
                with self.state_lock:
                    self.joint_cmd = self.arm.get_plan_joint_pos()
            except Exception as e:
                rospy.logwarn(f"Error getting plan joint pos: {e}")
            time.sleep(rate)
    
    def _update_arm_state(self):
        """更新机械臂状态"""
        gripper = self.arm.get_gripper_pos()
        pose = self.arm.get_cart_pose()
        joint = self.arm.get_joint_pos()
        
        joint_vel = self.arm.get_joint_vel()
        gripper_vel = self.arm.get_gripper_vel()
        
        with self.state_lock:
            self.end_state = pose + [gripper]       # [x,y,z,qx,qy,qz,qw,gripper]
            self.joint_state = joint + [gripper]    # [j1-j6, gripper]
            self.joint_vel = joint_vel
            self.gripper_vel = gripper_vel
    
    def get_observation(self):
        """
        获取当前观测
        
        Returns:
            dict: 包含 stamp, images, qpos_joint, qpos_end
                  如果数据不完整返回 None
        """
        with self.state_lock:
            if self.end_state is None:
                rospy.logwarn_throttle(1.0, "end_state is None")
                return None
            
            if self.joint_state is None:
                rospy.logwarn_throttle(1.0, "joint_state is None")
                return None
            
            qpos_joint = self.joint_state.copy()
            qpos_end = self.end_state.copy()
        
        # 获取图像
        stamp, images = self.image_sync.get_images()
        if images is None:
            rospy.logwarn_throttle(1.0, "latest_imgs is None")
            return None
        
        # 可视化
        if self.vis:
            for idx, img in enumerate(images):
                # image_sync 返回 RGB 格式，OpenCV 需要 BGR 格式
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"image_{idx}", img_bgr)
                cv2.waitKey(1)
        
        return {
            "stamp": stamp,
            "images": images,
            "qpos_joint": qpos_joint,
            "qpos_end": qpos_end,
            "gripper": qpos_joint[-1],  # 夹爪状态在 joint_state 最后一位
            "qpos": np.concatenate([qpos_joint, qpos_end], axis=0),  # 兼容旧版格式
        }

    def get_state_observation(self):
        """
        获取仅包含机械臂状态的观测快照（无需相机）

        Returns:
            dict: 包含 stamp, qpos_joint, qpos_end, gripper, qpos
                  如果状态不完整返回 None
        """
        with self.state_lock:
            if self.end_state is None or self.joint_state is None:
                return None

            qpos_joint = self.joint_state.copy()
            qpos_end = self.end_state.copy()

        return {
            "stamp": time.time(),
            "qpos_joint": qpos_joint,
            "qpos_end": qpos_end,
            "gripper": qpos_joint[-1],
            "qpos": np.concatenate([qpos_joint, qpos_end], axis=0),
        }
    
    def get_last_action(self):
        """
        获取最后的动作命令（用于数据采集）
        
        Returns:
            numpy array: [joints(6), gripper(1), end_pose(7), gripper(1)] = 15D
        """
        with self.state_lock:
            if self.joint_cmd is None or self.end_state is None:
                return None
            
            # 计算末端位姿命令
            end_cmd = self.arm.forward_kine(0, self.joint_cmd)[1]
            gripper = self.arm.get_gripper_pos()
            
            joints_cmd = list(self.joint_cmd) + [gripper]
            end_cmd = list(end_cmd) + [gripper]
            
            return np.concatenate([joints_cmd, end_cmd], axis=0)

    def get_teleop_action(self, backend_url: str = None) -> Optional[dict]:
        """从 backend 获取遥操作目标位姿（GAP-1 修复）

        通过 HTTP 请求 backend 的 /api/joystick/teleop_target 接口，
        获取遥操作者通过 track_pose() 发送给 SDK 的真实目标位姿。

        Args:
            backend_url: Backend URL, 默认 http://{robot_ip}:1999/api/joystick/teleop_target

        Returns:
            dict with:
                'target_pose': [x,y,z,qx,qy,qz,qw] or None
                'gripper_pose': float or None
                'scale': float
                'active': bool (是否在主动控制中)
            失败时返回 None
        """
        url = backend_url or f"http://{self.robot_ip}:1999/api/joystick/teleop_target"
        try:
            resp = self._teleop_http.get(url, timeout=0.05)
            if resp.status_code == 200:
                body = resp.json()
                return body.get('data', {})
        except requests.RequestException:
            pass
        return None

    def get_teleop_action_v2(self, backend_url: str = None) -> Optional[dict]:
        """从 backend 获取遥操作双通道状态（processed + raw）"""
        url = backend_url or f"http://{self.robot_ip}:1999/api/joystick/teleop_target_v2"
        try:
            resp = self._teleop_http.get(url, timeout=0.05)
            if resp.status_code == 200:
                body = resp.json()
                return body.get('data', {})
        except requests.RequestException:
            pass
        return None

    def end_control_nostep(self, action):
        """
        末端空间控制（不阻塞）
        
        Args:
            action: [x, y, z, qx, qy, qz, qw, gripper]
        """
        self.arm.track_pose(list(action[:7]))
        self.arm.set_gripper(action[-1], self.tau)
    
    def joint_control_nostep(self, action):
        """
        关节空间控制（不阻塞）
        
        Args:
            action: [j1, j2, j3, j4, j5, j6, gripper]
        """
        self.arm.track_joint(list(action[:6]))
        self.arm.set_gripper(action[-1], self.tau)
    
    def move_to_pose(self, pose, gripper=None):
        """
        移动到指定末端位姿（阻塞）
        
        Args:
            pose: [x, y, z, qx, qy, qz, qw]
            gripper: 夹爪开度，None 表示不变
        """
        self.arm.move_pose(list(pose))
        if gripper is not None:
            self.arm.set_gripper(gripper, self.tau)
    
    def move_to_joint(self, joint, gripper=None):
        """
        移动到指定关节位置（阻塞）
        
        Args:
            joint: [j1, j2, j3, j4, j5, j6]
            gripper: 夹爪开度，None 表示不变
        """
        self.arm.move_joint(list(joint))
        if gripper is not None:
            self.arm.set_gripper(gripper, self.tau)
    
    def set_gripper(self, pos, force=None):
        """
        设置夹爪
        
        Args:
            pos: 夹爪开度 (0-0.08m)
            force: 夹持力 (0-20N)，None 使用默认值
        """
        tau = force if force is not None else self.tau
        self.arm.set_gripper(pos, tau)
    
    def emergency_stop(self):
        """紧急停止"""
        self.arm.emergency_stop()
    
    def set_ready(self):
        """复位机械臂"""
        self.arm.set_ready()
    
    def shutdown(self):
        """关闭环境 - 默认回到零位但不断开连接"""
        rospy.loginfo("Shutting down RealEnvironment...")
        self.running = False
        
        # 等待线程结束
        if self.status_thread.is_alive():
            self.status_thread.join(timeout=1.0)
        if self.plan_thread.is_alive():
            self.plan_thread.join(timeout=1.0)
        
        # 被动模式下不回位
        if self.passive_mode:
            rospy.loginfo("Passive mode: skipping return to home")
        else:
            # 回位（不断开连接，不下使能）
            try:
                # 使用慢速回位
                self.arm.set_speed_level(self.init_speed)
                
                if self.return_to_zero:
                    # 回到零位（关节角度全为0）
                    rospy.loginfo(f"Moving to ZERO position (all joints = 0, speed level: {self.init_speed})...")
                    zero_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    self.arm.move_joint(zero_joints)
                    self.arm.set_gripper(0.0, self.tau)  # 夹爪也关闭
                else:
                    # 回到初始位置
                    rospy.loginfo(f"Moving to home position (speed level: {self.init_speed})...")
                    self.arm.move_pose(self.arm_init_pose)
                    self.arm.set_gripper(self.arm_init_gripper, self.tau)
                
                time.sleep(0.5)
                # 恢复运行期默认速度，避免 lower-machine teleop 延续慢速档位
                self.arm.set_speed_level(self.normal_speed_level)
                rospy.loginfo("Returned to position")
            except Exception as e:
                rospy.logwarn(f"Error returning to position: {e}")
        
        rospy.loginfo("RealEnvironment shutdown complete")


def create_environment_from_args(args):
    """
    从 argparse 参数创建环境
    兼容原始 carm_real/env_api.py 的参数格式
    
    Args:
        args: argparse.Namespace
        
    Returns:
        RealEnvironment 实例
    """
    config = {
        'robot_ip': getattr(args, 'robot_ip', '10.42.0.101'),
        'robot_mode': getattr(args, 'robot_mode', 2),
        'robot_tau': getattr(args, 'robot_tau', 10.0),
        'arm_init_pose': getattr(args, 'arm_init_pose', [0.26, -0.02, 0.22, 1, 0, 0, 0]),
        'arm_init_gripper': getattr(args, 'arm_init_gripper', 0.05),
        'init_speed': getattr(args, 'init_speed', 2.0),
        'normal_speed_level': getattr(args, 'normal_speed_level', 10.0),
        'camera_topics': normalize_camera_topics(getattr(args, 'camera_topics', None)),
        'sync_slop': float(getattr(args, 'sync_slop', DEFAULT_SYNC_SLOP)),
        'vis': getattr(args, 'vis', False),
    }

    return RealEnvironment(config)
