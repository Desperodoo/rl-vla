#!/usr/bin/env python3
"""
CARM 机械臂 ROS 数据记录程序
基于 carm_real/record_data_surreal3576.py 重构，使用 ROS 原生通信

功能:
- 记录相机图像（ROS 话题）
- 记录机械臂状态（关节角、末端位姿）
- 夹爪状态
- 时间戳同步
- 保存为 HDF5 格式（兼容 LeRobot）

使用方法:
    rosrun carm_ros_deploy record_data_ros.py --output_dir /path/to/data
"""

import argparse
import os
import time
import threading
import numpy as np
import cv2
import h5py
from datetime import datetime

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

# 本地模块
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_ros import RealEnvironment
from utils.image_sync import ImageSynchronizer, SingleImageSubscriber


class DataRecorder:
    """
    数据记录器
    """
    
    def __init__(self, config):
        """
        初始化记录器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 输出目录
        self.output_dir = config.get('output_dir', './recorded_data')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 记录参数
        self.record_freq = config.get('record_freq', 30)
        self.max_episodes = config.get('max_episodes', 100)
        self.max_steps = config.get('max_steps', 1000)
        
        # 图像参数
        self.image_width = config.get('image_width', 640)
        self.image_height = config.get('image_height', 480)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 初始化环境
        rospy.loginfo("Initializing environment...")
        self.env = RealEnvironment(config)
        
        # 数据缓冲
        self.episode_data = {
            'images': [],
            'qpos_joint': [],
            'qpos_end': [],
            'gripper': [],
            'timestamps': [],
        }
        
        # 控制状态
        self.recording = False
        self.episode_count = 0
        self.step_count = 0
        
        # 键盘监听
        self.keyboard_thread = None
        self.start_keyboard_listener()
        
        rospy.loginfo("DataRecorder initialized")
        rospy.loginfo(f"Output directory: {self.output_dir}")
        rospy.loginfo(f"Record frequency: {self.record_freq} Hz")
    
    def start_keyboard_listener(self):
        """启动键盘监听线程"""
        try:
            import termios
            import tty
            self.keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
            self.keyboard_thread.start()
            rospy.loginfo("Keyboard listener started (press 's' to start/stop, 'q' to quit)")
        except Exception as e:
            rospy.logwarn(f"Keyboard listener not available: {e}")
            rospy.logwarn("Use ROS service calls instead")
    
    def _keyboard_loop(self):
        """键盘监听循环"""
        import sys
        import termios
        import tty
        
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not rospy.is_shutdown():
                if sys.stdin in [sys.stdin]:
                    c = sys.stdin.read(1)
                    if c == 's':
                        self._toggle_recording()
                    elif c == 'q':
                        self._save_and_quit()
                        break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def _toggle_recording(self):
        """切换记录状态"""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def _save_and_quit(self):
        """保存并退出"""
        if self.recording:
            self.stop_recording()
        rospy.signal_shutdown("User quit")
    
    def start_recording(self):
        """开始记录"""
        if self.recording:
            rospy.logwarn("Already recording")
            return
        
        self.recording = True
        self.step_count = 0
        self.episode_data = {
            'images': [],
            'qpos_joint': [],
            'qpos_end': [],
            'gripper': [],
            'timestamps': [],
        }
        
        self.episode_count += 1
        rospy.loginfo(f"Recording started - Episode {self.episode_count}")
    
    def stop_recording(self):
        """停止记录并保存数据"""
        if not self.recording:
            rospy.logwarn("Not recording")
            return
        
        self.recording = False
        rospy.loginfo(f"Recording stopped - {self.step_count} steps collected")
        
        # 保存数据
        self.save_episode()
    
    def save_episode(self):
        """保存当前 episode 数据"""
        if len(self.episode_data['timestamps']) == 0:
            rospy.logwarn("No data to save")
            return
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{self.episode_count:04d}_{timestamp}.hdf5"
        filepath = os.path.join(self.output_dir, filename)
        
        rospy.loginfo(f"Saving episode to {filepath}...")
        
        # 转换为 numpy 数组
        num_steps = len(self.episode_data['timestamps'])
        
        with h5py.File(filepath, 'w') as f:
            # 创建数据组
            obs = f.create_group('observations')
            
            # 保存图像
            images = np.array(self.episode_data['images'])  # [T, H, W, C]
            obs.create_dataset('images', data=images, compression='gzip')
            
            # 保存状态
            qpos_joint = np.array(self.episode_data['qpos_joint'])  # [T, 7]
            obs.create_dataset('qpos_joint', data=qpos_joint)
            
            qpos_end = np.array(self.episode_data['qpos_end'])  # [T, 8]
            obs.create_dataset('qpos_end', data=qpos_end)
            
            gripper = np.array(self.episode_data['gripper'])  # [T]
            obs.create_dataset('gripper', data=gripper)
            
            timestamps = np.array(self.episode_data['timestamps'])  # [T]
            obs.create_dataset('timestamps', data=timestamps)
            
            # 元数据
            f.attrs['num_steps'] = num_steps
            f.attrs['record_freq'] = self.record_freq
            f.attrs['image_width'] = self.image_width
            f.attrs['image_height'] = self.image_height
            f.attrs['robot_ip'] = self.config.get('robot_ip', '')
            f.attrs['created_at'] = timestamp
        
        rospy.loginfo(f"Episode saved: {num_steps} steps, {images.nbytes / 1e6:.1f} MB")
    
    def record_step(self, obs):
        """
        记录一步数据
        
        Args:
            obs: 观测字典
        """
        if not self.recording:
            return
        
        if obs is None:
            return
        
        # 记录数据
        self.episode_data['images'].append(obs['images'][0])  # 第一个相机
        self.episode_data['qpos_joint'].append(obs['qpos_joint'])
        self.episode_data['qpos_end'].append(obs['qpos_end'])
        self.episode_data['gripper'].append(obs['gripper'])
        self.episode_data['timestamps'].append(obs['stamp'])
        
        self.step_count += 1
        
        # 检查是否达到最大步数
        if self.step_count >= self.max_steps:
            rospy.logwarn(f"Reached max steps ({self.max_steps}), stopping recording")
            self.stop_recording()
    
    def run(self):
        """运行记录循环"""
        rate = rospy.Rate(self.record_freq)
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("Data Recording Node Ready")
        rospy.loginfo("=" * 50)
        rospy.loginfo("Controls:")
        rospy.loginfo("  's' - Start/Stop recording")
        rospy.loginfo("  'q' - Save and quit")
        rospy.loginfo("=" * 50)
        
        while not rospy.is_shutdown():
            # 获取观测
            obs = self.env.get_observation()
            
            if obs is not None:
                # 显示状态
                if self.recording:
                    rospy.loginfo_throttle(1.0, 
                        f"Recording: Episode {self.episode_count}, Step {self.step_count}")
                
                # 记录数据
                self.record_step(obs)
                
                # 可视化
                if self.config.get('vis', False):
                    self._visualize(obs)
            
            rate.sleep()
    
    def _visualize(self, obs):
        """可视化当前观测"""
        if obs is None or len(obs['images']) == 0:
            return
        
        img = obs['images'][0].copy()
        
        # 添加状态文本
        status = "RECORDING" if self.recording else "PAUSED"
        color = (0, 0, 255) if self.recording else (255, 128, 0)
        
        cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"Episode: {self.episode_count}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"Step: {self.step_count}", (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 显示关节角
        qpos = obs['qpos_joint']
        qpos_str = ', '.join([f"{q:.2f}" for q in qpos[:6]])
        cv2.putText(img, f"Joints: [{qpos_str}]", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 显示夹爪
        cv2.putText(img, f"Gripper: {obs['gripper']:.3f}", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow("Recording", img)
        cv2.waitKey(1)
    
    def shutdown(self):
        """关闭记录器"""
        rospy.loginfo("Shutting down DataRecorder...")
        
        if self.recording:
            self.stop_recording()
        
        self.env.shutdown()
        cv2.destroyAllWindows()
        
        rospy.loginfo("DataRecorder shutdown complete")


class TeleopRecorder(DataRecorder):
    """
    遥操作数据记录器（拖动示教模式）
    """
    
    def __init__(self, config):
        """初始化遥操作记录器"""
        # 设置拖动模式
        config['robot_mode'] = 3  # DRAG mode
        super().__init__(config)
        rospy.loginfo("TeleopRecorder: Using DRAG mode for data collection")
    
    def run(self):
        """运行遥操作记录"""
        rospy.loginfo("=" * 50)
        rospy.loginfo("Teleop Data Recording Node Ready")
        rospy.loginfo("=" * 50)
        rospy.loginfo("Robot is in DRAG mode - manually guide the arm")
        rospy.loginfo("Controls:")
        rospy.loginfo("  's' - Start/Stop recording")
        rospy.loginfo("  'q' - Save and quit")
        rospy.loginfo("=" * 50)
        
        rate = rospy.Rate(self.record_freq)
        
        while not rospy.is_shutdown():
            # 获取观测
            obs = self.env.get_observation()
            
            if obs is not None:
                # 显示状态
                if self.recording:
                    rospy.loginfo_throttle(1.0, 
                        f"Recording: Episode {self.episode_count}, Step {self.step_count}")
                
                # 记录数据
                self.record_step(obs)
                
                # 可视化
                if self.config.get('vis', False):
                    self._visualize(obs)
            
            rate.sleep()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CARM Robot Data Recording (ROS)')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./recorded_data',
                        help='Output directory for recorded data')
    
    # 机械臂参数
    parser.add_argument('--robot_ip', type=str, default='10.42.0.101',
                        help='Robot IP address')
    parser.add_argument('--robot_mode', type=int, default=3,
                        help='Control mode (0=IDLE, 1=POSITION, 2=MIT, 3=DRAG)')
    
    # 相机参数
    parser.add_argument('--camera_topics', type=str,
                        default='/camera/color/image_raw',
                        help='Camera topic(s), comma separated')
    parser.add_argument('--sync_slop', type=float, default=0.1,
                        help='Image sync tolerance in seconds')
    
    # 记录参数
    parser.add_argument('--record_freq', type=int, default=30,
                        help='Recording frequency (Hz)')
    parser.add_argument('--max_episodes', type=int, default=100,
                        help='Maximum number of episodes')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per episode')
    
    # 图像参数
    parser.add_argument('--image_width', type=int, default=640,
                        help='Image width')
    parser.add_argument('--image_height', type=int, default=480,
                        help='Image height')
    
    # 模式
    parser.add_argument('--teleop', action='store_true',
                        help='Use teleop (drag) mode')
    parser.add_argument('--vis', action='store_true',
                        help='Visualize images')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 初始化 ROS 节点
    rospy.init_node('carm_data_recorder', anonymous=True)
    
    # 解析参数
    args = parse_args()
    
    # 转换为配置字典
    config = vars(args)
    
    # 处理相机话题
    if isinstance(config['camera_topics'], str):
        config['camera_topics'] = config['camera_topics'].split(',')
    
    rospy.loginfo("=" * 50)
    rospy.loginfo("CARM Data Recording Node")
    rospy.loginfo("=" * 50)
    rospy.loginfo(f"Robot IP: {config['robot_ip']}")
    rospy.loginfo(f"Camera topics: {config['camera_topics']}")
    rospy.loginfo(f"Output dir: {config['output_dir']}")
    rospy.loginfo(f"Mode: {'Teleop' if config['teleop'] else 'Normal'}")
    rospy.loginfo("=" * 50)
    
    # 创建记录器
    if config['teleop']:
        recorder = TeleopRecorder(config)
    else:
        recorder = DataRecorder(config)
    
    # 注册关闭回调
    rospy.on_shutdown(recorder.shutdown)
    
    try:
        recorder.run()
    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by user")
    finally:
        recorder.shutdown()


if __name__ == '__main__':
    main()
