#!/usr/bin/env python3
"""
CARM 机械臂 ROS 数据记录程序（被动模式）
不干扰网页手柄遥操作，只记录数据

功能:
- 记录相机图像（ROS 话题）
- 记录机械臂状态（关节角、末端位姿）
- 夹爪状态
- 动作命令
- 时间戳同步
- 保存为 HDF5 格式

使用方法:
    rosrun carm_deploy record_data_ros.py --output_dir /path/to/data --vis

遥操作:
    通过网页 http://10.42.0.101 使用手柄进行遥操作
    本脚本只记录数据，不控制机械臂
"""

import argparse
import os
import sys
import time
import threading
import signal
import atexit
import json
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
import os
# 添加 carm_deploy 根目录到路径
carm_deploy_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, carm_deploy_root)

from core.env_ros import RealEnvironment
from utils.image_sync import ImageSynchronizer, SingleImageSubscriber
from utils.timeline_logger import TimelineLogger


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
        
        # 输出目录（展开 ~ 和环境变量）
        raw_output_dir = config.get('output_dir', './recorded_data')
        self.output_dir = os.path.expandvars(os.path.expanduser(raw_output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 记录参数
        self.record_freq = config.get('record_freq', 30)
        self.max_episodes = config.get('max_episodes', 100)
        self.max_steps = config.get('max_steps', 1000)
        self.backend_url = config.get('backend_url', None)  # None = auto-detect from robot_ip
        
        # 图像参数
        self.image_width = config.get('image_width', 640)
        self.image_height = config.get('image_height', 480)

        # 相机参数
        raw_camera_topics = config.get('camera_topics', ['/camera/color/image_raw'])
        if isinstance(raw_camera_topics, str):
            raw_camera_topics = raw_camera_topics.split(',')
        self.camera_topics = [topic.strip() for topic in raw_camera_topics if topic.strip()]

        raw_camera_names = config.get('camera_names', '')
        if isinstance(raw_camera_names, str):
            raw_camera_names = [name.strip() for name in raw_camera_names.split(',') if name.strip()]
        elif raw_camera_names is None:
            raw_camera_names = []

        if len(raw_camera_names) > 0:
            if len(raw_camera_names) != len(self.camera_topics):
                raise ValueError(
                    f"camera_names count ({len(raw_camera_names)}) does not match "
                    f"camera_topics count ({len(self.camera_topics)})"
                )
            self.camera_names = raw_camera_names
        else:
            self.camera_names = [self._topic_to_camera_name(topic, idx)
                                 for idx, topic in enumerate(self.camera_topics)]

        self.camera_index = {name: idx for idx, name in enumerate(self.camera_names)}

        requested_primary = str(config.get('primary_camera', self.camera_names[0])).strip()
        if requested_primary not in self.camera_index:
            rospy.logwarn(
                f"primary_camera '{requested_primary}' not found in camera_names {self.camera_names}, "
                f"fallback to '{self.camera_names[0]}'"
            )
            requested_primary = self.camera_names[0]
        self.primary_camera = requested_primary
        self.primary_camera_idx = self.camera_index[self.primary_camera]

        config['camera_topics'] = self.camera_topics
        config['camera_names'] = self.camera_names
        config['primary_camera'] = self.primary_camera
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 启用被动模式（不干扰手柄遥操作）
        config['passive_mode'] = True
        
        # 初始化环境
        rospy.loginfo("Initializing environment...")
        self.env = RealEnvironment(config)

        # 时间线日志（用于分析采集时间语义）
        self.timeline_enabled = config.get('timeline_enabled', True)
        self.timeline_logger = None
        if self.timeline_enabled:
            timeline_path = config.get('timeline_log', '')
            if not timeline_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                timeline_path = os.path.join(self.output_dir, f'timeline_record_{timestamp}.jsonl')
            self.timeline_logger = TimelineLogger(timeline_path)
            self.timeline_logger.log(
                'init',
                record_freq=self.record_freq,
                output_dir=self.output_dir,
            )
        
        # 数据缓冲
        self.episode_data = {
            'images': [],
            'images_by_camera': {name: [] for name in self.camera_names},
            'qpos_joint': [],
            'qpos_end': [],
            'qpos': [],           # 兼容旧版格式
            'action': [],         # 遥操作目标位姿 [target_pose(7), gripper(1)] = 8D
            'teleop_scale': [],   # 遥操作 scale 值 (0 表示非活跃)
            'gripper': [],
            'timestamps': [],
        }
        
        # 控制状态
        self.recording = False
        self.episode_count = 0
        self.step_count = 0
        self.pending_save = False  # 等待用户确认保存
        self.pending_episode_data = None  # 待确认的 episode 数据
        self.streams_ready = False
        
        # 键盘监听
        self.keyboard_thread = None
        self._old_terminal_settings = None  # 终端原始设置，用于恢复
        self.start_keyboard_listener()
        
        rospy.loginfo("DataRecorder initialized")
        rospy.loginfo(f"Output directory: {self.output_dir}")
        rospy.loginfo(f"Record frequency: {self.record_freq} Hz")
        rospy.loginfo(f"Camera topics: {self.camera_topics}")
        rospy.loginfo(f"Camera names: {self.camera_names}")
        rospy.loginfo(f"Primary camera: {self.primary_camera}")

    def _topic_to_camera_name(self, topic, index):
        """从 topic 自动生成稳定的相机名"""
        name = topic.strip('/').replace('/', '_').replace('-', '_')
        if not name:
            name = f"camera_{index}"
        return name
    
    def _restore_terminal(self):
        """恢复终端设置（确保任何退出方式都能恢复）"""
        if self._old_terminal_settings is not None:
            import termios
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_terminal_settings)
            except Exception:
                pass
            self._old_terminal_settings = None

    def start_keyboard_listener(self):
        """启动键盘监听线程"""
        try:
            import termios
            import tty
            # 保存终端设置到实例变量，用于 atexit/signal 恢复
            self._old_terminal_settings = termios.tcgetattr(sys.stdin)
            # 注册多重恢复保障
            atexit.register(self._restore_terminal)
            original_sigint = signal.getsignal(signal.SIGINT)
            def _sigint_handler(signum, frame):
                self._restore_terminal()
                if callable(original_sigint) and original_sigint not in (signal.SIG_IGN, signal.SIG_DFL):
                    original_sigint(signum, frame)
                else:
                    raise KeyboardInterrupt
            signal.signal(signal.SIGINT, _sigint_handler)
            self.keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
            self.keyboard_thread.start()
            rospy.loginfo("Keyboard listener started (press 's' to start/stop, 'q' to quit)")
        except Exception as e:
            rospy.logwarn(f"Keyboard listener not available: {e}")
            rospy.logwarn("Use ROS service calls instead")
    
    def _keyboard_loop(self):
        """键盘监听循环"""
        import termios
        import tty
        
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not rospy.is_shutdown():
                c = sys.stdin.read(1)
                if self.pending_save:
                    # 等待用户确认保存
                    if c == 'y' or c == 'Y':
                        self._confirm_save(True)
                    elif c == 'n' or c == 'N':
                        self._confirm_save(False)
                    # 其他按键忽略
                else:
                    if c == 's':
                        self._toggle_recording()
                    elif c == 'q':
                        self._quit()
                        break
        finally:
            self._restore_terminal()
    
    def _toggle_recording(self):
        """切换记录状态"""
        if self.pending_save:
            rospy.logwarn("Please confirm save first (y/n)")
            return

        if not self.streams_ready:
            rospy.logwarn("Camera streams are not ready yet. Waiting for first frames from all camera topics.")
            return
        
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def _wait_for_camera_first_frames(self):
        """等待所有相机话题收到首帧后再允许录制。"""
        timeout_per_topic = float(self.config.get('camera_ready_timeout', 30.0))
        rospy.loginfo("Waiting for first frame from all camera topics before enabling recording...")
        for topic in self.camera_topics:
            while not rospy.is_shutdown():
                try:
                    rospy.loginfo(f"  waiting topic: {topic}")
                    rospy.wait_for_message(topic, Image, timeout=timeout_per_topic)
                    rospy.loginfo(f"  first frame received: {topic}")
                    break
                except rospy.ROSException:
                    rospy.logwarn(f"  timeout waiting for topic: {topic}. Retrying...")

        self.streams_ready = not rospy.is_shutdown()
        if self.streams_ready:
            rospy.loginfo("All camera topics are ready. Recording can be started.")
    
    def _confirm_save(self, save):
        """确认是否保存 episode"""
        if not self.pending_save:
            return
        
        if save:
            rospy.loginfo(">>> Saving episode...")
            self._do_save_episode()
        else:
            rospy.loginfo(">>> Episode discarded")
        
        self.pending_save = False
        self.pending_episode_data = None
        rospy.loginfo(">>> Press 's' to start next episode")
    
    def _quit(self):
        """退出"""
        if self.recording:
            self.stop_recording()
            # 如果正在等待确认，询问是否保存
            if self.pending_save:
                rospy.loginfo("Discarding pending episode on quit")
                self.pending_save = False
                self.pending_episode_data = None
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
            'images_by_camera': {name: [] for name in self.camera_names},
            'qpos_joint': [],
            'qpos_end': [],
            'qpos': [],           # 兼容旧版格式
            'action': [],         # 遥操作目标位姿 [target_pose(7), gripper(1)] = 8D
            'teleop_scale': [],   # 遥操作 scale 值 (0 表示非活跃)
            'gripper': [],
            'timestamps': [],
        }
        
        self.episode_count += 1
        rospy.loginfo(f"Recording started - Episode {self.episode_count}")
    
    def stop_recording(self):
        """停止记录并等待确认"""
        if not self.recording:
            rospy.logwarn("Not recording")
            return
        
        self.recording = False
        rospy.loginfo(f">>> Recording stopped - {self.step_count} steps collected")
        
        if len(self.episode_data['timestamps']) == 0:
            rospy.logwarn("No data recorded, nothing to save")
            rospy.loginfo(">>> Press 's' to start next episode")
            return
        
        # 保存数据到待确认状态
        self.pending_episode_data = self.episode_data.copy()
        self.pending_save = True
        
        rospy.loginfo("="*50)
        rospy.loginfo(f">>> Episode {self.episode_count}: {self.step_count} steps")
        rospy.loginfo(">>> Save this episode? (y/n)")
        rospy.loginfo("="*50)
    
    def _do_save_episode(self):
        """实际执行保存 episode 数据"""
        if self.pending_episode_data is None:
            rospy.logwarn("No pending data to save")
            return
        
        episode_data = self.pending_episode_data
        
        if len(episode_data['timestamps']) == 0:
            rospy.logwarn("No data to save")
            return
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{self.episode_count:04d}_{timestamp}.hdf5"
        filepath = os.path.join(self.output_dir, filename)
        
        rospy.loginfo(f"Saving episode to {filepath}...")
        
        # 转换为 numpy 数组
        num_steps = len(episode_data['timestamps'])
        
        with h5py.File(filepath, 'w') as f:
            # 创建数据组
            obs = f.create_group('observations')
            
            # 保存主视角图像（兼容旧版）
            images = np.array(episode_data['images'])  # [T, H, W, C]
            obs.create_dataset('images', data=images, compression='gzip')

            # 保存多相机图像（新格式）
            images_by_camera = episode_data.get('images_by_camera', {})
            if len(images_by_camera) > 0:
                cameras_group = obs.create_group('images_by_camera')
                for camera_name, camera_images in images_by_camera.items():
                    if len(camera_images) == 0:
                        continue
                    camera_array = np.array(camera_images)
                    cameras_group.create_dataset(camera_name, data=camera_array, compression='gzip')
            
            # 保存状态
            qpos_joint = np.array(episode_data['qpos_joint'])  # [T, 7]
            obs.create_dataset('qpos_joint', data=qpos_joint)
            
            qpos_end = np.array(episode_data['qpos_end'])  # [T, 8]
            obs.create_dataset('qpos_end', data=qpos_end)
            
            # 兼容旧版格式: qpos = [joints(7), end_pose(8)]
            qpos = np.array(episode_data['qpos'])  # [T, 15]
            obs.create_dataset('qpos', data=qpos)
            
            gripper = np.array(episode_data['gripper'])  # [T]
            obs.create_dataset('gripper', data=gripper)
            
            timestamps = np.array(episode_data['timestamps'])  # [T]
            obs.create_dataset('timestamps', data=timestamps)
            
            # 保存动作命令 (新格式: 8D = target_pose(7) + gripper(1))
            if len(episode_data['action']) > 0:
                action = np.array(episode_data['action'])  # [T, 8]
                f.create_dataset('action', data=action)

            # 保存遥操作 scale
            if len(episode_data.get('teleop_scale', [])) > 0:
                teleop_scale = np.array(episode_data['teleop_scale'])  # [T]
                f.create_dataset('teleop_scale', data=teleop_scale)

            # 元数据
            f.attrs['num_steps'] = num_steps
            f.attrs['record_freq'] = self.record_freq
            f.attrs['image_width'] = self.image_width
            f.attrs['image_height'] = self.image_height
            f.attrs['camera_topics'] = json.dumps(self.camera_topics)
            f.attrs['camera_names'] = json.dumps(self.camera_names)
            f.attrs['primary_camera'] = self.primary_camera
            f.attrs['robot_ip'] = self.config.get('robot_ip', '')
            f.attrs['created_at'] = timestamp
            f.attrs['data_version'] = 'v3'  # 多视角格式标记（兼容 v2 主图像字段）
        
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

        t_obs_ready_sys = time.time()
        
        # 记录数据
        obs_images = obs['images']
        if len(obs_images) <= self.primary_camera_idx:
            rospy.logwarn_throttle(
                1.0,
                f"Primary camera index {self.primary_camera_idx} out of range, "
                f"fallback to index 0 (available={len(obs_images)})"
            )
            primary_img = obs_images[0]
        else:
            primary_img = obs_images[self.primary_camera_idx]

        self.episode_data['images'].append(primary_img)  # 兼容字段：主视角

        for camera_name, camera_idx in self.camera_index.items():
            if camera_idx < len(obs_images):
                self.episode_data['images_by_camera'][camera_name].append(obs_images[camera_idx])
        self.episode_data['qpos_joint'].append(obs['qpos_joint'])
        self.episode_data['qpos_end'].append(obs['qpos_end'])
        self.episode_data['qpos'].append(obs['qpos'])  # 兼容旧版格式
        self.episode_data['gripper'].append(obs['gripper'])
        self.episode_data['timestamps'].append(obs['stamp'])
        
        # 记录遥操作目标位姿（GAP-1 修复：直接从 backend 获取 track_pose 目标）
        t_action_query_sys = time.time()
        teleop_state = self.env.get_teleop_action(backend_url=self.backend_url)

        if teleop_state and teleop_state.get('active') and teleop_state.get('target_pose') is not None:
            target_pose = teleop_state['target_pose']    # [7] xyz+quat
            gripper = teleop_state.get('gripper_pose', 0.0) or 0.0
            teleop_scale = teleop_state.get('scale', 0.0) or 0.0

            action = np.array(target_pose + [gripper], dtype=np.float64)
            self.episode_data['action'].append(action)
            self.episode_data['teleop_scale'].append(teleop_scale)
        else:
            # 遥操作未激活（离合器松开等），使用当前实际状态作为 fallback
            action = np.array(obs['qpos_end'], dtype=np.float64)  # 8D
            self.episode_data['action'].append(action)
            self.episode_data['teleop_scale'].append(0.0)

        if self.timeline_logger is not None:
            obs_stamp_ros = obs.get('stamp', None)
            delta_obs = None
            delta_action_obs = None
            if obs_stamp_ros is not None:
                delta_obs = t_obs_ready_sys - obs_stamp_ros
                delta_action_obs = t_action_query_sys - obs_stamp_ros
            self.timeline_logger.log(
                'record_step',
                episode=self.episode_count,
                step=self.step_count,
                obs_stamp_ros=obs_stamp_ros,
                t_obs_ready_sys=t_obs_ready_sys,
                delta_obs=delta_obs,
                t_action_query_sys=t_action_query_sys,
                delta_action_obs=delta_action_obs,
                action_present=action is not None,
            )
        
        self.step_count += 1
        
        # 检查是否达到最大步数
        if self.step_count >= self.max_steps:
            rospy.logwarn(f"Reached max steps ({self.max_steps}), stopping recording")
            self.stop_recording()
    
    def run(self):
        """运行记录循环"""
        rate = rospy.Rate(self.record_freq)

        # 启动门禁：先等待两路（或多路）相机首帧就绪。
        self._wait_for_camera_first_frames()
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("Data Recording Node Ready")
        rospy.loginfo("=" * 50)
        rospy.loginfo("Controls:")
        rospy.loginfo("  's' - Start/Stop recording")
        rospy.loginfo("  'y' - Confirm save episode")
        rospy.loginfo("  'n' - Discard episode")
        rospy.loginfo("  'q' - Quit")
        rospy.loginfo("=" * 50)
        rospy.loginfo(">>> Press 's' to start recording")
        
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
        
        # image_sync 返回 RGB 格式，OpenCV 需要 BGR 格式
        img = cv2.cvtColor(obs['images'][0], cv2.COLOR_RGB2BGR)
        
        # 添加状态文本
        if self.recording:
            status = "RECORDING"
            color = (0, 0, 255)  # 红色
        elif self.pending_save:
            status = "CONFIRM? (y/n)"
            color = (0, 165, 255)  # 橙色
        else:
            status = "READY"
            color = (0, 255, 0)  # 绿色
        
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

        if self.timeline_logger is not None:
            self.timeline_logger.close()
        
        rospy.loginfo("DataRecorder shutdown complete")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CARM Robot Data Recording (ROS)')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./recorded_data',
                        help='Output directory for recorded data')
    
    # 机械臂参数
    parser.add_argument('--robot_ip', type=str, default='10.42.0.101',
                        help='Robot IP address')
    
    # 相机参数
    parser.add_argument('--camera_topics', type=str,
                        default='/camera/color/image_raw',
                        help='Camera topic(s), comma separated')
    parser.add_argument('--camera_names', type=str, default='',
                        help='Camera name(s), comma separated (must align with camera_topics order)')
    parser.add_argument('--primary_camera', type=str, default='',
                        help='Primary camera name used for legacy observations/images (default: first camera)')
    parser.add_argument('--sync_slop', type=float, default=0.02,
                        help='Image sync tolerance in seconds')
    
    # 记录参数
    parser.add_argument('--record_freq', type=int, default=30,
                        help='Recording frequency (Hz)')
    parser.add_argument('--max_episodes', type=int, default=100,
                        help='Maximum number of episodes')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per episode')
    
    # 图像参数
    parser.add_argument('--image_width', type=int, default=320,
                        help='Image width')
    parser.add_argument('--image_height', type=int, default=240,
                        help='Image height')
    
    # 可视化
    parser.add_argument('--vis', action='store_true',
                        help='Visualize images')

    # Backend URL（遥操作目标位姿获取）
    parser.add_argument('--backend_url', type=str, default='',
                        help='Backend API URL (default: http://{robot_ip}:1999/api/joystick/teleop_target)')

    # 时间线日志
    parser.add_argument('--timeline_enabled', action='store_true',
                        help='Enable timeline logging (default: enabled)')
    parser.add_argument('--timeline_disabled', action='store_true',
                        help='Disable timeline logging')
    parser.add_argument('--timeline_log', type=str, default='',
                        help='Timeline log path (JSONL). Empty uses output_dir')
    
    # 兼容 roslaunch remap 参数
    return parser.parse_args(args=rospy.myargv()[1:])


def main():
    """主函数"""
    # 初始化 ROS 节点
    rospy.init_node('carm_data_recorder', anonymous=True)
    
    # 解析参数
    args = parse_args()
    
    # 转换为配置字典
    config = vars(args)

    # 从 ROS 参数覆盖（支持 roslaunch <param> 方式）
    for key in [
        'output_dir', 'robot_ip', 'robot_mode', 'camera_topics', 'camera_names', 'primary_camera', 'sync_slop',
        'record_freq', 'max_episodes', 'max_steps', 'image_width', 'image_height',
        'teleop', 'vis', 'timeline_log', 'timeline_enabled', 'timeline_disabled'
    ]:
        if rospy.has_param(f'~{key}'):
            config[key] = rospy.get_param(f'~{key}')

    # 时间线日志开关：默认开启，除非显式禁用
    if config.get('timeline_disabled', False):
        config['timeline_enabled'] = False
    else:
        config['timeline_enabled'] = True
    
    # 处理相机话题
    if isinstance(config['camera_topics'], str):
        config['camera_topics'] = [topic.strip() for topic in config['camera_topics'].split(',') if topic.strip()]

    if not config.get('primary_camera', '').strip() and isinstance(config.get('camera_names', ''), str):
        camera_names = [name.strip() for name in config['camera_names'].split(',') if name.strip()]
        if len(camera_names) > 0:
            config['primary_camera'] = camera_names[0]
    
    rospy.loginfo("=" * 50)
    rospy.loginfo("CARM Data Recording Node")
    rospy.loginfo("=" * 50)
    rospy.loginfo(f"Robot IP: {config['robot_ip']}")
    rospy.loginfo(f"Camera topics: {config['camera_topics']}")
    rospy.loginfo(f"Output dir: {config['output_dir']}")
    rospy.loginfo("Mode: Passive (does NOT control robot)")
    rospy.loginfo("=" * 50)
    
    # 创建记录器
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
