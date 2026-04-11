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
import copy
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

from core.camera_config import resolve_camera_config
from core.env_ros import RealEnvironment
from data.teleop_bridge import (
    TeleopShadowTransformer,
    TeleopSignalClient,
    TeleopUpperControlBridge,
)
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
        
        self.camera_config = config['camera_config']

        # 输出目录（展开 ~ 和环境变量）
        raw_output_dir = config.get('output_dir', './recorded_data')
        self.output_dir = os.path.expandvars(os.path.expanduser(raw_output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 记录参数
        self.record_freq = config.get('record_freq', 30)
        self.max_episodes = config.get('max_episodes', 100)
        self.max_steps = config.get('max_steps', 1000)
        self.backend_url = config.get('backend_url', None)  # None = auto-detect from robot_ip
        self.backend_url_v2 = config.get('backend_url_v2', None)
        self.events_v2_url = config.get('events_v2_url', None)
        self.teleop_bridge_mode = config.get('teleop_bridge_mode', 'passive_shadow')
        self.pred_horizon = int(config.get('pred_horizon', 16))
        self.act_horizon = int(config.get('act_horizon', self.pred_horizon))
        self.teleop_signal_timeout_ms = float(config.get('teleop_signal_timeout_ms', 150.0))
        self.teleop_candidate_control_freq = float(config.get('teleop_candidate_control_freq', 50.0))
        self.upper_control_enabled = bool(config.get('upper_control_enabled', False))
        self.enable_teleop_sse = bool(config.get('enable_teleop_sse', False))

        # 图像参数
        self.image_width = config.get('image_width', 640)
        self.image_height = config.get('image_height', 480)

        # 相机参数
        self.camera_topics = list(self.camera_config.topics)
        self.camera_names = list(self.camera_config.names)
        self.camera_index = {name: idx for idx, name in enumerate(self.camera_names)}
        self.primary_camera = self.camera_config.primary_name
        self.primary_camera_idx = self.camera_config.primary_index

        config['camera_topics'] = self.camera_topics
        config['camera_names'] = self.camera_names
        config['primary_camera'] = self.primary_camera
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # recorder 阶段默认仍保持 passive_mode，避免初始化/回零等副作用
        config['passive_mode'] = True
        if self.teleop_bridge_mode == 'upper_control':
            config['return_to_zero'] = False

        # 初始化环境
        rospy.loginfo("Initializing environment...")
        self.env = RealEnvironment(config)

        self.teleop_signal_client = TeleopSignalClient(
            robot_ip=config.get('robot_ip', '10.42.0.101'),
            backend_url_v2=self.backend_url_v2,
            events_v2_url=self.events_v2_url,
            enable_sse=self.enable_teleop_sse,
        )
        self.teleop_shadow_transformer = TeleopShadowTransformer(self.pred_horizon)
        self.teleop_bridge = None
        if self.teleop_bridge_mode == 'upper_control':
            self.teleop_bridge = TeleopUpperControlBridge(
                env=self.env,
                signal_client=self.teleop_signal_client,
                control_freq=self.teleop_candidate_control_freq,
                signal_timeout_ms=self.teleop_signal_timeout_ms,
                live_enabled=self.upper_control_enabled,
            )
            self.teleop_bridge.start()
        self.start_control_state = self.teleop_signal_client.get_control_state()

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
                teleop_bridge_mode=self.teleop_bridge_mode,
                pred_horizon=self.pred_horizon,
                act_horizon=self.act_horizon,
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
            'teleop_valid_mask': [],
            'teleop_processed_target_abs': [],
            'teleop_human_chunk_abs': [],
            'teleop_human_chunk_rel': [],
            'teleop_reconstructed_target_abs': [],
            'teleop_active': [],
            'teleop_processed_sequence': [],
            'teleop_raw_sequence': [],
            'teleop_signal_age_ms': [],
            'teleop_abs_reconstruction_pos_error': [],
            'teleop_abs_reconstruction_rot_error': [],
            'upper_candidate_target_abs': [],
            'upper_candidate_pos_error': [],
            'upper_candidate_rot_error': [],
            'upper_executed_target_abs': [],
            'teleop_candidate_loop_dt_ms': [],
            'teleop_candidate_stale': [],
            'teleop_candidate_applied': [],
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
        self.last_candidate_stale = False
        self.shutting_down = False
        self._shutdown_complete = False
        
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
            'teleop_valid_mask': [],
            'teleop_processed_target_abs': [],
            'teleop_human_chunk_abs': [],
            'teleop_human_chunk_rel': [],
            'teleop_reconstructed_target_abs': [],
            'teleop_active': [],
            'teleop_processed_sequence': [],
            'teleop_raw_sequence': [],
            'teleop_signal_age_ms': [],
            'teleop_abs_reconstruction_pos_error': [],
            'teleop_abs_reconstruction_rot_error': [],
            'upper_candidate_target_abs': [],
            'upper_candidate_pos_error': [],
            'upper_candidate_rot_error': [],
            'upper_executed_target_abs': [],
            'teleop_candidate_loop_dt_ms': [],
            'teleop_candidate_stale': [],
            'teleop_candidate_applied': [],
            'gripper': [],
            'timestamps': [],
        }
        
        self.episode_count += 1
        if self.teleop_bridge is not None:
            self.teleop_bridge.activate_for_recording()
        rospy.loginfo(f"Recording started - Episode {self.episode_count}")
    
    def stop_recording(self):
        """停止记录并等待确认"""
        if not self.recording:
            rospy.logwarn("Not recording")
            return
        
        self.recording = False
        if self.teleop_bridge is not None:
            self.teleop_bridge.deactivate_owner()
        rospy.loginfo(f">>> Recording stopped - {self.step_count} steps collected")
        
        if len(self.episode_data['timestamps']) == 0:
            rospy.logwarn("No data recorded, nothing to save")
            rospy.loginfo(">>> Press 's' to start next episode")
            return
        
        # 保存数据到待确认状态
        self.pending_episode_data = copy.deepcopy(self.episode_data)
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

            if len(episode_data.get('teleop_valid_mask', [])) > 0:
                f.create_dataset('teleop_valid_mask', data=np.array(episode_data['teleop_valid_mask'], dtype=np.bool_))
                f.create_dataset('teleop_processed_target_abs', data=np.array(episode_data['teleop_processed_target_abs']))
                f.create_dataset('teleop_human_chunk_abs', data=np.array(episode_data['teleop_human_chunk_abs']))
                f.create_dataset('teleop_human_chunk_rel', data=np.array(episode_data['teleop_human_chunk_rel']))
                f.create_dataset('teleop_reconstructed_target_abs', data=np.array(episode_data['teleop_reconstructed_target_abs']))
                f.create_dataset('teleop_active', data=np.array(episode_data['teleop_active'], dtype=np.bool_))
                f.create_dataset('teleop_processed_sequence', data=np.array(episode_data['teleop_processed_sequence'], dtype=np.int64))
                f.create_dataset('teleop_raw_sequence', data=np.array(episode_data['teleop_raw_sequence'], dtype=np.int64))
                f.create_dataset('teleop_signal_age_ms', data=np.array(episode_data['teleop_signal_age_ms'], dtype=np.float64))
                f.create_dataset(
                    'teleop_abs_reconstruction_pos_error',
                    data=np.array(episode_data['teleop_abs_reconstruction_pos_error'], dtype=np.float64),
                )
                f.create_dataset(
                    'teleop_abs_reconstruction_rot_error',
                    data=np.array(episode_data['teleop_abs_reconstruction_rot_error'], dtype=np.float64),
                )
                f.create_dataset('upper_candidate_target_abs', data=np.array(episode_data['upper_candidate_target_abs']))
                f.create_dataset(
                    'upper_candidate_pos_error',
                    data=np.array(episode_data['upper_candidate_pos_error'], dtype=np.float64),
                )
                f.create_dataset(
                    'upper_candidate_rot_error',
                    data=np.array(episode_data['upper_candidate_rot_error'], dtype=np.float64),
                )
                f.create_dataset('upper_executed_target_abs', data=np.array(episode_data['upper_executed_target_abs']))
                f.create_dataset(
                    'teleop_candidate_loop_dt_ms',
                    data=np.array(episode_data['teleop_candidate_loop_dt_ms'], dtype=np.float64),
                )
                f.create_dataset(
                    'teleop_candidate_stale',
                    data=np.array(episode_data['teleop_candidate_stale'], dtype=np.bool_),
                )
                f.create_dataset(
                    'teleop_candidate_applied',
                    data=np.array(episode_data['teleop_candidate_applied'], dtype=np.bool_),
                )

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
            f.attrs['data_version'] = 'v4'  # teleop uplift shadow / candidate metadata
            f.attrs['teleop_bridge_mode'] = self.teleop_bridge_mode
            f.attrs['backend_url_v2'] = self.teleop_signal_client.backend_url_v2
            f.attrs['events_v2_url'] = self.teleop_signal_client.events_v2_url
            f.attrs['teleop_signal_timeout_ms'] = self.teleop_signal_timeout_ms
            f.attrs['pred_horizon'] = self.pred_horizon
            f.attrs['act_horizon'] = self.act_horizon
            f.attrs['lower_control_enabled_at_start'] = bool(
                self.start_control_state.get('local_control_enabled')
            ) if self.start_control_state.get('local_control_enabled') is not None else True
            f.attrs['control_owner_at_start'] = self.start_control_state.get('control_owner', 'lower_machine')
            f.attrs['upper_control_enabled_at_start'] = bool(self.upper_control_enabled)
        
        rospy.loginfo(f"Episode saved: {num_steps} steps, {images.nbytes / 1e6:.1f} MB")
    
    def record_step(self, obs):
        """
        记录一步数据
        
        Args:
            obs: 观测字典
        """
        if self.shutting_down:
            return
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
        
        # 记录遥操作目标位姿（v2 双通道主路径）
        t_action_query_sys = time.time()
        teleop_snapshot = self.teleop_signal_client.fetch_snapshot()
        teleop_state_v2 = teleop_snapshot.get('teleop_state_v2')
        transformed = self.teleop_shadow_transformer.build(obs['qpos_end'][:7], teleop_state_v2)
        if self.teleop_bridge is not None:
            bridge_input = transformed['processed_target_abs'] if transformed['teleop_valid'] else None
            self.teleop_bridge.update_signal(
                processed_target_abs=bridge_input,
                signal_age_ms=teleop_snapshot.get('signal_age_ms'),
                teleop_active=teleop_snapshot.get('teleop_active'),
                processed_sequence=teleop_snapshot.get('processed_sequence'),
            )
            bridge_snapshot = self.teleop_bridge.snapshot()
        else:
            bridge_snapshot = {
                'upper_candidate_target_abs': np.zeros(8, dtype=np.float64),
                'upper_executed_target_abs': np.zeros(8, dtype=np.float64),
                'upper_candidate_pos_error': 0.0,
                'upper_candidate_rot_error': 0.0,
                'teleop_candidate_loop_dt_ms': 0.0,
                'teleop_candidate_stale': False,
                'teleop_candidate_applied': False,
            }

        processed = teleop_snapshot.get('processed') or {}
        teleop_scale = processed.get('scale', 0.0) or 0.0
        if transformed['teleop_valid']:
            action = np.array(transformed['processed_target_abs'], dtype=np.float64)
            self.episode_data['teleop_scale'].append(teleop_scale)
        else:
            action = np.array(obs['qpos_end'], dtype=np.float64)  # 8D
            self.episode_data['teleop_scale'].append(0.0)
        self.episode_data['action'].append(action)

        self.episode_data['teleop_valid_mask'].append(bool(transformed['teleop_valid']))
        self.episode_data['teleop_processed_target_abs'].append(transformed['processed_target_abs'])
        self.episode_data['teleop_human_chunk_abs'].append(transformed['human_chunk_abs'])
        self.episode_data['teleop_human_chunk_rel'].append(transformed['human_chunk_rel'])
        self.episode_data['teleop_reconstructed_target_abs'].append(transformed['reconstructed_target_abs'])
        self.episode_data['teleop_active'].append(bool(teleop_snapshot.get('teleop_active')))
        processed_sequence = teleop_snapshot.get('processed_sequence')
        raw_sequence = teleop_snapshot.get('raw_sequence')
        self.episode_data['teleop_processed_sequence'].append(
            -1 if processed_sequence is None else int(processed_sequence)
        )
        self.episode_data['teleop_raw_sequence'].append(
            -1 if raw_sequence is None else int(raw_sequence)
        )
        self.episode_data['teleop_signal_age_ms'].append(
            float(teleop_snapshot.get('signal_age_ms') or 0.0)
        )
        self.episode_data['teleop_abs_reconstruction_pos_error'].append(
            float(transformed['abs_reconstruction_pos_error'])
        )
        self.episode_data['teleop_abs_reconstruction_rot_error'].append(
            float(transformed['abs_reconstruction_rot_error'])
        )
        self.episode_data['upper_candidate_target_abs'].append(
            np.asarray(bridge_snapshot['upper_candidate_target_abs'], dtype=np.float64)
        )
        self.episode_data['upper_candidate_pos_error'].append(
            float(bridge_snapshot['upper_candidate_pos_error'])
        )
        self.episode_data['upper_candidate_rot_error'].append(
            float(bridge_snapshot['upper_candidate_rot_error'])
        )
        self.episode_data['upper_executed_target_abs'].append(
            np.asarray(bridge_snapshot['upper_executed_target_abs'], dtype=np.float64)
        )
        self.episode_data['teleop_candidate_loop_dt_ms'].append(
            float(bridge_snapshot['teleop_candidate_loop_dt_ms'])
        )
        self.episode_data['teleop_candidate_stale'].append(
            bool(bridge_snapshot['teleop_candidate_stale'])
        )
        self.episode_data['teleop_candidate_applied'].append(
            bool(bridge_snapshot['teleop_candidate_applied'])
        )

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
                teleop_processed_sequence=teleop_snapshot.get('processed_sequence'),
                teleop_raw_sequence=teleop_snapshot.get('raw_sequence'),
                teleop_active=bool(teleop_snapshot.get('teleop_active')),
                teleop_signal_age_ms=teleop_snapshot.get('signal_age_ms'),
                processed_target_abs_present=bool(transformed['teleop_valid']),
                human_chunk_rel_present=bool(transformed['teleop_valid']),
                abs_reconstruction_pos_error=float(transformed['abs_reconstruction_pos_error']),
                abs_reconstruction_rot_error=float(transformed['abs_reconstruction_rot_error']),
                upper_candidate_pos_error=float(bridge_snapshot['upper_candidate_pos_error']),
                upper_candidate_rot_error=float(bridge_snapshot['upper_candidate_rot_error']),
                teleop_candidate_loop_dt_ms=float(bridge_snapshot['teleop_candidate_loop_dt_ms']),
                teleop_candidate_stale=bool(bridge_snapshot['teleop_candidate_stale']),
                teleop_candidate_applied=bool(bridge_snapshot['teleop_candidate_applied']),
            )
            if bridge_snapshot['teleop_candidate_stale'] and not self.last_candidate_stale:
                self.timeline_logger.log(
                    'stale_signal_event',
                    episode=self.episode_count,
                    step=self.step_count,
                    teleop_processed_sequence=teleop_snapshot.get('processed_sequence'),
                    teleop_signal_age_ms=teleop_snapshot.get('signal_age_ms'),
                )
        self.last_candidate_stale = bool(bridge_snapshot['teleop_candidate_stale'])
        
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
            if self.shutting_down:
                break
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
        if self._shutdown_complete:
            return
        self.shutting_down = True
        rospy.loginfo("Shutting down DataRecorder...")
        
        if self.recording:
            self.stop_recording()

        if self.teleop_bridge is not None:
            self.teleop_bridge.stop()
        self.teleop_signal_client.close()
        self.env.shutdown()
        cv2.destroyAllWindows()

        if self.timeline_logger is not None:
            self.timeline_logger.close()
            self.timeline_logger = None
        
        self._shutdown_complete = True
        
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
    parser.add_argument('--backend_url_v2', type=str, default='',
                        help='Backend v2 API URL (default: http://{robot_ip}:1999/api/joystick/teleop_target_v2)')
    parser.add_argument('--events_v2_url', type=str, default='',
                        help='Backend SSE v2 URL (default: http://{robot_ip}:1999/api/joystick/events_v2)')
    parser.add_argument('--teleop_bridge_mode', type=str, default='passive_shadow',
                        choices=['passive_shadow', 'upper_control'],
                        help='Teleop bridge mode for recorder')
    parser.add_argument('--pred_horizon', type=int, default=16,
                        help='Prediction horizon used for teleop shadow chunk construction')
    parser.add_argument('--act_horizon', type=int, default=16,
                        help='Action horizon metadata for teleop uplift runs')
    parser.add_argument('--teleop_candidate_control_freq', type=float, default=50.0,
                        help='Candidate/live upper control loop frequency in Hz')
    parser.add_argument('--teleop_signal_timeout_ms', type=float, default=150.0,
                        help='Teleop signal stale timeout in milliseconds')
    parser.add_argument('--upper_control_enabled', action='store_true',
                        help='Enable live upper control in upper_control mode')
    parser.add_argument('--enable_teleop_sse', action='store_true',
                        help='Enable background SSE monitor for teleop v2')
    parser.add_argument('--return_to_zero', action='store_true',
                        help='Return to zero on shutdown (disabled automatically in upper_control mode)')

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
        'teleop', 'vis', 'timeline_log', 'timeline_enabled', 'timeline_disabled',
        'backend_url', 'backend_url_v2', 'events_v2_url', 'teleop_bridge_mode',
        'pred_horizon', 'act_horizon', 'teleop_candidate_control_freq',
        'teleop_signal_timeout_ms', 'upper_control_enabled', 'enable_teleop_sse',
        'return_to_zero',
    ]:
        if rospy.has_param(f'~{key}'):
            config[key] = rospy.get_param(f'~{key}')

    # 时间线日志开关：默认开启，除非显式禁用
    if config.get('timeline_disabled', False):
        config['timeline_enabled'] = False
    else:
        config['timeline_enabled'] = True
    
    camera_config = resolve_camera_config(
        config,
        ros_has_param=rospy.has_param,
        ros_get_param=rospy.get_param,
        logwarn=rospy.logwarn,
    )
    config.update(camera_config.to_runtime_dict())
    config['camera_config'] = camera_config

    rospy.loginfo("=" * 50)
    rospy.loginfo("CARM Data Recording Node")
    rospy.loginfo("=" * 50)
    rospy.loginfo(f"Robot IP: {config['robot_ip']}")
    rospy.loginfo(f"Camera topics: {config['camera_topics']}")
    rospy.loginfo(f"Output dir: {config['output_dir']}")
    rospy.loginfo(f"Teleop bridge mode: {config.get('teleop_bridge_mode', 'passive_shadow')}")
    rospy.loginfo(f"Upper control enabled: {config.get('upper_control_enabled', False)}")
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
