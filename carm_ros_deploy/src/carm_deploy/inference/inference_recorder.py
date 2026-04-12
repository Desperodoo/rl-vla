#!/usr/bin/env python3
"""
推理数据采集模块 - 在推理过程中记录数据用于后续训练

数据格式设计原则:
1. action 定义遵循 inference_ros 的模型输出格式（保持训练-推理一致性）
2. 记录模型原始输出 (action_model) 和实际执行输出 (action_executed)
3. observations/images 记录 primary camera 兼容字段，images_by_camera 保存多视角
4. observations/timestamps 记录 ROS observation stamp

HDF5 文件结构:
    episode_{XXXX}_{timestamp}.hdf5
    ├── observations/
    │   ├── images        [T, H, W, C]      # 主视角图像
    │   ├── images_by_camera/
    │   │   └── ...       [T, H, W, C]      # 多相机图像
    │   ├── qpos_joint    [T, 7]            # 关节角度 (6 joints + 1 gripper)
    │   ├── qpos_end      [T, 8]            # 末端位姿 (x,y,z,qx,qy,qz,qw,gripper)
    │   ├── qpos          [T, 15]           # 兼容旧版: [joints(7), end_pose(8)]
    │   ├── gripper       [T]               # 夹爪状态
    │   └── timestamps    [T]               # ROS observation stamp
    ├── action_model      [T, pred_horizon, action_dim]  # 模型原始输出
    ├── action_executed   [T, pred_horizon, action_dim]  # 实际执行 action
    └── attrs:
        ├── num_steps
        ├── pred_horizon
        ├── action_dim
        └── ...
"""

import glob
import os
import time
import json
import threading
import numpy as np
import h5py
from datetime import datetime
from typing import Optional, Dict, List, Any

try:
    import rospy
    HAS_ROSPY = True
except ImportError:
    HAS_ROSPY = False


def _log_info(msg: str):
    if HAS_ROSPY:
        rospy.loginfo(msg)
    else:
        print(f"[INFO] {msg}")


def _log_warn(msg: str):
    if HAS_ROSPY:
        rospy.logwarn(msg)
    else:
        print(f"[WARN] {msg}")


class InferenceRecorder:
    """
    推理数据记录器
    
    在推理过程中记录:
    - 观测数据 (图像、关节状态、末端位姿)
    - 模型原始输出 action
    - 实际执行的 action
    """
    
    def __init__(
        self,
        output_dir: str,
        pred_horizon: int = 16,
        action_dim: int = 15,
        image_size: tuple = (128, 128),
        max_steps: int = 2000,
        camera_topics: Optional[List[str]] = None,
        camera_names: Optional[List[str]] = None,
        primary_camera: Optional[str] = None,
        hitl_enabled: bool = False,
        hitl_mode: str = 'disabled',
        hitl_signal_source: str = 'teleop_v2_processed',
        hitl_arbitration_mode: str = 'source_select',
        hitl_record_full_provenance: bool = True,
    ):
        """
        初始化记录器
        
        Args:
            output_dir: 输出目录
            pred_horizon: 预测 horizon
            action_dim: action 维度
            image_size: 图像尺寸 (H, W)
            max_steps: 单个 episode 最大步数
        """
        self.output_dir = os.path.expandvars(os.path.expanduser(output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.image_size = image_size
        self.max_steps = max_steps
        self.camera_topics = [str(topic) for topic in (camera_topics or [])]
        self.camera_names = [str(name) for name in (camera_names or [])]
        self.camera_index = {name: idx for idx, name in enumerate(self.camera_names)}
        self.primary_camera = primary_camera or (self.camera_names[0] if self.camera_names else None)
        self.hitl_enabled = bool(hitl_enabled)
        self.hitl_mode = str(hitl_mode)
        self.hitl_signal_source = str(hitl_signal_source)
        self.hitl_arbitration_mode = str(hitl_arbitration_mode)
        self.hitl_record_full_provenance = bool(hitl_record_full_provenance)
        if self.camera_topics and len(self.camera_names) != len(self.camera_topics):
            raise ValueError("normalized camera_names count must match camera_topics count")
        if self.primary_camera is not None and self.primary_camera not in self.camera_index:
            raise ValueError(f"normalized primary_camera '{self.primary_camera}' not found in camera_names")
        self.primary_camera_idx = self.camera_index[self.primary_camera] if self.primary_camera is not None else 0
        self._lock = threading.RLock()
        
        # 状态
        self.recording = False
        self.episode_count = 0
        self.step_count = 0
        
        # 数据缓冲
        self._reset_buffer()
        
        # 待确认保存的数据
        self.pending_save = False
        self.pending_data = None
        
        _log_info(f"InferenceRecorder initialized, output_dir: {self.output_dir}")
    
    def _reset_buffer(self):
        """重置数据缓冲"""
        self.episode_data = {
            # 观测
            'images': [],           # [T, H, W, C]
            'images_by_camera': {name: [] for name in self.camera_names},
            'qpos_joint': [],       # [T, 7]
            'qpos_end': [],         # [T, 8]
            'qpos': [],             # [T, 15] 兼容旧版
            'gripper': [],          # [T]
            'timestamps': [],       # [T] ROS observation stamp

            # Action
            'action_model': [],      # [T, pred_horizon, action_dim] 模型输出
            'action_executed': [],   # [T, pred_horizon, action_dim] 实际执行输出

            # HITL provenance
            'action_policy_chunk': [],
            'action_human_chunk': [],
            'action_shared_chunk': [],
            'action_sched_candidate': [],
            'action_exec_candidate': [],
            'action_human_direct_target': [],
            'action_human_sched_target': [],
            'action_human_exec_target': [],
            'action_live_execute_target': [],
            'hitl_human_active': [],
            'hitl_human_valid': [],
            'hitl_signal_age_ms': [],
            'hitl_human_history_count': [],
            'hitl_human_history_span_ms': [],
            'hitl_human_history_usable': [],
            'hitl_human_rollout_step_count': [],
            'hitl_human_rollout_dt_ms': [],
            'hitl_human_linear_velocity': [],
            'hitl_human_angular_velocity': [],
            'hitl_human_gripper_velocity': [],
            'hitl_policy_sequence': [],
            'hitl_human_sequence': [],
            'hitl_shared_source': [],
            'hitl_shared_valid_mask': [],
            'hitl_live_execute_source': [],
        }
        self.control_data = {
            'timestamps': [],
            't_send_sys': [],
            'execute_source': [],
            'human_execute_mode': [],
            'live_execute_target': [],
            'human_direct_target': [],
            'human_sched_target': [],
            'human_exec_target': [],
            'shared_source': [],
        }
        self.step_count = 0

    @staticmethod
    def _copy_pending_value(value):
        if isinstance(value, list):
            return value.copy()
        if isinstance(value, dict):
            return {k: InferenceRecorder._copy_pending_value(v) for k, v in value.items()}
        return value
    
    def start_recording(self) -> bool:
        """
        开始记录
        
        Returns:
            是否成功开始
        """
        with self._lock:
            if self.recording:
                _log_warn("Already recording")
                return False
            
            if self.pending_save:
                _log_warn("Please confirm save first (y/n)")
                return False
            
            self.recording = True
            self._reset_buffer()
            self.episode_count += 1
        
        _log_info(f"Recording started - Episode {self.episode_count}")
        return True
    
    def stop_recording(self) -> bool:
        """
        停止记录，等待确认保存
        
        Returns:
            是否有数据等待保存
        """
        with self._lock:
            if not self.recording:
                _log_warn("Not recording")
                return False
            
            self.recording = False
            _log_info(f"Recording stopped - {self.step_count} steps collected")
            
            if self.step_count == 0:
                _log_warn("No data recorded, nothing to save")
                return False
            
            # 保存到待确认状态
            self.pending_data = {
                'episode_data': self._copy_pending_value(self.episode_data),
                'control_data': self._copy_pending_value(self.control_data),
            }
            self.pending_save = True
        
        _log_info("=" * 50)
        _log_info(f"Episode {self.episode_count}: {self.step_count} steps")
        _log_info("Save this episode? (y/n)")
        _log_info("=" * 50)
        
        return True
    
    def confirm_save(self, success: bool, outcome_label: Optional[str] = None) -> Optional[str]:
        """
        确认保存 episode

        Args:
            success: episode 是否成功
            outcome_label: outcome 文本标签，默认按 success 推断

        Returns:
            保存的文件路径，如果没有数据返回 None
        """
        with self._lock:
            if not self.pending_save or self.pending_data is None:
                _log_warn("No pending data to save")
                return None

            filepath = self._do_save(success=success, outcome_label=outcome_label)
            self.pending_save = False
            self.pending_data = None

            return filepath
    
    def discard(self):
        """丢弃当前待保存的 episode"""
        with self._lock:
            if not self.pending_save:
                return
            
            _log_info("Episode discarded")
            self.pending_save = False
            self.pending_data = None

    def record_control_step(
        self,
        *,
        query_time: float,
        t_send_sys: Optional[float] = None,
        execute_source: str = 'policy',
        human_execute_mode: Optional[str] = None,
        live_execute_target: Optional[np.ndarray] = None,
        human_direct_target: Optional[np.ndarray] = None,
        human_sched_target: Optional[np.ndarray] = None,
        human_exec_target: Optional[np.ndarray] = None,
        shared_source: Optional[str] = None,
    ) -> bool:
        """Record control-loop aligned provenance."""
        with self._lock:
            if not self.recording:
                return False

            zeros_step = np.zeros((self.action_dim,), dtype=np.float64)
            self.control_data['timestamps'].append(float(query_time))
            self.control_data['t_send_sys'].append(
                float(time.time() if t_send_sys is None else t_send_sys)
            )
            self.control_data['execute_source'].append(str(execute_source or 'policy'))
            self.control_data['human_execute_mode'].append('' if human_execute_mode is None else str(human_execute_mode))
            self.control_data['live_execute_target'].append(
                np.asarray(zeros_step if live_execute_target is None else live_execute_target, dtype=np.float64)
            )
            self.control_data['human_direct_target'].append(
                np.asarray(zeros_step if human_direct_target is None else human_direct_target, dtype=np.float64)
            )
            self.control_data['human_sched_target'].append(
                np.asarray(zeros_step if human_sched_target is None else human_sched_target, dtype=np.float64)
            )
            self.control_data['human_exec_target'].append(
                np.asarray(zeros_step if human_exec_target is None else human_exec_target, dtype=np.float64)
            )
            self.control_data['shared_source'].append('' if shared_source is None else str(shared_source))
            return True
    
    def record_step(
        self,
        obs: Dict[str, Any],
        action_model: np.ndarray,
        action_executed: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        hitl_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        记录一步数据
        
        Args:
            obs: 观测字典，包含 images, qpos_joint, qpos_end, qpos, gripper
            action_model: 模型原始输出 [pred_horizon, action_dim]
            action_executed: 实际执行的 action [pred_horizon, action_dim]，None 表示与模型输出一致
            timestamp: 时间戳，None 使用当前系统时间
            hitl_data: HITL provenance 数据
        """
        with self._lock:
            if not self.recording:
                return False
            
            if self.step_count >= self.max_steps:
                _log_warn(f"Reached max steps ({self.max_steps}), stopping recording")
                self.stop_recording()
                return False
            
            # 时间戳
            if timestamp is None:
                timestamp = time.time()
            obs_images = obs['images']
            # 观测
            if len(obs_images) == 0:
                raise ValueError('obs["images"] is empty')
            if self.primary_camera_idx >= len(obs_images):
                _log_warn(
                    f"Primary camera index {self.primary_camera_idx} out of range, fallback to index 0 (available={len(obs_images)})"
                )
                primary_img = obs_images[0]
            else:
                primary_img = obs_images[self.primary_camera_idx]
            self.episode_data['images'].append(primary_img)  # 兼容字段：主视角

            for camera_name, camera_idx in self.camera_index.items():
                if camera_idx < len(obs_images):
                    self.episode_data['images_by_camera'][camera_name].append(obs_images[camera_idx])
            self.episode_data['qpos_joint'].append(np.array(obs['qpos_joint']))
            self.episode_data['qpos_end'].append(np.array(obs['qpos_end']))
            self.episode_data['qpos'].append(np.array(obs['qpos']))
            self.episode_data['gripper'].append(float(obs['gripper']))
            self.episode_data['timestamps'].append(timestamp)
            
            # Action
            self.episode_data['action_model'].append(action_model.copy())
            
            if action_executed is not None:
                self.episode_data['action_executed'].append(action_executed.copy())
            else:
                self.episode_data['action_executed'].append(action_model.copy())

            if self.hitl_enabled:
                hitl_data = hitl_data or {}
                zeros_chunk = np.zeros_like(action_model, dtype=np.float64)
                zeros_step = np.zeros((action_model.shape[-1],), dtype=np.float64)
                self.episode_data['action_policy_chunk'].append(
                    np.asarray(hitl_data.get('action_policy_chunk', action_model), dtype=np.float64)
                )
                self.episode_data['action_human_chunk'].append(
                    np.asarray(hitl_data.get('action_human_chunk', zeros_chunk), dtype=np.float64)
                )
                self.episode_data['action_shared_chunk'].append(
                    np.asarray(hitl_data.get('action_shared_chunk', action_model), dtype=np.float64)
                )
                self.episode_data['action_sched_candidate'].append(
                    np.asarray(hitl_data.get('action_sched_candidate', zeros_step), dtype=np.float64)
                )
                self.episode_data['action_exec_candidate'].append(
                    np.asarray(hitl_data.get('action_exec_candidate', zeros_step), dtype=np.float64)
                )
                self.episode_data['action_human_direct_target'].append(
                    np.asarray(hitl_data.get('action_human_direct_target', zeros_step), dtype=np.float64)
                )
                human_sched_target = hitl_data.get('action_human_sched_target')
                if human_sched_target is None:
                    human_sched_target = zeros_step
                self.episode_data['action_human_sched_target'].append(
                    np.asarray(human_sched_target, dtype=np.float64)
                )
                human_exec_target = hitl_data.get('action_human_exec_target')
                if human_exec_target is None:
                    human_exec_target = zeros_step
                self.episode_data['action_human_exec_target'].append(
                    np.asarray(human_exec_target, dtype=np.float64)
                )
                live_execute_target = hitl_data.get('action_live_execute_target')
                if live_execute_target is None:
                    if action_executed is not None:
                        live_execute_target = np.asarray(action_executed[0], dtype=np.float64)
                    else:
                        live_execute_target = np.asarray(action_model[0], dtype=np.float64)
                self.episode_data['action_live_execute_target'].append(
                    np.asarray(live_execute_target, dtype=np.float64)
                )
                self.episode_data['hitl_human_active'].append(bool(hitl_data.get('hitl_human_active', False)))
                self.episode_data['hitl_human_valid'].append(bool(hitl_data.get('hitl_human_valid', False)))
                signal_age_ms = hitl_data.get('hitl_signal_age_ms')
                self.episode_data['hitl_signal_age_ms'].append(
                    np.nan if signal_age_ms is None else float(signal_age_ms)
                )
                self.episode_data['hitl_human_history_count'].append(int(hitl_data.get('hitl_human_history_count', 0)))
                history_span_ms = hitl_data.get('hitl_human_history_span_ms')
                self.episode_data['hitl_human_history_span_ms'].append(
                    np.nan if history_span_ms is None else float(history_span_ms)
                )
                self.episode_data['hitl_human_history_usable'].append(bool(hitl_data.get('hitl_human_history_usable', False)))
                self.episode_data['hitl_human_rollout_step_count'].append(int(hitl_data.get('hitl_human_rollout_step_count', 0)))
                rollout_dt_ms = hitl_data.get('hitl_human_rollout_dt_ms')
                self.episode_data['hitl_human_rollout_dt_ms'].append(
                    np.nan if rollout_dt_ms is None else float(rollout_dt_ms)
                )
                self.episode_data['hitl_human_linear_velocity'].append(
                    np.asarray(hitl_data.get('hitl_human_linear_velocity', np.zeros((3,), dtype=np.float64)), dtype=np.float64)
                )
                self.episode_data['hitl_human_angular_velocity'].append(
                    np.asarray(hitl_data.get('hitl_human_angular_velocity', np.zeros((3,), dtype=np.float64)), dtype=np.float64)
                )
                gripper_velocity = hitl_data.get('hitl_human_gripper_velocity')
                self.episode_data['hitl_human_gripper_velocity'].append(
                    np.nan if gripper_velocity is None else float(gripper_velocity)
                )
                self.episode_data['hitl_policy_sequence'].append(int(hitl_data.get('hitl_policy_sequence', self.step_count)))
                self.episode_data['hitl_human_sequence'].append(int(hitl_data.get('hitl_human_sequence', -1)))
                self.episode_data['hitl_shared_source'].append(int(hitl_data.get('hitl_shared_source', 0)))
                self.episode_data['hitl_shared_valid_mask'].append(bool(hitl_data.get('hitl_shared_valid_mask', True)))
                self.episode_data['hitl_live_execute_source'].append(int(hitl_data.get('hitl_live_execute_source', 0)))
            
            self.step_count += 1
            return True
    
    def _do_save(self, success: bool, outcome_label: Optional[str] = None) -> str:
        """实际执行保存"""
        if self.pending_data is None:
            return None
        
        data = self.pending_data['episode_data']
        control_data = self.pending_data.get('control_data', {})
        outcome_label = outcome_label or ('success' if success else 'failure')
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"inference_episode_{self.episode_count:04d}_{timestamp}.hdf5"
        filepath = os.path.join(self.output_dir, filename)
        
        _log_info(f"Saving episode to {filepath}...")
        
        num_steps = len(data['timestamps'])
        
        with h5py.File(filepath, 'w') as f:
            # 观测数据
            obs_grp = f.create_group('observations')
            
            images = np.array(data['images'])  # [T, H, W, C]
            obs_grp.create_dataset('images', data=images, compression='gzip')

            images_by_camera = data.get('images_by_camera', {})
            if len(images_by_camera) > 0:
                cameras_grp = obs_grp.create_group('images_by_camera')
                for camera_name, camera_images in images_by_camera.items():
                    if len(camera_images) == 0:
                        continue
                    cameras_grp.create_dataset(camera_name, data=np.array(camera_images), compression='gzip')
            
            qpos_joint = np.array(data['qpos_joint'])  # [T, 7]
            obs_grp.create_dataset('qpos_joint', data=qpos_joint)

            qpos_end = np.array(data['qpos_end'])  # [T, 8]
            obs_grp.create_dataset('qpos_end', data=qpos_end)
            
            qpos = np.array(data['qpos'])  # [T, 15]
            obs_grp.create_dataset('qpos', data=qpos)
            
            gripper = np.array(data['gripper'])  # [T]
            obs_grp.create_dataset('gripper', data=gripper)
            
            timestamps = np.array(data['timestamps'])  # [T]
            obs_grp.create_dataset('timestamps', data=timestamps)
            
            # Action 数据
            action_model = np.array(data['action_model'])  # [T, pred_horizon, action_dim]
            f.create_dataset('action_model', data=action_model)
            
            action_executed = np.array(data['action_executed'])  # [T, pred_horizon, action_dim]
            f.create_dataset('action_executed', data=action_executed)

            if self.hitl_enabled and self.hitl_record_full_provenance:
                f.create_dataset('action_policy_chunk', data=np.array(data['action_policy_chunk']))
                f.create_dataset('action_human_chunk', data=np.array(data['action_human_chunk']))
                f.create_dataset('action_shared_chunk', data=np.array(data['action_shared_chunk']))
                f.create_dataset('action_sched_candidate', data=np.array(data['action_sched_candidate']))
                f.create_dataset('action_exec_candidate', data=np.array(data['action_exec_candidate']))
                f.create_dataset('action_human_direct_target', data=np.array(data['action_human_direct_target']))
                f.create_dataset('action_human_sched_target', data=np.array(data['action_human_sched_target']))
                f.create_dataset('action_human_exec_target', data=np.array(data['action_human_exec_target']))
                f.create_dataset('action_live_execute_target', data=np.array(data['action_live_execute_target']))
                f.create_dataset('hitl_human_active', data=np.array(data['hitl_human_active'], dtype=np.bool_))
                f.create_dataset('hitl_human_valid', data=np.array(data['hitl_human_valid'], dtype=np.bool_))
                f.create_dataset('hitl_signal_age_ms', data=np.array(data['hitl_signal_age_ms'], dtype=np.float64))
                f.create_dataset('hitl_human_history_count', data=np.array(data['hitl_human_history_count'], dtype=np.int64))
                f.create_dataset('hitl_human_history_span_ms', data=np.array(data['hitl_human_history_span_ms'], dtype=np.float64))
                f.create_dataset('hitl_human_history_usable', data=np.array(data['hitl_human_history_usable'], dtype=np.bool_))
                f.create_dataset('hitl_human_rollout_step_count', data=np.array(data['hitl_human_rollout_step_count'], dtype=np.int64))
                f.create_dataset('hitl_human_rollout_dt_ms', data=np.array(data['hitl_human_rollout_dt_ms'], dtype=np.float64))
                f.create_dataset('hitl_human_linear_velocity', data=np.array(data['hitl_human_linear_velocity'], dtype=np.float64))
                f.create_dataset('hitl_human_angular_velocity', data=np.array(data['hitl_human_angular_velocity'], dtype=np.float64))
                f.create_dataset('hitl_human_gripper_velocity', data=np.array(data['hitl_human_gripper_velocity'], dtype=np.float64))
                f.create_dataset('hitl_policy_sequence', data=np.array(data['hitl_policy_sequence'], dtype=np.int64))
                f.create_dataset('hitl_human_sequence', data=np.array(data['hitl_human_sequence'], dtype=np.int64))
                f.create_dataset('hitl_shared_source', data=np.array(data['hitl_shared_source'], dtype=np.int64))
                f.create_dataset('hitl_shared_valid_mask', data=np.array(data['hitl_shared_valid_mask'], dtype=np.bool_))
                f.create_dataset('hitl_live_execute_source', data=np.array(data['hitl_live_execute_source'], dtype=np.int64))
                control_grp = f.create_group('control_provenance')
                control_grp.create_dataset('timestamps', data=np.array(control_data.get('timestamps', []), dtype=np.float64))
                control_grp.create_dataset('t_send_sys', data=np.array(control_data.get('t_send_sys', []), dtype=np.float64))
                control_grp.create_dataset('execute_source', data=np.array(control_data.get('execute_source', []), dtype=h5py.string_dtype(encoding='utf-8')))
                control_grp.create_dataset('human_execute_mode', data=np.array(control_data.get('human_execute_mode', []), dtype=h5py.string_dtype(encoding='utf-8')))
                control_grp.create_dataset('live_execute_target', data=np.array(control_data.get('live_execute_target', []), dtype=np.float64))
                control_grp.create_dataset('human_direct_target', data=np.array(control_data.get('human_direct_target', []), dtype=np.float64))
                control_grp.create_dataset('human_sched_target', data=np.array(control_data.get('human_sched_target', []), dtype=np.float64))
                control_grp.create_dataset('human_exec_target', data=np.array(control_data.get('human_exec_target', []), dtype=np.float64))
                control_grp.create_dataset('shared_source', data=np.array(control_data.get('shared_source', []), dtype=h5py.string_dtype(encoding='utf-8')))
            
            # 兼容旧格式: action = action_executed[:, 0, :]
            # 取每步的第一个 action 作为该步的 action
            action = action_executed[:, 0, :]  # [T, action_dim]
            f.create_dataset('action', data=action)
            
            # 元数据
            f.attrs['num_steps'] = num_steps
            f.attrs['pred_horizon'] = self.pred_horizon
            f.attrs['action_dim'] = self.action_dim
            f.attrs['image_height'] = images.shape[1] if len(images.shape) > 1 else 0
            f.attrs['image_width'] = images.shape[2] if len(images.shape) > 2 else 0
            f.attrs['created_at'] = timestamp
            f.attrs['data_source'] = 'inference_rollout'
            f.attrs['timestamp_semantics'] = 'obs_stamp_ros'
            f.attrs['action_semantics_version'] = 'absolute_ee_target_pose_v2'
            f.attrs['action_space'] = 'ee_target_pose_absolute'
            f.attrs['compat_action_source'] = 'action_executed[:,0,:]'
            f.attrs['camera_topics'] = json.dumps(self.camera_topics)
            f.attrs['camera_names'] = json.dumps(self.camera_names)
            f.attrs['primary_camera'] = self.primary_camera or ''
            f.attrs['success'] = bool(success)
            f.attrs['outcome_label'] = outcome_label
            f.attrs['hitl_enabled'] = bool(self.hitl_enabled)
            f.attrs['hitl_mode'] = self.hitl_mode
            f.attrs['hitl_signal_source'] = self.hitl_signal_source
            f.attrs['hitl_arbitration_mode'] = self.hitl_arbitration_mode
            f.attrs['hitl_live_execute_enabled'] = bool(self.hitl_mode == 'live')
            f.attrs['control_provenance_aligned'] = bool(self.hitl_enabled and self.hitl_record_full_provenance)
            f.attrs['num_control_steps'] = int(len(control_data.get('timestamps', [])))
        
        _log_info(
            f"Episode saved: {num_steps} steps, "
            f"outcome={outcome_label}"
        )
        
        return filepath
    
    @property
    def is_recording(self) -> bool:
        """是否正在记录"""
        return self.recording
    
    @property
    def is_pending_save(self) -> bool:
        """是否有待保存的数据"""
        return self.pending_save


class InferenceDatasetConverter:
    """
    将推理采集的数据转换为标准训练格式

    主要处理:
    1. action 从 chunk 格式转为单步格式
    2. 对齐观测和 action 的时间戳
    3. 为训练准入保留 metadata / sidecar
    """

    STAGING_SCHEMA_VERSION = 'inference_staging_v2'
    ADMISSION_LABEL = 'train_admission_v1'
    ADMISSION_POLICY_VERSION = 'carm_inference_admission_v1'
    DEFAULT_MIN_STEPS = 32
    DEFAULT_GOLD_MAX_SAFETY_CLIP_RATE = 0.01
    DEFAULT_SILVER_MAX_SAFETY_CLIP_RATE = 0.05

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, bytes):
            return value.decode('utf-8')
        return value

    @staticmethod
    def _normalize_attrs(attrs: Any) -> Dict[str, Any]:
        return {
            str(k): InferenceDatasetConverter._to_serializable(v)
            for k, v in attrs.items()
        }

    @staticmethod
    def _extract_run_info_summary(run_info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'path': run_info.get('_path'),
            'match_status': run_info.get('_match_status', 'unknown'),
            'timeline': run_info.get('files', {}).get('timeline'),
            'run_info': run_info.get('files', {}).get('run_info'),
            'model_path': run_info.get('model', {}).get('path'),
            'algorithm': run_info.get('model', {}).get('algorithm'),
            'pred_horizon': run_info.get('model', {}).get('pred_horizon'),
            'action_dim': run_info.get('model', {}).get('action_dim_full', run_info.get('model', {}).get('action_dim')),
            'avg_inference_time': run_info.get('summary', {}).get('avg_inference_time'),
            'max_inference_time': run_info.get('summary', {}).get('max_inference_time'),
            'safety_clips': run_info.get('summary', {}).get('safety_clips'),
            'safety_clip_rate': run_info.get('summary', {}).get('safety_clip_rate'),
            'desire_inference_freq': run_info.get('execution', {}).get('desire_inference_freq'),
            'act_horizon': run_info.get('execution', {}).get('act_horizon'),
            'control_freq': run_info.get('control', {}).get('control_freq'),
            'truncate_at_act_horizon': run_info.get('execution', {}).get('truncate_at_act_horizon'),
        }

    @staticmethod
    def _find_run_info(input_path: str) -> Dict[str, Any]:
        input_dir = os.path.dirname(input_path)
        input_name = os.path.basename(input_path)
        run_info_paths = sorted(glob.glob(os.path.join(input_dir, 'run_info_*.json')))
        matches: list[Dict[str, Any]] = []

        for run_info_path in run_info_paths:
            try:
                with open(run_info_path, 'r') as f:
                    run_info = json.load(f)
            except Exception:
                continue

            episode_files = run_info.get('files', {}).get('episode_hdf5', []) or []
            if input_name in episode_files:
                run_info['_path'] = run_info_path
                run_info['_match_status'] = 'matched_by_episode_hdf5'
                matches.append(run_info)

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            chosen = matches[-1]
            chosen['_match_status'] = 'multiple_matches_by_episode_hdf5'
            return chosen

        return {
            '_path': None,
            '_match_status': 'unmatched',
            'files': {},
            'summary': {},
            'execution': {},
            'control': {},
            'model': {},
        }

    @staticmethod
    def _evaluate_admission(
        *,
        policy: str,
        policy_version: str,
        kept_steps: int,
        safety_clip_rate: Optional[float],
        required_ok: bool,
        required_reasons: List[str],
        action: np.ndarray,
        qpos_joint: np.ndarray,
        qpos_end: np.ndarray,
        timestamps: np.ndarray,
        min_steps: int,
        gold_max_safety_clip_rate: float,
        silver_max_safety_clip_rate: float,
    ) -> Dict[str, Any]:
        reasons: list[str] = list(required_reasons)
        if kept_steps < min_steps:
            reasons.append('too_short')
        for name, data in (
            ('action', action),
            ('qpos_joint', qpos_joint),
            ('qpos_end', qpos_end),
            ('timestamps', timestamps),
        ):
            if np.isnan(data).any():
                reasons.append(f'nan_detected:{name}')
                break

        admission_bucket = 'reject'
        if len(reasons) == 0:
            safety_clip_rate_value = 0.0 if safety_clip_rate is None or np.isnan(safety_clip_rate) else float(safety_clip_rate)
            if safety_clip_rate_value <= gold_max_safety_clip_rate:
                admission_bucket = 'gold'
            elif safety_clip_rate_value <= silver_max_safety_clip_rate:
                admission_bucket = 'silver'
            else:
                reasons.append('high_safety_clip_rate')

        if policy == 'none':
            admission_pass = bool(required_ok and len([r for r in reasons if r.startswith('missing_required_data') or r.startswith('too_short') or r.startswith('nan_detected:')]) == 0)
            if admission_pass:
                admission_bucket = 'silver'
                admission_reason = 'ok'
            else:
                admission_bucket = 'reject'
                admission_reason = '|'.join(reasons or ['missing_required_data'])
        else:
            admission_pass = admission_bucket != 'reject'
            admission_reason = 'ok' if admission_pass else '|'.join(reasons or ['rejected'])

        return {
            'admission_label': InferenceDatasetConverter.ADMISSION_LABEL,
            'admission_pass': bool(admission_pass),
            'admission_reason': admission_reason,
            'admission_bucket': admission_bucket,
            'policy_level': 'episode',
            'admission_policy': policy,
            'policy_version': policy_version,
            'min_steps': int(min_steps),
            'gold_max_safety_clip_rate': float(gold_max_safety_clip_rate),
            'silver_max_safety_clip_rate': float(silver_max_safety_clip_rate),
            'safety_clip_rate': None if safety_clip_rate is None else float(safety_clip_rate),
        }

    @staticmethod
    def convert_to_training_format(
        input_path: str,
        output_path: str,
        admission_policy: str = 'none',
        policy_version: Optional[str] = None,
        min_steps: int = 32,
        gold_max_safety_clip_rate: float = 0.01,
        silver_max_safety_clip_rate: float = 0.05,
        drop_failed_episode: bool = False,
    ) -> Dict[str, Any]:
        """
        转换为训练格式并返回 episode-level metadata。

        Args:
            input_path: 输入 HDF5 文件路径
            output_path: 输出 HDF5 文件路径
            admission_policy: 准入策略，'none' 或 'episode'
            policy_version: 写入 metadata 的准入策略版本
            min_steps: episode-level admission 的最小步数
            gold_max_safety_clip_rate: gold bucket 最大 safety clip rate
            silver_max_safety_clip_rate: silver bucket 最大 safety clip rate
            drop_failed_episode: 若准入失败，是否不写出 staging HDF5
        """
        policy_version = policy_version or InferenceDatasetConverter.ADMISSION_POLICY_VERSION
        output_path = os.path.expandvars(os.path.expanduser(output_path))
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        sidecar_path = os.path.splitext(output_path)[0] + '.meta.json'

        run_info = InferenceDatasetConverter._find_run_info(input_path)
        run_summary = InferenceDatasetConverter._extract_run_info_summary(run_info)

        with h5py.File(input_path, 'r') as f_in:
            input_attrs = InferenceDatasetConverter._normalize_attrs(f_in.attrs)
            num_steps = int(f_in.attrs.get('num_steps', 0))
            obs_group = f_in.get('observations')
            if 'action_executed' not in f_in:
                raise KeyError('inference rollout missing required dataset action_executed')
            action_source = f_in['action_executed'][:]
            action_source_name = 'action_executed[:,0,:]'

            action = action_source[:, 0, :]
            keep_idx = np.ones(num_steps, dtype=bool)

            kept_steps = int(keep_idx.sum())
            dropped_steps = int(num_steps - kept_steps)
            action_kept = action[keep_idx]

            required_obs_keys = ['images', 'qpos_joint', 'qpos_end', 'gripper', 'timestamps']
            required_reasons: list[str] = []
            if obs_group is None:
                required_reasons.append('missing_observations_group')
            else:
                for key in required_obs_keys:
                    if key not in obs_group:
                        required_reasons.append(f'missing_required_obs:{key}')
            if 'action_executed' not in f_in:
                required_reasons.append('missing_required_action_executed')
            if f_in.attrs.get('timestamp_semantics', None) != 'obs_stamp_ros':
                required_reasons.append('bad_timestamp_semantics')
            if not str(f_in.attrs.get('primary_camera', '')).strip():
                required_reasons.append('missing_primary_camera')
            for attr_name in ('camera_names', 'camera_topics', 'action_semantics_version', 'action_space'):
                if attr_name not in f_in.attrs:
                    required_reasons.append(f'missing_attr:{attr_name}')
            required_ok = len(required_reasons) == 0

            if required_ok:
                qpos_joint_kept = obs_group['qpos_joint'][:][keep_idx]
                qpos_end_kept = obs_group['qpos_end'][:][keep_idx]
                timestamps_kept = obs_group['timestamps'][:][keep_idx]
            else:
                qpos_joint_kept = np.empty((0,), dtype=np.float32)
                qpos_end_kept = np.empty((0,), dtype=np.float32)
                timestamps_kept = np.empty((0,), dtype=np.float64)

            admission = InferenceDatasetConverter._evaluate_admission(
                policy=admission_policy,
                policy_version=policy_version,
                kept_steps=kept_steps,
                safety_clip_rate=run_summary.get('safety_clip_rate'),
                required_ok=required_ok,
                required_reasons=required_reasons,
                action=action_kept,
                qpos_joint=qpos_joint_kept,
                qpos_end=qpos_end_kept,
                timestamps=timestamps_kept,
                min_steps=min_steps,
                gold_max_safety_clip_rate=gold_max_safety_clip_rate,
                silver_max_safety_clip_rate=silver_max_safety_clip_rate,
            )

            converted = admission['admission_pass'] or not drop_failed_episode
            if converted:
                with h5py.File(output_path, 'w') as f_out:
                    obs_grp = f_out.create_group('observations')

                    for key in f_in['observations'].keys():
                        node = f_in['observations'][key]
                        if isinstance(node, h5py.Group):
                            out_group = obs_grp.create_group(key)
                            for subkey in node.keys():
                                subdata = node[subkey][:][keep_idx]
                                out_group.create_dataset(subkey, data=subdata, compression='gzip')
                            continue

                        data = node[:][keep_idx]
                        if key == 'images':
                            obs_grp.create_dataset(key, data=data, compression='gzip')
                        else:
                            obs_grp.create_dataset(key, data=data)

                    f_out.create_dataset('action', data=action_kept)

                    f_out.attrs['num_steps'] = kept_steps
                    f_out.attrs['source_file'] = os.path.basename(input_path)
                    f_out.attrs['dataset_type'] = 'inference_staging'
                    f_out.attrs['staging_schema_version'] = InferenceDatasetConverter.STAGING_SCHEMA_VERSION
                    f_out.attrs['source_num_steps'] = num_steps
                    f_out.attrs['kept_steps'] = kept_steps
                    f_out.attrs['dropped_steps'] = dropped_steps
                    f_out.attrs['action_source_used'] = action_source_name
                    f_out.attrs['source_run_info'] = os.path.basename(run_summary['path']) if run_summary['path'] else ''
                    f_out.attrs['source_timeline'] = run_summary.get('timeline') or ''

                    for key in [
                        'timestamp_semantics',
                        'camera_topics',
                        'camera_names',
                        'primary_camera',
                        'action_semantics_version',
                        'action_space',
                        'compat_action_source',
                        'data_source',
                        'success',
                        'outcome_label',
                    ]:
                        if key in f_in.attrs:
                            f_out.attrs[key] = f_in.attrs[key]

                    f_out.attrs['admission_label'] = admission['admission_label']
                    f_out.attrs['admission_pass'] = admission['admission_pass']
                    f_out.attrs['admission_reason'] = admission['admission_reason']
                    f_out.attrs['admission_bucket'] = admission['admission_bucket']
                    f_out.attrs['policy_version'] = admission['policy_version']
                    f_out.attrs['policy_level'] = admission['policy_level']
                    f_out.attrs['admission_policy'] = admission['admission_policy']
                    f_out.attrs['min_steps'] = admission['min_steps']
                    f_out.attrs['gold_max_safety_clip_rate'] = admission['gold_max_safety_clip_rate']
                    f_out.attrs['silver_max_safety_clip_rate'] = admission['silver_max_safety_clip_rate']

                    for key in ['safety_clips', 'safety_clip_rate', 'avg_inference_time', 'desire_inference_freq', 'act_horizon', 'control_freq']:
                        value = run_summary.get(key)
                        if value is not None:
                            f_out.attrs[key] = value

        sidecar = {
            'input_path': os.path.abspath(os.path.expandvars(os.path.expanduser(input_path))),
            'output_path': os.path.abspath(output_path),
            'sidecar_path': os.path.abspath(sidecar_path),
            'converted': bool(converted),
            'drop_failed_episode': bool(drop_failed_episode),
            'conversion': {
                'admission_policy': admission_policy,
                'policy_version': policy_version,
                'min_steps': int(min_steps),
                'gold_max_safety_clip_rate': float(gold_max_safety_clip_rate),
                'silver_max_safety_clip_rate': float(silver_max_safety_clip_rate),
            },
            'stats': {
                'num_steps_raw': num_steps,
                'kept_steps': kept_steps,
                'dropped_steps': dropped_steps,
                'action_source_used': action_source_name,
                'success': input_attrs.get('success'),
                'outcome_label': input_attrs.get('outcome_label'),
            },
            'admission': admission,
            'source': {
                'file': os.path.basename(input_path),
                'run_info': run_summary.get('run_info'),
                'timeline': run_summary.get('timeline'),
                'run_info_path': run_summary.get('path'),
                'run_info_match_status': run_summary.get('match_status'),
            },
            'run_info_summary': run_summary,
            'input_attrs': input_attrs,
        }
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)

        return {
            'output_path': output_path,
            'sidecar_path': sidecar_path,
            'converted': bool(converted),
            'admission_pass': bool(admission['admission_pass']),
            'admission_reason': admission['admission_reason'],
            'admission_bucket': admission['admission_bucket'],
            'policy_version': admission['policy_version'],
            'success': input_attrs.get('success'),
            'outcome_label': input_attrs.get('outcome_label'),
            'source_file': os.path.basename(input_path),
        }

    @staticmethod
    def convert_directory_to_training_format(
        input_dir: str,
        output_dir: str,
        admission_policy: str = 'none',
        policy_version: Optional[str] = None,
        min_steps: int = 32,
        gold_max_safety_clip_rate: float = 0.01,
        silver_max_safety_clip_rate: float = 0.05,
        drop_failed_episode: bool = False,
    ) -> list[Dict[str, Any]]:
        """批量将 inference_episode_*.hdf5 转换到训练 staging 目录。"""
        input_dir = os.path.expandvars(os.path.expanduser(input_dir))
        output_dir = os.path.expandvars(os.path.expanduser(output_dir))
        os.makedirs(output_dir, exist_ok=True)

        input_paths = sorted(
            glob.glob(os.path.join(input_dir, 'inference_episode_*.hdf5'))
        )
        converted_records: list[Dict[str, Any]] = []
        for idx, input_path in enumerate(input_paths, start=1):
            output_name = f'episode_{idx:04d}_{os.path.basename(input_path).replace("inference_episode_", "")}'
            output_path = os.path.join(output_dir, output_name)
            try:
                record = InferenceDatasetConverter.convert_to_training_format(
                    input_path=input_path,
                    output_path=output_path,
                    admission_policy=admission_policy,
                    policy_version=policy_version,
                    min_steps=min_steps,
                    gold_max_safety_clip_rate=gold_max_safety_clip_rate,
                    silver_max_safety_clip_rate=silver_max_safety_clip_rate,
                    drop_failed_episode=drop_failed_episode,
                )
            except Exception as exc:
                record = {
                    'output_path': output_path,
                    'sidecar_path': '',
                    'converted': False,
                    'admission_pass': False,
                    'admission_reason': f'conversion_error:{type(exc).__name__}',
                    'admission_bucket': 'reject',
                    'policy_version': policy_version or InferenceDatasetConverter.ADMISSION_POLICY_VERSION,
                    'success': None,
                    'outcome_label': None,
                    'source_file': os.path.basename(input_path),
                    'error': str(exc),
                }
            converted_records.append(record)
        return converted_records


if __name__ == '__main__':
    # 简单测试
    print("Testing InferenceRecorder...")
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = InferenceRecorder(
            output_dir=tmpdir,
            pred_horizon=16,
            action_dim=15,
        )
        
        # 模拟记录
        recorder.start_recording()
        
        for i in range(10):
            obs = {
                'images': [np.random.rand(128, 128, 3).astype(np.float32)],
                'qpos_joint': np.random.rand(7),
                'qpos_end': np.random.rand(8),
                'qpos': np.random.rand(15),
                'gripper': 0.5,
            }
            action_model = np.random.rand(16, 15)
            
            action_executed = action_model.copy()
            recorder.record_step(obs, action_model, action_executed)
        
        recorder.stop_recording()
        
        # 确认保存
        filepath = recorder.confirm_save()
        
        if filepath:
            print(f"Saved to: {filepath}")
            
            # 读取验证
            with h5py.File(filepath, 'r') as f:
                print(f"num_steps: {f.attrs['num_steps']}")
                print(f"action_model shape: {f['action_model'].shape}")
                print(f"action_executed shape: {f['action_executed'].shape}")
    
    print("Test complete")
