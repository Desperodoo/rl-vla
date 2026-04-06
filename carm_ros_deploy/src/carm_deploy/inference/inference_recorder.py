#!/usr/bin/env python3
"""
推理数据采集模块 - 在推理过程中记录数据用于后续训练

数据格式设计原则:
1. action 定义遵循 inference_ros 的模型输出格式（保持训练-推理一致性）
2. 记录模型原始输出 (action_model) 和干预后输出 (action_intervened)
3. 通过 intervention_mask 标记哪些维度被人工干预
4. observations/images 记录 primary camera 兼容字段，images_by_camera 保存多视角
5. observations/timestamps 记录 ROS observation stamp

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
    ├── action_intervened [T, pred_horizon, action_dim]  # 干预后 action
    ├── intervention_mask [T, pred_horizon, action_dim]  # 干预掩码 (bool)
    └── attrs:
        ├── num_steps
        ├── pred_horizon
        ├── action_dim
        ├── has_intervention
        └── ...
"""

import glob
import os
import time
import json
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
    - 干预后的 action
    - 干预掩码
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
        if self.camera_topics and len(self.camera_names) != len(self.camera_topics):
            raise ValueError("normalized camera_names count must match camera_topics count")
        if self.primary_camera is not None and self.primary_camera not in self.camera_index:
            raise ValueError(f"normalized primary_camera '{self.primary_camera}' not found in camera_names")
        self.primary_camera_idx = self.camera_index[self.primary_camera] if self.primary_camera is not None else 0
        
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
            'action_model': [],     # [T, pred_horizon, action_dim] 模型输出
            'action_intervened': [],# [T, pred_horizon, action_dim] 干预后
            'intervention_mask': [],# [T, pred_horizon, action_dim] 干预掩码
        }
        self.step_count = 0
    
    def start_recording(self) -> bool:
        """
        开始记录
        
        Returns:
            是否成功开始
        """
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
        if not self.recording:
            _log_warn("Not recording")
            return False
        
        self.recording = False
        _log_info(f"Recording stopped - {self.step_count} steps collected")
        
        if self.step_count == 0:
            _log_warn("No data recorded, nothing to save")
            return False
        
        # 保存到待确认状态
        self.pending_data = {k: v.copy() if isinstance(v, list) else v 
                           for k, v in self.episode_data.items()}
        self.pending_save = True
        
        _log_info("=" * 50)
        _log_info(f"Episode {self.episode_count}: {self.step_count} steps")
        _log_info("Save this episode? (y/n)")
        _log_info("=" * 50)
        
        return True
    
    def confirm_save(self) -> Optional[str]:
        """
        确认保存 episode
        
        Returns:
            保存的文件路径，如果没有数据返回 None
        """
        if not self.pending_save or self.pending_data is None:
            _log_warn("No pending data to save")
            return None
        
        filepath = self._do_save()
        self.pending_save = False
        self.pending_data = None
        
        return filepath
    
    def discard(self):
        """丢弃当前待保存的 episode"""
        if not self.pending_save:
            return
        
        _log_info("Episode discarded")
        self.pending_save = False
        self.pending_data = None
    
    def record_step(
        self,
        obs: Dict[str, Any],
        action_model: np.ndarray,
        action_intervened: Optional[np.ndarray] = None,
        intervention_mask: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ):
        """
        记录一步数据
        
        Args:
            obs: 观测字典，包含 images, qpos_joint, qpos_end, qpos, gripper
            action_model: 模型原始输出 [pred_horizon, action_dim]
            action_intervened: 干预后的 action [pred_horizon, action_dim]，None 表示无干预
            intervention_mask: 干预掩码 [pred_horizon, action_dim]，None 表示无干预
            timestamp: 时间戳，None 使用当前系统时间
        """
        if not self.recording:
            return
        
        if self.step_count >= self.max_steps:
            _log_warn(f"Reached max steps ({self.max_steps}), stopping recording")
            self.stop_recording()
            return
        
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
        
        if action_intervened is not None:
            self.episode_data['action_intervened'].append(action_intervened.copy())
        else:
            self.episode_data['action_intervened'].append(action_model.copy())
        
        if intervention_mask is not None:
            self.episode_data['intervention_mask'].append(intervention_mask.copy())
        else:
            self.episode_data['intervention_mask'].append(
                np.zeros_like(action_model, dtype=bool)
            )
        
        self.step_count += 1
    
    def _do_save(self) -> str:
        """实际执行保存"""
        if self.pending_data is None:
            return None
        
        data = self.pending_data
        
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
            
            action_intervened = np.array(data['action_intervened'])  # [T, pred_horizon, action_dim]
            f.create_dataset('action_intervened', data=action_intervened)
            
            intervention_mask = np.array(data['intervention_mask'])  # [T, pred_horizon, action_dim]
            f.create_dataset('intervention_mask', data=intervention_mask, dtype='bool')
            
            # 兼容旧格式: action = action_intervened[:, 0, :]
            # 取每步的第一个 action 作为该步的 action
            action = action_intervened[:, 0, :]  # [T, action_dim]
            f.create_dataset('action', data=action)
            
            # 元数据
            f.attrs['num_steps'] = num_steps
            f.attrs['pred_horizon'] = self.pred_horizon
            f.attrs['action_dim'] = self.action_dim
            f.attrs['has_intervention'] = bool(np.any(intervention_mask))
            f.attrs['intervention_ratio'] = float(np.mean(intervention_mask))
            f.attrs['image_height'] = images.shape[1] if len(images.shape) > 1 else 0
            f.attrs['image_width'] = images.shape[2] if len(images.shape) > 2 else 0
            f.attrs['created_at'] = timestamp
            f.attrs['data_source'] = 'inference_with_intervention'
            f.attrs['timestamp_semantics'] = 'obs_stamp_ros'
            f.attrs['action_semantics_version'] = 'absolute_ee_target_pose_v2'
            f.attrs['action_space'] = 'ee_target_pose_absolute'
            f.attrs['compat_action_source'] = 'action_intervened[:,0,:]'
            f.attrs['camera_topics'] = json.dumps(self.camera_topics)
            f.attrs['camera_names'] = json.dumps(self.camera_names)
            f.attrs['primary_camera'] = self.primary_camera or ''
        
        _log_info(f"Episode saved: {num_steps} steps, "
                  f"intervention_ratio: {np.mean(intervention_mask):.2%}")
        
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
    2. 可选过滤/保留干预数据
    3. 对齐观测和 action 的时间戳
    4. 为训练准入保留 metadata / sidecar
    """

    STAGING_SCHEMA_VERSION = 'inference_staging_v1'
    ADMISSION_LABEL = 'train_admission_v1'

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
        kept_steps: int,
        intervention_ratio_raw: float,
        required_ok: bool,
        action: np.ndarray,
        qpos_joint: np.ndarray,
        qpos_end: np.ndarray,
        timestamps: np.ndarray,
        min_steps: int,
        max_intervention_ratio: float,
    ) -> Dict[str, Any]:
        reasons: list[str] = []
        if not required_ok:
            reasons.append('missing_required_data')
        if kept_steps < min_steps:
            reasons.append('too_short')
        if intervention_ratio_raw > max_intervention_ratio:
            reasons.append('high_intervention')
        for name, data in (
            ('action', action),
            ('qpos_joint', qpos_joint),
            ('qpos_end', qpos_end),
            ('timestamps', timestamps),
        ):
            if np.isnan(data).any():
                reasons.append(f'nan_detected:{name}')
                break

        if policy == 'none':
            admission_pass = True if required_ok else False
            admission_reason = 'ok' if admission_pass else '|'.join(reasons or ['missing_required_data'])
        else:
            admission_pass = len(reasons) == 0
            admission_reason = 'ok' if admission_pass else '|'.join(reasons)

        return {
            'admission_label': InferenceDatasetConverter.ADMISSION_LABEL,
            'admission_pass': bool(admission_pass),
            'admission_reason': admission_reason,
            'policy_level': 'episode',
            'admission_policy': policy,
            'min_steps': int(min_steps),
            'max_intervention_ratio': float(max_intervention_ratio),
        }

    @staticmethod
    def convert_to_training_format(
        input_path: str,
        output_path: str,
        use_intervened_action: bool = True,
        filter_intervention: bool = False,
        admission_policy: str = 'none',
        min_steps: int = 32,
        max_intervention_ratio: float = 0.4,
        drop_failed_episode: bool = False,
    ) -> Dict[str, Any]:
        """
        转换为训练格式并返回 episode-level metadata。

        Args:
            input_path: 输入 HDF5 文件路径
            output_path: 输出 HDF5 文件路径
            use_intervened_action: 使用干预后的 action
            filter_intervention: 是否过滤掉有干预的帧
            admission_policy: 准入策略，'none' 或 'episode'
            min_steps: episode-level admission 的最小步数
            max_intervention_ratio: 原始 episode 最大允许 intervention ratio
            drop_failed_episode: 若准入失败，是否不写出 staging HDF5
        """
        output_path = os.path.expandvars(os.path.expanduser(output_path))
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        sidecar_path = os.path.splitext(output_path)[0] + '.meta.json'

        run_info = InferenceDatasetConverter._find_run_info(input_path)
        run_summary = InferenceDatasetConverter._extract_run_info_summary(run_info)

        with h5py.File(input_path, 'r') as f_in:
            input_attrs = InferenceDatasetConverter._normalize_attrs(f_in.attrs)
            num_steps = int(f_in.attrs.get('num_steps', 0))
            obs_group = f_in.get('observations')

            if use_intervened_action:
                action_source = f_in['action_intervened'][:]
                action_source_name = 'action_intervened[:,0,:]'
            else:
                action_source = f_in['action_model'][:]
                action_source_name = 'action_model[:,0,:]'

            action = action_source[:, 0, :]
            intervention_mask = f_in['intervention_mask'][:]
            has_intervention = intervention_mask[:, 0, :].any(axis=1)
            intervention_ratio_raw = float(has_intervention.mean()) if len(has_intervention) > 0 else 0.0

            if filter_intervention:
                keep_idx = ~has_intervention
                _log_info(f"Filtering intervention frames: {has_intervention.sum()}/{num_steps}")
            else:
                keep_idx = np.ones(num_steps, dtype=bool)

            kept_steps = int(keep_idx.sum())
            dropped_steps = int(num_steps - kept_steps)
            action_kept = action[keep_idx]

            required_obs_keys = ['images', 'qpos_joint', 'qpos_end', 'gripper', 'timestamps']
            required_ok = obs_group is not None and all(key in obs_group for key in required_obs_keys) and 'intervention_mask' in f_in

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
                kept_steps=kept_steps,
                intervention_ratio_raw=intervention_ratio_raw,
                required_ok=required_ok,
                action=action_kept,
                qpos_joint=qpos_joint_kept,
                qpos_end=qpos_end_kept,
                timestamps=timestamps_kept,
                min_steps=min_steps,
                max_intervention_ratio=max_intervention_ratio,
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
                    f_out.attrs['filtered_intervention'] = filter_intervention
                    f_out.attrs['source_file'] = os.path.basename(input_path)
                    f_out.attrs['dataset_type'] = 'inference_staging'
                    f_out.attrs['staging_schema_version'] = InferenceDatasetConverter.STAGING_SCHEMA_VERSION
                    f_out.attrs['kept_steps'] = kept_steps
                    f_out.attrs['dropped_steps'] = dropped_steps
                    f_out.attrs['intervention_ratio_raw'] = intervention_ratio_raw
                    f_out.attrs['intervention_ratio_kept'] = float(np.mean(has_intervention[keep_idx])) if kept_steps > 0 else 0.0
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
                        'has_intervention',
                        'intervention_ratio',
                    ]:
                        if key in f_in.attrs:
                            f_out.attrs[key] = f_in.attrs[key]

                    f_out.attrs['admission_label'] = admission['admission_label']
                    f_out.attrs['admission_pass'] = admission['admission_pass']
                    f_out.attrs['admission_reason'] = admission['admission_reason']
                    f_out.attrs['policy_level'] = admission['policy_level']
                    f_out.attrs['admission_policy'] = admission['admission_policy']
                    f_out.attrs['min_steps'] = admission['min_steps']
                    f_out.attrs['max_intervention_ratio'] = admission['max_intervention_ratio']

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
                'use_intervened_action': bool(use_intervened_action),
                'filter_intervention': bool(filter_intervention),
                'admission_policy': admission_policy,
                'min_steps': int(min_steps),
                'max_intervention_ratio': float(max_intervention_ratio),
            },
            'stats': {
                'num_steps_raw': num_steps,
                'kept_steps': kept_steps,
                'dropped_steps': dropped_steps,
                'intervention_ratio_raw': intervention_ratio_raw,
                'intervention_ratio_kept': float(np.mean(has_intervention[keep_idx])) if kept_steps > 0 else 0.0,
                'use_intervened_action': bool(use_intervened_action),
                'filter_intervention': bool(filter_intervention),
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
            'source_file': os.path.basename(input_path),
        }

    @staticmethod
    def convert_directory_to_training_format(
        input_dir: str,
        output_dir: str,
        use_intervened_action: bool = True,
        filter_intervention: bool = False,
        admission_policy: str = 'none',
        min_steps: int = 32,
        max_intervention_ratio: float = 0.4,
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
            record = InferenceDatasetConverter.convert_to_training_format(
                input_path=input_path,
                output_path=output_path,
                use_intervened_action=use_intervened_action,
                filter_intervention=filter_intervention,
                admission_policy=admission_policy,
                min_steps=min_steps,
                max_intervention_ratio=max_intervention_ratio,
                drop_failed_episode=drop_failed_episode,
            )
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
            
            # 模拟干预
            if i % 3 == 0:
                action_intervened = action_model.copy()
                action_intervened[:, 7:10] += 0.01  # xyz 干预
                mask = np.zeros_like(action_model, dtype=bool)
                mask[:, 7:10] = True
            else:
                action_intervened = None
                mask = None
            
            recorder.record_step(obs, action_model, action_intervened, mask)
        
        recorder.stop_recording()
        
        # 确认保存
        filepath = recorder.confirm_save()
        
        if filepath:
            print(f"Saved to: {filepath}")
            
            # 读取验证
            with h5py.File(filepath, 'r') as f:
                print(f"num_steps: {f.attrs['num_steps']}")
                print(f"has_intervention: {f.attrs['has_intervention']}")
                print(f"intervention_ratio: {f.attrs['intervention_ratio']:.2%}")
                print(f"action_model shape: {f['action_model'].shape}")
                print(f"action_intervened shape: {f['action_intervened'].shape}")
    
    print("Test complete")
