#!/usr/bin/env python3
"""
推理数据采集模块 - 在推理过程中记录数据用于后续训练

数据格式设计原则:
1. action 定义遵循 inference_ros 的模型输出格式（保持训练-推理一致性）
2. 记录模型原始输出 (action_model) 和干预后输出 (action_intervened)
3. 通过 intervention_mask 标记哪些维度被人工干预

HDF5 文件结构:
    episode_{XXXX}_{timestamp}.hdf5
    ├── observations/
    │   ├── images        [T, H, W, C]      # 图像
    │   ├── qpos_joint    [T, 7]            # 关节角度 (6 joints + 1 gripper)
    │   ├── qpos_end      [T, 8]            # 末端位姿 (x,y,z,qx,qy,qz,qw,gripper)
    │   ├── qpos          [T, 15]           # 兼容旧版: [joints(7), end_pose(8)]
    │   ├── gripper       [T]               # 夹爪状态
    │   └── timestamps    [T]               # 系统时间戳
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

import os
import time
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
            'qpos_joint': [],       # [T, 7]
            'qpos_end': [],         # [T, 8]
            'qpos': [],             # [T, 15] 兼容旧版
            'gripper': [],          # [T]
            'timestamps': [],       # [T] 系统时间
            
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
        
        # 观测
        self.episode_data['images'].append(obs['images'][0])  # 第一个相机
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
    """
    
    @staticmethod
    def convert_to_training_format(
        input_path: str,
        output_path: str,
        use_intervened_action: bool = True,
        filter_intervention: bool = False,
    ):
        """
        转换为训练格式
        
        Args:
            input_path: 输入 HDF5 文件路径
            output_path: 输出 HDF5 文件路径
            use_intervened_action: 使用干预后的 action
            filter_intervention: 是否过滤掉有干预的帧
        """
        with h5py.File(input_path, 'r') as f_in:
            num_steps = f_in.attrs['num_steps']
            
            # 选择 action 源
            if use_intervened_action:
                action_source = f_in['action_intervened'][:]
            else:
                action_source = f_in['action_model'][:]
            
            # 取每步的第一个 action
            action = action_source[:, 0, :]  # [T, action_dim]
            
            # 干预掩码
            intervention_mask = f_in['intervention_mask'][:]
            has_intervention = intervention_mask[:, 0, :].any(axis=1)  # [T]
            
            # 过滤干预帧
            if filter_intervention:
                keep_idx = ~has_intervention
                _log_info(f"Filtering intervention frames: {has_intervention.sum()}/{num_steps}")
            else:
                keep_idx = np.ones(num_steps, dtype=bool)
            
            with h5py.File(output_path, 'w') as f_out:
                obs_grp = f_out.create_group('observations')
                
                # 复制观测数据
                for key in f_in['observations'].keys():
                    data = f_in['observations'][key][:][keep_idx]
                    if key == 'images':
                        obs_grp.create_dataset(key, data=data, compression='gzip')
                    else:
                        obs_grp.create_dataset(key, data=data)
                
                # Action
                f_out.create_dataset('action', data=action[keep_idx])
                
                # 元数据
                f_out.attrs['num_steps'] = int(keep_idx.sum())
                f_out.attrs['filtered_intervention'] = filter_intervention
                f_out.attrs['source_file'] = os.path.basename(input_path)


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
