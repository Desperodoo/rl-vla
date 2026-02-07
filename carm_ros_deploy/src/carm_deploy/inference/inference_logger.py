#!/usr/bin/env python3
"""
CARM 推理日志记录器

记录推理过程中的所有数据用于后续分析。

功能:
    - 记录观测 (图像、关节状态、末端位姿)
    - 记录预测动作和执行动作
    - 记录时间信息
    - 记录安全事件
    - 保存为 HDF5 格式

使用方法:
    from inference_logger import InferenceLogger
    logger = InferenceLogger(log_dir='inference_logs')
    logger.start_episode()
    logger.log_step(timestamp, obs, pred_action, exec_action, ...)
    logger.end_episode()
"""

import os
import time
import json
import numpy as np
import h5py
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# 日志格式版本
LOG_FORMAT_VERSION = "2.0"


@dataclass
class StepData:
    """单步数据"""
    timestamp: float = 0.0
    
    # 观测
    image: Optional[np.ndarray] = None  # [H, W, C]
    qpos_joint: Optional[np.ndarray] = None  # [7]
    qpos_end: Optional[np.ndarray] = None  # [8]
    
    # 动作
    raw_action: Optional[np.ndarray] = None  # 模型原始输出（反归一化后）
    executed_action: Optional[np.ndarray] = None  # 实际发送给机械臂的动作
    
    # 时间
    inference_time: float = 0.0
    
    # 安全
    safety_clipped: bool = False
    safety_warnings: List[str] = field(default_factory=list)


class InferenceLogger:
    """
    推理日志记录器 (v2.0)
    
    记录推理过程中的所有数据，支持实时保存和后续分析。
    生成三类文件：
        - run_info_*.json: 运行配置和元数据
        - inference_*.hdf5: 详细数值数据
        - timeline_*.jsonl: 时间线事件（由 TimelineLogger 生成）
    """
    
    def __init__(
        self,
        log_dir: str = 'inference_logs',
        save_images: bool = False,  # 图像占用空间大，默认不保存
        max_image_size: tuple = (128, 128),  # 如果保存图像，下采样到此尺寸
        buffer_size: int = 100,  # 缓冲区大小，达到后自动保存
    ):
        self.log_dir = log_dir
        self.save_images = save_images
        self.max_image_size = max_image_size
        self.buffer_size = buffer_size
        
        os.makedirs(log_dir, exist_ok=True)
        
        # 当前 episode 数据
        self.current_episode: List[StepData] = []
        self.episode_count = 0
        self.step_count = 0
        
        # 运行配置（用于生成 run_info.json）
        self.run_info = {
            'version': LOG_FORMAT_VERSION,
            'created_at': None,
            'model': {},
            'normalizer': {},
            'control': {},
            'execution': {},
            'safety': {},
            'files': {},
        }
        
        # 元数据（兼容旧版本）
        self.metadata = {
            'start_time': None,
            'end_time': None,
            'model_path': None,
            'config': {},
        }
        
        # 当前 episode 文件
        self.current_file: Optional[h5py.File] = None
        self.current_file_path: Optional[str] = None
        self._timestamp_suffix: Optional[str] = None
    
    def set_metadata(
        self,
        model_path: str = None,
        config: Dict = None,
        # 新增：详细配置
        model_config: Dict = None,
        normalizer_config: Dict = None,
        control_config: Dict = None,
        execution_config: Dict = None,
        safety_config: Dict = None,
    ):
        """
        设置元数据和运行配置
        
        Args:
            model_path: 模型路径（兼容旧版本）
            config: 配置字典（兼容旧版本）
            model_config: 模型配置 (algorithm, action_mode, state_mode, etc.)
            normalizer_config: 归一化器配置 (enabled, mode, action_stats)
            control_config: 控制配置 (control_freq, teleop_scale, etc.)
            execution_config: 执行配置 (mode, act_horizon, etc.)
            safety_config: 安全配置 (config_path, check_workspace, etc.)
        """
        if model_path:
            self.metadata['model_path'] = model_path
            self.run_info['model']['path'] = model_path
        if config:
            self.metadata['config'] = config
        
        # 新增配置
        if model_config:
            self.run_info['model'].update(model_config)
        if normalizer_config:
            self.run_info['normalizer'].update(normalizer_config)
        if control_config:
            self.run_info['control'].update(control_config)
        if execution_config:
            self.run_info['execution'].update(execution_config)
        if safety_config:
            self.run_info['safety'].update(safety_config)
    
    def start_episode(self, episode_name: str = None, timeline_path: str = None):
        """
        开始新的 episode
        
        Args:
            episode_name: episode 名称（可选）
            timeline_path: timeline 文件路径（用于记录到 run_info）
        """
        self.current_episode = []
        self.step_count = 0
        self.metadata['start_time'] = datetime.now().isoformat()
        self.run_info['created_at'] = datetime.now().isoformat()
        
        # 创建文件
        if episode_name is None:
            self._timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
            episode_name = f'inference_{self._timestamp_suffix}'
        else:
            self._timestamp_suffix = episode_name.replace('inference_', '')
        
        self.current_file_path = os.path.join(self.log_dir, f'{episode_name}.hdf5')
        # 关闭可能残留的旧文件句柄，避免资源泄漏
        if self.current_file is not None:
            try:
                self.current_file.close()
            except Exception:
                pass
            self.current_file = None
        self.current_file = h5py.File(self.current_file_path, 'w')
        
        # 创建数据组
        self.current_file.create_group('observations')
        self.current_file.create_group('predictions')
        self.current_file.create_group('timing')
        self.current_file.create_group('safety')
        
        # 记录文件名到 run_info
        self.run_info['files']['hdf5'] = os.path.basename(self.current_file_path)
        if timeline_path:
            self.run_info['files']['timeline'] = os.path.basename(timeline_path)
        
        print(f"Started logging episode: {self.current_file_path}")
    
    def log_step(
        self,
        timestamp: float,
        obs: Optional[Dict] = None,
        raw_action: Optional[np.ndarray] = None,
        executed_action: Optional[np.ndarray] = None,
        inference_time: float = 0.0,
        safety_clipped: bool = False,
        safety_warnings: List[str] = None,
    ):
        """
        记录单步数据
        
        Args:
            timestamp: 时间戳
            obs: 观测字典 (包含 images, qpos_joint, qpos_end)
            raw_action: 模型原始输出（反归一化后，相对位姿）
            executed_action: 实际发送给机械臂的动作（绝对位姿）
            inference_time: 推理时间
            safety_clipped: 是否被安全裁剪
            safety_warnings: 安全警告列表
        """
        step = StepData(
            timestamp=timestamp,
            inference_time=inference_time,
            safety_clipped=safety_clipped,
            safety_warnings=safety_warnings or [],
        )
        
        if obs is not None:
            if 'images' in obs and obs['images']:
                # images 可能是 dict (按相机名) 或 list/array
                images = obs['images']
                if isinstance(images, dict):
                    # 取第一个相机的图像
                    first_key = next(iter(images.keys()))
                    step.image = images[first_key]
                elif len(images) > 0:
                    step.image = images[0]  # 取第一个相机
            if 'qpos_joint' in obs:
                step.qpos_joint = np.array(obs['qpos_joint'])
            if 'qpos_end' in obs:
                step.qpos_end = np.array(obs['qpos_end'])
        
        if raw_action is not None:
            step.raw_action = np.array(raw_action)
        if executed_action is not None:
            step.executed_action = np.array(executed_action)
        
        self.current_episode.append(step)
        self.step_count += 1
        
        # 缓冲区满时保存
        if len(self.current_episode) >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """将缓冲区数据写入文件"""
        if not self.current_file or len(self.current_episode) == 0:
            return
        
        start_idx = self.step_count - len(self.current_episode)
        
        for i, step in enumerate(self.current_episode):
            idx = start_idx + i
            prefix = f'step_{idx:06d}'
            
            # 观测
            obs_grp = self.current_file['observations']
            if step.qpos_joint is not None:
                obs_grp.create_dataset(f'{prefix}/qpos_joint', data=step.qpos_joint)
            if step.qpos_end is not None:
                obs_grp.create_dataset(f'{prefix}/qpos_end', data=step.qpos_end)
            if self.save_images and step.image is not None:
                import cv2
                image = cv2.resize(step.image, self.max_image_size)
                obs_grp.create_dataset(f'{prefix}/image', data=image, compression='gzip')
            
            # 预测
            pred_grp = self.current_file['predictions']
            if step.raw_action is not None:
                pred_grp.create_dataset(f'{prefix}/raw_action', data=step.raw_action)
            if step.executed_action is not None:
                pred_grp.create_dataset(f'{prefix}/executed_action', data=step.executed_action)
            
            # 时间
            time_grp = self.current_file['timing']
            time_grp.create_dataset(f'{prefix}/timestamp', data=step.timestamp)
            time_grp.create_dataset(f'{prefix}/inference_time', data=step.inference_time)
            
            # 安全
            safety_grp = self.current_file['safety']
            safety_grp.create_dataset(f'{prefix}/clipped', data=step.safety_clipped)
            if step.safety_warnings:
                safety_grp.create_dataset(f'{prefix}/warnings', 
                                          data=json.dumps(step.safety_warnings))
        
        self.current_episode = []
    
    def end_episode(self) -> str:
        """
        结束当前 episode
        
        Returns:
            保存的文件路径
        """
        self._flush_buffer()
        
        if self.current_file:
            # 保存元数据
            self.metadata['end_time'] = datetime.now().isoformat()
            self.metadata['num_steps'] = self.step_count
            
            for key, value in self.metadata.items():
                if value is not None:
                    if isinstance(value, dict):
                        self.current_file.attrs[key] = json.dumps(value)
                    else:
                        self.current_file.attrs[key] = str(value)
            
            self.current_file.close()
            self.current_file = None
            
            print(f"Episode saved: {self.current_file_path} ({self.step_count} steps)")
        
        # 保存 run_info.json
        self._save_run_info()
        
        self.episode_count += 1
        return self.current_file_path
    
    def _save_run_info(self):
        """保存 run_info.json 文件"""
        if self._timestamp_suffix is None:
            return
        
        run_info_path = os.path.join(
            self.log_dir, 
            f'run_info_{self._timestamp_suffix}.json'
        )
        
        # 添加结束信息
        self.run_info['ended_at'] = datetime.now().isoformat()
        self.run_info['total_steps'] = self.step_count
        
        # 转换 numpy 数组为列表
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        run_info_serializable = convert_numpy(self.run_info)
        
        with open(run_info_path, 'w') as f:
            json.dump(run_info_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"Run info saved: {run_info_path}")
    
    def get_summary(self) -> Dict:
        """获取当前 episode 的摘要"""
        if len(self.current_episode) == 0:
            return {}
        
        inference_times = [s.inference_time for s in self.current_episode if s.inference_time > 0]
        safety_clips = sum(1 for s in self.current_episode if s.safety_clipped)
        
        return {
            'num_steps': self.step_count,
            'avg_inference_time': np.mean(inference_times) if inference_times else 0,
            'max_inference_time': np.max(inference_times) if inference_times else 0,
            'safety_clips': safety_clips,
            'safety_clip_rate': safety_clips / self.step_count if self.step_count > 0 else 0,
        }


class InferenceLogAnalyzer:
    """
    推理日志分析器
    
    分析保存的推理日志，生成报告和可视化
    """
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.data = self._load_log()
    
    def _load_log(self) -> Dict:
        """加载日志文件"""
        data = {
            'timestamps': [],
            'qpos_joint': [],
            'qpos_end': [],
            'raw_actions': [],
            'processed_actions': [],
            'executed_actions': [],
            'inference_times': [],
            'control_times': [],
            'safety_clipped': [],
        }
        
        with h5py.File(self.log_path, 'r') as f:
            # 获取元数据
            self.metadata = {k: f.attrs[k] for k in f.attrs.keys()}
            
            # 遍历步骤
            timing_grp = f['timing']
            obs_grp = f['observations']
            pred_grp = f['predictions']
            safety_grp = f['safety']
            
            step_keys = sorted([k for k in timing_grp.keys() if k.startswith('step_')])
            
            for key in step_keys:
                step_name = key.split('/')[0] if '/' in key else key
                
                # 时间
                if f'{step_name}/timestamp' in timing_grp:
                    data['timestamps'].append(timing_grp[f'{step_name}/timestamp'][()])
                if f'{step_name}/inference_time' in timing_grp:
                    data['inference_times'].append(timing_grp[f'{step_name}/inference_time'][()])
                if f'{step_name}/control_time' in timing_grp:
                    data['control_times'].append(timing_grp[f'{step_name}/control_time'][()])
                
                # 观测
                if f'{step_name}/qpos_joint' in obs_grp:
                    data['qpos_joint'].append(obs_grp[f'{step_name}/qpos_joint'][:])
                if f'{step_name}/qpos_end' in obs_grp:
                    data['qpos_end'].append(obs_grp[f'{step_name}/qpos_end'][:])
                
                # 预测
                if f'{step_name}/raw_action' in pred_grp:
                    data['raw_actions'].append(pred_grp[f'{step_name}/raw_action'][:])
                if f'{step_name}/executed_action' in pred_grp:
                    data['executed_actions'].append(pred_grp[f'{step_name}/executed_action'][:])
                
                # 安全
                if f'{step_name}/clipped' in safety_grp:
                    data['safety_clipped'].append(safety_grp[f'{step_name}/clipped'][()])
        
        # 转换为 numpy 数组
        for key in data:
            if len(data[key]) > 0:
                data[key] = np.array(data[key])
        
        return data
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'num_steps': len(self.data['timestamps']),
        }
        
        if len(self.data['inference_times']) > 0:
            stats['inference_time'] = {
                'mean': np.mean(self.data['inference_times']),
                'std': np.std(self.data['inference_times']),
                'min': np.min(self.data['inference_times']),
                'max': np.max(self.data['inference_times']),
            }
        
        if len(self.data['safety_clipped']) > 0:
            stats['safety'] = {
                'total_clips': np.sum(self.data['safety_clipped']),
                'clip_rate': np.mean(self.data['safety_clipped']),
            }
        
        return stats
    
    def plot_joint_trajectory(self, save_path: str = None):
        """绘制关节轨迹"""
        import matplotlib.pyplot as plt
        
        if len(self.data['qpos_joint']) == 0:
            print("No joint data to plot")
            return
        
        qpos = self.data['qpos_joint']
        num_steps = len(qpos)
        time_steps = np.arange(num_steps)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Joint Trajectory During Inference', fontsize=14)
        
        # 关节 1-6
        for i in range(6):
            ax = axes[i // 4, i % 4]
            ax.plot(time_steps, qpos[:, i], 'b-', alpha=0.7)
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'Joint {i+1} (rad)')
            ax.set_title(f'Joint {i+1}')
            ax.grid(True, alpha=0.3)
        
        # 夹爪
        ax = axes[1, 2]
        ax.plot(time_steps, qpos[:, 6], 'g-', alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Gripper (m)')
        ax.set_title('Gripper')
        ax.grid(True, alpha=0.3)
        
        # 推理时间
        if len(self.data['inference_times']) > 0:
            ax = axes[1, 3]
            ax.plot(time_steps, self.data['inference_times'] * 1000, 'r-', alpha=0.7)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Inference Time (ms)')
            ax.set_title('Inference Time')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_action_comparison(self, save_path: str = None):
        """绘制动作对比 (预测 vs 执行)"""
        import matplotlib.pyplot as plt
        
        if len(self.data['raw_actions']) == 0 or len(self.data['executed_actions']) == 0:
            print("No action data to plot")
            return
        
        raw = self.data['raw_actions']
        executed = self.data['executed_actions']
        num_steps = len(raw)
        time_steps = np.arange(num_steps)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Raw vs Executed Actions', fontsize=14)
        
        for i in range(min(7, raw.shape[1])):
            ax = axes[i // 4, i % 4]
            ax.plot(time_steps, raw[:, i], 'b-', label='Raw', alpha=0.5)
            ax.plot(time_steps, executed[:, i], 'r-', label='Executed', alpha=0.5)
            
            if i < 6:
                ax.set_ylabel(f'Joint {i+1}')
            else:
                ax.set_ylabel('Gripper')
            
            ax.set_xlabel('Time Step')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 安全裁剪标记
        if len(self.data['safety_clipped']) > 0:
            ax = axes[1, 3]
            ax.fill_between(time_steps, 0, self.data['safety_clipped'], 
                           alpha=0.3, color='red', label='Safety Clip')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Clipped')
            ax.set_title('Safety Clips')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, output_dir: str):
        """生成完整分析报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 统计信息
        stats = self.get_statistics()
        stats_path = os.path.join(output_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {stats_path}")
        
        # 可视化
        self.plot_joint_trajectory(os.path.join(output_dir, 'joint_trajectory.png'))
        self.plot_action_comparison(os.path.join(output_dir, 'action_comparison.png'))
        
        print(f"\nReport generated in: {output_dir}")


# 测试代码
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['log', 'analyze'], default='log')
    parser.add_argument('--log_dir', type=str, default='inference_logs')
    parser.add_argument('--log_path', type=str, help='Path to log file for analysis')
    args = parser.parse_args()
    
    if args.mode == 'log':
        # 测试日志记录
        logger = InferenceLogger(log_dir=args.log_dir)
        logger.set_metadata(model_path='test_model.pt', config={'test': True})
        
        logger.start_episode()
        
        for i in range(10):
            obs = {
                'qpos_joint': np.random.randn(7),
                'qpos_end': np.random.randn(8),
            }
            logger.log_step(
                timestamp=time.time(),
                obs=obs,
                raw_action=np.random.randn(15),
                executed_action=np.random.randn(8),
                inference_time=0.05 + np.random.rand() * 0.01,
                safety_clipped=(i % 3 == 0),
            )
        
        logger.end_episode()
        print("Test logging complete!")
    
    else:
        # 测试分析
        if not args.log_path:
            print("Please provide --log_path for analysis")
        else:
            analyzer = InferenceLogAnalyzer(args.log_path)
            stats = analyzer.get_statistics()
            print("Statistics:", json.dumps(stats, indent=2))
            analyzer.generate_report('analysis_output')
