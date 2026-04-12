#!/usr/bin/env python3
"""
CARM 推理运行信息记录器。

职责收敛为两件事：
- 维护 run_info.json / 运行摘要
- 记录 canonical 训练回流产物与 timeline 的文件名映射

训练数据由 InferenceRecorder 负责，时间线由 TimelineLogger 负责。
"""

import json
import os
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

LOG_FORMAT_VERSION = "2.1"


def _log_info(msg: str):
    print(f"[INFO] {msg}")


class InferenceLogger:
    """推理运行信息记录器。"""

    def __init__(
        self,
        log_dir: str = 'inference_logs',
        save_images: bool = False,
        max_image_size: tuple = (128, 128),
        buffer_size: int = 100,
    ):
        self.log_dir = log_dir
        self.save_images = save_images
        self.max_image_size = max_image_size
        self.buffer_size = buffer_size

        os.makedirs(self.log_dir, exist_ok=True)

        self.episode_count = 0
        self.step_count = 0
        self._episode_started = False
        self._timestamp_suffix: Optional[str] = None
        self._run_info_path: Optional[str] = None
        self._inference_times: List[float] = []
        self._safety_clips = 0
        self._safety_reason_counts: Counter[str] = Counter()

        self.run_info = {
            'version': LOG_FORMAT_VERSION,
            'created_at': None,
            'model': {},
            'normalizer': {},
            'control': {},
            'execution': {},
            'hitl': {},
            'safety': {},
            'files': {
                'episode_hdf5': [],
                'timeline': None,
                'run_info': None,
            },
            'summary': {},
        }

        self.metadata = {
            'start_time': None,
            'end_time': None,
            'model_path': None,
            'config': {},
        }

        _log_info(f"InferenceLogger initialized, log_dir: {os.path.abspath(self.log_dir)}")

    def set_metadata(
        self,
        model_path: str = None,
        config: Dict = None,
        model_config: Dict = None,
        normalizer_config: Dict = None,
        control_config: Dict = None,
        execution_config: Dict = None,
        hitl_config: Dict = None,
        safety_config: Dict = None,
    ):
        """设置运行信息与配置快照。"""
        if model_path:
            self.metadata['model_path'] = model_path
            self.run_info['model']['path'] = model_path
        if config:
            self.metadata['config'] = config

        if model_config:
            self.run_info['model'].update(model_config)
        if normalizer_config:
            self.run_info['normalizer'].update(normalizer_config)
        if control_config:
            self.run_info['control'].update(control_config)
        if execution_config:
            self.run_info['execution'].update(execution_config)
        if hitl_config:
            self.run_info['hitl'].update(hitl_config)
        if safety_config:
            self.run_info['safety'].update(safety_config)

    def start_episode(self, episode_name: str = None, timeline_path: str = None):
        """开始新的运行记录会话。"""
        if self._episode_started:
            return

        self._episode_started = True
        self.episode_count += 1
        self.step_count = 0
        self._inference_times = []
        self._safety_clips = 0
        self._safety_reason_counts = Counter()

        self.metadata['start_time'] = datetime.now().isoformat()
        self.run_info['created_at'] = self.metadata['start_time']
        self.run_info['files'] = {
            'episode_hdf5': [],
            'timeline': None,
            'run_info': None,
        }
        self.run_info['summary'] = {}

        if episode_name is None:
            self._timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self._timestamp_suffix = episode_name.replace('inference_', '')

        if timeline_path:
            self.run_info['files']['timeline'] = os.path.basename(timeline_path)

        _log_info(f"Run-info session started: run_info_{self._timestamp_suffix}.json")

    def record_episode_file(self, episode_path: str, success: Optional[bool] = None, outcome_label: Optional[str] = None):
        """记录 canonical recorder episode 文件名与 outcome。"""
        basename = os.path.basename(episode_path)
        episode_files = self.run_info['files'].setdefault('episode_hdf5', [])
        if basename not in episode_files:
            episode_files.append(basename)
        if success is not None:
            self.run_info.setdefault('summary', {})['episode_success'] = bool(success)
        if outcome_label is not None:
            self.run_info.setdefault('summary', {})['episode_outcome_label'] = str(outcome_label)

    def log_step(
        self,
        timestamp: float,
        obs: Optional[Dict] = None,
        raw_action: Optional[np.ndarray] = None,
        executed_action: Optional[np.ndarray] = None,
        inference_time: float = 0.0,
        safety_clipped: bool = False,
        safety_warnings: List[str] = None,
        safety_reason_counts: Optional[Dict[str, int]] = None,
    ):
        """记录单步摘要，HDF5 数据由 InferenceRecorder 单独负责。"""
        self.step_count += 1
        if inference_time > 0:
            self._inference_times.append(float(inference_time))
        if safety_clipped:
            self._safety_clips += 1
        if safety_reason_counts:
            self._safety_reason_counts.update(safety_reason_counts)

    def end_episode(self) -> str:
        """结束会话并保存 run_info.json。"""
        if self._timestamp_suffix is None:
            self._timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')

        run_info_path = os.path.join(self.log_dir, f'run_info_{self._timestamp_suffix}.json')
        self._run_info_path = run_info_path

        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['num_steps'] = self.step_count

        self.run_info['ended_at'] = self.metadata['end_time']
        self.run_info['total_steps'] = self.step_count
        self.run_info['summary'] = self.get_summary()
        self.run_info['files']['run_info'] = os.path.basename(run_info_path)

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        with open(run_info_path, 'w') as f:
            json.dump(convert_numpy(self.run_info), f, indent=2, ensure_ascii=False)

        _log_info(f"Run info saved: {run_info_path}")

        self._episode_started = False
        return run_info_path

    def get_summary(self) -> Dict:
        """获取当前运行摘要。"""
        if self.step_count == 0:
            return {}

        return {
            'num_steps': self.step_count,
            'avg_inference_time': np.mean(self._inference_times) if self._inference_times else 0,
            'max_inference_time': np.max(self._inference_times) if self._inference_times else 0,
            'safety_clips': self._safety_clips,
            'safety_clip_rate': self._safety_clips / self.step_count if self.step_count > 0 else 0,
            'safety_reason_counts': dict(self._safety_reason_counts),
            'episode_success': self.run_info.get('summary', {}).get('episode_success'),
            'episode_outcome_label': self.run_info.get('summary', {}).get('episode_outcome_label'),
        }


if __name__ == '__main__':
    logger = InferenceLogger(log_dir='inference_logs')
    logger.set_metadata(model_path='test_model.pt', config={'test': True})
    logger.start_episode()
    for i in range(10):
        logger.log_step(timestamp=float(i), inference_time=0.05 + 0.01 * i, safety_clipped=(i % 3 == 0))
    logger.end_episode()
    print("Run-info logging test complete!")
