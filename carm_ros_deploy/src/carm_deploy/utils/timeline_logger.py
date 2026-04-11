#!/usr/bin/env python3
"""
轻量级时间线日志记录器 (v2.0)

以 JSONL 格式记录关键时间点，用于分析时间线与 action chunking 关系。

v2.0 变更：
- 精简 control 事件记录频率（默认每 100 步记录一次）
- 添加 action_norm 字段到 inference 事件
- 移除冗余字段
"""

import json
import os
import threading
import time
from typing import Any, Dict, Optional


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


class TimelineLogger:
    """JSONL 时间线记录器 (v2.0)"""

    def __init__(
        self, 
        log_path: str,
        control_log_interval: int = 100,  # control 事件记录间隔
    ):
        self.log_path = log_path
        self.control_log_interval = control_log_interval
        self._control_step_count = 0
        
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._fp = open(log_path, 'a', buffering=1)
        self._closed = False

    def log(self, event: str, **fields: Dict[str, Any]):
        """
        记录一条事件

        Args:
            event: 事件名称
            **fields: 字段
        """
        payload = {
            'event': event,
            't_sys': time.time(),
        }
        for k, v in fields.items():
            payload[k] = _to_jsonable(v)
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            if self._closed or self._fp.closed:
                return
            self._fp.write(line + '\n')
    
    def log_control(self, **fields: Dict[str, Any]) -> bool:
        """
        记录 control 事件（按间隔记录）
        
        Returns:
            是否实际记录了（用于调用方判断）
        """
        self._control_step_count += 1
        if self._control_step_count % self.control_log_interval == 0:
            self.log('control', **fields)
            return True
        return False
    
    def log_control_force(self, **fields: Dict[str, Any]):
        """强制记录 control 事件（用于异常情况）"""
        self.log('control', force=True, **fields)

    def close(self):
        with self._lock:
            if not self._closed and not self._fp.closed:
                self._fp.close()
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
