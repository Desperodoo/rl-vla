#!/usr/bin/env python3
"""
轻量级时间线日志记录器

以 JSONL 格式记录关键时间点，用于分析时间线与 action chunking 关系。
"""

import json
import os
import threading
import time
from typing import Any, Dict


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


class TimelineLogger:
    """JSONL 时间线记录器"""

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._lock = threading.Lock()
        self._fp = open(log_path, 'a', buffering=1)

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
            self._fp.write(line + '\n')

    def close(self):
        with self._lock:
            if not self._fp.closed:
                self._fp.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
