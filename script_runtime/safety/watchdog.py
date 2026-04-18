"""Execution watchdog."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Watchdog:
    timeout_s: float = 5.0
    starts: Dict[str, float] = field(default_factory=dict)

    def start(self, key: str) -> None:
        self.starts[key] = time.monotonic()

    def elapsed(self, key: str) -> float:
        start = self.starts.get(key)
        if start is None:
            return 0.0
        return time.monotonic() - start

    def expired(self, key: str) -> bool:
        return self.elapsed(key) > self.timeout_s

    def clear(self, key: str) -> None:
        self.starts.pop(key, None)
