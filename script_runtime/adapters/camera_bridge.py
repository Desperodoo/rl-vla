"""Camera/perception bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CameraSnapshot:
    rgb: Optional[Any] = None
    depth: Optional[Any] = None
    calibration_version: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class CameraBridge:
    """Minimal camera bridge for world-state refresh."""

    def __init__(self, snapshot: Optional[CameraSnapshot] = None):
        self._snapshot = snapshot or CameraSnapshot()

    def get_snapshot(self) -> CameraSnapshot:
        return self._snapshot

    def set_snapshot(self, snapshot: CameraSnapshot) -> None:
        self._snapshot = snapshot
