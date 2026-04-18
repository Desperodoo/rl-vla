"""ROS bridge exposing status needed by script runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RosStatusSnapshot:
    active_source: str = "policy"
    control_owner: str = "upper_machine"
    timeline_enabled: bool = True
    hitl_mode: str = "disabled"


class RosBridge:
    """Thin adapter for owner/source/timeline state."""

    def __init__(self, snapshot: RosStatusSnapshot | None = None):
        self._snapshot = snapshot or RosStatusSnapshot()

    def get_status(self) -> Dict[str, Any]:
        return {
            "active_source": self._snapshot.active_source,
            "control_owner": self._snapshot.control_owner,
            "timeline_enabled": self._snapshot.timeline_enabled,
            "hitl_mode": self._snapshot.hitl_mode,
        }

    def set_status(self, **fields: Any) -> None:
        for key, value in fields.items():
            setattr(self._snapshot, key, value)
