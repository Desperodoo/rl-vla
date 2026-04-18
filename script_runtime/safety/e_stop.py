"""Emergency stop control primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class EmergencyStopController:
    sdk_bridge: Any

    def trigger(self) -> Dict[str, Any]:
        return self.sdk_bridge.stop()
