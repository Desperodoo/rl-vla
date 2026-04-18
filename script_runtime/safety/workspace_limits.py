"""Workspace safety gate bridging to existing safety controller."""

from __future__ import annotations

from typing import Any, Dict, List

from script_runtime.core.failure_codes import FailureCode


class WorkspaceGate:
    """Thin wrapper around the existing SafetyController."""

    def __init__(self, safety_controller: Any | None = None):
        self.safety_controller = safety_controller

    def validate_pose(self, pose: List[float]) -> tuple[bool, FailureCode, Dict[str, Any]]:
        if self.safety_controller is None:
            return True, FailureCode.NONE, {}
        warnings = []
        if hasattr(self.safety_controller, "check_workspace_limits"):
            ok, warnings = self.safety_controller.check_workspace_limits(pose)
            if not ok:
                return False, FailureCode.WORKSPACE_VIOLATION, {"warnings": warnings}
        return True, FailureCode.NONE, {"warnings": warnings}
