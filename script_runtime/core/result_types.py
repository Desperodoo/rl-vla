"""Shared result and tracing types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from .failure_codes import FailureCode


class SkillStatus(str, Enum):
    """Lifecycle states returned by skills and executor nodes."""

    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclass
class RecoveryAction:
    """Recovery instruction emitted by a failed skill."""

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    retryable: bool = True


@dataclass
class SkillResult:
    """Normalized outcome returned by a skill invocation."""

    status: SkillStatus
    failure_code: FailureCode = FailureCode.NONE
    message: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    recovery_action: Optional[RecoveryAction] = None

    @classmethod
    def running(cls, message: str = "", **payload: Any) -> "SkillResult":
        return cls(status=SkillStatus.RUNNING, message=message, payload=dict(payload))

    @classmethod
    def success(cls, message: str = "", **payload: Any) -> "SkillResult":
        return cls(status=SkillStatus.SUCCESS, message=message, payload=dict(payload))

    @classmethod
    def failure(
        cls,
        failure_code: FailureCode,
        message: str = "",
        recovery_action: Optional[RecoveryAction] = None,
        **payload: Any,
    ) -> "SkillResult":
        return cls(
            status=SkillStatus.FAILURE,
            failure_code=failure_code,
            message=message,
            payload=dict(payload),
            recovery_action=recovery_action,
        )


@dataclass
class TraceEvent:
    """Serialized trace record for tasks, skills, and executor transitions."""

    task_id: str
    skill_name: str
    result: str
    failure_code: str = FailureCode.NONE.value
    duration_s: float = 0.0
    retry_count: int = 0
    active_source: str = "policy"
    control_owner: str = "upper_machine"
    inputs_summary: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)
