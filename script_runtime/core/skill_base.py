"""Base skill abstractions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .failure_codes import FailureCode
from .result_types import RecoveryAction, SkillResult, TraceEvent


@dataclass
class SkillContext:
    """Runtime context injected into skills."""

    blackboard: Any
    adapters: Dict[str, Any] = field(default_factory=dict)
    task_id: str = ""
    trace_sink: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def world_state(self):  # pragma: no cover - trivial
        return self.blackboard.world_state

    def emit_trace(self, event: TraceEvent) -> None:
        if self.trace_sink is not None:
            self.trace_sink.record(event)


class Skill:
    """Base skill contract for script runtime."""

    def __init__(
        self,
        name: str,
        timeout_s: float = 5.0,
        failure_code: FailureCode = FailureCode.UNKNOWN,
        safety_limits: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.timeout_s = float(timeout_s)
        self.default_failure_code = failure_code
        self.safety_limits = dict(safety_limits or {})

    def preconditions(self, world_state: Any, context: SkillContext) -> Tuple[bool, str]:
        return True, ""

    def run(self, context: SkillContext) -> SkillResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def recover(self, context: SkillContext) -> Optional[RecoveryAction]:
        return None

    def trace_payload(self, context: SkillContext, result: SkillResult) -> Dict[str, Any]:
        return {
            "message": result.message,
            "payload": result.payload,
        }

    def execute(self, context: SkillContext) -> SkillResult:
        ok, reason = self.preconditions(context.world_state, context)
        started = time.monotonic()
        if not ok:
            result = SkillResult.failure(
                FailureCode.PRECONDITION_FAILED,
                message=reason or f"Preconditions failed for {self.name}",
                recovery_action=self.recover(context),
            )
            self._emit_trace(context, result, time.monotonic() - started)
            return result

        result = self.run(context)
        if result.status is result.status.FAILURE and result.recovery_action is None:
            result.recovery_action = self.recover(context)
        self._emit_trace(context, result, time.monotonic() - started)
        return result

    def _emit_trace(self, context: SkillContext, result: SkillResult, duration_s: float) -> None:
        execution = context.world_state.execution
        retry_count = execution.retry_counts.get(self.name, 0)
        event = TraceEvent(
            task_id=context.task_id or execution.task_id,
            skill_name=self.name,
            result=result.status.value,
            failure_code=result.failure_code.value,
            duration_s=duration_s,
            retry_count=retry_count,
            active_source=execution.active_source,
            control_owner=execution.control_owner,
            inputs_summary=self.trace_payload(context, result),
            payload=result.payload,
        )
        context.emit_trace(event)
