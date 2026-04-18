"""Behavior-tree inspired executor with optional py_trees integration."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import py_trees  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    py_trees = None

from script_runtime.core.failure_codes import FailureCode
from script_runtime.core.result_types import RecoveryAction, SkillResult, SkillStatus, TraceEvent
from script_runtime.core.skill_base import SkillContext


@dataclass
class TraceRecorder:
    """In-memory trace collector that can be persisted to JSONL."""

    events: List[TraceEvent] = field(default_factory=list)

    def record(self, event: TraceEvent) -> None:
        self.events.append(event)

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(_jsonify(asdict(event)), ensure_ascii=False) for event in self.events)


def _jsonify(value: Any) -> Any:
    """Best-effort conversion for numpy / torch / dataclass-heavy payloads."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _jsonify(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _jsonify(vars(value))
        except Exception:
            pass
    return repr(value)


class Node:
    """Minimal executor node interface."""

    def __init__(self, name: str):
        self.name = name

    def tick(self, context: SkillContext) -> SkillResult:  # pragma: no cover - abstract
        raise NotImplementedError


class ConditionNode(Node):
    def __init__(self, name: str, predicate: Callable[[SkillContext], tuple[bool, str] | bool]):
        super().__init__(name)
        self.predicate = predicate

    def tick(self, context: SkillContext) -> SkillResult:
        out = self.predicate(context)
        if isinstance(out, tuple):
            ok, reason = out
        else:
            ok, reason = bool(out), ""
        if ok:
            return SkillResult.success(message=reason or f"{self.name} passed")
        return SkillResult.failure(FailureCode.PRECONDITION_FAILED, message=reason or f"{self.name} failed")


class SkillNode(Node):
    def __init__(self, name: str, skill_name: str):
        super().__init__(name)
        self.skill_name = skill_name

    def tick(self, context: SkillContext) -> SkillResult:
        skill = context.metadata["registry"].get(self.skill_name)
        context.world_state.execution.current_skill = skill.name
        return skill.execute(context)


class SequenceNode(Node):
    def __init__(self, name: str, children: List[Node]):
        super().__init__(name)
        self.children = children

    def tick(self, context: SkillContext) -> SkillResult:
        for child in self.children:
            result = child.tick(context)
            if result.status != SkillStatus.SUCCESS:
                return result
        return SkillResult.success(message=f"{self.name} complete")


class SelectorNode(Node):
    def __init__(self, name: str, children: List[Node]):
        super().__init__(name)
        self.children = children

    def tick(self, context: SkillContext) -> SkillResult:
        last_failure = SkillResult.failure(FailureCode.UNKNOWN, message=f"{self.name} had no successful branch")
        for child in self.children:
            result = child.tick(context)
            if result.status == SkillStatus.SUCCESS:
                return result
            last_failure = result
        return last_failure


class RetryNode(Node):
    def __init__(self, name: str, child: Node, max_attempts: int = 2):
        super().__init__(name)
        self.child = child
        self.max_attempts = max_attempts

    def tick(self, context: SkillContext) -> SkillResult:
        last_result = SkillResult.failure(FailureCode.UNKNOWN, message=f"{self.name} failed")
        for _ in range(self.max_attempts):
            result = self.child.tick(context)
            if result.status == SkillStatus.SUCCESS:
                return result
            skill_name = context.world_state.execution.current_skill or self.child.name
            context.blackboard.increment_retry(skill_name)
            last_result = result
        return last_result


class TimeoutNode(Node):
    def __init__(self, name: str, child: Node, timeout_s: float):
        super().__init__(name)
        self.child = child
        self.timeout_s = timeout_s

    def tick(self, context: SkillContext) -> SkillResult:
        started = time.monotonic()
        result = self.child.tick(context)
        duration = time.monotonic() - started
        if duration > self.timeout_s:
            return SkillResult.failure(FailureCode.TIMEOUT, message=f"{self.name} exceeded timeout", duration_s=duration)
        return result


class RecoveryNode(Node):
    def __init__(self, name: str, child: Node, recovery_builder: Callable[[SkillContext, SkillResult], Optional[Node]]):
        super().__init__(name)
        self.child = child
        self.recovery_builder = recovery_builder

    def tick(self, context: SkillContext) -> SkillResult:
        result = self.child.tick(context)
        if result.status == SkillStatus.SUCCESS:
            return result
        recovery = self.recovery_builder(context, result)
        if recovery is None:
            return result
        recovery_result = recovery.tick(context)
        if recovery_result.status == SkillStatus.SUCCESS:
            return self.child.tick(context)
        return recovery_result


class TreeExecutor:
    """Executes behavior-tree style nodes with trace hooks."""

    def __init__(self, root: Node, registry: Any, trace_recorder: Optional[TraceRecorder] = None):
        self.root = root
        self.registry = registry
        self.trace_recorder = trace_recorder or TraceRecorder()

    @property
    def backend_name(self) -> str:
        return "py_trees" if py_trees is not None else "lightweight_bt"

    def run(self, context: SkillContext) -> SkillResult:
        context.metadata["registry"] = self.registry
        context.trace_sink = self.trace_recorder
        return self.root.tick(context)
