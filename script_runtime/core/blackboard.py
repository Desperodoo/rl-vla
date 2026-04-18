"""World state and blackboard primitives."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .failure_codes import FailureCode


@dataclass
class RobotState:
    joint_positions: List[float] = field(default_factory=list)
    eef_pose: List[float] = field(default_factory=list)
    gripper_width: float = 0.0
    velocities: Dict[str, Any] = field(default_factory=dict)
    fault_flag: bool = False
    mode: str = "unknown"


@dataclass
class SceneState:
    object_pose: Optional[List[float]] = None
    container_pose: Optional[List[float]] = None
    place_pose: Optional[List[float]] = None
    grasped: bool = False
    contact_state: str = "unknown"
    workspace_ready: bool = True


@dataclass
class PerceptionState:
    detection_confidence: float = 0.0
    tracking_lost: bool = False
    depth_anomaly: bool = False
    calibration_version: str = "unknown"


@dataclass
class ExecutionContextState:
    current_skill: str = ""
    previous_failure_code: FailureCode = FailureCode.NONE
    retry_counts: Dict[str, int] = field(default_factory=dict)
    task_goal: Dict[str, Any] = field(default_factory=dict)
    active_source: str = "policy"
    control_owner: str = "upper_machine"
    task_id: str = ""


@dataclass
class LearnedHintsState:
    grasp_candidates: List[Dict[str, Any]] = field(default_factory=list)
    place_candidates: List[Dict[str, Any]] = field(default_factory=list)
    risk_score: float = 0.0
    success_probability: float = 0.0


@dataclass
class WorldState:
    robot: RobotState = field(default_factory=RobotState)
    scene: SceneState = field(default_factory=SceneState)
    perception: PerceptionState = field(default_factory=PerceptionState)
    execution: ExecutionContextState = field(default_factory=ExecutionContextState)
    learned: LearnedHintsState = field(default_factory=LearnedHintsState)
    last_update_time: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TaskBlackboard:
    """Centralized mutable world state with namespaced scratch storage."""

    def __init__(self, world_state: Optional[WorldState] = None):
        self._world_state = world_state or WorldState()
        self._scratch: Dict[str, Any] = {}

    @property
    def world_state(self) -> WorldState:
        return self._world_state

    def update_world(self, **sections: Any) -> WorldState:
        for section_name, values in sections.items():
            target = getattr(self._world_state, section_name)
            for key, value in values.items():
                setattr(target, key, value)
        self._world_state.last_update_time = time.time()
        return self._world_state

    def set(self, key: str, value: Any) -> None:
        self._scratch[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._scratch.get(key, default)

    def delete(self, key: str) -> None:
        self._scratch.pop(key, None)

    def increment_retry(self, skill_name: str) -> int:
        retries = self._world_state.execution.retry_counts
        retries[skill_name] = retries.get(skill_name, 0) + 1
        return retries[skill_name]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "world_state": copy.deepcopy(self._world_state.to_dict()),
            "scratch": copy.deepcopy(self._scratch),
        }
