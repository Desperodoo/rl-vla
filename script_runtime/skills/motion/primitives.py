"""Primitive motion skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from script_runtime.core.failure_codes import FailureCode
from script_runtime.core.result_types import RecoveryAction, SkillResult
from script_runtime.core.skill_base import Skill, SkillContext


@dataclass
class MotionTargetMixin:
    target_pose: Optional[List[float]] = None
    target_joints: Optional[List[float]] = None
    speed: float = 1.0


class MoveJ(Skill):
    def __init__(
        self,
        name: str = "MoveJ",
        target_joints: Optional[List[float]] = None,
        speed: float = 1.0,
        blackboard_key: str = "target_joints",
    ):
        super().__init__(name=name, timeout_s=8.0, failure_code=FailureCode.NO_IK)
        self.target_joints = target_joints
        self.speed = speed
        self.blackboard_key = blackboard_key

    def run(self, context: SkillContext) -> SkillResult:
        if self.name in {"Retreat", "GoHome"} and context.blackboard.get("skip_release_sequence", False):
            return SkillResult.success(skipped=True, reason="holding_object_for_goal")
        target = self.target_joints or context.blackboard.get(self.blackboard_key)
        if not target:
            return SkillResult.failure(FailureCode.NO_IK, message="Missing joint target")
        sdk = context.adapters["sdk"]
        result = sdk.move_j(target, speed=self.speed)
        if not result.get("ok", False):
            return SkillResult.failure(FailureCode.SDK_ERROR, message="move_j failed", sdk_result=result)
        if hasattr(sdk, "refresh_world"):
            sdk.refresh_world(context.blackboard)
        return SkillResult.success(command=result["command"])

    def recover(self, context: SkillContext):
        return RecoveryAction(name="SafeRetreat")


class MoveL(Skill):
    def __init__(
        self,
        name: str = "MoveL",
        target_pose: Optional[List[float]] = None,
        speed: float = 1.0,
        blackboard_key: str = "target_pose",
    ):
        super().__init__(name=name, timeout_s=8.0, failure_code=FailureCode.NO_IK)
        self.target_pose = target_pose
        self.speed = speed
        self.blackboard_key = blackboard_key

    def run(self, context: SkillContext) -> SkillResult:
        if self.name in {"Retreat", "GoHome"} and context.blackboard.get("skip_release_sequence", False):
            return SkillResult.success(skipped=True, reason="holding_object_for_goal")
        target = self.target_pose or context.blackboard.get(self.blackboard_key)
        if not target:
            return SkillResult.failure(FailureCode.NO_IK, message="Missing pose target")
        sdk = context.adapters["sdk"]
        result = sdk.move_l(target, speed=self.speed)
        if not result.get("ok", False):
            return SkillResult.failure(FailureCode.SDK_ERROR, message="move_l failed", sdk_result=result)
        if hasattr(sdk, "refresh_world"):
            sdk.refresh_world(context.blackboard)
        return SkillResult.success(command=result["command"])

    def recover(self, context: SkillContext):
        return RecoveryAction(name="SafeRetreat")


class ServoDelta(Skill):
    def __init__(self, name: str = "ServoDelta", delta_pose: Optional[List[float]] = None):
        super().__init__(name=name, timeout_s=2.0, failure_code=FailureCode.COLLISION_RISK)
        self.delta_pose = delta_pose

    def run(self, context: SkillContext) -> SkillResult:
        delta = self.delta_pose or context.blackboard.get("servo_delta")
        if not delta:
            return SkillResult.failure(FailureCode.NO_IK, message="Missing servo delta")
        sdk = context.adapters["sdk"]
        result = sdk.servo_delta(delta)
        if not result.get("ok", False):
            return SkillResult.failure(FailureCode.SDK_ERROR, message="servo_delta failed", sdk_result=result)
        if hasattr(sdk, "refresh_world"):
            sdk.refresh_world(context.blackboard)
        return SkillResult.success(command=result["command"])

    def recover(self, context: SkillContext):
        return RecoveryAction(name="Stop")


class Stop(Skill):
    def __init__(self, name: str = "Stop"):
        super().__init__(name=name, timeout_s=1.0, failure_code=FailureCode.SDK_ERROR)

    def run(self, context: SkillContext) -> SkillResult:
        result = context.adapters["sdk"].stop()
        if not result.get("ok", False):
            return SkillResult.failure(FailureCode.SDK_ERROR, message="stop failed", sdk_result=result)
        return SkillResult.success(command=result["command"])


class GoHome(MoveJ):
    def __init__(self, home_joints: Optional[List[float]] = None):
        super().__init__(
            name="GoHome",
            target_joints=home_joints,
            speed=0.8,
            blackboard_key="home_joints",
        )


class GoPregrasp(MoveL):
    def __init__(self, target_pose: Optional[List[float]] = None):
        super().__init__(
            name="GoPregrasp",
            target_pose=target_pose,
            speed=0.5,
            blackboard_key="pregrasp_pose",
        )


class Retreat(MoveL):
    def __init__(self, target_pose: Optional[List[float]] = None):
        super().__init__(
            name="Retreat",
            target_pose=target_pose,
            speed=0.5,
            blackboard_key="retreat_pose",
        )


class Lift(MoveL):
    def __init__(self, target_pose: Optional[List[float]] = None):
        super().__init__(
            name="Lift",
            target_pose=target_pose,
            speed=0.4,
            blackboard_key="lift_pose",
        )


class PlaceApproach(MoveL):
    def __init__(self, target_pose: Optional[List[float]] = None):
        super().__init__(
            name="PlaceApproach",
            target_pose=target_pose,
            speed=0.4,
            blackboard_key="place_pose",
        )


class PlaceRelease(MoveL):
    def __init__(self, target_pose: Optional[List[float]] = None):
        super().__init__(
            name="PlaceRelease",
            target_pose=target_pose,
            speed=0.25,
            blackboard_key="place_release_pose",
        )


class ApproachActiveGrasp(MoveL):
    def __init__(self, target_pose: Optional[List[float]] = None):
        super().__init__(
            name="ApproachActiveGrasp",
            target_pose=target_pose,
            speed=0.2,
            blackboard_key="active_grasp_pose",
        )


class ResetArm(MoveJ):
    def __init__(self, target_joints: Optional[List[float]] = None):
        super().__init__(
            name="ResetArm",
            target_joints=target_joints,
            speed=0.5,
            blackboard_key="reset_joints",
        )
