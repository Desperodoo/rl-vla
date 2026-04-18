"""Factory helpers for default runtime assembly."""

from __future__ import annotations

from script_runtime.core.registry import SkillRegistry
from script_runtime.skills.checks import CheckContact, CheckGrasp, CheckSceneReady, CheckTaskSuccess, WaitForObjectStable
from script_runtime.skills.gripper import ExecuteGraspPhase, GuardedClose, OpenGripper, ResetGripper
from script_runtime.skills.learned import ScorePlaceCandidates, SuccessRiskCheck
from script_runtime.skills.motion import (
    ApproachActiveGrasp,
    GoHome,
    GoPregrasp,
    Lift,
    MoveJ,
    MoveL,
    PlaceApproach,
    PlaceRelease,
    ResetArm,
    Retreat,
    ServoDelta,
    Stop,
)
from script_runtime.skills.perception import GetGraspCandidates, GetObjectPose, ReacquirePerception
from script_runtime.skills.recovery import HumanTakeover, RetryWithNextCandidate, SafeRetreat


def build_default_skill_registry() -> SkillRegistry:
    """Construct a default registry with the first-wave primitive skills."""

    registry = SkillRegistry()
    registry.register_many(
        [
            MoveJ(),
            MoveL(),
            ServoDelta(),
            Stop(),
            GoHome(),
            GoPregrasp(),
            ApproachActiveGrasp(),
            Lift(),
            PlaceApproach(),
            PlaceRelease(),
            Retreat(),
            ResetArm(),
            OpenGripper(),
            GuardedClose(),
            ExecuteGraspPhase(),
            ResetGripper(),
            CheckSceneReady(),
            WaitForObjectStable(),
            CheckGrasp(),
            CheckContact(),
            CheckTaskSuccess(),
            GetObjectPose(),
            GetGraspCandidates(),
            ReacquirePerception(),
            SafeRetreat(),
            RetryWithNextCandidate(),
            HumanTakeover(),
            ScorePlaceCandidates(),
            SuccessRiskCheck(),
        ]
    )
    return registry
