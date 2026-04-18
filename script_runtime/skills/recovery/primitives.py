"""Recovery skills."""

from __future__ import annotations

from script_runtime.core.failure_codes import FailureCode
from script_runtime.core.result_types import SkillResult
from script_runtime.core.skill_base import Skill, SkillContext


class SafeRetreat(Skill):
    def __init__(self):
        super().__init__(name="SafeRetreat", timeout_s=2.0, failure_code=FailureCode.COLLISION_RISK)

    def run(self, context: SkillContext) -> SkillResult:
        retreat_pose = context.blackboard.get("retreat_pose", context.world_state.scene.place_pose)
        if retreat_pose is None:
            return SkillResult.failure(FailureCode.COLLISION_RISK, message="No retreat pose available")
        result = context.adapters["sdk"].move_l(retreat_pose, speed=0.3)
        if not result.get("ok", False):
            return SkillResult.failure(FailureCode.SDK_ERROR, message="Safe retreat failed", sdk_result=result)
        return SkillResult.success(command=result["command"])


class RetryWithNextCandidate(Skill):
    def __init__(self):
        super().__init__(name="RetryWithNextCandidate", timeout_s=0.2, failure_code=FailureCode.NO_GRASP_CANDIDATE)

    def run(self, context: SkillContext) -> SkillResult:
        candidates = list(context.world_state.learned.grasp_candidates)
        if len(candidates) <= 1:
            return SkillResult.failure(FailureCode.NO_GRASP_CANDIDATE, message="No next candidate available")
        candidates.pop(0)
        context.blackboard.update_world(learned={"grasp_candidates": candidates})
        context.blackboard.set("active_grasp_candidate", candidates[0])
        if candidates[0].get("pose") is not None:
            context.blackboard.set("active_grasp_pose", candidates[0]["pose"])
        if candidates[0].get("pregrasp_pose") is not None:
            context.blackboard.set("pregrasp_pose", candidates[0]["pregrasp_pose"])
        return SkillResult.success(next_candidate=candidates[0])


class HumanTakeover(Skill):
    def __init__(self):
        super().__init__(name="HumanTakeover", timeout_s=0.1, failure_code=FailureCode.HUMAN_ABORT)

    def run(self, context: SkillContext) -> SkillResult:
        context.blackboard.update_world(
            execution={
                "active_source": "human",
                "control_owner": "upper_machine",
                "previous_failure_code": FailureCode.HUMAN_ABORT,
            }
        )
        return SkillResult.failure(FailureCode.HUMAN_ABORT, message="Escalated to human takeover")
