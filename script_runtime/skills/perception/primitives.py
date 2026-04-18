"""Perception-oriented skills."""

from __future__ import annotations

from typing import Any, List

from script_runtime.core.failure_codes import FailureCode
from script_runtime.core.result_types import RecoveryAction, SkillResult
from script_runtime.core.skill_base import Skill, SkillContext


class GetObjectPose(Skill):
    def __init__(self):
        super().__init__(name="GetObjectPose", timeout_s=1.0, failure_code=FailureCode.NO_OBJECT_DETECTED)

    def run(self, context: SkillContext) -> SkillResult:
        maniskill = context.adapters.get("maniskill")
        pose = None
        if maniskill is not None and hasattr(maniskill, "get_object_pose"):
            pose = maniskill.get_object_pose()
            if hasattr(maniskill, "get_place_pose"):
                place_pose = maniskill.get_place_pose()
                if place_pose is not None:
                    context.blackboard.update_world(scene={"place_pose": place_pose})
                    context.blackboard.set("place_pose", place_pose)
            if hasattr(maniskill, "get_place_release_pose"):
                release_pose = maniskill.get_place_release_pose()
                if release_pose is not None:
                    context.blackboard.set("place_release_pose", release_pose)
            if hasattr(maniskill, "get_retreat_pose"):
                retreat_pose = maniskill.get_retreat_pose()
                if retreat_pose is not None:
                    context.blackboard.set("retreat_pose", retreat_pose)
        if pose is None:
            pose = context.blackboard.get("object_pose", context.world_state.scene.object_pose)
        if pose is None:
            return SkillResult.failure(FailureCode.NO_OBJECT_DETECTED, message="Object pose unavailable")
        context.blackboard.update_world(scene={"object_pose": pose})
        return SkillResult.success(object_pose=pose)

    def recover(self, context: SkillContext):
        return RecoveryAction(name="ReacquirePerception")


class GetGraspCandidates(Skill):
    def __init__(self):
        super().__init__(name="GetGraspCandidates", timeout_s=1.0, failure_code=FailureCode.NO_GRASP_CANDIDATE)

    def run(self, context: SkillContext) -> SkillResult:
        maniskill = context.adapters.get("maniskill")
        candidates = None
        if maniskill is not None and hasattr(maniskill, "get_grasp_candidates"):
            candidates = maniskill.get_grasp_candidates()
        if candidates is None:
            candidates = context.blackboard.get("grasp_candidates", context.world_state.learned.grasp_candidates)
        if not candidates:
            return SkillResult.failure(FailureCode.NO_GRASP_CANDIDATE, message="No grasp candidates")

        learned = context.adapters.get("learned")
        if learned is not None:
            scores = learned.score(candidates, context.world_state, context)
            enriched = []
            for candidate, score in zip(candidates, scores):
                item = dict(candidate)
                item["score"] = score
                enriched.append(item)
            candidates = sorted(enriched, key=lambda item: item.get("score", 0.0), reverse=True)
        context.blackboard.update_world(learned={"grasp_candidates": candidates})
        context.blackboard.set("active_grasp_candidate", candidates[0])
        if candidates[0].get("pose") is not None:
            context.blackboard.set("active_grasp_pose", candidates[0]["pose"])
        if candidates[0].get("pregrasp_pose") is not None:
            context.blackboard.set("pregrasp_pose", candidates[0]["pregrasp_pose"])
        return SkillResult.success(grasp_candidates=candidates)

    def recover(self, context: SkillContext):
        return RecoveryAction(name="ReacquirePerception")


class ReacquirePerception(Skill):
    def __init__(self):
        super().__init__(name="ReacquirePerception", timeout_s=2.0, failure_code=FailureCode.PERCEPTION_LOST)

    def run(self, context: SkillContext) -> SkillResult:
        camera = context.adapters.get("camera")
        snapshot = camera.get_snapshot() if camera is not None else None
        if snapshot is None:
            return SkillResult.failure(FailureCode.PERCEPTION_LOST, message="Camera snapshot unavailable")
        context.blackboard.update_world(
            perception={"tracking_lost": False, "calibration_version": snapshot.calibration_version}
        )
        return SkillResult.success(calibration_version=snapshot.calibration_version)
