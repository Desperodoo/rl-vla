"""Check and guard skills."""

from __future__ import annotations

from script_runtime.core.failure_codes import FailureCode
from script_runtime.core.result_types import SkillResult
from script_runtime.core.skill_base import Skill, SkillContext


class CheckSceneReady(Skill):
    def __init__(self):
        super().__init__(name="CheckSceneReady", timeout_s=0.2, failure_code=FailureCode.PERCEPTION_LOST)

    def run(self, context: SkillContext) -> SkillResult:
        scene = context.world_state.scene
        perception = context.world_state.perception
        if not scene.workspace_ready:
            return SkillResult.failure(FailureCode.WORKSPACE_VIOLATION, message="Workspace not ready")
        if perception.tracking_lost or perception.depth_anomaly:
            return SkillResult.failure(FailureCode.PERCEPTION_LOST, message="Perception degraded")
        return SkillResult.success(scene_ready=True)


class WaitForObjectStable(Skill):
    def __init__(self):
        super().__init__(name="WaitForObjectStable", timeout_s=0.5, failure_code=FailureCode.PERCEPTION_LOST)

    def run(self, context: SkillContext) -> SkillResult:
        conf = context.world_state.perception.detection_confidence
        if conf < 0.5:
            return SkillResult.failure(FailureCode.PERCEPTION_LOST, message="Object not stable enough")
        return SkillResult.success(stable=True)


class CheckGrasp(Skill):
    def __init__(self):
        super().__init__(name="CheckGrasp", timeout_s=0.5, failure_code=FailureCode.GRASP_FAIL)

    def run(self, context: SkillContext) -> SkillResult:
        maniskill = context.adapters.get("maniskill")
        if maniskill is not None and hasattr(maniskill, "is_grasped"):
            grasped = bool(maniskill.is_grasped())
            context.blackboard.update_world(scene={"grasped": grasped})
        if not context.world_state.scene.grasped:
            return SkillResult.failure(FailureCode.GRASP_FAIL, message="Grasp not confirmed")
        return SkillResult.success(grasp_confirmed=True)


class CheckContact(Skill):
    def __init__(self):
        super().__init__(name="CheckContact", timeout_s=0.5, failure_code=FailureCode.CONTACT_ANOMALY)

    def run(self, context: SkillContext) -> SkillResult:
        state = context.world_state.scene.contact_state
        if state in ("anomaly", "unexpected"):
            return SkillResult.failure(FailureCode.CONTACT_ANOMALY, message=f"Unexpected contact: {state}")
        return SkillResult.success(contact_state=state)


class CheckTaskSuccess(Skill):
    def __init__(self):
        super().__init__(name="CheckTaskSuccess", timeout_s=2.0, failure_code=FailureCode.UNKNOWN)

    def run(self, context: SkillContext) -> SkillResult:
        sdk = context.adapters.get("sdk")
        if sdk is not None and hasattr(sdk, "settle"):
            sdk.settle()
            if hasattr(sdk, "refresh_world"):
                sdk.refresh_world(context.blackboard)

        if sdk is None or not hasattr(sdk, "evaluate_task_success"):
            return SkillResult.success(env_success=True, mode="no_eval_hook")

        result = sdk.evaluate_task_success()
        if not result.get("ok", False):
            return SkillResult.failure(FailureCode.SDK_ERROR, message="Task success evaluation failed", sdk_result=result)
        if not result.get("success", False):
            return SkillResult.failure(FailureCode.UNKNOWN, message="Environment success check failed", env_result=result)
        return SkillResult.success(env_success=True, env_result=result)
