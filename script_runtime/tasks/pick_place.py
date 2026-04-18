"""Pick-and-place task tree."""

from __future__ import annotations

from typing import Any

from script_runtime.core.task_base import ScriptTask
from script_runtime.executors.pytrees_executor import RecoveryNode, RetryNode, SequenceNode, SkillNode, TimeoutNode


class PickPlaceTask(ScriptTask):
    """First milestone task line for the runtime."""

    def __init__(self, goal: dict | None = None):
        super().__init__(name="pick_place", goal=goal or {})

    def build(self, registry: Any):
        grasp_attempt = SequenceNode(
            "grasp_attempt",
            [
                TimeoutNode("go_pregrasp_timeout", SkillNode("go_pregrasp", "GoPregrasp"), timeout_s=8.0),
                TimeoutNode("grasp_phase_timeout", SkillNode("execute_grasp_phase", "ExecuteGraspPhase"), timeout_s=8.0),
                SkillNode("check_grasp", "CheckGrasp"),
                SkillNode("lift", "Lift"),
                SkillNode("check_grasp_after_lift", "CheckGrasp"),
            ],
        )

        def build_grasp_recovery(_context, _result):
            return SequenceNode(
                "grasp_recovery_sequence",
                [
                    SkillNode("safe_retreat", "SafeRetreat"),
                    SkillNode("retry_next_candidate", "RetryWithNextCandidate"),
                ],
            )

        self.root = SequenceNode(
            "pick_place_root",
            [
                SkillNode("check_scene_ready", "CheckSceneReady"),
                RetryNode(
                    "acquire_object",
                    SequenceNode(
                        "perception_sequence",
                        [
                            SkillNode("get_object_pose", "GetObjectPose"),
                            SkillNode("get_grasp_candidates", "GetGraspCandidates"),
                        ],
                    ),
                    max_attempts=2,
                ),
                RetryNode(
                    "grasp_retry_budget",
                    RecoveryNode("grasp_recovery", grasp_attempt, build_grasp_recovery),
                    max_attempts=3,
                ),
                SkillNode("place_approach", "PlaceApproach"),
                SkillNode("place_release", "PlaceRelease"),
                SkillNode("open_gripper", "OpenGripper"),
                SkillNode("retreat", "Retreat"),
                SkillNode("go_home", "GoHome"),
                SkillNode("check_task_success", "CheckTaskSuccess"),
            ],
        )
        return self.root
