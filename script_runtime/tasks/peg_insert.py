"""Peg insertion task placeholder."""

from __future__ import annotations

from typing import Any

from script_runtime.core.task_base import ScriptTask
from script_runtime.executors.pytrees_executor import SequenceNode, SkillNode


class PegInsertTask(ScriptTask):
    """Contact-rich placeholder task for future expansion."""

    def __init__(self, goal: dict | None = None):
        super().__init__(name="peg_insert", goal=goal or {})

    def build(self, registry: Any):
        self.root = SequenceNode(
            "peg_insert_root",
            [
                SkillNode("check_scene_ready", "CheckSceneReady"),
                SkillNode("get_object_pose", "GetObjectPose"),
                SkillNode("go_pregrasp", "GoPregrasp"),
                SkillNode("check_contact", "CheckContact"),
            ],
        )
        return self.root
