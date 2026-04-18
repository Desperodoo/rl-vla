"""Drawer-open then pick placeholder task."""

from __future__ import annotations

from typing import Any

from script_runtime.core.task_base import ScriptTask
from script_runtime.executors.pytrees_executor import SequenceNode, SkillNode


class DrawerOpenPickTask(ScriptTask):
    def __init__(self, goal: dict | None = None):
        super().__init__(name="drawer_open_pick", goal=goal or {})

    def build(self, registry: Any):
        self.root = SequenceNode(
            "drawer_open_pick_root",
            [
                SkillNode("check_scene_ready", "CheckSceneReady"),
                SkillNode("get_object_pose", "GetObjectPose"),
                SkillNode("go_pregrasp", "GoPregrasp"),
            ],
        )
        return self.root
