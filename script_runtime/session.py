"""Standalone runtime session helpers for SDK-first script execution."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from script_runtime.adapters import MockSDKBridge, NullLearnedModuleAdapter, SDKBridge
from script_runtime.core import SkillContext, TaskBlackboard, WorldState
from script_runtime.executors import TraceRecorder, TreeExecutor
from script_runtime.factory import build_default_skill_registry
from script_runtime.tasks.pick_place import PickPlaceTask


def load_runtime_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".json":
            return json.load(handle)
        return yaml.safe_load(handle) or {}


def seed_pick_place_blackboard(blackboard: TaskBlackboard, config: Dict[str, Any]) -> TaskBlackboard:
    scene = config.get("scene", {})
    execution = config.get("execution", {})
    task_goal = config.get("task_goal", {})
    poses = config.get("poses", {})
    gripper = config.get("gripper", {})

    blackboard.update_world(
        perception={
            "detection_confidence": float(scene.get("detection_confidence", 1.0 if scene.get("object_pose") else 0.0)),
            "tracking_lost": bool(scene.get("tracking_lost", False)),
            "depth_anomaly": bool(scene.get("depth_anomaly", False)),
            "calibration_version": str(scene.get("calibration_version", "static-sdk")),
        },
        scene={
            "object_pose": scene.get("object_pose"),
            "container_pose": scene.get("container_pose"),
            "place_pose": poses.get("place_pose"),
            "grasped": bool(scene.get("grasped", False)),
            "contact_state": str(scene.get("contact_state", "nominal")),
            "workspace_ready": bool(scene.get("workspace_ready", True)),
        },
        execution={
            "task_goal": dict(task_goal),
            "active_source": str(execution.get("active_source", "policy")),
            "control_owner": str(execution.get("control_owner", "script_runtime")),
        },
    )

    for key in ("home_joints", "reset_joints", "pregrasp_pose", "lift_pose", "place_pose", "retreat_pose"):
        if key in poses:
            blackboard.set(key, poses[key])
    if "place_release_pose" in poses:
        blackboard.set("place_release_pose", poses["place_release_pose"])
    elif "place_pose" in poses:
        blackboard.set("place_release_pose", poses["place_pose"])

    if "object_pose" in scene:
        blackboard.set("object_pose", scene["object_pose"])
    if "grasp_candidates" in scene:
        blackboard.set("grasp_candidates", scene["grasp_candidates"])
    if "servo_delta" in poses:
        blackboard.set("servo_delta", poses["servo_delta"])
    if "open_width" in gripper:
        blackboard.set("open_gripper_width", gripper["open_width"])
    if "close_width" in gripper:
        blackboard.set("close_gripper_width", gripper["close_width"])
    return blackboard


@dataclass
class ScriptRuntimeSession:
    """Runtime session that assembles executor, registry, adapters, and trace."""

    task: Any
    registry: Any
    blackboard: TaskBlackboard
    adapters: Dict[str, Any]
    trace_recorder: TraceRecorder
    trace_path: Optional[Path] = None
    task_id: str = ""
    runtime_artifacts: Dict[str, Any] = None

    def run(self):
        task_id = self.task_id or f"{self.task.name}-{uuid.uuid4().hex[:8]}"
        self.blackboard.update_world(execution={"task_id": task_id})
        sdk = self.adapters.get("sdk")
        if sdk is not None:
            init_result = sdk.initialize()
            if not init_result.get("ok", False):
                raise RuntimeError(f"Failed to initialize SDK runtime: {init_result}")
            if hasattr(sdk, "attach_blackboard"):
                sdk.attach_blackboard(self.blackboard)
            sdk.refresh_world(self.blackboard)

        root = self.task.root or self.task.build(self.registry)
        executor = TreeExecutor(root=root, registry=self.registry, trace_recorder=self.trace_recorder)
        context = SkillContext(
            blackboard=self.blackboard,
            adapters=self.adapters,
            task_id=task_id,
            metadata={"session_type": "sdk_first"},
        )
        result = executor.run(context)
        if sdk is not None:
            sdk.refresh_world(self.blackboard)
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_path.write_text(self.trace_recorder.to_jsonl(), encoding="utf-8")
        self.runtime_artifacts = {}
        if sdk is not None and hasattr(sdk, "export_episode_artifacts"):
            artifact_dir = self.trace_path.parent if self.trace_path is not None else None
            self.runtime_artifacts = dict(sdk.export_episode_artifacts(task_id=task_id, output_dir=artifact_dir) or {})
        return result

    def shutdown(self) -> Dict[str, Any]:
        sdk = self.adapters.get("sdk")
        if sdk is None:
            return {"ok": True, "skipped": True}
        return sdk.shutdown()


def build_pick_place_session(
    config: Dict[str, Any],
    sdk_bridge: Optional[SDKBridge] = None,
    registry: Optional[Any] = None,
    use_mock: bool = False,
) -> ScriptRuntimeSession:
    world = WorldState()
    blackboard = TaskBlackboard(world)
    seed_pick_place_blackboard(blackboard, config)

    runtime = config.get("runtime", {})
    task = PickPlaceTask(goal=config.get("task_goal", {}))
    registry = registry or build_default_skill_registry()
    trace_path = runtime.get("trace_path")

    if sdk_bridge is None:
        sdk_bridge = MockSDKBridge() if use_mock else _build_sdk_bridge_from_config(config)

    adapters = {
        "sdk": sdk_bridge,
        "learned": NullLearnedModuleAdapter(),
    }
    return ScriptRuntimeSession(
        task=task,
        registry=registry,
        blackboard=blackboard,
        adapters=adapters,
        trace_recorder=TraceRecorder(),
        trace_path=Path(trace_path) if trace_path else None,
        task_id=str(runtime.get("task_id", "")),
        runtime_artifacts={},
    )


def _build_sdk_bridge_from_config(config: Dict[str, Any]) -> SDKBridge:
    from script_runtime.adapters.sdk_bridge import CArmSDKBridge

    robot = config.get("robot", {})
    gripper = config.get("gripper", {})
    runtime = config.get("runtime", {})
    return CArmSDKBridge(
        robot_ip=str(robot["robot_ip"]),
        port=int(robot.get("port", 8090)),
        timeout=float(robot.get("timeout", 1.0)),
        control_mode=int(robot.get("control_mode", 1)),
        speed_level=float(robot.get("speed_level", 3.0)),
        response_level=int(robot.get("response_level", 20)),
        tool_index=robot.get("tool_index"),
        collision_enable=robot.get("collision_enable"),
        collision_sensitivity=int(robot.get("collision_sensitivity", 0)),
        ready_on_initialize=bool(runtime.get("ready_on_initialize", True)),
        servo_enable_on_initialize=bool(runtime.get("servo_enable_on_initialize", True)),
        open_width=float(gripper.get("open_width", 0.08)),
        close_width=float(gripper.get("close_width", 0.0)),
        gripper_tau=float(gripper.get("tau", 10.0)),
        use_linear_for_move_l=bool(robot.get("use_linear_for_move_l", True)),
        sync=bool(robot.get("sync", True)),
    )
