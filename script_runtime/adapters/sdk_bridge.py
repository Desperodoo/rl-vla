"""SDK bridge wrapping robot control primitives."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class SDKBridge:
    """Interface expected by motion and gripper skills."""

    def connect(self) -> Dict[str, Any]:
        raise NotImplementedError

    def initialize(self) -> Dict[str, Any]:
        raise NotImplementedError

    def shutdown(self) -> Dict[str, Any]:
        raise NotImplementedError

    def move_j(self, joints: List[float], speed: float = 1.0) -> Dict[str, Any]:
        raise NotImplementedError

    def move_l(self, pose: List[float], speed: float = 1.0) -> Dict[str, Any]:
        raise NotImplementedError

    def servo_delta(self, delta_pose: List[float]) -> Dict[str, Any]:
        raise NotImplementedError

    def stop(self) -> Dict[str, Any]:
        raise NotImplementedError

    def open_gripper(self, width: Optional[float] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def close_gripper(self, width: Optional[float] = None, guarded: bool = False) -> Dict[str, Any]:
        raise NotImplementedError

    def get_status(self) -> Dict[str, Any]:
        raise NotImplementedError

    def refresh_world(self, blackboard: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def execute_grasp_phase(self, target_pose: List[float], context: Any | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    def attach_blackboard(self, blackboard: Any) -> None:
        return None

    def settle(self, steps: Optional[int] = None) -> Dict[str, Any]:
        return {"ok": True, "action": "settle", "steps": steps or 0}

    def evaluate_task_success(self) -> Dict[str, Any]:
        return {"ok": True, "success": True}

    def export_episode_artifacts(self, task_id: str, output_dir: Any | None = None) -> Dict[str, Any]:
        return {}

    def should_release_object(self) -> bool:
        return True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _import_carm_module() -> Any:
    candidates = []
    env_value = os.environ.get("CARM_PYTHON_LIB", "").strip()
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(_repo_root() / "arm_control_sdk" / "python" / "lib")

    for candidate in candidates:
        if candidate and candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

    try:
        return importlib.import_module("carm")
    except ModuleNotFoundError as exc:
        searched = ", ".join(str(path) for path in candidates)
        raise ModuleNotFoundError(
            "Unable to import 'carm'. Build the arm_control_sdk Python extension first, "
            "or point CARM_PYTHON_LIB to arm_control_sdk/python/lib. "
            f"Searched: {searched}"
        ) from exc


@dataclass
class CArmSDKBridge(SDKBridge):
    """Direct bridge to ``arm_control_sdk`` for script-runtime execution.

    This bridge intentionally keeps ROS out of the critical path. It talks to the
    pybind-exposed ``carm.CArmSingleCol`` API directly and normalizes its return
    values into the runtime's adapter contract.
    """

    robot_ip: str
    port: int = 8090
    timeout: float = 1.0
    control_mode: int = 1
    speed_level: float = 3.0
    response_level: int = 20
    tool_index: Optional[int] = None
    collision_enable: Optional[bool] = None
    collision_sensitivity: int = 0
    ready_on_initialize: bool = True
    servo_enable_on_initialize: bool = True
    open_width: float = 0.08
    close_width: float = 0.0
    gripper_tau: float = 10.0
    use_linear_for_move_l: bool = True
    sync: bool = True
    client: Any = field(init=False, default=None)
    last_status: Dict[str, Any] = field(default_factory=dict)
    blackboard: Any = field(init=False, default=None)

    def connect(self) -> Dict[str, Any]:
        if self.client is None:
            module = _import_carm_module()
            self.client = module.CArmSingleCol(self.robot_ip, self.port, self.timeout)
        ret = self.client.connect(self.robot_ip, self.port, self.timeout)
        ok = self._ret_ok(ret)
        return self._result("connect", ok, ret=ret, robot_ip=self.robot_ip, port=self.port)

    def initialize(self) -> Dict[str, Any]:
        if self.client is None or not self.client.is_connected():
            connect_result = self.connect()
            if not connect_result["ok"]:
                return connect_result

        if self.ready_on_initialize:
            ready_ret = self.client.set_ready()
            if not self._ret_ok(ready_ret):
                return self._result("set_ready", False, ret=ready_ret)

        if self.servo_enable_on_initialize:
            servo_ret = self.client.set_servo_enable(True)
            if not self._ret_ok(servo_ret):
                return self._result("set_servo_enable", False, ret=servo_ret)

        mode_ret = self.client.set_control_mode(self.control_mode)
        if not self._ret_ok(mode_ret):
            return self._result("set_control_mode", False, ret=mode_ret, control_mode=self.control_mode)

        speed_ret = self.client.set_speed_level(self.speed_level, self.response_level)
        if not self._ret_ok(speed_ret):
            return self._result(
                "set_speed_level",
                False,
                ret=speed_ret,
                speed_level=self.speed_level,
                response_level=self.response_level,
            )

        if self.tool_index is not None:
            tool_ret = self.client.set_tool_index(self.tool_index)
            if not self._ret_ok(tool_ret):
                return self._result("set_tool_index", False, ret=tool_ret, tool_index=self.tool_index)

        if self.collision_enable is not None:
            collision_ret = self.client.set_collision_config(self.collision_enable, self.collision_sensitivity)
            if not self._ret_ok(collision_ret):
                return self._result(
                    "set_collision_config",
                    False,
                    ret=collision_ret,
                    collision_enable=self.collision_enable,
                    collision_sensitivity=self.collision_sensitivity,
                )

        return self._result(
            "initialize",
            True,
            control_mode=self.control_mode,
            speed_level=self.speed_level,
            tool_index=self.tool_index,
        )

    def shutdown(self) -> Dict[str, Any]:
        if self.client is None:
            return self._result("shutdown", True, skipped=True)
        ret = self.client.disconnect()
        ok = self._ret_ok(ret)
        return self._result("shutdown", ok, ret=ret)

    def move_j(self, joints: List[float], speed: float = 1.0) -> Dict[str, Any]:
        self._set_speed_from_runtime(speed)
        ret = self.client.move_joint(list(joints), -1, self.sync)
        return self._result("move_j", self._ret_ok(ret), ret=ret, command={"type": "move_j", "joints": list(joints), "speed": speed})

    def move_l(self, pose: List[float], speed: float = 1.0) -> Dict[str, Any]:
        self._set_speed_from_runtime(speed)
        pose = list(pose)
        if self.use_linear_for_move_l and hasattr(self.client, "move_line_pose"):
            ret = self.client.move_line_pose(pose, self.sync)
        else:
            ret = self.client.move_pose(pose, -1, self.sync)
        return self._result("move_l", self._ret_ok(ret), ret=ret, command={"type": "move_l", "pose": pose, "speed": speed})

    def servo_delta(self, delta_pose: List[float]) -> Dict[str, Any]:
        current_pose = list(self.client.get_cart_pose())
        if not current_pose:
            return self._result("servo_delta", False, message="Current cartesian pose unavailable")
        target = current_pose[:]
        for index, value in enumerate(list(delta_pose)):
            if index >= len(target):
                break
            target[index] += value
        ret = self.client.track_pose(target, -1)
        return self._result(
            "servo_delta",
            self._ret_ok(ret),
            ret=ret,
            command={"type": "servo_delta", "delta_pose": list(delta_pose), "target_pose": target},
        )

    def stop(self) -> Dict[str, Any]:
        ret = self.client.emergency_stop()
        return self._result("stop", self._ret_ok(ret), ret=ret, command={"type": "stop"})

    def open_gripper(self, width: Optional[float] = None) -> Dict[str, Any]:
        target_width = self.open_width if width is None else width
        ret = self.client.set_gripper(target_width, self.gripper_tau)
        return self._result(
            "open_gripper",
            self._ret_ok(ret),
            ret=ret,
            command={"type": "open_gripper", "width": target_width, "tau": self.gripper_tau},
        )

    def close_gripper(self, width: Optional[float] = None, guarded: bool = False) -> Dict[str, Any]:
        target_width = self.close_width if width is None else width
        ret = self.client.set_gripper(target_width, self.gripper_tau)
        return self._result(
            "close_gripper",
            self._ret_ok(ret),
            ret=ret,
            command={"type": "close_gripper", "width": target_width, "guarded": guarded, "tau": self.gripper_tau},
        )

    def get_status(self) -> Dict[str, Any]:
        if self.client is None:
            return {"ok": False, "connected": False, "fault": True, "message": "SDK client not initialized"}

        raw_status = self.client.get_status()
        status = {
            "ok": True,
            "connected": bool(self.client.is_connected()),
            "fault": False,
            "joint_positions": list(self.client.get_joint_pos()),
            "eef_pose": list(self.client.get_cart_pose()),
            "gripper_width": float(self.client.get_gripper_pos()),
            "servo_status": getattr(raw_status, "servo_status", None),
            "state": getattr(raw_status, "state", None),
            "fsm_state": getattr(raw_status, "fsm_state", None),
            "speed_percentage": getattr(raw_status, "speed_percentage", None),
            "arm_name": getattr(raw_status, "arm_name", ""),
            "arm_index": getattr(raw_status, "arm_index", None),
            "mode": f"control_mode_{self.control_mode}",
            "raw_status": self._status_to_dict(raw_status),
        }
        status["fault"] = not status["connected"] or bool(status["servo_status"] in (False, 0))
        self.last_status = status
        return status

    def refresh_world(self, blackboard: Any) -> Dict[str, Any]:
        status = self.get_status()
        if not status.get("ok", False):
            return status
        blackboard.update_world(
            robot={
                "joint_positions": status.get("joint_positions", []),
                "eef_pose": status.get("eef_pose", []),
                "gripper_width": status.get("gripper_width", 0.0),
                "fault_flag": status.get("fault", False),
                "mode": status.get("mode", "unknown"),
                "velocities": {"speed_percentage": status.get("speed_percentage")},
            }
        )
        return status

    def execute_grasp_phase(self, target_pose: List[float], context: Any | None = None) -> Dict[str, Any]:
        move_result = self.move_l(target_pose, speed=0.2)
        if not move_result.get("ok", False):
            return move_result
        close_result = self.close_gripper(guarded=True)
        if not close_result.get("ok", False):
            return close_result
        return {
            "ok": True,
            "action": "execute_grasp_phase",
            "command": {"type": "execute_grasp_phase", "target_pose": list(target_pose)},
            "substeps": [move_result, close_result],
        }

    def attach_blackboard(self, blackboard: Any) -> None:
        self.blackboard = blackboard

    def evaluate_task_success(self) -> Dict[str, Any]:
        # Real robot acceptance is task-specific and not yet standardized.
        return {"ok": True, "success": True, "mode": "sdk_placeholder"}

    def _set_speed_from_runtime(self, speed: float) -> None:
        if self.client is None:
            return
        if speed <= 0:
            return
        level = max(0.1, min(10.0, float(speed) * 10.0))
        self.client.set_speed_level(level, self.response_level)

    @staticmethod
    def _ret_ok(ret: Any) -> bool:
        if isinstance(ret, bool):
            return ret
        if isinstance(ret, (int, float)):
            return ret == 0
        return ret is None

    @staticmethod
    def _status_to_dict(status: Any) -> Dict[str, Any]:
        if status is None:
            return {}
        keys = [
            "arm_index",
            "arm_name",
            "arm_is_connected",
            "arm_dof",
            "servo_status",
            "state",
            "fsm_state",
            "speed_percentage",
            "on_debug_mode",
        ]
        return {key: getattr(status, key, None) for key in keys}

    @staticmethod
    def _result(action: str, ok: bool, **payload: Any) -> Dict[str, Any]:
        return {"ok": ok, "action": action, **payload}


@dataclass
class MockSDKBridge(SDKBridge):
    """Test-friendly bridge that records issued commands."""

    status: Dict[str, Any] = field(
        default_factory=lambda: {
            "ok": True,
            "fault": False,
            "connected": True,
            "joint_positions": [0.0] * 6,
            "eef_pose": [0.0] * 7,
            "gripper_width": 0.08,
            "mode": "control_mode_1",
        }
    )
    commands: List[Dict[str, Any]] = field(default_factory=list)
    blackboard: Any = field(init=False, default=None)

    def connect(self) -> Dict[str, Any]:
        cmd = {"type": "connect"}
        self.commands.append(cmd)
        self.status["connected"] = True
        return {"ok": True, "command": cmd}

    def initialize(self) -> Dict[str, Any]:
        cmd = {"type": "initialize"}
        self.commands.append(cmd)
        self.status["connected"] = True
        return {"ok": True, "command": cmd}

    def shutdown(self) -> Dict[str, Any]:
        cmd = {"type": "shutdown"}
        self.commands.append(cmd)
        self.status["connected"] = False
        return {"ok": True, "command": cmd}

    def move_j(self, joints: List[float], speed: float = 1.0) -> Dict[str, Any]:
        cmd = {"type": "move_j", "joints": list(joints), "speed": speed}
        self.commands.append(cmd)
        self.status["joint_positions"] = list(joints)
        return {"ok": True, "command": cmd}

    def move_l(self, pose: List[float], speed: float = 1.0) -> Dict[str, Any]:
        cmd = {"type": "move_l", "pose": list(pose), "speed": speed}
        self.commands.append(cmd)
        self.status["eef_pose"] = list(pose)
        return {"ok": True, "command": cmd}

    def servo_delta(self, delta_pose: List[float]) -> Dict[str, Any]:
        cmd = {"type": "servo_delta", "delta_pose": list(delta_pose)}
        self.commands.append(cmd)
        current = list(self.status.get("eef_pose", []))
        if current:
            for index, value in enumerate(delta_pose):
                if index >= len(current):
                    break
                current[index] += value
            self.status["eef_pose"] = current
        return {"ok": True, "command": cmd}

    def stop(self) -> Dict[str, Any]:
        cmd = {"type": "stop"}
        self.commands.append(cmd)
        return {"ok": True, "command": cmd}

    def open_gripper(self, width: Optional[float] = None) -> Dict[str, Any]:
        cmd = {"type": "open_gripper", "width": width}
        self.commands.append(cmd)
        if width is not None:
            self.status["gripper_width"] = width
        return {"ok": True, "command": cmd}

    def close_gripper(self, width: Optional[float] = None, guarded: bool = False) -> Dict[str, Any]:
        cmd = {"type": "close_gripper", "width": width, "guarded": guarded}
        self.commands.append(cmd)
        if width is not None:
            self.status["gripper_width"] = width
        return {"ok": True, "command": cmd}

    def get_status(self) -> Dict[str, Any]:
        return dict(self.status)

    def refresh_world(self, blackboard: Any) -> Dict[str, Any]:
        status = self.get_status()
        blackboard.update_world(
            robot={
                "joint_positions": status.get("joint_positions", []),
                "eef_pose": status.get("eef_pose", []),
                "gripper_width": status.get("gripper_width", 0.0),
                "fault_flag": status.get("fault", False),
                "mode": status.get("mode", "mock"),
            }
        )
        return status

    def execute_grasp_phase(self, target_pose: List[float], context: Any | None = None) -> Dict[str, Any]:
        descend = self.move_l(target_pose, speed=0.2)
        if not descend.get("ok", False):
            return descend
        close = self.close_gripper(guarded=True)
        if not close.get("ok", False):
            return close
        self.status["gripper_width"] = 0.0
        return {
            "ok": True,
            "action": "execute_grasp_phase",
            "command": {"type": "execute_grasp_phase", "target_pose": list(target_pose)},
            "substeps": [descend, close],
        }

    def attach_blackboard(self, blackboard: Any) -> None:
        self.blackboard = blackboard

    def evaluate_task_success(self) -> Dict[str, Any]:
        return {"ok": True, "success": True, "mode": "mock"}
