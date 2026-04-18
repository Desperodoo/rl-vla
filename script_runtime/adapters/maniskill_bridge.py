"""Optional ManiSkill bridge for SDK-free script-runtime validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from script_runtime.adapters.sdk_bridge import SDKBridge


def _quat_xyzw_to_euler_xyz(quat_xyzw: List[float]) -> List[float]:
    x, y, z, w = [float(v) for v in quat_xyzw]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return [float(roll), float(pitch), float(yaw)]


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _quat_xyzw_conjugate(quat_xyzw: List[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    return np.asarray([-x, -y, -z, w], dtype=np.float64)


def _quat_xyzw_multiply(lhs: List[float] | np.ndarray, rhs: List[float] | np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = [float(v) for v in lhs]
    rx, ry, rz, rw = [float(v) for v in rhs]
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _quat_xyzw_rotate(quat_xyzw: List[float] | np.ndarray, vec_xyz: List[float] | np.ndarray) -> np.ndarray:
    vec_quat = np.asarray([float(vec_xyz[0]), float(vec_xyz[1]), float(vec_xyz[2]), 0.0], dtype=np.float64)
    quat = np.asarray([float(v) for v in quat_xyzw], dtype=np.float64)
    rotated = _quat_xyzw_multiply(_quat_xyzw_multiply(quat, vec_quat), _quat_xyzw_conjugate(quat))
    return rotated[:3]


def _pose_to_list(pose_like: Any) -> Optional[List[float]]:
    if pose_like is None:
        return None
    if isinstance(pose_like, (list, tuple)) and len(pose_like) in (7, 8):
        return [float(v) for v in pose_like[:7]]

    pos = getattr(pose_like, "p", None)
    quat = getattr(pose_like, "q", None)
    if pos is None or quat is None:
        return None

    pos = np.asarray(pos).reshape(-1).tolist()
    quat = np.asarray(quat).reshape(-1).tolist()
    if len(quat) == 4:
        # ManiSkill Pose often exposes wxyz; runtime uses xyzw.
        quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]
    else:
        quat_xyzw = quat
    return [float(v) for v in (pos + quat_xyzw)]


@dataclass
class ManiSkillBridge(SDKBridge):
    """Bridge that maps runtime motion/gripper calls to ManiSkill actions.

    This is intentionally a validation bridge, not a production control stack.
    It is best used with privileged state observations so we can validate task
    logic before trusting perception or real hardware.
    """

    env_id: str = "PickCube-v1"
    obs_mode: str = "state"
    control_mode: str = "pd_ee_pose"
    render_mode: Optional[str] = None
    sim_backend: str = "physx_cpu"
    num_envs: int = 1
    max_episode_steps: Optional[int] = None
    env: Any = field(init=False, default=None)
    obs: Any = field(init=False, default=None)
    info: Dict[str, Any] = field(init=False, default_factory=dict)
    last_result: Dict[str, Any] = field(init=False, default_factory=dict)
    current_gripper_action: float = field(init=False, default=1.0)
    cached_reset_options: Dict[str, Any] = field(default_factory=dict)
    reset_seed: Optional[int] = None
    grasp_height_offset: float = 0.0
    pregrasp_height_offset: float = 0.08
    place_height_offset: float = 0.06
    pose_substeps: int = 60
    gripper_substeps: int = 15
    settle_substeps: int = 25
    position_tolerance: float = 0.01
    orientation_tolerance: float = 0.15
    guarded_close_descend_delta: float = 0.02
    guarded_close_descent_steps: int = 6
    guarded_close_hold_steps: int = 10
    lock_orientation: bool = True
    delta_translation_limit: float = 0.02
    delta_rotation_limit: float = 0.05
    grasp_phase_descend_segments: int = 5
    grasp_phase_close_hold_steps: int = 12
    grasp_phase_lift_delta: float = 0.03
    grasp_phase_lift_steps: int = 6
    place_release_height_offset: float = 0.015
    retreat_height_offset: float = 0.08
    capture_video: bool = False
    capture_every_n_steps: int = 2
    artifact_dir: Optional[str] = None
    episode_name: str = "episode"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    blackboard: Any = field(init=False, default=None)
    captured_frames: List[Any] = field(init=False, default_factory=list)
    debug_rows: List[Dict[str, Any]] = field(init=False, default_factory=list)
    step_counter: int = field(init=False, default=0)

    def connect(self) -> Dict[str, Any]:
        if self.env is None:
            import gymnasium as gym
            import mani_skill.envs  # noqa: F401

            kwargs = dict(
                obs_mode=self.obs_mode,
                control_mode=self.control_mode,
                render_mode=self.render_mode,
                sim_backend=self.sim_backend,
            )
            kwargs.update(dict(self.env_kwargs or {}))
            if self.num_envs is not None:
                kwargs["num_envs"] = self.num_envs
            if self.max_episode_steps is not None:
                kwargs["max_episode_steps"] = self.max_episode_steps
            self.env = gym.make(self.env_id, **kwargs)
        return {"ok": True, "action": "connect", "env_id": self.env_id}

    def initialize(self) -> Dict[str, Any]:
        self.connect()
        reset_kwargs: Dict[str, Any] = {}
        if self.cached_reset_options:
            reset_kwargs["options"] = self.cached_reset_options
        if self.reset_seed is not None:
            reset_kwargs["seed"] = self.reset_seed
        self.obs, self.info = self.env.reset(**reset_kwargs)
        self.current_gripper_action = self._infer_gripper_action()
        self.captured_frames = []
        self.debug_rows = []
        self.step_counter = 0
        self._capture_visual(command={"type": "reset"})
        return {"ok": True, "action": "initialize", "env_id": self.env_id}

    def shutdown(self) -> Dict[str, Any]:
        if self.env is not None:
            self.env.close()
        self.env = None
        return {"ok": True, "action": "shutdown"}

    def move_j(self, joints: List[float], speed: float = 1.0) -> Dict[str, Any]:
        # Validation fallback: we do not yet solve joint-space commands in sim,
        # so use the final task step as a settle/hold window for success checks.
        current_pose = self.get_status().get("eef_pose", [])
        result = self._step_repeated(
            current_pose,
            {"type": "move_j", "joints": list(joints), "speed": speed, "sim_fallback": True},
            steps=self.settle_substeps,
        )
        return result

    def move_l(self, pose: List[float], speed: float = 1.0) -> Dict[str, Any]:
        return self._step_pose_action({"type": "move_l", "pose": list(pose), "speed": speed}, target_pose=list(pose))

    def servo_delta(self, delta_pose: List[float]) -> Dict[str, Any]:
        current_pose = self.get_status().get("eef_pose", [])
        target = list(current_pose)
        for index, value in enumerate(delta_pose):
            if index >= len(target):
                break
            target[index] += float(value)
        return self._step_pose_action(
            {"type": "servo_delta", "delta_pose": list(delta_pose), "target_pose": target},
            target_pose=target,
        )

    def stop(self) -> Dict[str, Any]:
        return {"ok": True, "action": "stop", "command": {"type": "stop"}}

    def open_gripper(self, width: Optional[float] = None) -> Dict[str, Any]:
        self.current_gripper_action = 1.0
        current_pose = self.get_status().get("eef_pose", [])
        return self._step_repeated(current_pose, {"type": "open_gripper", "width": width}, steps=self.gripper_substeps)

    def close_gripper(self, width: Optional[float] = None, guarded: bool = False) -> Dict[str, Any]:
        self.current_gripper_action = -1.0
        current_pose = self.get_status().get("eef_pose", [])
        if guarded:
            return self._execute_guarded_close_sequence(
                current_pose=current_pose,
                command={"type": "close_gripper", "width": width, "guarded": guarded},
            )
        return self._step_repeated(current_pose, {"type": "close_gripper", "width": width, "guarded": guarded}, steps=self.gripper_substeps)

    def execute_grasp_phase(self, target_pose: List[float], context: Any | None = None) -> Dict[str, Any]:
        current_pose = self.get_status().get("eef_pose", [])
        if len(current_pose) < 7:
            return {"ok": False, "action": "execute_grasp_phase", "message": "Current ee pose unavailable"}

        start_pose = list(current_pose)
        descend_results: List[Dict[str, Any]] = []
        for step_index in range(max(1, self.grasp_phase_descend_segments)):
            alpha = float(step_index + 1) / float(max(1, self.grasp_phase_descend_segments))
            live_target_pose = self._get_live_grasp_pose(target_pose)
            intermediate = [
                float(start_pose[i] + (live_target_pose[i] - start_pose[i]) * alpha)
                for i in range(3)
            ] + list(live_target_pose[3:7])
            result = self._step_pose_action({"type": "grasp_descend", "pose": intermediate}, target_pose=intermediate)
            descend_results.append(result)
            if self.is_grasped() or self._episode_ended():
                return {
                    "ok": self.is_grasped(),
                    "action": "execute_grasp_phase",
                    "command": {"type": "execute_grasp_phase", "target_pose": list(target_pose)},
                    "substeps": descend_results,
                }

        self.current_gripper_action = -1.0
        settle_pose = self._get_live_grasp_pose(target_pose)
        close_result = self._step_repeated(
            settle_pose,
            {"type": "grasp_close_hold", "pose": settle_pose},
            steps=self.grasp_phase_close_hold_steps,
        )
        if self.is_grasped():
            lift_pose = list(self.get_status().get("eef_pose", settle_pose))
            lift_pose[2] += self.grasp_phase_lift_delta
            lift_result = self._step_repeated(
                lift_pose,
                {"type": "post_grasp_lift", "pose": lift_pose},
                steps=self.grasp_phase_lift_steps,
            )
            return {
                "ok": True,
                "action": "execute_grasp_phase",
                "command": {"type": "execute_grasp_phase", "target_pose": list(target_pose)},
                "substeps": [*descend_results, close_result, lift_result],
            }
        return {
            "ok": False,
            "action": "execute_grasp_phase",
            "command": {"type": "execute_grasp_phase", "target_pose": list(target_pose)},
            "substeps": [*descend_results, close_result],
            "message": "Grasp phase completed without confirmed grasp",
        }

    def get_status(self) -> Dict[str, Any]:
        if self.env is None:
            return {"ok": False, "connected": False, "fault": True, "message": "ManiSkill env not initialized"}

        tcp_pose = self._get_tcp_pose()
        qpos = self._get_qpos()
        gripper_width = 0.04 * (self.current_gripper_action + 1.0) / 2.0
        return {
            "ok": True,
            "connected": True,
            "fault": False,
            "joint_positions": qpos,
            "eef_pose": tcp_pose,
            "gripper_width": gripper_width,
            "mode": self.control_mode,
            "success": self._extract_success_flag(),
            "is_grasped": self.is_grasped(),
        }

    def refresh_world(self, blackboard: Any) -> Dict[str, Any]:
        self.blackboard = blackboard
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
            },
            scene={"grasped": self.is_grasped()},
        )
        return status

    def attach_blackboard(self, blackboard: Any) -> None:
        self.blackboard = blackboard

    def get_object_pose(self) -> Optional[List[float]]:
        extra = self._get_extra_dict()
        pose_keys = ("cubeA_pose", "cube_pose", "obj_pose", "object_pose")
        for key in pose_keys:
            if key in extra:
                value = np.asarray(extra[key]).reshape(-1).tolist()
                if len(value) >= 7:
                    return self._world_pose_to_base([float(v) for v in value[:7]])

        env_unwrapped = self.env.unwrapped
        for attr_name in ("cubeA", "cube", "obj", "object"):
            actor = getattr(env_unwrapped, attr_name, None)
            pose = _pose_to_list(getattr(actor, "pose", None))
            if pose is not None:
                return self._world_pose_to_base(pose)
        return None

    def get_support_pose(self) -> Optional[List[float]]:
        extra = self._get_extra_dict()
        for key in ("cubeB_pose", "support_pose", "container_pose", "bin_pose"):
            if key in extra:
                value = np.asarray(extra[key]).reshape(-1).tolist()
                if len(value) >= 7:
                    return self._world_pose_to_base([float(v) for v in value[:7]])
        env_unwrapped = self.env.unwrapped
        for attr_name in ("cubeB", "container", "bin", "goal_site"):
            actor = getattr(env_unwrapped, attr_name, None)
            pose = _pose_to_list(getattr(actor, "pose", None) or actor)
            if pose is not None:
                return self._world_pose_to_base(pose)
        return None

    def get_place_pose(self) -> Optional[List[float]]:
        extra = self._get_extra_dict()
        for key in ("goal_pose", "goal_pos", "target_pose"):
            if key not in extra:
                continue
            value = np.asarray(extra[key]).reshape(-1).tolist()
            if len(value) == 3:
                target_pos = self._world_pos_to_base([float(value[0]), float(value[1]), float(value[2])])
                tcp_pose = self.get_status().get("eef_pose", [])
                quat = tcp_pose[3:7] if len(tcp_pose) >= 7 else [0.0, 0.0, 0.0, 1.0]
                return [
                    float(target_pos[0]),
                    float(target_pos[1]),
                    float(target_pos[2] + self.place_height_offset),
                    *[float(v) for v in quat],
                ]
            if len(value) >= 7:
                pose = self._world_pose_to_base([float(v) for v in value[:7]])
                pose[2] += self.place_height_offset
                return pose
        env_unwrapped = self.env.unwrapped
        for attr_name in ("goal_site", "goal_region", "goal_pose"):
            goal = getattr(env_unwrapped, attr_name, None)
            pose = _pose_to_list(getattr(goal, "pose", goal))
            if pose is not None:
                pose = self._world_pose_to_base(pose)
                pose[2] += self.place_height_offset
                return pose
        support_pose = self.get_support_pose()
        if support_pose is not None:
            pose = list(support_pose)
            cube_half_size = getattr(self.env.unwrapped, "cube_half_size", None) if self.env is not None else None
            if cube_half_size is not None:
                cube_half_size = float(np.asarray(cube_half_size).reshape(-1)[0])
            else:
                cube_half_size = 0.02
            pose[2] += cube_half_size * 2 + self.place_height_offset
            tcp_pose = self.get_status().get("eef_pose", [])
            if len(tcp_pose) >= 7:
                pose[3:7] = tcp_pose[3:7]
            return pose
        return None

    def get_place_release_pose(self) -> Optional[List[float]]:
        extra = self._get_extra_dict()
        for key in ("goal_pose", "goal_pos", "target_pose"):
            if key not in extra:
                continue
            value = np.asarray(extra[key]).reshape(-1).tolist()
            if len(value) == 3:
                target_pos = self._world_pos_to_base([float(value[0]), float(value[1]), float(value[2])])
                tcp_pose = self.get_status().get("eef_pose", [])
                quat = tcp_pose[3:7] if len(tcp_pose) >= 7 else [0.0, 0.0, 0.0, 1.0]
                return [
                    float(target_pos[0]),
                    float(target_pos[1]),
                    float(target_pos[2] + self.place_release_height_offset),
                    *[float(v) for v in quat],
                ]
            if len(value) >= 7:
                pose = self._world_pose_to_base([float(v) for v in value[:7]])
                pose[2] += self.place_release_height_offset
                return pose
        env_unwrapped = self.env.unwrapped
        for attr_name in ("goal_site", "goal_region", "goal_pose"):
            goal = getattr(env_unwrapped, attr_name, None)
            pose = _pose_to_list(getattr(goal, "pose", goal))
            if pose is not None:
                pose = self._world_pose_to_base(pose)
                pose[2] += self.place_release_height_offset
                return pose
        support_pose = self.get_support_pose()
        if support_pose is not None:
            pose = list(support_pose)
            cube_half_size = getattr(self.env.unwrapped, "cube_half_size", None) if self.env is not None else None
            if cube_half_size is not None:
                cube_half_size = float(np.asarray(cube_half_size).reshape(-1)[0])
            else:
                cube_half_size = 0.02
            pose[2] += cube_half_size * 2 + self.place_release_height_offset
            tcp_pose = self.get_status().get("eef_pose", [])
            if len(tcp_pose) >= 7:
                pose[3:7] = tcp_pose[3:7]
            return pose
        return None

    def get_retreat_pose(self) -> Optional[List[float]]:
        release_pose = self.get_place_release_pose()
        if release_pose is None:
            return None
        retreat_pose = list(release_pose)
        retreat_pose[2] += self.retreat_height_offset
        return retreat_pose

    def get_grasp_candidates(self) -> List[Dict[str, Any]]:
        object_pose = self.get_object_pose()
        if object_pose is None:
            return []
        tcp_pose = self.get_status().get("eef_pose", [])
        orientation = tcp_pose[3:7] if len(tcp_pose) >= 7 else [0.0, 0.0, 0.0, 1.0]
        height_offset = self.grasp_height_offset
        cube_half_size = getattr(self.env.unwrapped, "cube_half_size", None) if self.env is not None else None
        if cube_half_size is not None and abs(height_offset) < 1e-6:
            height_offset = 0.0
        candidates: List[Dict[str, Any]] = []
        xy_offsets = [(0.0, 0.0)]
        if self.get_support_pose() is not None:
            xy_offsets.extend([(0.004, 0.0), (-0.004, 0.0), (0.0, 0.004), (0.0, -0.004)])
        z_offsets = [0.0]
        if self.get_support_pose() is not None:
            z_offsets.append(-0.003)
        score = 1.0
        for dx, dy in xy_offsets:
            for dz in z_offsets:
                grasp_pose = [
                    float(object_pose[0] + dx),
                    float(object_pose[1] + dy),
                    float(object_pose[2] + height_offset + dz),
                    *[float(v) for v in orientation],
                ]
                pregrasp_pose = list(grasp_pose)
                pregrasp_pose[2] += self.pregrasp_height_offset
                candidates.append(
                    {
                        "pose": grasp_pose,
                        "pregrasp_pose": pregrasp_pose,
                        "score": score,
                        "source": "oracle_state",
                    }
                )
                score -= 0.05
        return candidates

    def is_grasped(self) -> bool:
        if self.env is None:
            return False
        info = self.info or {}
        for key in ("is_grasped", "is_grasping", "is_cubeA_grasped"):
            if key in info:
                value = np.asarray(info[key]).reshape(-1)[0]
                return bool(value)

        env_unwrapped = self.env.unwrapped
        obj = (
            getattr(env_unwrapped, "cubeA", None)
            or getattr(env_unwrapped, "cube", None)
            or getattr(env_unwrapped, "obj", None)
        )
        agent = getattr(env_unwrapped, "agent", None)
        if obj is not None and agent is not None and hasattr(agent, "is_grasping"):
            try:
                return bool(agent.is_grasping(obj))
            except Exception:
                return False
        return False

    def _step(self, action: np.ndarray, command: Dict[str, Any]) -> Dict[str, Any]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.info = info if isinstance(info, dict) else {}
        self.step_counter += 1
        self.last_result = {
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }
        self._capture_visual(command=command)
        return {"ok": True, "action": "env.step", "command": command, **self.last_result}

    def _step_repeated(self, target_pose: List[float], command: Dict[str, Any], steps: int) -> Dict[str, Any]:
        result = None
        for _ in range(max(1, steps)):
            action = self._compose_control_action(target_pose, self.current_gripper_action)
            result = self._step(action, command)
            if self._episode_ended() or self.is_grasped():
                break
        return result or {"ok": False, "action": "env.step", "command": command}

    def _step_pose_action(self, command: Dict[str, Any], target_pose: List[float]) -> Dict[str, Any]:
        result = None
        for _ in range(max(1, self.pose_substeps)):
            action = self._compose_control_action(target_pose, self.current_gripper_action)
            result = self._step(action, command)
            if self._pose_reached(target_pose) or self._episode_ended():
                break
        return result or {"ok": False, "action": "env.step", "command": command}

    def _execute_guarded_close_sequence(self, current_pose: List[float], command: Dict[str, Any]) -> Dict[str, Any]:
        if len(current_pose) < 7:
            return self._step_repeated(current_pose, command, steps=self.gripper_substeps)

        result = None
        start_pose = list(current_pose)
        for step_index in range(max(1, self.guarded_close_descent_steps)):
            fraction = float(step_index + 1) / float(max(1, self.guarded_close_descent_steps))
            target_pose = list(start_pose)
            target_pose[2] = start_pose[2] - self.guarded_close_descend_delta * fraction
            action = self._compose_control_action(target_pose, self.current_gripper_action)
            result = self._step(action, command)
            if self.is_grasped() or self._episode_ended():
                return result

        settle_pose = list(self.get_status().get("eef_pose", start_pose))
        return self._step_repeated(settle_pose, command, steps=self.guarded_close_hold_steps)

    def _episode_ended(self) -> bool:
        if not self.last_result:
            return False
        terminated = self.last_result.get("terminated")
        truncated = self.last_result.get("truncated")
        return self._tensor_flag(terminated) or self._tensor_flag(truncated)

    def _pose_reached(self, target_pose: List[float]) -> bool:
        current = self.get_status().get("eef_pose", [])
        if len(current) < 7 or len(target_pose) < 7:
            return False
        pos_err = np.linalg.norm(np.asarray(current[:3]) - np.asarray(target_pose[:3]))
        if self.lock_orientation:
            return pos_err <= self.position_tolerance
        quat_dot = abs(float(np.dot(np.asarray(current[3:7]), np.asarray(target_pose[3:7]))))
        quat_dot = min(1.0, max(-1.0, quat_dot))
        ang_err = 2.0 * np.arccos(quat_dot)
        return pos_err <= self.position_tolerance and ang_err <= self.orientation_tolerance

    @staticmethod
    def _tensor_flag(value: Any) -> bool:
        if value is None:
            return False
        array = np.asarray(value).reshape(-1)
        if array.size == 0:
            return False
        return bool(array[0])

    def _compose_control_action(self, pose: List[float], gripper_action: float) -> np.ndarray:
        if "delta" in self.control_mode:
            return self._compose_pd_ee_delta_pose_action(pose, gripper_action)
        return self._compose_pd_ee_pose_action(pose, gripper_action)

    def _compose_pd_ee_pose_action(self, pose: List[float], gripper_action: float) -> np.ndarray:
        pose = list(pose)
        if len(pose) < 7:
            raise ValueError(f"Expected pose with 7 dims [xyz quat_xyzw], got {pose}")
        xyz = pose[:3]
        quat = pose[3:7]
        if self.lock_orientation:
            current_pose = self.get_status().get("eef_pose", [])
            if len(current_pose) >= 7:
                quat = current_pose[3:7]
        euler = _quat_xyzw_to_euler_xyz(quat)
        action = np.asarray(xyz + euler + [float(gripper_action)], dtype=np.float32)
        if getattr(self.env.action_space, "shape", None) and self.env.action_space.shape[-1] != action.shape[-1]:
            raise ValueError(
                f"ManiSkill action dim mismatch: env expects {self.env.action_space.shape[-1]}, "
                f"but runtime composed {action.shape[-1]} dims"
            )
        return action

    def _compose_pd_ee_delta_pose_action(self, target_pose: List[float], gripper_action: float) -> np.ndarray:
        target_pose = list(target_pose)
        current_pose = self.get_status().get("eef_pose", [])
        if len(target_pose) < 7 or len(current_pose) < 7:
            raise ValueError(f"Expected pose with 7 dims [xyz quat_xyzw], got target={target_pose}, current={current_pose}")

        delta_pos = [
            float(target_pose[i] - current_pose[i]) / self.delta_translation_limit
            for i in range(3)
        ]
        if self.lock_orientation:
            delta_euler = [0.0, 0.0, 0.0]
        else:
            current_euler = _quat_xyzw_to_euler_xyz(current_pose[3:7])
            target_euler = _quat_xyzw_to_euler_xyz(target_pose[3:7])
            delta_euler = [
                _wrap_angle(float(target_euler[i] - current_euler[i])) / self.delta_rotation_limit
                for i in range(3)
            ]
        action = np.asarray(delta_pos + delta_euler + [float(gripper_action)], dtype=np.float32)
        action[:6] = np.clip(action[:6], -1.0, 1.0)
        action[6] = float(np.clip(action[6], -1.0, 1.0))
        return action

    def _get_qpos(self) -> List[float]:
        robot = self.env.unwrapped.agent.robot
        qpos = robot.get_qpos()
        return np.asarray(qpos).reshape(-1).astype(np.float32).tolist()

    def _get_tcp_pose(self) -> List[float]:
        ctrl = self.env.unwrapped.agent.controller.controllers["arm"]
        pos = np.asarray(ctrl.ee_pose_at_base.p).reshape(-1).tolist()
        quat_wxyz = np.asarray(ctrl.ee_pose_at_base.q).reshape(-1).tolist()
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        return [float(v) for v in (pos + quat_xyzw)]

    def _infer_gripper_action(self) -> float:
        qpos = self._get_qpos()
        if len(qpos) < 8:
            return 1.0
        raw = float(qpos[7])
        return float(np.clip((raw / 0.04) * 2.0 - 1.0, -1.0, 1.0))

    def _get_extra_dict(self) -> Dict[str, Any]:
        if isinstance(self.obs, dict):
            extra = self.obs.get("extra", {})
            if isinstance(extra, dict):
                return extra
        return {}

    def _get_live_grasp_pose(self, fallback_pose: List[float]) -> List[float]:
        object_pose = self.get_object_pose()
        if object_pose is None:
            return list(fallback_pose)
        current_pose = self.get_status().get("eef_pose", [])
        quat = current_pose[3:7] if len(current_pose) >= 7 else fallback_pose[3:7]
        return [
            float(object_pose[0]),
            float(object_pose[1]),
            float(object_pose[2] + self.grasp_height_offset),
            *[float(v) for v in quat],
        ]

    def _extract_success_flag(self) -> bool:
        info = self.info or {}
        for key in ("success", "success_once", "env_success"):
            if key in info:
                value = np.asarray(info[key]).reshape(-1)[0]
                return bool(value)
        return False

    def settle(self, steps: Optional[int] = None) -> Dict[str, Any]:
        current_pose = self.get_status().get("eef_pose", [])
        result = self._step_repeated(
            current_pose,
            {"type": "settle", "pose": current_pose},
            steps=steps if steps is not None else self.settle_substeps,
        )
        return result

    def evaluate_task_success(self) -> Dict[str, Any]:
        status = self.get_status()
        return {
            "ok": True,
            "success": bool(status.get("success", False)),
            "is_grasped": bool(status.get("is_grasped", False)),
            "eef_pose": status.get("eef_pose", []),
            "object_pose": self.get_object_pose(),
            "place_pose": self.get_place_release_pose(),
        }

    def should_release_object(self) -> bool:
        if self.get_support_pose() is not None:
            return True
        release_pose = self.get_place_release_pose()
        if release_pose is None:
            return True
        if self.env is not None:
            raw = getattr(self.env.unwrapped, "cube_half_size", 0.02)
            cube_half_size = float(np.asarray(raw).reshape(-1)[0])
        else:
            cube_half_size = 0.02
        return float(release_pose[2]) <= cube_half_size + 0.035

    def export_episode_artifacts(self, task_id: str, output_dir: Any | None = None) -> Dict[str, Any]:
        out_dir = Path(output_dir or self.artifact_dir or "script_runtime/artifacts/maniskill_media")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = self.episode_name or task_id or "episode"
        outputs: Dict[str, Any] = {}

        if self.debug_rows:
            debug_path = out_dir / f"{stem}_grounding.json"
            debug_path.write_text(json.dumps(self.debug_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            outputs["grounding_json"] = str(debug_path)

        if self.captured_frames:
            gif_path = out_dir / f"{stem}_rollout.gif"
            try:
                from PIL import Image

                pil_frames = [Image.fromarray(frame) for frame in self.captured_frames]
                pil_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=90,
                    loop=0,
                )
                outputs["rollout_gif"] = str(gif_path)
            except Exception:
                pass

            topdown_path = out_dir / f"{stem}_grounding_topdown.png"
            image = self._render_topdown_summary()
            if image is not None:
                image.save(topdown_path)
                outputs["grounding_topdown_png"] = str(topdown_path)
        return outputs

    def _get_robot_base_pose_world(self) -> Optional[List[float]]:
        if self.env is None:
            return None
        robot_pose = getattr(self.env.unwrapped.agent.robot, "pose", None)
        return _pose_to_list(robot_pose)

    def _world_pos_to_base(self, world_pos: List[float]) -> List[float]:
        base_pose = self._get_robot_base_pose_world()
        if base_pose is None:
            return [float(v) for v in world_pos]
        base_pos = np.asarray(base_pose[:3], dtype=np.float64)
        base_quat = np.asarray(base_pose[3:7], dtype=np.float64)
        rel = np.asarray(world_pos, dtype=np.float64) - base_pos
        rel_base = _quat_xyzw_rotate(_quat_xyzw_conjugate(base_quat), rel)
        return [float(v) for v in rel_base]

    def _world_pose_to_base(self, world_pose: List[float]) -> List[float]:
        base_pose = self._get_robot_base_pose_world()
        if base_pose is None:
            return [float(v) for v in world_pose]
        pos_base = self._world_pos_to_base(world_pose[:3])
        base_quat = np.asarray(base_pose[3:7], dtype=np.float64)
        world_quat = np.asarray(world_pose[3:7], dtype=np.float64)
        quat_base = _quat_xyzw_multiply(_quat_xyzw_conjugate(base_quat), world_quat)
        return [*pos_base, *[float(v) for v in quat_base]]

    def _capture_visual(self, command: Dict[str, Any]) -> None:
        if not self.capture_video:
            return
        if self.render_mode != "rgb_array":
            return
        if self.step_counter % max(1, self.capture_every_n_steps) != 0 and command.get("type") != "reset":
            return
        try:
            frame = self.env.render()
        except Exception:
            return
        if frame is None:
            return
        if hasattr(frame, "detach"):
            frame = frame.detach()
        if hasattr(frame, "cpu"):
            frame = frame.cpu()
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        frame_array = np.asarray(frame)
        if frame_array.ndim == 4:
            frame_array = frame_array[0]
        annotated = self._annotate_frame(frame_array.astype(np.uint8), command)
        if annotated is not None:
            self.captured_frames.append(np.asarray(annotated, dtype=np.uint8))

    def _annotate_frame(self, frame: np.ndarray, command: Dict[str, Any]):
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return frame
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        panel_width = 360
        canvas = Image.new("RGB", (image.width + panel_width, image.height), (248, 249, 251))
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)

        current_skill = ""
        if self.blackboard is not None:
            current_skill = self.blackboard.world_state.execution.current_skill
        tcp_pose = self.get_status().get("eef_pose", [])
        object_pose = self.get_object_pose()
        support_pose = self.get_support_pose()
        goal_pose = self.get_place_release_pose() or self.get_place_pose()
        grasp_pose = None
        if self.blackboard is not None:
            grasp_pose = self.blackboard.get("active_grasp_pose")

        lines = [
            f"step: {self.step_counter}",
            f"skill: {current_skill or 'idle'}",
            f"command: {command.get('type', 'unknown')}",
            f"success: {self._extract_success_flag()}",
            f"is_grasped: {self.is_grasped()}",
            f"tcp: {self._format_pose(tcp_pose)}",
            f"object: {self._format_pose(object_pose)}",
            f"goal: {self._format_pose(goal_pose)}",
            f"grasp: {self._format_pose(grasp_pose)}",
            f"support: {self._format_pose(support_pose)}",
        ]
        x0 = image.width + 16
        y = 16
        for line in lines:
            draw.text((x0, y), line, fill=(25, 32, 44))
            y += 18

        top_left = (image.width + 16, max(y + 8, 180))
        box_w = panel_width - 32
        box_h = min(260, image.height - top_left[1] - 16)
        draw.rectangle([top_left, (top_left[0] + box_w, top_left[1] + box_h)], outline=(120, 130, 140), width=2)
        self._draw_topdown(draw, top_left, box_w, box_h, tcp_pose, object_pose, goal_pose, grasp_pose)
        self._draw_projected_markers(
            draw,
            image.width,
            image.height,
            tcp_pose=tcp_pose,
            object_pose=object_pose,
            goal_pose=goal_pose,
            grasp_pose=grasp_pose,
            support_pose=support_pose,
        )

        self.debug_rows.append(
            {
                "step": self.step_counter,
                "skill": current_skill,
                "command": dict(command),
                "success": self._extract_success_flag(),
                "is_grasped": self.is_grasped(),
                "tcp_pose": tcp_pose,
                "object_pose": object_pose,
                "goal_pose": goal_pose,
                "grasp_pose": grasp_pose,
                "support_pose": support_pose,
            }
        )
        return canvas

    def _draw_topdown(
        self,
        draw: Any,
        top_left: tuple[int, int],
        width: int,
        height: int,
        tcp_pose: Optional[List[float]],
        object_pose: Optional[List[float]],
        goal_pose: Optional[List[float]],
        grasp_pose: Optional[List[float]],
    ) -> None:
        x0, y0 = top_left
        x_min, x_max = -0.2, 0.7
        y_min, y_max = -0.35, 0.35

        def project(pose: Optional[List[float]]) -> Optional[tuple[int, int]]:
            if pose is None or len(pose) < 2:
                return None
            px = int(x0 + ((float(pose[0]) - x_min) / (x_max - x_min)) * width)
            py = int(y0 + height - ((float(pose[1]) - y_min) / (y_max - y_min)) * height)
            return px, py

        def dot(pose: Optional[List[float]], color: tuple[int, int, int], label: str) -> None:
            point = project(pose)
            if point is None:
                return
            px, py = point
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color, outline=(0, 0, 0))
            draw.text((px + 8, py - 8), label, fill=color)

        dot(goal_pose, (46, 125, 50), "goal")
        dot(object_pose, (198, 40, 40), "obj")
        dot(tcp_pose, (21, 101, 192), "tcp")
        dot(grasp_pose, (245, 124, 0), "grasp")
        support_pose = self.get_support_pose()
        dot(support_pose, (76, 175, 80), "support")

    @staticmethod
    def _format_pose(pose: Optional[List[float]]) -> str:
        if pose is None or len(pose) < 3:
            return "n/a"
        return f"[{pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f}]"

    def _draw_projected_markers(
        self,
        draw: Any,
        image_width: int,
        image_height: int,
        tcp_pose: Optional[List[float]],
        object_pose: Optional[List[float]],
        goal_pose: Optional[List[float]],
        grasp_pose: Optional[List[float]],
        support_pose: Optional[List[float]],
    ) -> None:
        camera = self._get_render_camera()
        if camera is None:
            return
        for pose, color, label, radius in (
            (goal_pose, (46, 204, 64), "goal", 8),
            (support_pose, (102, 187, 106), "support", 6),
            (object_pose, (244, 67, 54), "obj", 7),
            (grasp_pose, (255, 152, 0), "grasp", 6),
            (tcp_pose, (33, 150, 243), "tcp", 7),
        ):
            point = self._project_pose_to_render_image(pose, camera)
            if point is None:
                continue
            px, py = point
            if not (0 <= px < image_width and 0 <= py < image_height):
                continue
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=color, width=3)
            draw.text((px + radius + 3, py - radius - 3), label, fill=color)

    def _get_render_camera(self) -> Any:
        if self.env is None:
            return None
        env = self.env.unwrapped
        cameras = getattr(env, "_human_render_cameras", {})
        if "render_camera" in cameras:
            return cameras["render_camera"]
        if cameras:
            return next(iter(cameras.values()))
        return None

    def _project_pose_to_render_image(self, pose: Optional[List[float]], camera: Any) -> Optional[tuple[int, int]]:
        if pose is None or len(pose) < 3:
            return None
        try:
            params = camera.get_params()
            intrinsic = self._to_numpy(params["intrinsic_cv"])
            extrinsic = self._to_numpy(params["extrinsic_cv"])
        except Exception:
            return None
        if intrinsic.ndim == 3:
            intrinsic = intrinsic[0]
        if extrinsic.ndim == 3:
            extrinsic = extrinsic[0]
        point_world = np.asarray([pose[0], pose[1], pose[2], 1.0], dtype=np.float64)
        point_cam = extrinsic @ point_world
        if point_cam.shape[0] < 3:
            return None
        x_c, y_c, z_c = point_cam[:3]
        if z_c <= 1e-6:
            return None
        uvw = intrinsic @ np.asarray([x_c, y_c, z_c], dtype=np.float64)
        return int(round(float(uvw[0] / uvw[2]))), int(round(float(uvw[1] / uvw[2])))

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def _render_topdown_summary(self):
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return None
        image = Image.new("RGB", (640, 480), (250, 250, 252))
        draw = ImageDraw.Draw(image)
        draw.text((20, 16), f"Episode: {self.episode_name}", fill=(20, 20, 24))
        self._draw_topdown(draw, (20, 60), 600, 380, None, None, None, None)

        def project(pose: Optional[List[float]]) -> Optional[tuple[int, int]]:
            if pose is None or len(pose) < 2:
                return None
            x_min, x_max = -0.2, 0.7
            y_min, y_max = -0.35, 0.35
            px = int(20 + ((float(pose[0]) - x_min) / (x_max - x_min)) * 600)
            py = int(60 + 380 - ((float(pose[1]) - y_min) / (y_max - y_min)) * 380)
            return px, py

        tcp_points = [project(row.get("tcp_pose")) for row in self.debug_rows]
        tcp_points = [point for point in tcp_points if point is not None]
        if len(tcp_points) >= 2:
            draw.line(tcp_points, fill=(21, 101, 192), width=3)
        for row in self.debug_rows[:: max(1, len(self.debug_rows) // 6)]:
            point = project(row.get("object_pose"))
            if point is not None:
                draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=(198, 40, 40))
        if self.debug_rows:
            final = self.debug_rows[-1]
            for pose, color, label in (
                (final.get("goal_pose"), (46, 125, 50), "goal"),
                (final.get("object_pose"), (198, 40, 40), "obj"),
                (final.get("tcp_pose"), (21, 101, 192), "tcp"),
            ):
                point = project(pose)
                if point is not None:
                    draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=color)
                    draw.text((point[0] + 8, point[1] - 8), label, fill=color)
        return image
