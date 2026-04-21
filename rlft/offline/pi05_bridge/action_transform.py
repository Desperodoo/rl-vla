from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation as R

ActionRepresentation = Literal["ee_delta_pose_gripper", "absolute_pose_gripper"]


@dataclass(frozen=True)
class Pi05ActionRepresentationSpec:
    representation: ActionRepresentation
    target_dim: int
    pose_slice: tuple[int, int]
    gripper_index: int
    rotation_mode: Literal["rotvec", "quat"]
    description: str


_ACTION_REPRESENTATION_SPECS: dict[ActionRepresentation, Pi05ActionRepresentationSpec] = {
    "ee_delta_pose_gripper": Pi05ActionRepresentationSpec(
        representation="ee_delta_pose_gripper",
        target_dim=7,
        pose_slice=(0, 6),
        gripper_index=6,
        rotation_mode="rotvec",
        description="relative end-effector delta pose [dx, dy, dz, d_rx, d_ry, d_rz] + gripper",
    ),
    "absolute_pose_gripper": Pi05ActionRepresentationSpec(
        representation="absolute_pose_gripper",
        target_dim=8,
        pose_slice=(0, 7),
        gripper_index=7,
        rotation_mode="quat",
        description="absolute end-effector target pose [x, y, z, qx, qy, qz, qw] + gripper",
    ),
}


def _pose_to_transform_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    rotation = R.from_quat(quaternion).as_matrix()
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation.astype(np.float32)
    transform[:3, 3] = position.astype(np.float32)
    return transform


def _compute_relative_pose_transform(pose_current: np.ndarray, pose_target: np.ndarray) -> np.ndarray:
    t_current = _pose_to_transform_matrix(pose_current[:3], pose_current[3:7])
    t_target = _pose_to_transform_matrix(pose_target[:3], pose_target[3:7])
    t_relative = np.linalg.inv(t_current) @ t_target
    position = t_relative[:3, 3].astype(np.float32)
    quaternion = R.from_matrix(t_relative[:3, :3]).as_quat().astype(np.float32)
    return np.concatenate([position, quaternion], axis=0)


def get_pi05_action_representation_spec(representation: ActionRepresentation) -> Pi05ActionRepresentationSpec:
    try:
        return _ACTION_REPRESENTATION_SPECS[representation]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Unsupported pi05 action representation '{representation}'. "
            f"Expected one of: {sorted(_ACTION_REPRESENTATION_SPECS)}"
        ) from exc


def infer_carm_raw_action_layout(action_dim: int) -> tuple[slice, int]:
    """Infer raw CARM action pose/gripper indices from stored action dimensionality."""
    if action_dim == 8:
        return slice(0, 7), 7
    if action_dim >= 15:
        return slice(7, 14), 14
    raise ValueError(f"Unsupported CARM raw action dim: {action_dim}")


def _absolute_pose_with_gripper(raw_action: np.ndarray) -> tuple[np.ndarray, np.float32]:
    pose_slice, gripper_index = infer_carm_raw_action_layout(raw_action.shape[-1])
    target_pose = np.asarray(raw_action[pose_slice], dtype=np.float32)
    gripper = np.float32(raw_action[gripper_index])
    return target_pose, gripper


def transform_carm_raw_action(
    raw_action: np.ndarray,
    ref_ee_pose: np.ndarray,
    representation: ActionRepresentation,
) -> np.ndarray:
    """Transform a stored CARM action into the bridge action contract."""
    raw_action = np.asarray(raw_action, dtype=np.float32)
    ref_ee_pose = np.asarray(ref_ee_pose, dtype=np.float32)[:7]
    target_pose, gripper = _absolute_pose_with_gripper(raw_action)

    if representation == "absolute_pose_gripper":
        return np.concatenate([target_pose, np.array([gripper], dtype=np.float32)], axis=0).astype(np.float32)

    if representation == "ee_delta_pose_gripper":
        relative_pose = _compute_relative_pose_transform(ref_ee_pose, target_pose).astype(np.float32)
        relative_rotvec = R.from_quat(relative_pose[3:7]).as_rotvec().astype(np.float32)
        return np.concatenate([relative_pose[:3], relative_rotvec, np.array([gripper], dtype=np.float32)], axis=0)

    raise ValueError(f"Unsupported pi05 action representation: {representation}")


def transform_carm_raw_action_sequence(
    raw_actions: np.ndarray,
    ee_poses: np.ndarray,
    representation: ActionRepresentation,
) -> np.ndarray:
    """Vectorize per-step raw CARM action conversion for bridge/export code."""
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    ee_poses = np.asarray(ee_poses, dtype=np.float32)
    if raw_actions.ndim != 2:
        raise ValueError(f"Expected raw_actions rank 2, got shape {raw_actions.shape}")
    if ee_poses.ndim != 2:
        raise ValueError(f"Expected ee_poses rank 2, got shape {ee_poses.shape}")
    if raw_actions.shape[0] != ee_poses.shape[0]:
        raise ValueError(
            f"raw_actions and ee_poses must have the same length, got {raw_actions.shape[0]} and {ee_poses.shape[0]}"
        )

    spec = get_pi05_action_representation_spec(representation)
    converted = np.zeros((raw_actions.shape[0], spec.target_dim), dtype=np.float32)
    for index in range(raw_actions.shape[0]):
        converted[index] = transform_carm_raw_action(raw_actions[index], ee_poses[index], representation)
    return converted
