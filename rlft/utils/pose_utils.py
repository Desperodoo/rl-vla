"""
Pose Transformation Utilities.

Shared utility functions for SE(3) pose transformations used by both
training (carm_dataset.py) and inference (inference_ros.py).

Quaternion convention: [qx, qy, qz, qw] (scipy convention)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_to_transform_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Convert pose (xyz + quaternion) to 4x4 transformation matrix.
    
    Args:
        position: Translation [x, y, z]
        quaternion: Quaternion [qx, qy, qz, qw]
    
    Returns:
        4x4 homogeneous transformation matrix
    """
    rotation = R.from_quat(quaternion).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def transform_matrix_to_pose(transform: np.ndarray) -> tuple:
    """Convert 4x4 transformation matrix to pose (xyz + quaternion).
    
    Args:
        transform: 4x4 homogeneous transformation matrix
    
    Returns:
        (position, quaternion) tuple where position=[x,y,z] and quaternion=[qx,qy,qz,qw]
    """
    position = transform[:3, 3]
    quaternion = R.from_matrix(transform[:3, :3]).as_quat()
    return position, quaternion


def compute_relative_pose_transform(pose_current: np.ndarray, pose_target: np.ndarray) -> np.ndarray:
    """Compute relative pose transformation from current to target.
    
    relative_transform = current_pose^{-1} @ target_pose
    
    At inference: target_pose = current_pose @ relative_transform
    
    Args:
        pose_current: Current pose [x, y, z, qx, qy, qz, qw]
        pose_target: Target pose [x, y, z, qx, qy, qz, qw]
    
    Returns:
        Relative pose [x, y, z, qx, qy, qz, qw] (7D)
    """
    T_current = pose_to_transform_matrix(pose_current[:3], pose_current[3:7])
    T_target = pose_to_transform_matrix(pose_target[:3], pose_target[3:7])
    T_relative = np.linalg.inv(T_current) @ T_target
    position, quaternion = transform_matrix_to_pose(T_relative)
    return np.concatenate([position, quaternion])


def apply_relative_transform(relative_pose: np.ndarray, current_pose: np.ndarray, 
                             gripper: float = None) -> np.ndarray:
    """Apply relative pose transformation to current pose to get target absolute pose.
    
    target_pose = current_pose @ relative_transform
    
    This is the inverse operation of compute_relative_pose_transform().
    
    Args:
        relative_pose: Model output relative pose [x, y, z, qx, qy, qz, qw]
        current_pose: Current end-effector pose [x, y, z, qx, qy, qz, qw]
        gripper: Gripper value (optional, appended to result if provided)
    
    Returns:
        Target absolute pose. If gripper is provided: [x,y,z,qx,qy,qz,qw,gripper] (8D)
        Otherwise: [x,y,z,qx,qy,qz,qw] (7D)
    """
    T_relative = pose_to_transform_matrix(relative_pose[:3], relative_pose[3:])
    T_current = pose_to_transform_matrix(current_pose[:3], current_pose[3:])
    
    T_target = T_current @ T_relative
    
    target_position = T_target[:3, 3]
    target_quat = R.from_matrix(T_target[:3, :3]).as_quat()
    
    if gripper is not None:
        return np.array(target_position.tolist() + target_quat.tolist() + [gripper])
    else:
        return np.concatenate([target_position, target_quat])


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation (Slerp) for quaternions.
    
    Args:
        q0: Start quaternion [qx, qy, qz, qw]
        q1: End quaternion [qx, qy, qz, qw]
        t: Interpolation factor [0, 1]
    
    Returns:
        Interpolated quaternion [qx, qy, qz, qw]
    """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)

    if theta < 1e-7:
        return (1 - t) * q0 + t * q1
    
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def apply_teleop_scale(delta_pose: np.ndarray, scale: float) -> np.ndarray:
    """Apply joystick-style scaling to an end-effector delta pose.
    
    - Translation: linear scaling delta_pos *= scale
    - Rotation: slerp from identity quaternion to delta quaternion
    
    Args:
        delta_pose: Relative pose [x, y, z, qx, qy, qz, qw] (7D)
        scale: Scale factor (0, 1]. 1.0 means no scaling.
    
    Returns:
        Scaled relative pose [x, y, z, qx, qy, qz, qw]
    """
    if scale >= 1.0:
        return delta_pose.copy()
    
    scaled_pose = delta_pose.copy()
    
    # Translation: linear scaling
    scaled_pose[:3] = delta_pose[:3] * scale
    
    # Rotation: slerp from identity to target
    identity_quat = np.array([0.0, 0.0, 0.0, 1.0])
    delta_quat = delta_pose[3:7]
    scaled_quat = quaternion_slerp(identity_quat, delta_quat, scale)
    scaled_pose[3:7] = scaled_quat
    
    return scaled_pose
