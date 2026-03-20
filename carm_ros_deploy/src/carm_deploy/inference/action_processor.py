"""Action processing pipeline for inference.

Extracts the action post-processing logic from the monolithic _inference_loop:
  speed scaling → safety checks → relative-to-absolute conversion.

Fixes:
  R-2: is_full_mode index calculation computed once (ActionIndices).
  BUG-3: Gripper limits checked directly, not via dummy joint array.
"""

import numpy as np
from dataclasses import dataclass
from typing import List

from scipy.spatial.transform import Rotation as R

from rlft.utils.pose_utils import (
    pose_to_transform_matrix,
    apply_relative_transform,
    apply_teleop_scale,
)
from utils.log_compat import log_warn


# ---------------------------------------------------------------------------
# Pre-computed action dimension indices (R-2 fix)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionIndices:
    """Immutable mapping from action_dim_full to column indices.

    Computed once at init, eliminates the 3× repeated calculation in
    the original _inference_loop.
    """
    is_full_mode: bool
    rel_pose_start: int
    rel_pose_end: int
    gripper_idx: int

    @classmethod
    def from_action_dim(cls, action_dim_full: int) -> 'ActionIndices':
        is_full = (action_dim_full == 15)
        return cls(
            is_full_mode=is_full,
            rel_pose_start=7 if is_full else 0,
            rel_pose_end=14 if is_full else 7,
            gripper_idx=14 if is_full else 7,
        )


# ---------------------------------------------------------------------------
# Safety result
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    """Outcome of the safety checking pass."""
    actions: np.ndarray
    clipped: bool
    events: List[str]


# ---------------------------------------------------------------------------
# ActionProcessor
# ---------------------------------------------------------------------------

class ActionProcessor:
    """Stateless processor that transforms raw policy output into executable
    robot actions.

    Pipeline order (matches the original inference_ros.py):
        1. apply_speed_scale   – optional runtime speed scaling
        2. apply_safety_checks – translation cap + workspace clip + gripper clip
        3. convert_to_absolute – relative SE(3) → absolute end-effector pose
    """

    def __init__(
        self,
        action_dim_full: int,
        safety_controller,
        inference_speed_scale: float = 1.0,
        check_workspace: bool = True,
        max_relative_translation: float = 0.1,
    ):
        self.indices = ActionIndices.from_action_dim(action_dim_full)
        self.safety = safety_controller
        self.speed_scale = inference_speed_scale
        self.check_workspace = check_workspace
        self.max_trans = max_relative_translation

    # ---- 1) speed scaling -----------------------------------------------

    def apply_speed_scale(self, actions: np.ndarray) -> np.ndarray:
        """Scale relative pose components by ``inference_speed_scale``."""
        if self.speed_scale == 1.0:
            return actions
        out = actions.copy()
        idx = self.indices
        for i in range(len(out)):
            rel = out[i, idx.rel_pose_start:idx.rel_pose_end].copy()
            out[i, idx.rel_pose_start:idx.rel_pose_end] = apply_teleop_scale(
                rel, self.speed_scale,
            )
        return out

    # ---- 2) safety checks -----------------------------------------------

    def apply_safety_checks(
        self,
        actions: np.ndarray,
        qpos_end: List[float],
    ) -> SafetyResult:
        """Run safety pipeline: translation clamp → workspace clip → gripper clip.

        BUG-3 fix: gripper limits are checked directly via ``np.clip``
        instead of constructing a dummy 7-D joint array.
        """
        idx = self.indices
        events: List[str] = []
        clipped = False
        out = actions.copy()

        for i in range(len(out)):
            rel_pose = out[i, idx.rel_pose_start:idx.rel_pose_end]
            grip = float(out[i, idx.gripper_idx])

            # (a) clamp relative translation magnitude
            trans_norm = float(np.linalg.norm(rel_pose[:3]))
            if trans_norm > self.max_trans:
                scale = self.max_trans / trans_norm
                out[i, idx.rel_pose_start:idx.rel_pose_start + 3] *= scale
                rel_pose = out[i, idx.rel_pose_start:idx.rel_pose_end]
                if i == 0:
                    events.append(
                        f"Translation scaled: {trans_norm:.3f}m -> {self.max_trans}m",
                    )
                    log_warn(
                        f"Safety: Translation scaled from {trans_norm:.3f}m "
                        f"to {self.max_trans}m",
                    )
                clipped = True

            # (b) workspace check on the computed absolute target
            target_pose = apply_relative_transform(rel_pose, qpos_end[:7], grip)
            target_np = np.array(target_pose[:7])

            if self.check_workspace:
                clipped_pose, ws_warnings = self.safety.check_workspace(target_np)
                if ws_warnings:
                    clipped = True
                    if i == 0:
                        events.extend(ws_warnings)
                        for w in ws_warnings:
                            log_warn(f"Workspace clip: {w}")
                    # recompute relative pose from the clipped absolute
                    T_cur = pose_to_transform_matrix(qpos_end[:3], qpos_end[3:7])
                    T_clip = pose_to_transform_matrix(
                        clipped_pose[:3], clipped_pose[3:7],
                    )
                    T_rel = np.linalg.inv(T_cur) @ T_clip
                    out[i, idx.rel_pose_start:idx.rel_pose_start + 3] = T_rel[:3, 3]
                    out[i, idx.rel_pose_start + 3:idx.rel_pose_end] = (
                        R.from_matrix(T_rel[:3, :3]).as_quat()
                    )

            # (c) gripper limits — BUG-3 fix: direct clip, no dummy joints
            grip_min = self.safety.joint_limits.gripper_min
            grip_max = self.safety.joint_limits.gripper_max
            clamped_grip = float(np.clip(grip, grip_min, grip_max))
            if clamped_grip != grip:
                out[i, idx.gripper_idx] = clamped_grip
                if idx.is_full_mode:
                    out[i, 6] = clamped_grip  # first gripper slot in full mode
                if i == 0:
                    events.append(f"Gripper clipped: {grip:.3f} -> {clamped_grip:.3f}")
                clipped = True

        return SafetyResult(actions=out, clipped=clipped, events=events)

    # ---- 3) relative → absolute conversion ------------------------------

    def convert_to_absolute(
        self,
        actions: np.ndarray,
        qpos_end: List[float],
    ) -> np.ndarray:
        """Convert relative pose actions to absolute end-effector target poses.

        Returns:
            Array of shape ``[N, 8]`` where each row is
            ``[x, y, z, qx, qy, qz, qw, gripper]``.
        """
        idx = self.indices
        abs_actions = []
        for i in range(actions.shape[0]):
            rel_pose = actions[i][idx.rel_pose_start:idx.rel_pose_end]
            grip = actions[i][idx.gripper_idx]
            target = apply_relative_transform(rel_pose, qpos_end[:7], grip)
            abs_actions.append(target)
        return np.array(abs_actions)
