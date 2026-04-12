#!/usr/bin/env python3
"""
Shared Human-in-the-loop utilities for inference_ros.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from data.teleop_shadow_utils import (
    apply_relative_transform,
    compute_relative_pose_transform,
    quaternion_angle_distance,
)
from data.teleop_bridge import TeleopShadowTransformer


HITL_SHARED_SOURCE_MAP = {
    "policy": 0,
    "human": 1,
    "policy_fallback": 2,
    "human_unavailable": 3,
}


class HumanChunkProposalBuilder:
    """Lift teleop processed signal into chunk-level human proposal semantics."""

    def __init__(
        self,
        pred_horizon: int,
        control_freq: float = 50.0,
        act_horizon: Optional[int] = None,
        stale_timeout_ms: float = 150.0,
        require_active: bool = True,
        history_window_ms: float = 120.0,
        min_history_samples: int = 2,
        min_history_span_ms: float = 30.0,
    ):
        self.pred_horizon = int(pred_horizon)
        self.control_freq = float(control_freq)
        self.control_dt_s = 1.0 / self.control_freq if self.control_freq > 0 else 0.02
        self.act_horizon = int(act_horizon or pred_horizon)
        self.stale_timeout_ms = float(stale_timeout_ms)
        self.require_active = bool(require_active)
        self.history_window_ms = float(history_window_ms)
        self.min_history_samples = int(min_history_samples)
        self.min_history_span_ms = float(min_history_span_ms)
        self.shadow_transformer = TeleopShadowTransformer(self.pred_horizon)
        self.history: Deque[Dict[str, Any]] = deque()

    @staticmethod
    def _normalize_quat(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm <= 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        quat = quat / norm
        if quat[3] < 0.0:
            quat = -quat
        return quat

    def _prune_history(self, now_s: float):
        window_s = max(self.history_window_ms, self.min_history_span_ms) / 1000.0
        while self.history and (now_s - self.history[0]["t_recv_sys"]) > window_s:
            self.history.popleft()

    def _update_history(
        self,
        now_s: float,
        processed_target_abs: np.ndarray,
        processed_sequence: int,
        active: bool,
        teleop_valid: bool,
        stale: bool,
    ):
        self._prune_history(now_s)
        if not teleop_valid or stale or not active:
            return

        sample = {
            "t_recv_sys": float(now_s),
            "processed_sequence": int(processed_sequence),
            "target_abs": np.asarray(processed_target_abs, dtype=np.float64).copy(),
        }
        if self.history and self.history[-1]["processed_sequence"] == sample["processed_sequence"]:
            self.history[-1] = sample
        else:
            self.history.append(sample)

    def _estimate_rollout_diagnostics(self) -> Dict[str, Any]:
        samples = list(self.history)
        history_count = len(samples)
        if history_count < self.min_history_samples:
            return {
                "history_count": history_count,
                "history_span_ms": 0.0,
                "history_usable": False,
                "rollout_step_count": 1,
                "rollout_dt_ms": self.control_dt_s * 1000.0,
                "linear_velocity": np.zeros(3, dtype=np.float64),
                "angular_velocity": np.zeros(3, dtype=np.float64),
                "gripper_velocity": 0.0,
            }

        first = samples[0]
        last = samples[-1]
        dt_s = float(last["t_recv_sys"] - first["t_recv_sys"])
        history_span_ms = max(0.0, dt_s * 1000.0)
        if dt_s <= 1e-6 or history_span_ms < self.min_history_span_ms:
            return {
                "history_count": history_count,
                "history_span_ms": history_span_ms,
                "history_usable": False,
                "rollout_step_count": 1,
                "rollout_dt_ms": self.control_dt_s * 1000.0,
                "linear_velocity": np.zeros(3, dtype=np.float64),
                "angular_velocity": np.zeros(3, dtype=np.float64),
                "gripper_velocity": 0.0,
            }

        p0 = np.asarray(first["target_abs"], dtype=np.float64)
        p1 = np.asarray(last["target_abs"], dtype=np.float64)
        linear_velocity = (p1[:3] - p0[:3]) / dt_s
        grip_velocity = float((p1[7] - p0[7]) / dt_s)

        q0 = self._normalize_quat(p0[3:7])
        q1 = self._normalize_quat(p1[3:7])
        if np.dot(q0, q1) < 0.0:
            q1 = -q1
        angular_velocity = (R.from_quat(q1) * R.from_quat(q0).inv()).as_rotvec() / dt_s

        rollout_step_count = max(1, min(self.pred_horizon, self.act_horizon))
        return {
            "history_count": history_count,
            "history_span_ms": history_span_ms,
            "history_usable": True,
            "rollout_step_count": rollout_step_count,
            "rollout_dt_ms": self.control_dt_s * 1000.0,
            "linear_velocity": np.asarray(linear_velocity, dtype=np.float64),
            "angular_velocity": np.asarray(angular_velocity, dtype=np.float64),
            "gripper_velocity": grip_velocity,
        }

    def _build_rollout_chunk(
        self,
        processed_target_abs: np.ndarray,
        diagnostics: Dict[str, Any],
    ) -> np.ndarray:
        chunk = np.repeat(np.asarray(processed_target_abs, dtype=np.float64)[None, :], self.pred_horizon, axis=0)
        rollout_step_count = int(diagnostics["rollout_step_count"])
        if not diagnostics["history_usable"] or rollout_step_count <= 1:
            return chunk

        base_target = np.asarray(processed_target_abs, dtype=np.float64)
        base_rot = R.from_quat(self._normalize_quat(base_target[3:7]))
        linear_velocity = np.asarray(diagnostics["linear_velocity"], dtype=np.float64)
        angular_velocity = np.asarray(diagnostics["angular_velocity"], dtype=np.float64)
        gripper_velocity = float(diagnostics["gripper_velocity"])

        for step_idx in range(1, rollout_step_count):
            dt_s = step_idx * self.control_dt_s
            target = base_target.copy()
            target[:3] = base_target[:3] + linear_velocity * dt_s
            target[3:7] = self._normalize_quat(
                (R.from_rotvec(angular_velocity * dt_s) * base_rot).as_quat()
            )
            target[7] = base_target[7] + gripper_velocity * dt_s
            chunk[step_idx] = target

        if rollout_step_count < self.pred_horizon:
            chunk[rollout_step_count:] = chunk[rollout_step_count - 1]
        return chunk

    def _build_chunk_semantics(self, ref_pose: np.ndarray, human_chunk_abs: np.ndarray) -> Dict[str, Any]:
        human_chunk_abs = np.asarray(human_chunk_abs, dtype=np.float64)
        human_chunk_rel = np.zeros((self.pred_horizon, 7), dtype=np.float64)
        reconstructed = np.zeros((self.pred_horizon, 8), dtype=np.float64)

        for idx in range(self.pred_horizon):
            pose_abs = human_chunk_abs[idx]
            rel = compute_relative_pose_transform(ref_pose, pose_abs[:7])
            human_chunk_rel[idx] = rel
            reconstructed[idx] = apply_relative_transform(rel, ref_pose, pose_abs[7])

        pos_error = float(np.linalg.norm(reconstructed[0, :3] - human_chunk_abs[0, :3]))
        rot_error = quaternion_angle_distance(reconstructed[0, 3:7], human_chunk_abs[0, 3:7])
        return {
            "human_chunk_abs": human_chunk_abs,
            "human_chunk_rel": human_chunk_rel,
            "reconstructed_target_abs": reconstructed,
            "abs_reconstruction_pos_error": pos_error,
            "abs_reconstruction_rot_error": rot_error,
        }

    def build(self, ref_qpos_end: np.ndarray, teleop_snapshot: Optional[dict]) -> Dict[str, Any]:
        ref_qpos_end = np.asarray(ref_qpos_end, dtype=np.float64)
        teleop_snapshot = teleop_snapshot or {}
        teleop_state_v2 = teleop_snapshot.get("teleop_state_v2")
        transformed = self.shadow_transformer.build(ref_qpos_end[:7], teleop_state_v2)

        active = bool(teleop_snapshot.get("teleop_active"))
        signal_age_ms = teleop_snapshot.get("signal_age_ms")
        processed_sequence = teleop_snapshot.get("processed_sequence")
        raw_sequence = teleop_snapshot.get("raw_sequence")
        t_recv_sys = teleop_snapshot.get("t_recv_sys")
        if t_recv_sys is None:
            t_recv_sys = 0.0
        stale = (
            signal_age_ms is None
            or float(signal_age_ms) > self.stale_timeout_ms
        )
        self._update_history(
            now_s=float(t_recv_sys),
            processed_target_abs=np.asarray(transformed["processed_target_abs"], dtype=np.float64),
            processed_sequence=-1 if processed_sequence is None else int(processed_sequence),
            active=active,
            teleop_valid=bool(transformed["teleop_valid"]),
            stale=bool(stale),
        )
        diagnostics = self._estimate_rollout_diagnostics()
        if bool(transformed["teleop_valid"]):
            human_chunk_abs = self._build_rollout_chunk(transformed["processed_target_abs"], diagnostics)
            chunk_semantics = self._build_chunk_semantics(ref_qpos_end[:7], human_chunk_abs)
        else:
            chunk_semantics = {
                "human_chunk_abs": np.asarray(transformed["human_chunk_abs"], dtype=np.float64),
                "human_chunk_rel": np.asarray(transformed["human_chunk_rel"], dtype=np.float64),
                "reconstructed_target_abs": np.asarray(transformed["reconstructed_target_abs"], dtype=np.float64),
                "abs_reconstruction_pos_error": float(transformed["abs_reconstruction_pos_error"]),
                "abs_reconstruction_rot_error": float(transformed["abs_reconstruction_rot_error"]),
            }

        valid = bool(transformed["teleop_valid"]) and (active or not self.require_active) and not stale
        return {
            "human_chunk_proposal": np.asarray(chunk_semantics["human_chunk_abs"], dtype=np.float64),
            "human_chunk_rel": np.asarray(chunk_semantics["human_chunk_rel"], dtype=np.float64),
            "processed_target_abs": np.asarray(transformed["processed_target_abs"], dtype=np.float64),
            "reconstructed_target_abs": np.asarray(chunk_semantics["reconstructed_target_abs"], dtype=np.float64),
            "human_valid": bool(valid),
            "human_active": active,
            "human_stale": bool(stale),
            "signal_age_ms": None if signal_age_ms is None else float(signal_age_ms),
            "processed_sequence": -1 if processed_sequence is None else int(processed_sequence),
            "raw_sequence": -1 if raw_sequence is None else int(raw_sequence),
            "teleop_valid": bool(transformed["teleop_valid"]),
            "abs_reconstruction_pos_error": float(chunk_semantics["abs_reconstruction_pos_error"]),
            "abs_reconstruction_rot_error": float(chunk_semantics["abs_reconstruction_rot_error"]),
            "history_count": int(diagnostics["history_count"]),
            "history_span_ms": float(diagnostics["history_span_ms"]),
            "history_usable": bool(diagnostics["history_usable"]),
            "rollout_step_count": int(diagnostics["rollout_step_count"]),
            "rollout_dt_ms": float(diagnostics["rollout_dt_ms"]),
            "linear_velocity": np.asarray(diagnostics["linear_velocity"], dtype=np.float64),
            "angular_velocity": np.asarray(diagnostics["angular_velocity"], dtype=np.float64),
            "gripper_velocity": float(diagnostics["gripper_velocity"]),
        }


class HitlArbitrationBridge:
    """Minimal chunk-level source-select arbitration."""

    def arbitrate(
        self,
        policy_chunk: np.ndarray,
        human_proposal: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        policy_chunk = np.asarray(policy_chunk, dtype=np.float64)
        if human_proposal is None:
            return self._build_result(policy_chunk, "policy_fallback", False, "missing_human_proposal")

        human_chunk = np.asarray(human_proposal["human_chunk_proposal"], dtype=np.float64)
        if bool(human_proposal.get("human_valid")):
            return self._build_result(human_chunk, "human", True, "")

        reason = "human_unavailable"
        if bool(human_proposal.get("human_stale")):
            reason = "human_stale"
        elif not bool(human_proposal.get("human_active")):
            reason = "human_inactive"
        elif not bool(human_proposal.get("teleop_valid")):
            reason = "teleop_invalid"
        return self._build_result(policy_chunk, "policy_fallback", False, reason)

    @staticmethod
    def _build_result(shared_chunk: np.ndarray, shared_source: str, human_selected: bool, fallback_reason: str) -> Dict[str, Any]:
        return {
            "shared_chunk": np.asarray(shared_chunk, dtype=np.float64),
            "shared_source": shared_source,
            "shared_source_code": int(HITL_SHARED_SOURCE_MAP[shared_source]),
            "human_selected": bool(human_selected),
            "fallback_reason": fallback_reason,
            "shared_valid": True,
        }
