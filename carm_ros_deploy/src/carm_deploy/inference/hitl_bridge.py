#!/usr/bin/env python3
"""
Shared Human-in-the-loop utilities for inference_ros.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

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
        stale_timeout_ms: float = 150.0,
        require_active: bool = True,
    ):
        self.pred_horizon = int(pred_horizon)
        self.stale_timeout_ms = float(stale_timeout_ms)
        self.require_active = bool(require_active)
        self.shadow_transformer = TeleopShadowTransformer(self.pred_horizon)

    def build(self, ref_qpos_end: np.ndarray, teleop_snapshot: Optional[dict]) -> Dict[str, Any]:
        ref_qpos_end = np.asarray(ref_qpos_end, dtype=np.float64)
        teleop_snapshot = teleop_snapshot or {}
        teleop_state_v2 = teleop_snapshot.get("teleop_state_v2")
        transformed = self.shadow_transformer.build(ref_qpos_end[:7], teleop_state_v2)

        active = bool(teleop_snapshot.get("teleop_active"))
        signal_age_ms = teleop_snapshot.get("signal_age_ms")
        processed_sequence = teleop_snapshot.get("processed_sequence")
        raw_sequence = teleop_snapshot.get("raw_sequence")
        stale = (
            signal_age_ms is None
            or float(signal_age_ms) > self.stale_timeout_ms
        )

        valid = bool(transformed["teleop_valid"]) and (active or not self.require_active) and not stale
        return {
            "human_chunk_proposal": np.asarray(transformed["human_chunk_abs"], dtype=np.float64),
            "human_chunk_rel": np.asarray(transformed["human_chunk_rel"], dtype=np.float64),
            "processed_target_abs": np.asarray(transformed["processed_target_abs"], dtype=np.float64),
            "reconstructed_target_abs": np.asarray(transformed["reconstructed_target_abs"], dtype=np.float64),
            "human_valid": bool(valid),
            "human_active": active,
            "human_stale": bool(stale),
            "signal_age_ms": None if signal_age_ms is None else float(signal_age_ms),
            "processed_sequence": -1 if processed_sequence is None else int(processed_sequence),
            "raw_sequence": -1 if raw_sequence is None else int(raw_sequence),
            "teleop_valid": bool(transformed["teleop_valid"]),
            "abs_reconstruction_pos_error": float(transformed["abs_reconstruction_pos_error"]),
            "abs_reconstruction_rot_error": float(transformed["abs_reconstruction_rot_error"]),
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
