"""Standardized failure codes for script runtime."""

from __future__ import annotations

from enum import Enum


class FailureCode(str, Enum):
    """Canonical failure codes used across skills, tasks, and recovery."""

    NONE = "NONE"
    PERCEPTION_LOST = "PERCEPTION_LOST"
    NO_OBJECT_DETECTED = "NO_OBJECT_DETECTED"
    NO_GRASP_CANDIDATE = "NO_GRASP_CANDIDATE"
    NO_IK = "NO_IK"
    COLLISION_RISK = "COLLISION_RISK"
    WORKSPACE_VIOLATION = "WORKSPACE_VIOLATION"
    GRASP_FAIL = "GRASP_FAIL"
    CONTACT_ANOMALY = "CONTACT_ANOMALY"
    TIMEOUT = "TIMEOUT"
    ROBOT_FAULT = "ROBOT_FAULT"
    SDK_ERROR = "SDK_ERROR"
    HUMAN_ABORT = "HUMAN_ABORT"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    UNKNOWN = "UNKNOWN"
