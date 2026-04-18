"""Speed limit policy for runtime actions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeedLimitPolicy:
    joint_speed: float = 1.0
    cartesian_speed: float = 1.0
    servo_gain: float = 1.0

    def clamp_joint_speed(self, requested: float) -> float:
        return min(max(requested, 0.0), self.joint_speed)

    def clamp_cartesian_speed(self, requested: float) -> float:
        return min(max(requested, 0.0), self.cartesian_speed)
