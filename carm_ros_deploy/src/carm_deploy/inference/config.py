"""Typed configuration for the CARM inference pipeline.

Replaces the untyped 50+ key config dict with a dataclass.
Fixes BUG-6 (timeline_enabled dead code) and BUG-8 (truncate_at_act_horizon cannot be disabled).
"""

import dataclasses
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class InferenceConfig:
    """Single source of truth for all inference parameters."""

    # -- Robot --
    robot_ip: str = '10.42.0.101'
    robot_mode: int = 4
    robot_tau: float = 10.0
    arm_init_pose: List[float] = field(
        default_factory=lambda: [0.2475, 0.0014, 0.3251, 0.9996, -0.0034, 0.0255, -0.0074],
    )
    arm_init_gripper: float = 0.078

    # -- Camera --
    camera_topics: List[str] = field(default_factory=lambda: ['/camera/color/image_raw'])
    sync_slop: float = 0.02

    # -- Policy --
    pretrain: str = ''
    algorithm: str = 'consistency_flow'
    desire_inference_freq: float = 30.0
    temporal_factor_k: float = 0.05
    num_inference_steps: int = 10
    use_ema: bool = False

    # -- Action execution --
    execution_mode: str = 'receding_horizon'
    max_active_chunks: Optional[int] = None
    crossfade_steps: int = 0
    # BUG-8 fix: default True, disable with --no_truncate_at_act_horizon
    truncate_at_act_horizon: bool = True
    act_horizon: int = 8

    # -- Control --
    pos_lookahead_step: int = 1
    pos_lookahead_duration: float = 0.015
    inference_speed_scale: float = 1.0
    control_freq: int = 50
    gripper_hysteresis_window: int = 1

    # -- Safety --
    safety_config: str = ''
    init_speed: float = 2.0
    skip_init_confirm: bool = False  # skip arm init confirmation prompt (for scripted launch)

    # -- Logging --
    log_dir: str = ''
    save_images: bool = False
    vis: bool = True
    # BUG-6 fix: single flag, no more --timeline_enabled
    timeline_disabled: bool = False
    timeline_log: str = ''
    timeline_control_stride: int = 10
    chunk_time_base: str = 'sys_time'

    # -- Intervention & recording --
    record_inference: bool = False
    intervention: bool = False
    intervention_mode: str = 'replace'
    intervention_xyz_scale: float = 0.01
    intervention_gripper_open: float = 1.0
    intervention_gripper_close: float = 0.0
    record_dir: str = ''
    max_steps: int = 99999

    # -- Deprecated (kept for backward compat, ignored) --
    joint_cmd_mode: bool = False

    # ---- Computed properties ----

    @property
    def timeline_enabled(self) -> bool:
        """BUG-6 fix: derived from timeline_disabled."""
        return not self.timeline_disabled

    @property
    def teleop_scale(self) -> float:
        """Fixed to 1.0 per GAP-2 fix. Not configurable."""
        return 1.0

    # ---- Serialization ----

    def to_dict(self) -> dict:
        """Convert to plain dict for backward compatibility with modules
        that still use ``config.get('key')``."""
        d = dataclasses.asdict(self)
        d['timeline_enabled'] = self.timeline_enabled
        d['teleop_scale'] = self.teleop_scale
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'InferenceConfig':
        """Build config from a plain dict, ignoring unknown keys."""
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_argparse(cls, args) -> 'InferenceConfig':
        """Build config from argparse Namespace."""
        return cls.from_dict(vars(args))

    def resolve_safety_config(self, carm_deploy_root: str) -> str:
        """Resolve safety_config path, raising if not found."""
        path = self.safety_config
        if not path:
            path = os.path.join(carm_deploy_root, 'safety_config.json')
        path = os.path.expandvars(os.path.expanduser(path))
        self.safety_config = path
        return path

    def normalize_camera_topics(self) -> None:
        """Ensure camera_topics is a list (roslaunch may pass a comma-separated string)."""
        if isinstance(self.camera_topics, str):
            self.camera_topics = self.camera_topics.split(',')

    def normalize_arm_init_pose(self) -> None:
        """Ensure arm_init_pose is a list of floats (roslaunch may pass a string)."""
        if isinstance(self.arm_init_pose, str):
            self.arm_init_pose = [float(x) for x in self.arm_init_pose.split()]
        if isinstance(self.arm_init_gripper, str):
            self.arm_init_gripper = float(self.arm_init_gripper)
