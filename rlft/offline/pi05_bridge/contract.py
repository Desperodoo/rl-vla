from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass(frozen=True)
class Pi05ObservationContract:
    """Frozen observation contract for the first LeRobot bridge version."""

    image_key: str = "observation.image"
    state_key: str = "observation.state"
    ee_pose_key: str = "observation.ee_pose"
    image_layout: Literal["NCHW", "NHWC"] = "NCHW"
    state_mode: Literal["joint_only", "ee_only", "both"] = "joint_only"
    image_size: Tuple[int, int] = (224, 224)
    normalize_images: bool = True
    include_depth: bool = False


@dataclass(frozen=True)
class Pi05ActionContract:
    """Action contract for the first bridge version.

    We intentionally keep the current CARM v2 teleop semantics untouched:
    absolute target pose (7D) + gripper scalar (1D).
    """

    action_key: str = "action"
    target_dim: int = 8
    pose_slice: Tuple[int, int] = (0, 7)
    gripper_index: int = 7
    representation: Literal["absolute_pose_gripper"] = "absolute_pose_gripper"
    normalize_actions: bool = True
    action_norm_mode: Literal["standard", "minmax"] = "standard"


@dataclass(frozen=True)
class Pi05BridgeContract:
    """Top-level contract shared by train/eval/deploy bridges."""

    obs_horizon: int = 2
    action_horizon: int = 16
    window_stride: int = 1
    observation: Pi05ObservationContract = field(default_factory=Pi05ObservationContract)
    action: Pi05ActionContract = field(default_factory=Pi05ActionContract)

    def as_metadata(self) -> dict:
        return {
            "obs_horizon": self.obs_horizon,
            "action_horizon": self.action_horizon,
            "window_stride": self.window_stride,
            "observation": {
                "image_key": self.observation.image_key,
                "state_key": self.observation.state_key,
                "ee_pose_key": self.observation.ee_pose_key,
                "image_layout": self.observation.image_layout,
                "state_mode": self.observation.state_mode,
                "image_size": list(self.observation.image_size),
                "normalize_images": self.observation.normalize_images,
                "include_depth": self.observation.include_depth,
            },
            "action": {
                "action_key": self.action.action_key,
                "target_dim": self.action.target_dim,
                "pose_slice": list(self.action.pose_slice),
                "gripper_index": self.action.gripper_index,
                "representation": self.action.representation,
                "normalize_actions": self.action.normalize_actions,
                "action_norm_mode": self.action.action_norm_mode,
            },
        }
