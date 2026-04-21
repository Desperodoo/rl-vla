from dataclasses import dataclass, field
from typing import Literal, Tuple

from .action_transform import ActionRepresentation, get_pi05_action_representation_spec


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

    Mainline uses LIBERO-like end-effector delta actions, while keeping the
    legacy absolute target-pose route available as a baseline/ablation.
    """

    action_key: str = "action"
    representation: ActionRepresentation = "ee_delta_pose_gripper"
    normalize_actions: bool = True
    action_norm_mode: Literal["standard", "minmax"] = "standard"
    target_dim: int = field(init=False)
    pose_slice: Tuple[int, int] = field(init=False)
    gripper_index: int = field(init=False)
    rotation_mode: Literal["rotvec", "quat"] = field(init=False)
    description: str = field(init=False)

    def __post_init__(self) -> None:
        spec = get_pi05_action_representation_spec(self.representation)
        object.__setattr__(self, "target_dim", spec.target_dim)
        object.__setattr__(self, "pose_slice", spec.pose_slice)
        object.__setattr__(self, "gripper_index", spec.gripper_index)
        object.__setattr__(self, "rotation_mode", spec.rotation_mode)
        object.__setattr__(self, "description", spec.description)


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
                "rotation_mode": self.action.rotation_mode,
                "description": self.action.description,
                "normalize_actions": self.action.normalize_actions,
                "action_norm_mode": self.action.action_norm_mode,
            },
        }
