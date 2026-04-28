"""pi0.5 bridge utilities for LeRobot-first integration.

This package contains thin adapters that map the current CARM real-robot
pipeline to an external pi0.5/LeRobot training stack.
"""

from .contract import Pi05ActionContract, Pi05BridgeContract, Pi05ObservationContract
from .config_bridge import (
    DEFAULT_OPENPI_PI05_BASE_PRETRAINED_PATH,
    DEFAULT_OPENPI_PI05_DROID_PRETRAINED_PATH,
    DEFAULT_OPENPI_PI05_LIBERO_PRETRAINED_PATH,
    DEFAULT_OPENPI_PI05_PRETRAINED_PATHS,
    build_lerobot_train_command,
    build_pi05_run_config,
    resolve_default_openpi_pi05_pretrained_path,
    resolve_target_image_size,
)
from .dataset_bridge import Pi05EpisodeWindow, Pi05LeRobotDatasetBridge, build_pi05_dataset_bridge
from .env_probe import build_probe_environment, probe_lerobot_environment
from .export import export_carm_to_lerobot_dataset
from .openpi_checkpoint import prepare_openpi_pi05_checkpoint
from .task_semantics import Pi05TaskSemantics, load_pi05_task_semantics
from .validate import validate_bridge_dataset, validate_lerobot_dataset_path, validate_lerobot_train_command

__all__ = [
    "Pi05ActionContract",
    "Pi05BridgeContract",
    "Pi05ObservationContract",
    "Pi05EpisodeWindow",
    "Pi05LeRobotDatasetBridge",
    "build_pi05_dataset_bridge",
    "build_pi05_run_config",
    "build_lerobot_train_command",
    "resolve_target_image_size",
    "resolve_default_openpi_pi05_pretrained_path",
    "DEFAULT_OPENPI_PI05_BASE_PRETRAINED_PATH",
    "DEFAULT_OPENPI_PI05_DROID_PRETRAINED_PATH",
    "DEFAULT_OPENPI_PI05_LIBERO_PRETRAINED_PATH",
    "DEFAULT_OPENPI_PI05_PRETRAINED_PATHS",
    "build_probe_environment",
    "probe_lerobot_environment",
    "validate_bridge_dataset",
    "validate_lerobot_dataset_path",
    "validate_lerobot_train_command",
    "export_carm_to_lerobot_dataset",
    "prepare_openpi_pi05_checkpoint",
    "Pi05TaskSemantics",
    "load_pi05_task_semantics",
]
