"""
rlft.datasets - Dataset Classes for IL and Offline RL

Provides:
- ManiSkillDataset: ManiSkill3 HDF5 demo loading for IL
- OfflineRLDataset: ManiSkill3 with SMDP formulation for offline RL
- CARMDataset: CARM real robot demo loading with relative pose actions
"""

from .maniskill_dataset import ManiSkillDataset, OfflineRLDataset
from .carm_dataset import CARMDataset, ActionNormalizer
from .data_utils import (
    load_traj_hdf5,
    load_carm_dataset,
    create_obs_process_fn,
    create_carm_obs_process_fn,
    get_carm_data_info,
    ObservationStacker,
    IterationBasedBatchSampler,
    worker_init_fn,
)

__all__ = [
    "ManiSkillDataset",
    "OfflineRLDataset",
    "CARMDataset",
    "ActionNormalizer",
    "load_traj_hdf5",
    "load_carm_dataset",
    "create_obs_process_fn",
    "create_carm_obs_process_fn",
    "get_carm_data_info",
    "ObservationStacker",
    "IterationBasedBatchSampler",
    "worker_init_fn",
]
