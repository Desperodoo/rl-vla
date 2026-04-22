"""Dry-run smoke validation for the joint-control PI05 runtime contract.

This test validates that the runtime-facing PI05 loader can produce a
[batch, horizon, action_dim] chunk for joint-control execution.
"""

from pathlib import Path
import importlib.machinery
import sys

import numpy as np
import pytest
import torch


def _has_real_lerobot() -> bool:
    return importlib.machinery.PathFinder.find_spec("lerobot") is not None


if not _has_real_lerobot():
    pytest.skip("lerobot package not available for real-checkpoint runtime smoke", allow_module_level=True)


_CARM_DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(_CARM_DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(_CARM_DEPLOY_ROOT))

from inference.policy_loader_pi05 import LeRobotPi05Policy


CHECKPOINT_ROOT = Path("/mnt/disk_2/wjz/openpi/pi05_droid_pytorch")
DATASET_ROOT = Path("/mnt/disk_2/wjz/runs/pi05_full_export/train")


@pytest.mark.skipif(not CHECKPOINT_ROOT.exists() or not DATASET_ROOT.exists(), reason="real pi05 assets not available")
def test_real_checkpoint_chunk_matches_joint_runtime_contract():
    policy = LeRobotPi05Policy(
        {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "state_mode": "joint_only",
            "control_mode": "joint",
            "action_representation": "joint_absolute_gripper",
            "dataset_root": str(DATASET_ROOT),
            "repo_id": "carm/pi05_local",
            "task": "pick and place",
        }
    )
    policy.load_model(str(CHECKPOINT_ROOT))

    image_h, image_w = policy.target_image_size
    image = np.zeros((3, image_h, image_w), dtype=np.float32)
    qpos = np.zeros(7, dtype=np.float32)
    ee_pose = np.zeros(7, dtype=np.float32)

    out = policy.predict_action_chunk({"image": image, "qpos": qpos, "ee_pose": ee_pose, "task": "pick and place"})
    chunk = out["a_hat"]

    assert torch.is_tensor(chunk)
    assert chunk.ndim == 3
    assert chunk.shape[0] == 1
    assert chunk.shape[1] > 0
    assert chunk.shape[2] == policy.action_dim
    assert chunk.shape[2] == policy.action_dim_full
    assert torch.isfinite(chunk).all()

    chunk_np = chunk.squeeze(0).detach().cpu().numpy()
    assert np.all(np.isfinite(chunk_np))
