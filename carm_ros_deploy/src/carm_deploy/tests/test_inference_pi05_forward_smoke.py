"""Forward smoke test for the standalone pi05 policy loader.

Uses lightweight stubs for offline CI and only runs real-checkpoint validation
when local assets are available.
"""

from pathlib import Path
import importlib.machinery
import sys
import types

import numpy as np
import pytest
import torch


_CARM_DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(_CARM_DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(_CARM_DEPLOY_ROOT))

if importlib.machinery.PathFinder.find_spec("lerobot") is None:
    _lerobot_root = types.ModuleType("lerobot")
    _lerobot_configs = types.ModuleType("lerobot.configs")
    _lerobot_configs_policies = types.ModuleType("lerobot.configs.policies")
    _lerobot_datasets = types.ModuleType("lerobot.datasets")
    _lerobot_dataset_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    _lerobot_policies = types.ModuleType("lerobot.policies")
    _lerobot_factory_mod = types.ModuleType("lerobot.policies.factory")

    class _DummyPreTrainedConfig:
        @classmethod
        def from_pretrained(cls, path):
            cfg = cls()
            cfg.device = "cpu"
            cfg.pretrained_path = path
            cfg.input_features = {"observation.image": types.SimpleNamespace(shape=(3, 224, 224))}
            cfg.output_features = {"action": types.SimpleNamespace(shape=(8,))}
            cfg.normalization_mapping = {}
            cfg.n_action_steps = 15
            cfg.use_peft = False
            return cfg

    class _DummyDataset:
        def __init__(self, repo_id=None, root=None):
            self.repo_id = repo_id
            self.root = root
            self.meta = types.SimpleNamespace(stats={}, episodes=[])

    class _DummyPolicy:
        def __init__(self, cfg):
            self.config = cfg

        def eval(self):
            return self

        def select_action(self, processed):
            return torch.zeros((1, 8), dtype=torch.float32)

        def predict_action_chunk(self, processed, **kwargs):
            return torch.zeros((1, 15, 8), dtype=torch.float32)

    _lerobot_configs_policies.PreTrainedConfig = _DummyPreTrainedConfig
    _lerobot_dataset_mod.LeRobotDataset = _DummyDataset
    _lerobot_factory_mod.make_policy = lambda cfg, ds_meta=None, rename_map=None: _DummyPolicy(cfg)
    _lerobot_factory_mod.make_pre_post_processors = lambda **kwargs: (lambda x: x, lambda x: x)

    sys.modules["lerobot"] = _lerobot_root
    sys.modules["lerobot.configs"] = _lerobot_configs
    sys.modules["lerobot.configs.policies"] = _lerobot_configs_policies
    sys.modules["lerobot.datasets"] = _lerobot_datasets
    sys.modules["lerobot.datasets.lerobot_dataset"] = _lerobot_dataset_mod
    sys.modules["lerobot.policies"] = _lerobot_policies
    sys.modules["lerobot.policies.factory"] = _lerobot_factory_mod

if importlib.machinery.PathFinder.find_spec("peft") is None:
    _peft_mod = types.ModuleType("peft")
    _peft_mod.PeftConfig = type("PeftConfig", (), {"from_pretrained": staticmethod(lambda path: object())})
    sys.modules["peft"] = _peft_mod

from inference.policy_loader_pi05 import LeRobotPi05Policy


CHECKPOINT_ROOT = Path("/mnt/disk_2/wjz/openpi/pi05_droid_pytorch")
DATASET_ROOT = Path("/mnt/disk_2/wjz/runs/pi05_full_export/train")


def test_stubbed_forward_smoke():
    policy = LeRobotPi05Policy(
        {
            "device": "cpu",
            "state_mode": "joint_only",
            "control_mode": "joint",
            "action_representation": "joint_absolute_gripper",
            "dataset_root": "/tmp",
            "repo_id": "carm/pi05_local",
        }
    )
    policy.load_model("/tmp")

    image_h, image_w = policy.target_image_size
    image = np.zeros((3, image_h, image_w), dtype=np.float32)
    qpos = np.zeros(7, dtype=np.float32)
    ee_pose = np.zeros(7, dtype=np.float32)

    out = policy({"image": image, "qpos": qpos, "ee_pose": ee_pose})
    assert "a_hat" in out
    assert torch.is_tensor(out["a_hat"])
    assert out["a_hat"].ndim == 3
    assert out["a_hat"].shape[0] == 1
    assert out["a_hat"].shape[1] == 1
    assert out["a_hat"].shape[2] >= 0
    assert torch.isfinite(out["a_hat"]).all()


@pytest.mark.skipif(
    (not CHECKPOINT_ROOT.exists() or not DATASET_ROOT.exists() or importlib.machinery.PathFinder.find_spec("lerobot") is None),
    reason="real pi05 assets or lerobot package not available",
)
def test_real_checkpoint_forward_smoke():
    policy = LeRobotPi05Policy(
        {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "state_mode": "joint_only",
            "control_mode": "joint",
            "action_representation": "joint_absolute_gripper",
            "dataset_root": str(DATASET_ROOT),
            "repo_id": "carm/pi05_local",
        }
    )
    policy.load_model(str(CHECKPOINT_ROOT))

    image_h, image_w = policy.target_image_size
    image = np.zeros((3, image_h, image_w), dtype=np.float32)
    qpos = np.zeros(7, dtype=np.float32)
    ee_pose = np.zeros(7, dtype=np.float32)

    out = policy({"image": image, "qpos": qpos, "ee_pose": ee_pose})
    assert "a_hat" in out
    assert torch.is_tensor(out["a_hat"])
    assert out["a_hat"].ndim == 3
    assert out["a_hat"].shape[0] == 1
    assert out["a_hat"].shape[1] == 1
    assert out["a_hat"].shape[2] > 0
    assert torch.isfinite(out["a_hat"]).all()
