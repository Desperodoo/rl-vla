#!/usr/bin/env python3
"""
LeRobot/OpenPI pi0.5 policy loader for CARM ROS deployment.

This loader is intentionally separate from the legacy RLFT policy loader because
it consumes a different checkpoint/config/processor stack.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from peft import PeftConfig


try:
    import rospy

    _log_info = rospy.loginfo
    _log_warn = rospy.logwarn
    _log_err = rospy.logerr
except ImportError:
    _logger = logging.getLogger(__name__)
    _log_info = _logger.info
    _log_warn = _logger.warning
    _log_err = _logger.error


DEFAULT_TOKENIZER_PATH = "/home/wjz/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"
DEFAULT_REPO_ID = "carm/pi05_local"


class LeRobotPi05Policy:
    """Inference wrapper around a LeRobot/OpenPI pi0.5/pi05 policy."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.dataset = None
        self.loaded = False

        self.obs_horizon = int(config.get("obs_horizon", 1))
        self.pred_horizon = int(config.get("pred_horizon", 16))
        self.action_dim = int(config.get("action_dim", 7))
        self.action_dim_full = int(config.get("action_dim_full", self.action_dim))
        self.state_mode = config.get("state_mode", "joint_only")
        self.target_image_size = tuple(config.get("target_image_size", (224, 224)))
        self.use_ema = False
        self.algorithm = "lerobot_pi05"
        self.normalize_actions = True
        self.action_norm_mode = "dataset"
        self.gripper_hysteresis_window = 1
        self.control_mode = config.get("control_mode", "joint")
        self.action_representation = config.get("action_representation", "joint_absolute_gripper")
        self.tokenizer_path_override = config.get("tokenizer_path_override", DEFAULT_TOKENIZER_PATH)
        self.default_task = config.get("task", "pick and place")
        self.dataset_root = config.get("dataset_root")
        self.repo_id = config.get("repo_id", DEFAULT_REPO_ID)
        self.input_features: Dict[str, Any] = {}
        self.output_features: Dict[str, Any] = {}

    def load_model(self, model_path: str):
        policy_path = Path(model_path).expanduser().resolve()
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy path not found: {policy_path}")

        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.device = str(self.device)
        cfg.pretrained_path = policy_path
        cfg.use_peft = self._resolve_use_peft(policy_path)

        ds_meta = self._load_dataset_meta()
        _log_info(f"Loading LeRobot policy from: {policy_path}")
        self.policy = make_policy(cfg=cfg, ds_meta=ds_meta, rename_map={})

        preprocessor_overrides = {
            "rename_observations_processor": {"rename_map": {}},
        }
        if ds_meta is not None:
            preprocessor_overrides["normalizer_processor"] = {
                "stats": ds_meta.stats,
                "features": {**self.policy.config.input_features, **self.policy.config.output_features},
                "norm_map": self.policy.config.normalization_mapping,
            }
        if self.tokenizer_path_override:
            preprocessor_overrides["tokenizer_processor"] = {
                "tokenizer_name": self.tokenizer_path_override,
            }

        postprocessor_overrides = {}
        if ds_meta is not None:
            postprocessor_overrides["unnormalizer_processor"] = {
                "stats": ds_meta.stats,
                "features": self.policy.config.output_features,
                "norm_map": self.policy.config.normalization_mapping,
            }

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(policy_path),
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides=postprocessor_overrides,
        )

        self.policy.eval()
        self.input_features = dict(getattr(self.policy.config, "input_features", {}) or {})
        self.output_features = dict(getattr(self.policy.config, "output_features", {}) or {})
        self.pred_horizon = int(getattr(self.policy.config, "n_action_steps", self.pred_horizon))
        self.action_dim = self._infer_action_dim()
        self.action_dim_full = self.action_dim
        self.target_image_size = tuple(self._infer_image_size())
        self.loaded = True

        _log_info(
            "Loaded LeRobot pi0.5 policy: "
            f"pred_horizon={self.pred_horizon}, action_dim={self.action_dim}, "
            f"image_size={self.target_image_size}, control_mode={self.control_mode}"
        )

    def reset(self):
        """Stateless wrapper for interface parity with legacy policies."""
        return None

    def build_state_from_obs(self, qpos_joint: np.ndarray, qpos_end: np.ndarray) -> np.ndarray:
        if self.state_mode == "joint_only":
            return np.asarray(qpos_joint, dtype=np.float32)
        if self.state_mode == "ee_only":
            return np.asarray(qpos_end, dtype=np.float32)
        if self.state_mode == "both":
            return np.concatenate(
                [np.asarray(qpos_joint, dtype=np.float32), np.asarray(qpos_end[:7], dtype=np.float32)],
                axis=0,
            )
        raise ValueError(f"Unsupported state_mode: {self.state_mode}")

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if not self.loaded or self.policy is None:
            raise RuntimeError("Policy not loaded. Call load_model() first.")

        sample = self._build_sample(inputs)
        processed = self.preprocessor(sample) if self.preprocessor is not None else sample
        with torch.no_grad():
            pred = self.policy.select_action(processed)
            if self.postprocessor is not None:
                pred = self.postprocessor(pred)

        action = pred["action"] if isinstance(pred, dict) and "action" in pred else pred
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action = action.to(self.device).float()
        if action.ndim == 1:
            action = action.unsqueeze(0)
        return {"a_hat": action.unsqueeze(0)}

    def predict_action_chunk(self, inputs: Dict[str, Any], **kwargs: Any) -> Dict[str, torch.Tensor]:
        if not self.loaded or self.policy is None:
            raise RuntimeError("Policy not loaded. Call load_model() first.")

        sample = self._build_sample(inputs)
        processed = self.preprocessor(sample) if self.preprocessor is not None else sample
        with torch.no_grad():
            chunk = self.policy.predict_action_chunk(processed, **kwargs)
            if self.postprocessor is not None:
                chunk = self.postprocessor(chunk)

        action = chunk["action"] if isinstance(chunk, dict) and "action" in chunk else chunk
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        action = action.to(self.device).float()
        if action.ndim == 2:
            action = action.unsqueeze(0)
        return {"a_hat": action}

    def _build_sample(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        image = inputs["image"]
        state = inputs["qpos"]

        if isinstance(image, np.ndarray):
            image_t = torch.from_numpy(image)
        else:
            image_t = image.detach().clone()
        if image_t.ndim == 3:
            pass
        elif image_t.ndim == 4 and image_t.shape[0] == 1:
            image_t = image_t.squeeze(0)
        else:
            raise ValueError(f"Unexpected image shape for pi05 policy: {tuple(image_t.shape)}")
        image_t = image_t.to(self.device).float()

        if isinstance(state, np.ndarray):
            state_t = torch.from_numpy(state)
        else:
            state_t = state.detach().clone()
        state_t = state_t.to(self.device).float().reshape(-1)

        sample = {
            "observation.image": image_t,
            "observation.state": state_t,
        }
        if "ee_pose" in inputs:
            ee_pose = inputs["ee_pose"]
            if isinstance(ee_pose, np.ndarray):
                ee_pose_t = torch.from_numpy(ee_pose)
            else:
                ee_pose_t = ee_pose.detach().clone()
            sample["observation.ee_pose"] = ee_pose_t.to(self.device).float().reshape(-1)
        if "task" in inputs:
            sample["task"] = inputs["task"]
        else:
            sample["task"] = self.default_task
        return sample

    def _load_dataset_meta(self):
        if not self.dataset_root:
            raise ValueError("LeRobotPi05Policy requires dataset_root to load dataset metadata for make_policy().")
        dataset_root = Path(self.dataset_root).expanduser().resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
        self.dataset = LeRobotDataset(repo_id=self.repo_id, root=dataset_root)
        return self.dataset.meta

    def _resolve_use_peft(self, policy_path: Path) -> bool:
        if self.config.get("peft_adapter_path"):
            return True
        if (policy_path / "adapter_config.json").exists():
            try:
                _ = PeftConfig.from_pretrained(policy_path)
                return True
            except Exception:
                _log_warn("adapter_config.json found but PEFT config could not be loaded; falling back to non-PEFT")
        return False

    def _infer_action_dim(self) -> int:
        output_features = self.output_features or {}
        action_feature = output_features.get("action")
        shape = getattr(action_feature, "shape", None) if action_feature is not None else None
        if shape:
            if len(shape) == 1:
                return int(shape[0])
            return int(shape[-1])
        return int(self.config.get("action_dim", 7))

    def _infer_image_size(self) -> tuple[int, int]:
        input_features = self.input_features or {}
        image_feature = input_features.get("observation.image")
        shape = getattr(image_feature, "shape", None) if image_feature is not None else None
        if shape and len(shape) >= 2:
            return int(shape[-2]), int(shape[-1])
        return tuple(self.config.get("target_image_size", (224, 224)))
