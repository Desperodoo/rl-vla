"""Tests for inference.policy_loader — POLICY_DEFAULTS, RealPolicy."""

import json
import numpy as np
import pytest
import torch

from inference.policy_loader import (
    POLICY_DEFAULTS,
    PolicyInterface,
    RealPolicy,
)
from rlft.utils.model_factory import SUPPORTED_ALGORITHMS


# ═══════════════════════════════════════════════════════════════════════════
# POLICY_DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPolicyDefaults:
    EXPECTED_KEYS = {
        "obs_horizon", "pred_horizon", "action_dim", "action_dim_full",
        "state_mode", "target_image_size", "visual_feature_dim",
        "state_encoder_hidden_dim", "state_encoder_out_dim",
        "use_state_encoder", "algorithm", "num_inference_steps",
        "use_ema", "gripper_threshold", "gripper_open_val",
        "gripper_close_val", "gripper_head_hidden_dim",
        "gripper_hysteresis_window",
    }

    def test_all_keys_present(self):
        assert set(POLICY_DEFAULTS.keys()) == self.EXPECTED_KEYS

    def test_types(self):
        assert isinstance(POLICY_DEFAULTS["obs_horizon"], int)
        assert isinstance(POLICY_DEFAULTS["target_image_size"], tuple)
        assert isinstance(POLICY_DEFAULTS["use_ema"], bool)
        assert isinstance(POLICY_DEFAULTS["gripper_threshold"], float)

    def test_algorithm_in_supported(self):
        assert POLICY_DEFAULTS["algorithm"] in SUPPORTED_ALGORITHMS


# ═══════════════════════════════════════════════════════════════════════════
# _get_state_dim_for_mode
# ═══════════════════════════════════════════════════════════════════════════

class TestGetStateDim:
    @pytest.mark.parametrize("mode,expected", [
        ("joint_only", 7),
        ("ee_only", 8),
        ("both", 14),
    ])
    def test_known_modes(self, mode, expected, default_policy_config):
        policy = RealPolicy(default_policy_config)
        assert policy._get_state_dim_for_mode(mode) == expected

    def test_unknown_mode_defaults_to_7(self, default_policy_config):
        policy = RealPolicy(default_policy_config)
        assert policy._get_state_dim_for_mode("xyz_unknown") == 7


# ═══════════════════════════════════════════════════════════════════════════
# __init__ config override
# ═══════════════════════════════════════════════════════════════════════════

class TestRealPolicyInit:
    def test_defaults(self):
        policy = RealPolicy({})
        for key in ("obs_horizon", "pred_horizon", "action_dim", "algorithm"):
            assert getattr(policy, key) == POLICY_DEFAULTS[key]

    def test_config_overrides(self):
        config = {"obs_horizon": 4, "algorithm": "flow_matching", "pred_horizon": 8}
        policy = RealPolicy(config)
        assert policy.obs_horizon == 4
        assert policy.algorithm == "flow_matching"
        assert policy.pred_horizon == 8

    def test_device_set(self):
        policy = RealPolicy({})
        assert policy.device.type in ("cuda", "cpu")

    def test_not_loaded_initially(self):
        policy = RealPolicy({})
        assert policy.loaded is False


# ═══════════════════════════════════════════════════════════════════════════
# build_state_from_obs
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildState:
    @pytest.fixture()
    def qpos_joint(self):
        return np.arange(7, dtype=np.float32)

    @pytest.fixture()
    def qpos_end(self):
        return np.arange(8, dtype=np.float32) + 10

    def test_joint_only(self, qpos_joint, qpos_end):
        policy = RealPolicy({"state_mode": "joint_only"})
        s = policy.build_state_from_obs(qpos_joint, qpos_end)
        np.testing.assert_array_equal(s, qpos_joint)

    def test_ee_only(self, qpos_joint, qpos_end):
        policy = RealPolicy({"state_mode": "ee_only"})
        s = policy.build_state_from_obs(qpos_joint, qpos_end)
        np.testing.assert_array_equal(s, qpos_end)

    def test_both(self, qpos_joint, qpos_end):
        policy = RealPolicy({"state_mode": "both"})
        s = policy.build_state_from_obs(qpos_joint, qpos_end)
        assert s.shape == (14,)
        np.testing.assert_array_equal(s[:7], qpos_joint)
        np.testing.assert_array_equal(s[7:], qpos_end[:7])


# ═══════════════════════════════════════════════════════════════════════════
# _preprocess_image
# ═══════════════════════════════════════════════════════════════════════════

class TestPreprocessImage:
    def test_output_shape(self, default_policy_config, random_image):
        policy = RealPolicy(default_policy_config)
        img = random_image(256, 256)
        result = policy._preprocess_image(img)
        assert result.shape == (3, 128, 128)

    def test_custom_size(self, random_image):
        policy = RealPolicy({"target_image_size": (64, 64)})
        result = policy._preprocess_image(random_image(200, 300))
        assert result.shape == (3, 64, 64)


# ═══════════════════════════════════════════════════════════════════════════
# _update_obs_history / reset
# ═══════════════════════════════════════════════════════════════════════════

class TestObsHistory:
    def test_obs_horizon_trim(self, default_policy_config):
        policy = RealPolicy(default_policy_config)  # obs_horizon=2
        for i in range(5):
            policy._update_obs_history(np.zeros((3, 128, 128)), np.zeros(7))
        assert len(policy.obs_history["rgb"]) == 2

    def test_padding_on_first(self, default_policy_config):
        policy = RealPolicy(default_policy_config)
        policy._update_obs_history(np.ones((3, 128, 128)), np.ones(7))
        # Should pad to obs_horizon=2
        assert len(policy.obs_history["rgb"]) == 2

    def test_reset(self, default_policy_config):
        policy = RealPolicy(default_policy_config)
        policy._update_obs_history(np.zeros((3, 128, 128)), np.zeros(7))
        policy.reset()
        assert len(policy.obs_history["rgb"]) == 0
        assert policy._last_gripper_state == 0


# ═══════════════════════════════════════════════════════════════════════════
# load_model (fake checkpoint)
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadModel:
    def test_loads_all_components(self, default_policy_config, fake_checkpoint_dir):
        policy = RealPolicy(default_policy_config)
        ckpt_path = str(fake_checkpoint_dir / "model.pt")
        policy.load_model(ckpt_path)
        assert policy.loaded
        assert policy.visual_encoder is not None
        assert policy.state_encoder is not None
        assert policy.agent is not None
        assert policy.gripper_head is not None

    def test_file_not_found(self, default_policy_config):
        policy = RealPolicy(default_policy_config)
        with pytest.raises(FileNotFoundError):
            policy.load_model("/nonexistent/model.pt")

    def test_no_args_json_uses_defaults(self, default_policy_config, fake_checkpoint_dir, tmp_path):
        """When args.json is missing, load_model should still succeed with defaults."""
        import shutil
        ckpt_dir = tmp_path / "no_args"
        ckpt_dir.mkdir()
        shutil.copy(str(fake_checkpoint_dir / "model.pt"), str(ckpt_dir / "model.pt"))
        # Do NOT copy args.json
        policy = RealPolicy(default_policy_config)
        policy.load_model(str(ckpt_dir / "model.pt"))
        assert policy.loaded


# ═══════════════════════════════════════════════════════════════════════════
# _encode_observations
# ═══════════════════════════════════════════════════════════════════════════

class TestEncodeObservations:
    def test_output_shape(self, default_policy_config, fake_checkpoint_dir):
        policy = RealPolicy(default_policy_config)
        policy.load_model(str(fake_checkpoint_dir / "model.pt"))
        # Feed obs
        img = np.random.randint(0, 256, (3, 128, 128), dtype=np.uint8)
        state = np.random.randn(7).astype(np.float32)
        policy._update_obs_history(img, state)

        with torch.no_grad():
            features = policy._encode_observations()

        # obs_features: [B, T, visual_dim + state_encoder_out_dim]
        assert features.shape[0] == 1
        assert features.shape[1] == 2  # obs_horizon


# ═══════════════════════════════════════════════════════════════════════════
# _apply_gripper_hysteresis
# ═══════════════════════════════════════════════════════════════════════════

class TestGripperHysteresis:
    def test_all_open(self, default_policy_config):
        policy = RealPolicy(default_policy_config)
        cls = np.zeros(16, dtype=np.int64)  # all open
        vals = policy._apply_gripper_hysteresis(cls)
        assert vals[0] == pytest.approx(policy.gripper_open_val)

    def test_all_close(self, default_policy_config):
        policy = RealPolicy(default_policy_config)
        cls = np.ones(16, dtype=np.int64)  # all close
        vals = policy._apply_gripper_hysteresis(cls)
        assert vals[0] == pytest.approx(policy.gripper_close_val)

    def test_output_length(self, default_policy_config):
        policy = RealPolicy(default_policy_config)
        cls = np.zeros(16, dtype=np.int64)
        vals = policy._apply_gripper_hysteresis(cls)
        assert len(vals) == 16

    def test_hysteresis_window_delays(self):
        policy = RealPolicy({"gripper_hysteresis_window": 5})
        # 5 consecutive CLOSE should eventually switch
        for _ in range(5):
            vals = policy._apply_gripper_hysteresis(np.ones(16, dtype=np.int64))
        assert vals[0] == pytest.approx(policy.gripper_close_val)


# ═══════════════════════════════════════════════════════════════════════════
# _reconstruct_full_action
# ═══════════════════════════════════════════════════════════════════════════

class TestReconstructFullAction:
    def test_full_mode(self, device):
        policy = RealPolicy({"action_dim": 13, "action_dim_full": 15})
        policy.device = device
        actions_cont = torch.randn(16, 13, device=device)
        gripper_vals = np.full(16, 0.04, dtype=np.float32)
        result = policy._reconstruct_full_action(actions_cont, gripper_vals)
        assert result.shape == (16, 15)
        # Gripper slots
        np.testing.assert_allclose(result[:, 6].cpu().numpy(), 0.04, atol=1e-6)
        np.testing.assert_allclose(result[:, 14].cpu().numpy(), 0.04, atol=1e-6)
        # Joint actions preserved
        torch.testing.assert_close(result[:, :6], actions_cont[:, :6])

    def test_ee_only_mode(self, device):
        policy = RealPolicy({"action_dim": 7, "action_dim_full": 8})
        policy.device = device
        actions_cont = torch.randn(16, 7, device=device)
        gripper_vals = np.full(16, 0.078, dtype=np.float32)
        result = policy._reconstruct_full_action(actions_cont, gripper_vals)
        assert result.shape == (16, 8)
        np.testing.assert_allclose(result[:, 7].cpu().numpy(), 0.078, atol=1e-6)
        torch.testing.assert_close(result[:, :7], actions_cont[:, :7])


# ═══════════════════════════════════════════════════════════════════════════
# predict (end-to-end)
# ═══════════════════════════════════════════════════════════════════════════

class TestPredictE2E:
    def test_predict_returns_correct_shape(self, default_policy_config, fake_checkpoint_dir, device):
        policy = RealPolicy(default_policy_config)
        policy.load_model(str(fake_checkpoint_dir / "model.pt"))

        img = torch.randint(0, 256, (1, 128, 128, 3), dtype=torch.uint8)
        state = torch.randn(1, 7)

        result = policy({"image": img, "qpos": state})
        assert "a_hat" in result
        assert result["a_hat"].shape == (1, 16, 15)

    def test_predict_requires_loaded_model(self, default_policy_config):
        policy = RealPolicy(default_policy_config)
        with pytest.raises(RuntimeError, match="not loaded"):
            policy({"image": torch.zeros(1, 3, 128, 128), "qpos": torch.zeros(1, 7)})
