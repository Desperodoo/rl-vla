"""Tests for inference.inference_ros.InferenceNode — mock ROS, test pure computation."""

import sys
import os
import types
import importlib
import importlib.util
import numpy as np
import pytest
from unittest import mock

# ---------------------------------------------------------------------------
# Path setup — must come before any mocking / importing
# ---------------------------------------------------------------------------
_CARM_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RL_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_CARM_DEPLOY_ROOT)))
for p in (_CARM_DEPLOY_ROOT, _RL_VLA_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Mock rospy and ROS message packages BEFORE importing inference_ros
# ---------------------------------------------------------------------------
_rospy_mock = types.ModuleType("rospy")
_rospy_mock.loginfo = lambda *a, **kw: None
_rospy_mock.logwarn = lambda *a, **kw: None
_rospy_mock.logerr = lambda *a, **kw: None
_rospy_mock.loginfo_throttle = lambda *a, **kw: None
_rospy_mock.logwarn_throttle = lambda *a, **kw: None
_rospy_mock.is_shutdown = lambda: False
_rospy_mock.ROSInterruptException = type("ROSInterruptException", (Exception,), {})
_rospy_mock.Rate = lambda hz: mock.MagicMock()
_rospy_mock.Publisher = mock.MagicMock()
_rospy_mock.Subscriber = mock.MagicMock()
_rospy_mock.get_param = lambda *a, **kw: None
_rospy_mock.init_node = lambda *a, **kw: None

sys.modules["rospy"] = _rospy_mock

# Mock ROS messages
for mod_name in ("std_msgs", "std_msgs.msg",
                 "sensor_msgs", "sensor_msgs.msg",
                 "geometry_msgs", "geometry_msgs.msg"):
    sys.modules[mod_name] = mock.MagicMock()

# ---------------------------------------------------------------------------
# Mock core package to avoid env_ros importing the CARM hardware SDK
# ---------------------------------------------------------------------------
# Load safety_controller directly (it has no hardware deps)
_sc_path = os.path.join(_CARM_DEPLOY_ROOT, "core", "safety_controller.py")
_spec = importlib.util.spec_from_file_location("core.safety_controller", _sc_path)
_sc_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sc_mod)
sys.modules["core.safety_controller"] = _sc_mod

# Create a fake `core` package that exposes SafetyController without importing env_ros
_core_mock = types.ModuleType("core")
_core_mock.SafetyController = _sc_mod.SafetyController
_core_mock.safety_controller = _sc_mod
sys.modules["core"] = _core_mock

# Mock env_ros entirely
_env_ros_mock = types.ModuleType("core.env_ros")
_env_ros_mock.RealEnvironment = mock.MagicMock()
sys.modules["core.env_ros"] = _env_ros_mock

# Mock InferenceRecorder (may have ROS deps)
_rec_mock = types.ModuleType("inference.inference_recorder")
_rec_mock.InferenceRecorder = mock.MagicMock()
sys.modules["inference.inference_recorder"] = _rec_mock

# Mock keyboard_intervention
_ki_mock = types.ModuleType("utils.keyboard_intervention")
_ki_mock.KeyboardInterventionHandler = mock.MagicMock()
_ki_mock.InterventionApplier = mock.MagicMock()
sys.modules["utils.keyboard_intervention"] = _ki_mock

# ---------------------------------------------------------------------------
# NOW import InferenceNode
# ---------------------------------------------------------------------------
from inference.inference_ros import InferenceNode
import cv2


# ═══════════════════════════════════════════════════════════════════════════
# Test pure-computation methods of InferenceNode
# ═══════════════════════════════════════════════════════════════════════════

def _make_node(config_overrides=None, fake_checkpoint_dir=None):
    """
    Build an InferenceNode with maximum mocking.
    
    We monkey-patch everything that touches ROS, the real env, and file I/O
    so that __init__ completes without real hardware.
    """
    config = {
        "pretrain": "",
        "data_dir": "",
        "safety_config": "",
        "log_dir": "/tmp/carm_test_logs",
        "temporal_factor_k": 0.05,
        "desire_inference_freq": 30,
        "pos_lookahead_step": 1,
        "control_freq": 50,
        "execution_mode": "temporal_ensemble",
        "timeline_enabled": False,
        "record_inference": False,
        "intervention": False,
    }
    if config_overrides:
        config.update(config_overrides)

    # Patch heavy subsystems
    with mock.patch("inference.inference_ros.RealEnvironment") as MockEnv, \
         mock.patch.object(InferenceNode, "_create_policy") as mock_policy, \
         mock.patch.object(InferenceNode, "_create_safety_controller") as mock_safety, \
         mock.patch.object(InferenceNode, "_create_logger") as mock_logger, \
         mock.patch.object(InferenceNode, "_setup_logger_metadata"), \
         mock.patch.object(InferenceNode, "_init_intervention_and_recording", create=True):
        
        # Fake policy
        fake_policy = mock.MagicMock()
        fake_policy.pred_horizon = 16
        fake_policy.obs_horizon = 2
        fake_policy.action_dim_full = 15
        fake_policy.algorithm = "consistency_flow"
        fake_policy.num_inference_steps = 10
        fake_policy.state_mode = "joint_only"
        fake_policy.use_ema = False
        fake_policy.gripper_hysteresis_window = 1
        fake_policy.normalize_actions = False
        mock_policy.return_value = fake_policy

        from core.safety_controller import SafetyController
        mock_safety.return_value = SafetyController()

        from inference.inference_logger import InferenceLogger
        mock_logger.return_value = InferenceLogger(log_dir=config["log_dir"])

        node = InferenceNode(config)
    
    return node


# ═══════════════════════════════════════════════════════════════════════════
# _preprocess_image
# ═══════════════════════════════════════════════════════════════════════════

class TestNodePreprocessImage:
    def test_resize(self):
        node = _make_node()
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = node._preprocess_image(img, target_size=(128, 128))
        assert result.shape == (128, 128, 3)

    def test_preserves_dtype(self):
        node = _make_node()
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = node._preprocess_image(img, target_size=(32, 32))
        assert result.dtype == np.uint8


# ═══════════════════════════════════════════════════════════════════════════
# _normalize_images
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeNormalizeImages:
    def test_output_shape(self):
        node = _make_node()
        obs = {"images": [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)]}
        result = node._normalize_images(obs, target_size=(128, 128))
        assert result.shape == (3, 128, 128)

    def test_hwc_to_chw(self):
        node = _make_node()
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # R channel
        obs = {"images": [img]}
        result = node._normalize_images(obs, target_size=(64, 64))
        # First channel should be 255
        assert result[0, 0, 0] == 255


# ═══════════════════════════════════════════════════════════════════════════
# ActionChunkManager integration
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeActionManager:
    def test_action_manager_exists(self):
        node = _make_node()
        assert node.action_manager is not None

    def test_action_manager_mode(self):
        node = _make_node({"execution_mode": "receding_horizon"})
        assert node.action_manager.execution_mode == "receding_horizon"

    def test_crossfade_steps(self):
        node = _make_node({"execution_mode": "receding_horizon", "crossfade_steps": 5})
        assert node.action_manager.crossfade_steps == 5


# ═══════════════════════════════════════════════════════════════════════════
# Config propagation
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeConfigPropagation:
    def test_control_freq(self):
        node = _make_node({"control_freq": 100})
        assert node.control_freq == 100

    def test_teleop_scale(self):
        node = _make_node({"teleop_scale": 0.6})
        assert node.teleop_scale == 0.6

    def test_pred_horizon_from_policy(self):
        node = _make_node()
        assert node._pred_horizon == 16

    def test_act_horizon_override(self):
        node = _make_node({"act_horizon": 8})
        assert node._act_horizon == 8
