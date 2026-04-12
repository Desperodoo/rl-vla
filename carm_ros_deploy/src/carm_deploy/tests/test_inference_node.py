"""Tests for InferenceNode — mock ROS, test pure computation."""

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

_camera_config_mock = types.ModuleType("core.camera_config")
_camera_config_mock.resolve_camera_config = mock.MagicMock()
sys.modules["core.camera_config"] = _camera_config_mock

# Mock env_ros entirely
_env_ros_mock = types.ModuleType("core.env_ros")
_env_ros_mock.RealEnvironment = mock.MagicMock()
sys.modules["core.env_ros"] = _env_ros_mock

# Mock cv_bridge/image_sync import chain pulled in by utils package
_cv_bridge_mock = types.ModuleType("cv_bridge")
_cv_bridge_mock.CvBridge = mock.MagicMock()
sys.modules["cv_bridge"] = _cv_bridge_mock

_image_sync_mock = types.ModuleType("utils.image_sync")
_image_sync_mock.ImageSynchronizer = mock.MagicMock()
_image_sync_mock.SingleImageSubscriber = mock.MagicMock()
sys.modules["utils.image_sync"] = _image_sync_mock

# Mock policy_loader to avoid importing rlft/torchvision stack
_policy_loader_mock = types.ModuleType("inference.policy_loader")
_policy_loader_mock.PolicyInterface = mock.MagicMock()
_policy_loader_mock.RealPolicy = mock.MagicMock()
sys.modules["inference.policy_loader"] = _policy_loader_mock

# Mock rlft model factory used by inference_ros import
_model_factory_mock = types.ModuleType("rlft.utils.model_factory")
_model_factory_mock.SUPPORTED_ALGORITHMS = ["consistency_flow", "flow_matching"]
sys.modules["rlft.utils.model_factory"] = _model_factory_mock

# Mock InferenceRecorder (may have ROS deps)
_rec_mock = types.ModuleType("inference.inference_recorder")
_rec_mock.InferenceRecorder = mock.MagicMock()
sys.modules["inference.inference_recorder"] = _rec_mock

# Mock episode keyboard controls
_episode_keyboard_mock = types.ModuleType("utils.episode_keyboard")
_episode_keyboard_mock.EpisodeKeyboardHandler = mock.MagicMock()
sys.modules["utils.episode_keyboard"] = _episode_keyboard_mock

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
        "camera_config": mock.MagicMock(
            topics=['/camera/color/image_raw'],
            names=['camera_color_image_raw'],
            primary_name='camera_color_image_raw',
            primary_index=0,
        ),
    }
    if config_overrides:
        config.update(config_overrides)

    # Patch heavy subsystems
    with mock.patch("inference.inference_ros.RealEnvironment") as MockEnv, \
         mock.patch("inference.inference_ros.threading.Thread") as MockThread, \
         mock.patch.object(InferenceNode, "_create_policy") as mock_policy, \
         mock.patch.object(InferenceNode, "_create_safety_controller") as mock_safety, \
         mock.patch.object(InferenceNode, "_create_logger") as mock_logger, \
         mock.patch.object(InferenceNode, "_setup_logger_metadata"), \
         mock.patch.object(InferenceNode, "_init_recording_controls", create=True):
        
        MockThread.return_value = mock.MagicMock()

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

        fake_logger = mock.MagicMock()
        fake_logger.log_dir = config["log_dir"]
        fake_logger.current_file_path = None
        mock_logger.return_value = fake_logger

        node = InferenceNode(config)
    
    return node


# ═══════════════════════════════════════════════════════════════════════════
# _preprocess_image and _normalize_images removed (BUG-1 fix):
# Image preprocessing is now fully handled by RealPolicy.
# See tests/test_policy_loader.py for image preprocessing tests.
# ═══════════════════════════════════════════════════════════════════════════


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

    def test_teleop_scale_fixed(self):
        """teleop_scale is fixed to 1.0 per GAP-2 fix, ignores config input."""
        node = _make_node({"teleop_scale": 0.6})
        assert node.teleop_scale == 1.0

    def test_pred_horizon_from_policy(self):
        node = _make_node()
        assert node._pred_horizon == 16

    def test_act_horizon_override(self):
        node = _make_node({"act_horizon": 8})
        assert node._act_horizon == 8

    def test_hitl_defaults_disabled(self):
        node = _make_node()
        assert node.hitl_mode == "disabled"
        assert node.hitl_enabled is False
        assert node.hitl_human_execute_mode == "direct"

    def test_hitl_candidate_initializes_candidate_manager(self):
        node = _make_node({"hitl_mode": "candidate"})
        assert node.hitl_enabled is True
        assert node.hitl_candidate_manager is not None

    def test_hitl_live_initializes_owner_state(self):
        node = _make_node({"hitl_mode": "live"})
        assert node.hitl_enabled is True
        assert node.hitl_live_owner_active is False
        assert node.hitl_live_state["shared_source"] == "policy"

    def test_hitl_human_execute_mode_configurable(self):
        node = _make_node({"hitl_mode": "live", "hitl_human_execute_mode": "scheduled"})
        assert node.hitl_human_execute_mode == "scheduled"
        assert node.hitl_live_state["human_execute_mode"] == "scheduled"
