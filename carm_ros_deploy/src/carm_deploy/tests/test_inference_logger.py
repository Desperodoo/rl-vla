"""Tests for inference.inference_logger — InferenceLogger."""

import json
import numpy as np
import pytest
import h5py

import sys, os
_CARM_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CARM_DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, _CARM_DEPLOY_ROOT)

from inference.inference_logger import InferenceLogger


class TestStartEndEpisode:
    def test_creates_hdf5(self, tmp_path):
        logger = InferenceLogger(log_dir=str(tmp_path))
        logger.start_episode("test_ep")
        path = logger.end_episode()
        assert os.path.exists(path)
        assert path.endswith(".hdf5")

    def test_hdf5_has_groups(self, tmp_path):
        logger = InferenceLogger(log_dir=str(tmp_path))
        logger.start_episode("test_ep")
        logger.end_episode()
        with h5py.File(str(tmp_path / "test_ep.hdf5"), "r") as f:
            assert "observations" in f
            assert "predictions" in f
            assert "timing" in f
            assert "safety" in f


class TestLogStep:
    def test_log_and_read(self, tmp_path):
        logger = InferenceLogger(log_dir=str(tmp_path), buffer_size=10)
        logger.start_episode("ep1")

        for i in range(5):
            logger.log_step(
                timestamp=float(i),
                obs={"qpos_joint": np.zeros(7), "qpos_end": np.ones(8)},
                raw_action=np.random.randn(15),
                executed_action=np.random.randn(15),
                inference_time=0.01 * i,
                safety_clipped=(i % 2 == 0),
            )

        path = logger.end_episode()

        with h5py.File(path, "r") as f:
            # 5 steps written
            obs_keys = list(f["observations"].keys())
            assert len(obs_keys) == 5
            # Check data
            np.testing.assert_array_equal(
                f["observations/step_000000/qpos_joint"][()], np.zeros(7)
            )

    def test_auto_flush(self, tmp_path):
        """Buffer auto-flushes at buffer_size."""
        logger = InferenceLogger(log_dir=str(tmp_path), buffer_size=3)
        logger.start_episode("ep2")

        for i in range(5):
            logger.log_step(timestamp=float(i))

        # After 5 steps with buffer_size=3, first 3 should have been flushed
        # Internal buffer should have 2 remaining
        assert len(logger.current_episode) == 2
        logger.end_episode()


class TestRunInfo:
    def test_run_info_json(self, tmp_path):
        logger = InferenceLogger(log_dir=str(tmp_path))
        logger.set_metadata(model_path="/fake/model.pt", config={"lr": 0.001})
        logger.start_episode()
        logger.log_step(timestamp=0.0)
        logger.end_episode()

        run_info_files = [f for f in os.listdir(str(tmp_path)) if f.startswith("run_info")]
        assert len(run_info_files) >= 1

        with open(str(tmp_path / run_info_files[0]), "r") as f:
            info = json.load(f)
        assert info["model"]["path"] == "/fake/model.pt"
        assert info["total_steps"] == 1


class TestSetMetadata:
    def test_model_config(self, tmp_path):
        logger = InferenceLogger(log_dir=str(tmp_path))
        logger.set_metadata(
            model_config={"algorithm": "flow_matching"},
            safety_config={"check_workspace": True},
        )
        assert logger.run_info["model"]["algorithm"] == "flow_matching"
        assert logger.run_info["safety"]["check_workspace"] is True


class TestGetSummary:
    def test_empty_summary(self, tmp_path):
        logger = InferenceLogger(log_dir=str(tmp_path))
        assert logger.get_summary() == {}

    def test_summary_with_data(self, tmp_path):
        logger = InferenceLogger(log_dir=str(tmp_path), buffer_size=100)
        logger.start_episode("ep")
        for i in range(10):
            logger.log_step(timestamp=float(i), inference_time=0.01, safety_clipped=(i < 3))
        summary = logger.get_summary()
        assert summary["num_steps"] == 10
        assert summary["avg_inference_time"] == pytest.approx(0.01)
        assert summary["safety_clips"] == 3
        assert summary["safety_clip_rate"] == pytest.approx(0.3)
        logger.end_episode()


class TestMultipleEpisodes:
    def test_sequential_episodes(self, tmp_path):
        logger = InferenceLogger(log_dir=str(tmp_path))
        for ep_i in range(3):
            logger.start_episode(f"ep_{ep_i}")
            logger.log_step(timestamp=0.0)
            logger.end_episode()
        hdf5_files = [f for f in os.listdir(str(tmp_path)) if f.endswith(".hdf5")]
        assert len(hdf5_files) == 3
