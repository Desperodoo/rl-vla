"""Tests for inference.inference_logger — InferenceLogger."""

import json
import os
import sys

import numpy as np
import pytest

_CARM_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CARM_DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, _CARM_DEPLOY_ROOT)

from inference.inference_logger import InferenceLogger


class TestRunInfoOnly:
    def test_creates_run_info(self, tmp_path):
        logger = InferenceLogger(log_dir=str(tmp_path))
        logger.start_episode("test_ep")
        path = logger.end_episode()
        assert os.path.exists(path)
        assert path.endswith("run_info_test_ep.json")
        assert not os.path.exists(tmp_path / "test_ep.hdf5")

    def test_run_info_records_episode_file(self, tmp_path):
        logger = InferenceLogger(log_dir=str(tmp_path))
        logger.start_episode("test_ep")
        logger.record_episode_file("/some/path/inference_episode_0001_20260401_221942.hdf5")
        path = logger.end_episode()

        with open(path, "r") as f:
            info = json.load(f)
        assert info["files"]["episode_hdf5"] == ["inference_episode_0001_20260401_221942.hdf5"]
        assert info["files"]["run_info"] == "run_info_test_ep.json"


class TestLogStep:
    def test_updates_summary(self, tmp_path):
        logger = InferenceLogger(log_dir=str(tmp_path), buffer_size=3)
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

        summary = logger.get_summary()
        assert summary["num_steps"] == 5
        assert summary["safety_clips"] == 3
        assert summary["avg_inference_time"] == pytest.approx(np.mean([0.01, 0.02, 0.03, 0.04]))

        path = logger.end_episode()
        with open(path, "r") as f:
            info = json.load(f)
        assert info["total_steps"] == 5
        assert info["summary"]["num_steps"] == 5


class TestSetMetadata:
    def test_model_config(self, tmp_path):
        logger = InferenceLogger(log_dir=str(tmp_path))
        logger.set_metadata(
            model_config={"algorithm": "flow_matching"},
            safety_config={"check_workspace": True},
        )
        assert logger.run_info["model"]["algorithm"] == "flow_matching"
        assert logger.run_info["safety"]["check_workspace"] is True


class TestMultipleEpisodes:
    def test_sequential_episodes(self, tmp_path):
        logger = InferenceLogger(log_dir=str(tmp_path))
        for ep_i in range(3):
            logger.start_episode(f"ep_{ep_i}")
            logger.log_step(timestamp=0.0)
            logger.end_episode()
        json_files = [f for f in os.listdir(str(tmp_path)) if f.startswith("run_info_")]
        assert len(json_files) == 3
