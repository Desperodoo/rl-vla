"""Tests for inference.dry_run.DryRunEnvironment."""

import sys
import os
import types
import tempfile
import numpy as np
import pytest
import h5py

# Path setup
_CARM_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RL_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_CARM_DEPLOY_ROOT)))
for p in (_CARM_DEPLOY_ROOT, _RL_VLA_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Mock rospy
if "rospy" not in sys.modules:
    _rospy_mock = types.ModuleType("rospy")
    _rospy_mock.loginfo = lambda *a, **kw: None
    _rospy_mock.logwarn = lambda *a, **kw: None
    _rospy_mock.logerr = lambda *a, **kw: None
    sys.modules["rospy"] = _rospy_mock

from inference.dry_run import DryRunEnvironment


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic environment
# ═══════════════════════════════════════════════════════════════════════════

class TestSynthetic:
    def test_create(self):
        env = DryRunEnvironment.synthetic(num_frames=10)
        assert env.num_frames == 10

    def test_get_observation_keys(self):
        env = DryRunEnvironment.synthetic(num_frames=5)
        obs = env.get_observation()
        assert obs is not None
        assert 'stamp' in obs
        assert 'images' in obs
        assert 'qpos_joint' in obs
        assert 'qpos_end' in obs
        assert 'gripper' in obs
        assert 'qpos' in obs

    def test_observation_types(self):
        env = DryRunEnvironment.synthetic(num_frames=5, image_size=(120, 160))
        obs = env.get_observation()
        assert isinstance(obs['stamp'], float)
        assert isinstance(obs['images'], list)
        assert len(obs['images']) == 1
        assert obs['images'][0].shape == (120, 160, 3)
        assert obs['images'][0].dtype == np.uint8
        assert isinstance(obs['qpos_joint'], list)
        assert len(obs['qpos_joint']) == 7
        assert isinstance(obs['qpos_end'], list)
        assert len(obs['qpos_end']) == 8
        assert isinstance(obs['gripper'], float)
        assert isinstance(obs['qpos'], np.ndarray)
        assert obs['qpos'].shape == (15,)

    def test_loop(self):
        env = DryRunEnvironment.synthetic(num_frames=3, loop=True)
        obs1 = env.get_observation()
        env.get_observation()
        env.get_observation()  # exhaust
        obs4 = env.get_observation()  # should wrap around
        assert obs4 is not None
        assert obs1['stamp'] == obs4['stamp']

    def test_no_loop(self):
        env = DryRunEnvironment.synthetic(num_frames=2, loop=False)
        env.get_observation()
        env.get_observation()
        obs3 = env.get_observation()  # should be None
        assert obs3 is None


# ═══════════════════════════════════════════════════════════════════════════
# HDF5 loading
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def fake_hdf5(tmp_path):
    """Create a minimal HDF5 episode file."""
    path = str(tmp_path / 'test_episode.hdf5')
    T = 5
    with h5py.File(path, 'w') as f:
        obs = f.create_group('observations')
        obs.create_dataset('timestamps', data=np.arange(T, dtype=np.float64))
        obs.create_dataset('images', data=np.zeros((T, 60, 80, 3), dtype=np.uint8))
        obs.create_dataset('qpos_joint', data=np.random.randn(T, 7))
        obs.create_dataset('qpos_end', data=np.random.randn(T, 8))
        f.attrs['num_steps'] = T
    return path


class TestFromHDF5:
    def test_load(self, fake_hdf5):
        env = DryRunEnvironment.from_hdf5(fake_hdf5)
        assert env.num_frames == 5

    def test_observation_from_hdf5(self, fake_hdf5):
        env = DryRunEnvironment.from_hdf5(fake_hdf5)
        obs = env.get_observation()
        assert obs is not None
        assert obs['images'][0].shape == (60, 80, 3)
        assert len(obs['qpos_joint']) == 7
        assert len(obs['qpos_end']) == 8

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            DryRunEnvironment.from_hdf5('/nonexistent/path.hdf5')


# ═══════════════════════════════════════════════════════════════════════════
# Real HDF5 data (if available)
# ═══════════════════════════════════════════════════════════════════════════

_REAL_HDF5 = os.path.join(
    _RL_VLA_ROOT, 'data', 'test_gap_fix', 'episode_0001_20260312_162208.hdf5',
)


@pytest.mark.skipif(
    not os.path.exists(_REAL_HDF5), reason='Real HDF5 data not available',
)
class TestRealHDF5:
    def test_load_real(self):
        env = DryRunEnvironment.from_hdf5(_REAL_HDF5, loop=False)
        assert env.num_frames > 0

    def test_replay_all_frames(self):
        env = DryRunEnvironment.from_hdf5(_REAL_HDF5, loop=False)
        count = 0
        while True:
            obs = env.get_observation()
            if obs is None:
                break
            count += 1
            assert len(obs['qpos_joint']) == 7
            assert len(obs['qpos_end']) == 8
        assert count == env.num_frames


# ═══════════════════════════════════════════════════════════════════════════
# Action recording
# ═══════════════════════════════════════════════════════════════════════════

class TestActionRecording:
    def test_record_actions(self):
        env = DryRunEnvironment.synthetic(num_frames=3)
        env.end_control_nostep(np.zeros(8))
        env.end_control_nostep(np.ones(8))
        assert len(env.actions_sent) == 2
        np.testing.assert_array_equal(env.actions_sent[0], np.zeros(8))
        np.testing.assert_array_equal(env.actions_sent[1], np.ones(8))

    def test_reset_clears_actions(self):
        env = DryRunEnvironment.synthetic(num_frames=3)
        env.end_control_nostep(np.zeros(8))
        env.reset()
        assert len(env.actions_sent) == 0
        assert env.current_step == 0

    def test_shutdown_noop(self):
        env = DryRunEnvironment.synthetic(num_frames=3)
        env.shutdown()  # should not raise

    def test_init_status_noop(self):
        env = DryRunEnvironment.synthetic(num_frames=3)
        env.init_status()  # should not raise
