"""Tests for utils.trajectory_interpolator — TrajectoryInterpolator & ActionChunkManager."""

import time
import numpy as np
import pytest

import sys, os
_CARM_DEPLOY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CARM_DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, _CARM_DEPLOY_ROOT)

from utils.trajectory_interpolator import TrajectoryInterpolator, ActionChunkManager


# ═══════════════════════════════════════════════════════════════════════════
# TrajectoryInterpolator
# ═══════════════════════════════════════════════════════════════════════════

class TestTrajectoryInterpolator:
    def test_empty_query_returns_none(self):
        ti = TrajectoryInterpolator()
        assert ti.get_once(0.0) is None
        # get_interpolated with <2 points falls back to get_once
        assert ti.get_interpolated(0.0) is None

    def test_single_point_interpolated_fallback(self):
        """get_interpolated with 1 point falls back to get_once."""
        ti = TrajectoryInterpolator()
        ti.append(1.0, [10.0, 20.0])
        result = ti.get_interpolated(1.0)
        np.testing.assert_array_equal(result, [10.0, 20.0])
        # query before the single point
        result2 = ti.get_interpolated(0.5)
        np.testing.assert_array_equal(result2, [10.0, 20.0])
        # query after the single point — expired
        assert ti.get_interpolated(2.0) is None

    def test_append_and_query(self):
        ti = TrajectoryInterpolator()
        ti.append(1.0, [1, 2, 3])
        ti.append(2.0, [4, 5, 6])
        result = ti.get_once(1.5)
        np.testing.assert_array_equal(result, [4, 5, 6])

    def test_time_must_increase(self):
        ti = TrajectoryInterpolator()
        ti.append(2.0, [1])
        ti.append(1.0, [2])  # should be ignored
        assert len(ti) == 1

    def test_interpolation(self):
        ti = TrajectoryInterpolator()
        ti.append(0.0, [0.0])
        ti.append(1.0, [10.0])
        result = ti.get_interpolated(0.5)
        np.testing.assert_allclose(result, [5.0])

    def test_interpolation_out_of_range(self):
        ti = TrajectoryInterpolator()
        ti.append(1.0, [10.0])
        ti.append(2.0, [20.0])
        # Before first
        result = ti.get_interpolated(0.5)
        np.testing.assert_array_equal(result, [10.0])
        # After last
        result = ti.get_interpolated(3.0)
        assert result is None

    def test_clear(self):
        ti = TrajectoryInterpolator()
        ti.append(1.0, [1])
        ti.clear()
        assert len(ti) == 0
        assert ti.empty

    def test_clear_before(self):
        ti = TrajectoryInterpolator()
        for i in range(10):
            ti.append(float(i), [i])
        ti.clear_before(5.0)
        assert ti.oldest_timestamp == 5.0

    def test_latest_timestamp(self):
        ti = TrajectoryInterpolator()
        ti.append(1.0, [0])
        ti.append(3.0, [0])
        assert ti.latest_timestamp == 3.0

    def test_get_once_with_timestamp(self):
        ti = TrajectoryInterpolator()
        ti.append(1.0, [10])
        ti.append(2.0, [20])
        action, ts = ti.get_once_with_timestamp(1.5)
        np.testing.assert_array_equal(action, [20])
        assert ts == 2.0


# ═══════════════════════════════════════════════════════════════════════════
# ActionChunkManager — temporal_ensemble
# ═══════════════════════════════════════════════════════════════════════════

class TestActionChunkManagerTE:
    def _make_traj(self, start_t, values):
        """Create a TrajectoryInterpolator with evenly spaced timesteps."""
        traj = TrajectoryInterpolator()
        for i, v in enumerate(values):
            traj.append(start_t + i * 0.1, np.array(v, dtype=np.float64))
        return traj

    def test_empty_returns_none(self):
        acm = ActionChunkManager(execution_mode='temporal_ensemble')
        assert acm.get_fused_action(0.0) is None

    def test_single_chunk(self):
        acm = ActionChunkManager(execution_mode='temporal_ensemble')
        traj = self._make_traj(0.0, [[1.0], [2.0], [3.0]])
        acm.add_trajectory(traj)
        action = acm.get_fused_action(0.1)
        np.testing.assert_allclose(action, [2.0])

    def test_two_chunks_fuse(self):
        acm = ActionChunkManager(execution_mode='temporal_ensemble', temporal_factor_k=0.01)
        traj1 = self._make_traj(0.0, [[10.0], [10.0], [10.0]])
        traj2 = self._make_traj(0.0, [[20.0], [20.0], [20.0]])
        acm.add_trajectory(traj1)
        acm.add_trajectory(traj2)
        action = acm.get_fused_action(0.0)
        # Both valid → weighted average between 10 and 20
        assert 10.0 < action[0] < 20.0

    def test_all_expired_returns_none(self):
        acm = ActionChunkManager(execution_mode='temporal_ensemble')
        traj = self._make_traj(0.0, [[1.0]])
        acm.add_trajectory(traj)
        result = acm.get_fused_action(10.0)  # far in the future
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# ActionChunkManager — receding_horizon
# ═══════════════════════════════════════════════════════════════════════════

class TestActionChunkManagerRH:
    def _make_traj(self, start_t, values):
        traj = TrajectoryInterpolator()
        for i, v in enumerate(values):
            traj.append(start_t + i * 0.1, np.array(v, dtype=np.float64))
        return traj

    def test_single_chunk(self):
        acm = ActionChunkManager(execution_mode='receding_horizon')
        traj = self._make_traj(0.0, [[5.0], [6.0]])
        acm.add_trajectory(traj)
        action = acm.get_fused_action(0.0)
        np.testing.assert_allclose(action, [5.0])

    def test_latest_chunk_wins(self):
        acm = ActionChunkManager(execution_mode='receding_horizon')
        traj1 = self._make_traj(0.0, [[10.0], [10.0], [10.0]])
        traj2 = self._make_traj(0.0, [[99.0], [99.0], [99.0]])
        acm.add_trajectory(traj1)
        acm.add_trajectory(traj2)
        action = acm.get_fused_action(0.0)
        np.testing.assert_allclose(action, [99.0])

    def test_hold_position_on_expired(self):
        acm = ActionChunkManager(execution_mode='receding_horizon')
        traj = self._make_traj(0.0, [[7.0]])
        acm.add_trajectory(traj)
        acm.get_fused_action(0.0)  # sets _last_action
        action = acm.get_fused_action(10.0)  # expired
        np.testing.assert_allclose(action, [7.0])

    def test_crossfade(self):
        acm = ActionChunkManager(
            execution_mode='receding_horizon',
            crossfade_steps=4,
        )
        traj1 = self._make_traj(0.0, [[0.0], [0.0], [0.0]])
        acm.add_trajectory(traj1)
        _ = acm.get_fused_action(0.0)  # init with 0.0

        # New chunk with value 1.0
        traj2 = self._make_traj(0.3, [[1.0], [1.0], [1.0]])
        acm.add_trajectory(traj2)
        first = acm.get_fused_action(0.3)
        # Should be a blend, not immediately 1.0
        assert first[0] < 1.0
        assert first[0] > 0.0


class TestActionChunkManagerClear:
    def test_clear(self):
        acm = ActionChunkManager()
        traj = TrajectoryInterpolator()
        traj.append(0.0, [1])
        acm.add_trajectory(traj)
        acm.clear()
        assert len(acm) == 0

    def test_get_last_fused_chunk_ids(self):
        acm = ActionChunkManager(execution_mode='temporal_ensemble')
        traj = TrajectoryInterpolator()
        traj.append(0.0, [1])
        cid = acm.add_trajectory(traj)
        acm.get_fused_action(0.0)
        assert cid in acm.get_last_fused_chunk_ids()
