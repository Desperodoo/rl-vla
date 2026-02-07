"""Tests for ActionNormalizer — fit / transform / inverse / serialization."""

import json
import numpy as np
import pytest

from rlft.datasets.carm_dataset import ActionNormalizer


def _make_actions(n=200, d=13):
    return np.random.randn(n, d).astype(np.float32) * 5 + 2


class TestStandardMode:
    def test_roundtrip(self):
        norm = ActionNormalizer(mode="standard")
        data = _make_actions()
        norm.fit(data)
        transformed = norm.transform(data)
        recovered = norm.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, data, atol=1e-4)

    def test_transform_stats(self):
        norm = ActionNormalizer(mode="standard")
        data = _make_actions()
        norm.fit(data)
        t = norm.transform(data)
        np.testing.assert_allclose(t.mean(axis=0), 0, atol=0.15)
        np.testing.assert_allclose(t.std(axis=0), 1, atol=0.15)


class TestMinmaxMode:
    def test_roundtrip(self):
        norm = ActionNormalizer(mode="minmax")
        data = _make_actions()
        norm.fit(data)
        transformed = norm.transform(data)
        recovered = norm.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, data, atol=1e-4)

    def test_range(self):
        norm = ActionNormalizer(mode="minmax")
        data = _make_actions()
        norm.fit(data)
        t = norm.transform(data)
        assert t.min() >= -1.0 - 1e-6
        assert t.max() <= 1.0 + 1e-6


class TestSerialization:
    def test_save_load_json(self, tmp_path):
        norm = ActionNormalizer(mode="standard")
        norm.fit(_make_actions(d=7))
        path = str(tmp_path / "norm.json")
        norm.save(path)

        norm2 = ActionNormalizer()
        norm2.load(path)
        assert norm2.mode == "standard"
        np.testing.assert_allclose(norm2.stats["mean"], norm.stats["mean"])
        np.testing.assert_allclose(norm2.stats["std"], norm.stats["std"])

    def test_from_checkpoint(self):
        norm = ActionNormalizer(mode="minmax")
        norm.fit(_make_actions(d=13))
        ckpt_data = {
            "mode": norm.mode,
            "stats": {k: v.tolist() for k, v in norm.stats.items()},
        }
        norm2 = ActionNormalizer.from_checkpoint(ckpt_data)
        assert norm2.mode == "minmax"
        data = _make_actions(d=13)
        np.testing.assert_allclose(
            norm2.transform(data), norm.transform(data), atol=1e-6
        )


class TestErrors:
    def test_transform_before_fit(self):
        norm = ActionNormalizer()
        with pytest.raises(ValueError, match="fit"):
            norm.transform(np.zeros((5, 7)))

    def test_inverse_before_fit(self):
        norm = ActionNormalizer()
        with pytest.raises(ValueError, match="fit"):
            norm.inverse_transform(np.zeros((5, 7)))
