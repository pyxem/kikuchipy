#
# Copyright 2019-2026 the kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.
#

import dask.array as da
import numpy as np
import pytest

import kikuchipy as kp
from kikuchipy.indexing.similarity_metrics._normalized_cross_correlation import (
    _ncc_single_patterns_1d_float32_exp_centered,
    _ncc_single_patterns_2d_ref_normalized,
    _zero_mean_normalize_pattern_numpy,
)


class TestSimilarityMetric:
    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="Data type float16 not among"):
            kp.indexing.NormalizedCrossCorrelationMetric(
                dtype=np.float16
            ).raise_error_if_invalid()

    def test_metric_repr(self):
        ncc = kp.indexing.NormalizedCrossCorrelationMetric(1, 1)
        assert repr(ncc) == (
            "NormalizedCrossCorrelationMetric: float32, greater is better, "
            "rechunk: False, navigation mask: False, signal mask: False"
        )


class TestNumbaAcceleratedMetrics:
    def test_ncc_single_patterns_1d_float32(self):
        exp = np.linspace(0, 0.5, 100, dtype=np.float32)
        sim = np.linspace(0.5, 1, 100, dtype=np.float32)
        exp -= np.mean(exp)
        exp_squared_norm = np.square(exp).sum()

        r1 = _ncc_single_patterns_1d_float32_exp_centered(exp, sim, exp_squared_norm)
        r2 = _ncc_single_patterns_1d_float32_exp_centered.py_func(
            exp, sim, exp_squared_norm
        )
        assert np.isclose(r1, r2)
        assert np.isclose(r1, 0.99999994, atol=1e-8)


class TestNCCSinglePattern:
    def test_zero_mean_normalize_pattern(self):
        rng = np.random.default_rng(0)
        pattern = rng.random((10, 12))
        pattern_copy = pattern.copy()

        pattern_normalized = _zero_mean_normalize_pattern_numpy(pattern)
        assert pattern_normalized.shape == (pattern.size,)
        assert np.isclose(pattern_normalized.mean(), 0)
        assert np.isclose(np.linalg.norm(pattern_normalized), 1)
        # Input is not overwritten
        assert np.array_equal(pattern, pattern_copy)

        # Only pixels not masked out are kept
        signal_mask = np.zeros((10, 12), dtype=bool)
        signal_mask[:2] = True
        pattern_normalized2 = _zero_mean_normalize_pattern_numpy(pattern, signal_mask)
        assert pattern_normalized2.shape == (pattern.size - 2 * 12,)

    def test_ncc(self):
        rng = np.random.default_rng(0)
        pattern = rng.integers(0, 255, (10, 12), dtype=np.uint8)
        reference = _zero_mean_normalize_pattern_numpy(pattern)
        assert np.isclose(_ncc_single_patterns_2d_ref_normalized(pattern, reference), 1)

        # Negated pattern is anti-correlated
        pattern_neg = -pattern.astype(np.float64)
        ncc_neg = _ncc_single_patterns_2d_ref_normalized(pattern_neg, reference)
        assert np.isclose(ncc_neg, -1)

    def test_ncc_signal_mask(self):
        """Masked out pixels do not affect the NCC."""
        rng = np.random.default_rng(0)
        pattern = rng.random((10, 12))
        signal_mask = np.zeros((10, 12), dtype=bool)
        signal_mask[:2] = True
        reference = _zero_mean_normalize_pattern_numpy(pattern, signal_mask)

        pattern2 = pattern.copy()
        pattern2[:2] = 100
        ncc = _ncc_single_patterns_2d_ref_normalized(pattern2, reference, signal_mask)
        assert np.isclose(ncc, 1)

    def test_ncc_flat_pattern(self):
        """A flat pattern gives an NCC of zero, without warnings."""
        rng = np.random.default_rng(0)
        reference = _zero_mean_normalize_pattern_numpy(rng.random((10, 12)))
        flat = np.full((10, 12), 7, dtype=np.uint8)
        assert _zero_mean_normalize_pattern_numpy(flat).sum() == 0
        assert _ncc_single_patterns_2d_ref_normalized(flat, reference) == 0

    def test_metric_flat_pattern(self):
        """The NCC metric returns zero instead of NaN for flat patterns,
        for both NumPy and Dask arrays.
        """
        metric = kp.indexing.NormalizedCrossCorrelationMetric()
        patterns = np.ones((2, 3, 3), dtype=np.float32)
        patterns[1] = np.arange(9).reshape((3, 3))
        patterns = patterns.reshape((2, -1))
        for arr in [patterns, da.from_array(patterns)]:
            prepared = metric._zero_mean_normalize_patterns(arr.copy())
            prepared = np.asarray(prepared)
            assert not np.isnan(prepared).any()
            assert np.allclose(prepared[0], 0)
