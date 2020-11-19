# -*- coding: utf-8 -*-
# Copyright 2019-2020 The kikuchipy developers
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

import dask.array as da
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from kikuchipy.data import nickel_ebsd_small
from kikuchipy.indexing._pattern_matching import _pattern_match
from kikuchipy.indexing.similarity_metrics import (
    make_similarity_metric,
    MetricScope,
)


class TestPatternMatching:
    zncc_flat_metric = make_similarity_metric(
        lambda p, t: cdist(p, t, metric="correlation"),
        greater_is_better=False,
        flat=True,
    )
    dummy_metric = make_similarity_metric(lambda p, t: 1.0)

    def test_not_recognized_metric(self):
        with pytest.raises(ValueError):
            _pattern_match(
                np.zeros((2, 2)), np.zeros((2, 2)), metric="not_recognized"
            )

    def test_mismatching_signal_shapes(self):
        self.dummy_metric.scope = MetricScope.MANY_TO_MANY
        with pytest.raises(OSError):
            _pattern_match(
                np.zeros((2, 2)), np.zeros((3, 3)), metric=self.dummy_metric
            )

    def test_metric_not_compatible_with_data(self):
        self.dummy_metric.scope = MetricScope.ONE_TO_MANY
        with pytest.raises(OSError):
            _pattern_match(
                np.zeros((2, 2, 2, 2)),
                np.zeros((2, 2)),
                metric=self.dummy_metric,
            )

    @pytest.mark.parametrize("n_slices", [1, 2])
    def test_pattern_match_compute_true(self, n_slices):
        # Four patterns
        p = np.array(
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[9, 8], [1, 7]], [[5, 2], [2, 7]]],
            ],
            np.int8,
        )
        # Five templates
        t = np.array(
            [
                [[5, 3], [2, 7]],
                [[9, 8], [1, 7]],
                [[10, 2], [5, 3]],
                [[8, 4], [6, 12]],
                [[43, 0], [5, 3]],
            ],
            np.int8,
        )
        t_da = da.from_array(t)
        mr = _pattern_match(p, t_da, n_slices=n_slices, keep_n=1)
        assert mr[0][2] == 1  # Template index in t of perfect match
        assert np.allclose(mr[1][2], 1.0)  # ZNCC of perfect match

    def test_pattern_match_compute_false(self):
        p = np.arange(16).reshape((2, 2, 2, 2))
        t = np.arange(8).reshape((2, 2, 2))
        mr = _pattern_match(p, t, compute=False)
        assert len(mr) == 2
        assert isinstance(mr[0], da.Array) and isinstance(mr[1], da.Array)

    def test_pattern_match_slices_compute_false(self):
        p = np.arange(16).reshape((2, 2, 2, 2))
        t = np.arange(8).reshape((2, 2, 2))
        with pytest.raises(NotImplementedError):
            _pattern_match(p, t, n_slices=2, compute=False)

    def test_pattern_match_one_to_one(self):
        p = np.random.random(3 * 3).reshape((3, 3))
        mr = _pattern_match(p, p)
        assert mr[0][0] == 0

    def test_pattern_match_phase_name(self):
        """Ensure that the `phase_name` accepts different types."""
        exp = nickel_ebsd_small().data
        sim = exp.reshape((-1,) + exp.shape[-2:])

        sim_idx1, scores1 = _pattern_match(exp, sim, n_slices=2)
        sim_idx2, scores2 = _pattern_match(exp, sim, phase_name="a", n_slices=2)
        sim_idx3, scores3 = _pattern_match(exp, sim, phase_name="", n_slices=2)

        assert np.allclose(sim_idx1[0], [0, 3, 6, 4, 7, 1, 8, 5, 2])
        assert np.allclose(sim_idx2[0], [0, 3, 6, 4, 7, 1, 8, 5, 2])
        assert np.allclose(sim_idx3[0], [0, 3, 6, 4, 7, 1, 8, 5, 2])
