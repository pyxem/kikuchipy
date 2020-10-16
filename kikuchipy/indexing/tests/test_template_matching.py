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


from kikuchipy.indexing import make_similarity_metric, template_match

from kikuchipy.indexing.template_matching import _template_match_slice_templates
from kikuchipy.indexing.similarity_metrics import MetricScope


class TestTemplateMatching:
    zncc_flat_metric = make_similarity_metric(
        lambda p, t: cdist(p, t, metric="correlation"),
        greater_is_better=False,
        flat=True,
    )

    dummy_metric = make_similarity_metric(lambda p, t: 1.0)

    def test_not_recognized_metric(self):
        with pytest.raises(ValueError):
            template_match(
                np.zeros((2, 2)), np.zeros((2, 2)), metric="not_recognized"
            )

    def test_not_accepted_metric(self):
        dummy_metric = make_similarity_metric(
            lambda p, t: 1.0, scope=MetricScope.ONE_TO_ONE
        )
        with pytest.raises(ValueError):
            template_match(
                np.zeros((2, 2)), np.zeros((2, 2)), metric=dummy_metric
            )

    def test_mismatching_signal_shapes(self):
        self.dummy_metric.scope = MetricScope.MANY_TO_MANY
        with pytest.raises(OSError):
            template_match(
                np.zeros((2, 2)), np.zeros((3, 3)), metric=self.dummy_metric
            )

    def test_not_accepted_data(self):
        with pytest.raises(OSError):
            template_match(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_metric_not_compatible_with_data(self):
        self.dummy_metric.scope = MetricScope.ONE_TO_MANY
        with pytest.raises(OSError):
            template_match(np.zeros((2, 2)), np.zeros((2, 2)))

    @pytest.mark.parametrize(
        "n_slices",
        [None, 2],
    )
    def test_template_match_compute_true(self, n_slices):
        # Four patterns
        p = np.array(
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[9, 8], [1, 7]], [[5, 2], [2, 7]]],
            ],
            np.int8,
        )
        # One perfect match, at [1,0,1] in results, and one close match
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
        mr = template_match(p, t_da, n_slices=n_slices)
        assert mr[0][2] == 1  # Template index in t of perfect match
        assert pytest.approx(mr[1][2]) == 1.0  # ZNCC of perfect match

    def test_template_match_compute_false(self):
        p = np.arange(16).reshape((2, 2, 2, 2))
        t = np.arange(8).reshape((2, 2, 2))
        mr = template_match(p, t, compute=False)
        assert len(mr) == 2
        assert isinstance(mr[0], da.Array) and isinstance(mr[1], da.Array)
