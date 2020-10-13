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

import pytest
import numpy as np
import dask.array as da

from kikuchipy.indexing.similarity_metrics import (
    make_similarity_metric,
    SimilarityMetric,
    MetricScope,
    FlatSimilarityMetric,
    SIMILARITY_METRICS,
    _zncc_einsum,
    _ndp_einsum,
)
from kikuchipy.indexing._util import (
    _get_nav_shape,
    _get_sig_shape,
    _get_number_of_templates,
)
from scipy.spatial.distance import cdist


class TestSimilarityMetrics:
    @pytest.mark.parametrize(
        "flat,returned_class",
        [
            (False, SimilarityMetric),
            (True, FlatSimilarityMetric),
        ],
    )
    def test_make_similarity_metric(self, flat, returned_class):
        assert (
            type(
                make_similarity_metric(
                    lambda p, t: np.zeros((2, 4))
                    if flat
                    else np.zeros((2, 2, 2)),
                    flat=flat,
                    scope=MetricScope.MANY_TO_MANY,
                )
            )
            is returned_class
        )

    def test_zncc(self):
        zncc_metric = SIMILARITY_METRICS["zncc"]
        p = np.array(
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[9, 8], [1, 7]], [[5, 2], [2, 7]]],
            ],
            np.int8,
        )
        p_da = da.from_array(p)

        # One perfect match and one close match
        t = np.array(
            [[[5, 3], [2, 7]], [[9, 8], [1, 7]]],
            np.int8,
        )
        t_da = da.from_array(t)

        # many to many
        assert pytest.approx(zncc_metric(p_da, t_da).compute()[1, 1, 0]) == 1

        # Working with lower scopes, here one to many:
        assert pytest.approx(zncc_metric(p_da[1, 0], t_da).compute()[1]) == 1

    def test_ndp(self):
        ndp_metric = SIMILARITY_METRICS["ndp"]
        p = np.array(
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[9, 8], [1, 7]], [[5, 2], [2, 7]]],
            ],
            np.int8,
        )
        p_da = da.from_array(p)

        # One perfect match and one close match
        t = np.array(
            [[[5, 3], [2, 7]], [[9, 8], [1, 7]]],
            np.int8,
        )
        t_da = da.from_array(t)

        # many to many
        assert pytest.approx(ndp_metric(p_da, t_da).compute()[1, 1, 0]) == 1

    def test_flat_metric(self):
        p = np.array(
            [
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                [[[9, 8], [1, 7]], [[5, 2], [2, 7]]],
            ],
            np.int8,
        )
        t = np.array(
            [[[5, 3], [2, 7]], [[9, 8], [1, 7]]],
            np.int8,
        )
        euclidean_metric = make_similarity_metric(
            lambda p, t: cdist(p, t, metric="euclidean"),
            greater_is_better=False,
            flat=True,
            scope=MetricScope.MANY_TO_MANY,
            make_compatible_to_lower_scopes=True,
        )
        assert (
            euclidean_metric._is_compatible(p, t) == True
            and pytest.approx(euclidean_metric(p, t)[2, 1]) == 0
        )

    def test_make_compatible_to_lower_scopes(self):
        zncc_metric = SIMILARITY_METRICS["zncc"]
        assert zncc_metric._is_compatible(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_too_large_scoped_inputs(self):
        metric = make_similarity_metric(
            lambda p, t: 1.0, scope=MetricScope.ONE_TO_ONE
        )
        assert (
            metric._is_compatible(np.zeros((2, 2, 2, 2)), np.zeros((4, 2, 2)))
            == False
        )

    def test_too_small_scoped_inputs(self):
        metric = make_similarity_metric(
            lambda p, t: np.zeros((2, 2, 2)), scope=MetricScope.MANY_TO_MANY
        )
        assert (
            metric._is_compatible(np.zeros((2, 2)), np.zeros((2, 2))) == False
        )

    def test_get_number_of_templates(self):
        t = np.array(
            [[[5, 3], [2, 7]], [[9, 8], [1, 7]]],
            np.int8,
        )
        assert (
            _get_number_of_templates(t) == 2
            and _get_number_of_templates(t[0]) == 1
        )
