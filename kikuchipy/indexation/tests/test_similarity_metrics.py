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

from kikuchipy.indexation.similarity_metrics import (
    make_similarity_metric,
    SimilarityMetric,
    MetricScope,
    FlatSimilarityMetric,
    SIMILARITY_METRICS,
    normalize,
    zero_mean,
    expand_dims_to_many_to_many,
    _zncc_einsum,
    _ndp_einsum,
)


# working for many_to_one and lower
def sum_absolute_difference(p, t, flat):
    axis = 1 if flat else (2, 3)
    return np.sum(np.abs(p - t), axis=axis).squeeze()


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
                    lambda p, t: sum_absolute_difference(p, t, flat),
                    flat=flat,
                    scope=MetricScope.MANY_TO_ONE,
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
        assert 1 == pytest.approx(zncc_metric(p_da, t_da).compute()[1, 1, 0])

        # Working with lower scopes, here one to many:
        assert 1 == pytest.approx(zncc_metric(p_da[1, 0], t_da).compute()[1])
