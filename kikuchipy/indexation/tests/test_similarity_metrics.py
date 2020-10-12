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
        assert isinstance(
            make_similarity_metric(
                lambda p, t: sum_absolute_difference(p, t, flat),
                flat=flat,
                scope=MetricScope.MANY_TO_ONE,
            ),
            returned_class,
        )
