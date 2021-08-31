# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
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

import numpy as np
import pytest

from kikuchipy.indexing.similarity_metrics import (
    NormalizedCrossCorrelationMetric,
)
from kikuchipy.indexing.similarity_metrics._normalized_cross_correlation import (
    _ncc_single_patterns_2d_float32,
)


class TestSimilarityMetric:
    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="Data type float16 not among"):
            NormalizedCrossCorrelationMetric(dtype=np.float16).raise_error_if_invalid()

    def test_metric_repr(self):
        ncc = NormalizedCrossCorrelationMetric(1, 1)
        assert repr(ncc) == (
            "NormalizedCrossCorrelationMetric: float32, greater is better, "
            "rechunk: False, signal mask: False"
        )


class TestNumbaAcceleratedMetrics:
    def test_ncc_single_patterns_2d_float32(self):
        r = _ncc_single_patterns_2d_float32.py_func(
            exp=np.linspace(0, 0.5, 100, dtype=np.float32).reshape((10, 10)),
            sim=np.linspace(0.5, 1, 100, dtype=np.float32).reshape((10, 10)),
        )
        assert r == 1
