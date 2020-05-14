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

import numpy as np
import pytest

from kikuchipy.pattern.correlate import normalized_correlation_coefficient


class TestNormalizedCrossCorrelation:
    @pytest.mark.parametrize(
        "pattern_idx, template_idx, answer",
        [((0, 0), (0, 1), 0.4935737), ((0, 0), (0, 0), 1.0000000)],
    )
    def test_normalised_correlation_coefficient(
        self, dummy_signal, pattern_idx, template_idx, answer
    ):
        coefficient = normalized_correlation_coefficient(
            pattern=dummy_signal.inav[pattern_idx].data,
            template=dummy_signal.inav[template_idx].data,
            zero_normalised=True,
        )
        assert np.allclose(coefficient, answer, atol=1e-7)
