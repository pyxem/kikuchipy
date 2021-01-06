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

from kikuchipy.projections.hesse_normal_form import HesseNormalForm
from kikuchipy.projections.spherical_projection import _get_polar


class TestHesseNormalForm:
    @pytest.mark.parametrize(
        "radius, desired_result",
        [
            (
                10,
                np.array(
                    [
                        [0.4020, 1.5306],
                        [0.7127, 1.4995],
                        [0.4734, 1.5234],
                        [0.5902, 1.5117],
                        [1.2904, 1.4414],
                    ]
                ),
            ),
            (
                2,
                np.array(
                    [
                        [0.4020, 1.3684],
                        [0.7127, 1.2064],
                        [0.4734, 1.3318],
                        [0.5902, 1.2712],
                        [1.2904, 0.8695],
                    ]
                ),
            ),
        ],
    )
    def test_project_polar(self, radius, desired_result):
        """Project yields expected result."""
        v = np.array(
            [
                [0.5900, 0.8438, 0.4139],
                [0.4993, 0.7440, 0.6386],
                [0.8894, 0.7788, 0.5596],
                [0.6672, 0.8618, 0.6433],
                [0.7605, 0.0647, 0.9848],
            ]
        )
        polar = _get_polar(v)
        assert np.allclose(
            HesseNormalForm.project_polar(polar, radius=radius),
            desired_result,
            atol=1e-3,
        )

    @pytest.mark.parametrize(
        "radius, desired_result",
        [
            (
                10,
                np.array(
                    [
                        [0.4020, 1.5306],
                        [0.7127, 1.4995],
                        [0.4734, 1.5234],
                        [0.5902, 1.5117],
                        [1.2904, 1.4414],
                    ]
                ),
            ),
            (
                2,
                np.array(
                    [
                        [0.4020, 1.3684],
                        [0.7127, 1.2064],
                        [0.4734, 1.3318],
                        [0.5902, 1.2712],
                        [1.2904, 0.8695],
                    ]
                ),
            ),
        ],
    )
    def test_project_cartesian(self, radius, desired_result):
        """Inverse projection yields expected result."""
        v = np.array(
            [
                [0.5900, 0.8438, 0.4139],
                [0.4993, 0.7440, 0.6386],
                [0.8894, 0.7788, 0.5596],
                [0.6672, 0.8618, 0.6433],
                [0.7605, 0.0647, 0.9848],
            ]
        )
        assert np.allclose(
            HesseNormalForm.project_cartesian(v, radius=radius),
            desired_result,
            atol=1e-3,
        )
