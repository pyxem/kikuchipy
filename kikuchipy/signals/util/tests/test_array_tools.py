# Copyright 2019-2023 The kikuchipy developers
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

import kikuchipy as kp


class TestGridIndices:
    @pytest.mark.parametrize(
        "grid_shape, nav_shape, desired_idx, desired_spacing",
        [
            (
                (2, 2),
                (9, 9),
                np.array([[[3, 3], [6, 6]], [[3, 6], [3, 6]]]),
                (3, 3),
            ),
            (
                (4, 5),
                (55, 75),
                np.array(
                    [
                        [
                            [11, 11, 11, 11, 11],
                            [22, 22, 22, 22, 22],
                            [33, 33, 33, 33, 33],
                            [44, 44, 44, 44, 44],
                        ],
                        [
                            [12, 25, 38, 51, 64],
                            [12, 25, 38, 51, 64],
                            [12, 25, 38, 51, 64],
                            [12, 25, 38, 51, 64],
                        ],
                    ]
                ),
                (11, 13),
            ),
            (10, 105, np.linspace(8, 98, 10, dtype=int)[np.newaxis, :], 10),
        ],
    )
    def test_grid_indices(self, grid_shape, nav_shape, desired_idx, desired_spacing):
        idx, spacing = kp.signals.util.grid_indices(
            grid_shape, nav_shape, return_spacing=True
        )
        assert np.allclose(idx, desired_idx)
        assert np.allclose(spacing, desired_spacing)

    def test_grid_indices_raises(self):
        with pytest.raises(ValueError, match="`grid_shape` and `nav_shape` must both "):
            _ = kp.signals.util.grid_indices(4, (55, 75))
