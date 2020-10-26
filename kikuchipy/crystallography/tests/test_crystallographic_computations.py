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

from diffsims.crystallography import ReciprocalLatticePoint
import matplotlib.colors as mcolors
import numpy as np
import pytest
from orix.crystal_map import Phase

from kikuchipy.crystallography._computations import (
    _get_colors_for_allowed_bands,
    _get_hkl_family,
    _get_uvw_from_hkl,
)


class TestCrystallographicComputations:
    @pytest.mark.parametrize(
        "hkl, desired_uvw",
        [
            ([1, 1, 1], np.array([]).reshape((0, 3))),
            ([[1, 1, 1], [2, 2, 2]], np.array([]).reshape((0, 3))),
            ([[1, 1, 1], [1, 1, -1]], [[-1, 1, 0], [1, -1, 0]]),
        ],
    )
    def test_get_uvw_from_hkl(self, hkl, desired_uvw):
        """Desired uvw from the cross product of hkl."""
        assert np.allclose(_get_uvw_from_hkl(hkl), desired_uvw)

    @pytest.mark.parametrize(
        (
            "hkl, desired_family_keys, desired_family_values, desired_indices, "
            "reduce"
        ),
        [
            ([1, 1, 1], [[1, 1, 1]], [1, 1, 1], [0], False),
            ([1, 1, 1], [[1, 1, 1]], [1, 1, 1], [0], True),
            (
                [[1, 1, 1], [2, 0, 0]],
                [[1, 1, 1], [2, 0, 0]],
                [[[1, 1, 1]], [[2, 0, 0]]],
                [[0], [1]],
                False,
            ),
            (
                ReciprocalLatticePoint(
                    phase=Phase(space_group=225), hkl=[1, 1, 1]
                )
                .symmetrise()
                ._hkldata,
                [1, 1, 1],
                [
                    [
                        [1, 1, 1],
                        [-1, 1, 1],
                        [-1, -1, 1],
                        [1, -1, 1],
                        [1, -1, -1],
                        [1, 1, -1],
                        [-1, 1, -1],
                        [-1, -1, -1],
                    ]
                ],
                [np.arange(8)],
                False,
            ),
            (
                ReciprocalLatticePoint(
                    phase=Phase(space_group=225), hkl=[[1, 1, 1], [2, 0, 0]]
                )
                .symmetrise()
                ._hkldata,
                [[1, 1, 1], [2, 0, 0]],
                [
                    [
                        [1, 1, 1],
                        [-1, 1, 1],
                        [-1, -1, 1],
                        [1, -1, 1],
                        [1, -1, -1],
                        [1, 1, -1],
                        [-1, 1, -1],
                        [-1, -1, -1],
                    ],
                    [
                        [2, 0, 0],
                        [0, 2, 0],
                        [-2, 0, 0],
                        [0, -2, 0],
                        [0, 0, 2],
                        [0, 0, -2],
                    ],
                ],
                [np.arange(8).tolist(), np.arange(8, 14).tolist()],
                False,
            ),
            (
                ReciprocalLatticePoint(
                    phase=Phase(space_group=225), hkl=[[1, 1, 1], [2, 2, 2]]
                )
                .symmetrise()
                ._hkldata,
                [1, 1, 1],
                [
                    [
                        [1, 1, 1],
                        [-1, 1, 1],
                        [-1, -1, 1],
                        [1, -1, 1],
                        [1, -1, -1],
                        [1, 1, -1],
                        [-1, 1, -1],
                        [-1, -1, -1],
                        [2, 2, 2],
                        [-2, 2, 2],
                        [-2, -2, 2],
                        [2, -2, 2],
                        [2, -2, -2],
                        [2, 2, -2],
                        [-2, 2, -2],
                        [-2, -2, -2],
                    ]
                ],
                [np.arange(16)],
                True,
            ),
        ],
    )
    def test_get_hkl_family(
        self,
        hkl,
        desired_family_keys,
        desired_family_values,
        desired_indices,
        reduce,
    ):
        """Desired sets of families and indices."""
        families, families_idx = _get_hkl_family(hkl, reduce=reduce)

        for i, (k, v) in enumerate(families.items()):
            assert np.allclose(k, desired_family_keys[i])
            assert np.allclose(v, desired_family_values[i])
            assert np.allclose(families_idx[k], desired_indices[i])

    @pytest.mark.parametrize(
        "highest_hkl, color_cycle, desired_hkl_colors",
        [
            ([1, 1, 1], ["C0", "C1"], [[1, 1, 1], [0.12, 0.47, 0.71]]),
            (
                [2, 2, 2],
                ["g", "b"],
                [
                    [[1, 1, 1], [0, 0.5, 0]],
                    [[2, 0, 0], [0, 0, 1]],
                    [[2, 2, 0], [0, 0.5, 0]],
                    [[2, 2, 2], [0, 0.5, 0]],
                ],
            ),
            (
                [2, 2, 2],
                None,
                [
                    [[1, 1, 1], [1, 0, 0]],
                    [[2, 0, 0], [1, 1, 0]],
                    [[2, 2, 0], [0, 1, 0]],
                    [[2, 2, 2], [1, 0, 0]],
                ],
            ),
        ],
    )
    def test_get_colors_for_allowed_bands(
        self, nickel_phase, highest_hkl, color_cycle, desired_hkl_colors
    ):
        """Desired colors for bands."""
        hkl_colors = _get_colors_for_allowed_bands(
            phase=nickel_phase, highest_hkl=highest_hkl, color_cycle=color_cycle
        )

        assert np.allclose(hkl_colors, desired_hkl_colors, atol=1e-2)

    def test_get_colors_for_allowed_bands_999(self, nickel_phase):
        """Not passing `highest_hkl` works fine."""
        hkl_colors = _get_colors_for_allowed_bands(phase=nickel_phase)

        assert np.shape(hkl_colors) == (69, 2, 3)
