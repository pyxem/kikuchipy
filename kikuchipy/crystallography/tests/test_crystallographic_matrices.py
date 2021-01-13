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

from diffpy.structure import Lattice
import numpy as np
import pytest

from kikuchipy.crystallography import (
    get_direct_structure_matrix,
    get_reciprocal_structure_matrix,
    get_reciprocal_metric_tensor,
)


class TestCrystallographicMatrices:
    @pytest.mark.parametrize(
        "lattice, desired_matrix",
        [
            (Lattice(3.52, 3.52, 3.52, 90, 90, 90), np.eye(3) / 3.52),
            (
                Lattice(3.52, 3.52, 10.5, 90, 90, 120),
                np.array([[0.284, 0, 0], [0.164, 0.328, 0], [0, 0, 0.095]]),
            ),
        ],
    )
    def test_reciprocal_structure_matrix(self, lattice, desired_matrix):
        assert np.allclose(
            get_reciprocal_structure_matrix(lattice), desired_matrix, atol=1e-3
        )

    @pytest.mark.parametrize(
        "lattice, desired_matrix",
        [
            (Lattice(3.52, 3.52, 3.52, 90, 90, 90), np.eye(3) * 3.52),
            (
                Lattice(3.52, 3.52, 10.5, 90, 90, 120),
                np.array([[3.52, -1.76, 0], [0, 3.048, 0], [0, 0, 10.5]]),
            ),
        ],
    )
    def test_direct_structure_matrix(self, lattice, desired_matrix):
        assert np.allclose(
            get_direct_structure_matrix(lattice), desired_matrix, atol=1e-3
        )

    @pytest.mark.parametrize(
        "lattice, desired_matrix",
        [
            (Lattice(3.52, 3.52, 3.52, 90, 90, 90), np.eye(3) * 0.080),
            (
                Lattice(3.52, 3.52, 10.5, 90, 90, 120),
                np.array(
                    [[0.107, 0.053, -0], [0.053, 0.107, -0], [-0, -0, 0.009]]
                ),
            ),
        ],
    )
    def test_reciprocal_metric_tensor(self, lattice, desired_matrix):
        recip_metrics = get_reciprocal_metric_tensor(lattice)
        assert np.allclose(
            recip_metrics, lattice.reciprocal().metrics, atol=1e-3
        )
        assert np.allclose(recip_metrics, desired_matrix, atol=1e-3)
