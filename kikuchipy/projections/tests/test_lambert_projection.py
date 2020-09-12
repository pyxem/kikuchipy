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


from kikuchipy.projections.lambert_projection import LambertProjection
from kikuchipy.projections.lambert_projection import (
    _eq_c,
    _setCartesian,
    _setLambert,
)
from orix.vector import Vector3d

# !! WORK IN PROGRESS !!


class TestLambertProjection:
    def test_project_Vector3d(self):
        """Works for Vector3d objects with single and multiple vectors"""

        vector_one = Vector3d((0.578, 0.578, 0.578))
        output_a = LambertProjection.project(vector_one)
        expected_a = np.array((0.81417, 0.81417))

        assert output_a[..., 0][0] == pytest.approx(expected_a[..., 0], rel=1e4)
        assert output_a[..., 1][0] == pytest.approx(expected_a[..., 1], rel=1e4)

        vector_two = Vector3d(
            np.array(
                [[0.578, 0.578, 0.578], [0, 0.707, 0.707], [0.707, 0, 0.707]]
            )
        )
        output_b = LambertProjection.project(vector_two)

        expected_b = np.array(
            (
                (0.814172, 0.814172, 0.814172),
                (0, 0, 0),
                (0.6784123, 0, 0.6784123),
            )
        )

        assert output_b[..., 0][0] == pytest.approx(
            expected_b[..., 0][0], rel=1e4
        )
        assert output_b[..., 1][0] == pytest.approx(
            expected_b[..., 1][0], rel=1e4
        )
        assert output_b[..., 1][2] == pytest.approx(
            expected_b[..., 1][2], rel=1e4
        )
        assert output_b[..., 0][2] == pytest.approx(
            expected_b[..., 0][2], rel=1e4
        )

    def test_project_ndarray(self):
        "Works for numpy ndarrays"
        ipt = np.array((0.578, 0.578, 0.578))
        output = LambertProjection.project(ipt)
        expected = np.array((0.81417, 0.81417))
        assert output[..., 0][0] == pytest.approx(expected[..., 0], rel=1e4)

    def test_iproject(self):
        """Conversion from Lambert to Cartesian coordinates works"""
        vec = np.array((0.81417, 0.81417))
        expected = Vector3d((0.57702409, 0.577, 0.578))
        output = LambertProjection.iproject(vec)
        print(type(output.x.data))
        print(type(expected.x.data))
        assert output.x.data[0] == pytest.approx(expected.x.data[0])

    def test_eq_c(self):
        """Helper function works"""
        arr_1 = np.array((0.578, 0.578, 0.578))
        value = arr_1[..., 0]
        output_arr = np.array((0.61655, 0.61655, 0.61655))
        expected = output_arr[..., 0]
        output = _eq_c(value)
        assert expected == pytest.approx(output)

    def test_lambert_to_gnomonic(self):
        """Conversion from Lambert to Gnomonic works"""
        vec = np.array(
            (0.81417, 0.81417)
        )  # Should give x,y,z = 1/sqrt(3) (1, 1, 1)
        output = LambertProjection.lambert_to_gnomonic(vec)
        expected = np.array((1, 1))
        assert output[..., 0][0] == pytest.approx(expected[..., 0], rel=1e-2)

    def test_gnomonic_to_lambert(self):
        """Conversion from Gnomonic to Lambert works"""
        vec = np.array((1, 1))  # Similar to the case above
        output = LambertProjection.gnomonic_to_lambert(vec)
        expected = np.array((0.81417, 0.81417))
        assert output[..., 0][0] == pytest.approx(expected[..., 0], rel=1e-3)

    def test_setLambert(self):
        x = 1 / np.sqrt(3)
        y = 1 / np.sqrt(3)
        z = 1 / np.sqrt(3)
        X, Y = _setLambert(x, y, z)
        expX = 0.8147985301
        expY = 0.8147985301
        assert X == pytest.approx(expX, rel=1e-5)
        assert Y == pytest.approx(expY, rel=1e-5)

    def test_setCartesian(self):
        X = 0.8147985301
        Y = 0.8147985301

        x, y, z = _setCartesian(X, Y)
        assert x == pytest.approx(1 / np.sqrt(3), rel=1e-5)
        assert y == pytest.approx(1 / np.sqrt(3), rel=1e-5)
        assert z == pytest.approx(1 / np.sqrt(3), rel=1e-5)
