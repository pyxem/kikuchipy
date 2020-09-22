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
from orix.vector import Vector3d
import pytest

from kikuchipy.projections.lambert_projection import LambertProjection, _eq_c


class TestLambertProjection:
    def test_project_vector3d(self):
        """Works for Vector3d objects with single and multiple vectors"""

        vector_one = Vector3d((0.578, 0.578, 0.578))
        output_a = LambertProjection.project(vector_one)
        expected_a = np.array((0.81417, 0.81417))

        assert (output_a[..., 0][0])[0] == pytest.approx(
            expected_a[..., 0], rel=1e4
        )
        assert (output_a[..., 1][0])[0] == pytest.approx(
            expected_a[..., 1], rel=1e4
        )

        vector_two = Vector3d(
            np.array(
                [[0.578, 0.578, 0.578], [0, 0.707, 0.707], [0.707, 0, 0.707]]
            )
        )
        output_b = LambertProjection.project(vector_two)

        expected_x = np.array((0.81417, 0, 0.678))
        expected_y = np.array((0.81417, 0.678, 0))

        assert output_b[..., 0, 0] == pytest.approx(expected_x[0], rel=1e4)
        assert output_b[..., 0, 1] == pytest.approx(expected_y[0], rel=1e4)
        assert output_b[..., 1, 0] == pytest.approx(expected_x[1], rel=1e4)
        assert output_b[..., 1, 1] == pytest.approx(expected_y[1], rel=1e4)
        assert output_b[..., 2, 0] == pytest.approx(expected_x[2], rel=1e4)
        assert output_b[..., 2, 1] == pytest.approx(expected_y[2], rel=1e4)

    def test_project_ndarray(self):
        "Works for numpy ndarrays"
        ipt = np.array((0.578, 0.578, 0.578))
        output = LambertProjection.project(ipt)
        expected = np.array((0.81417, 0.81417))
        assert output[..., 0, 0] == pytest.approx(expected[0], rel=1e4)

    def test_iproject(self):
        """Conversion from Lambert to Cartesian coordinates works"""
        vec = np.array((0.81417, 0.81417))
        expected = Vector3d((0.57702409, 0.577, 0.578))
        output = LambertProjection.iproject(vec)
        assert np.allclose(output.x.data[0], expected.x.data[0], rtol=1e-05)

    def test_eq_c(self):
        """Helper function works"""
        arr_1 = np.array((0.578, 0.578, 0.578))
        value = arr_1[..., 0]
        output_arr = np.array((0.61655, 0.61655, 0.61655))
        expected = output_arr[..., 0]
        output = _eq_c(value)
        assert output == pytest.approx(expected, rel=1e-5)

    def test_lambert_to_gnomonic(self):
        """Conversion from Lambert to Gnomonic works"""
        vec = np.array(
            (0.81417, 0.81417)
        )  # Should give x,y,z = 1/sqrt(3) (1, 1, 1)
        output = LambertProjection.lambert_to_gnomonic(vec)
        expected = np.array((1, 1))
        assert output[..., 0, 0] == pytest.approx(expected[0], abs=1e-2)

    def test_gnomonic_to_lambert(self):
        """Conversion from Gnomonic to Lambert works"""
        vec = np.array((1, 1))  # Similar to the case above
        output = LambertProjection.gnomonic_to_lambert(vec)
        expected = np.array((0.81417, 0.81417))
        assert output[..., 0, 0] == pytest.approx(expected[0], rel=1e-3)

    def test_shape_respect(self):
        """Check that LambertProjection.project() respects navigation axes"""
        sx = 60
        a = np.arange(sx ** 2).reshape((sx, sx))
        v = Vector3d(np.dstack([a, a, a]))
        assert v.shape == (sx, sx)
        assert v.data.shape == (sx, sx, 3)
        xy_lambert = LambertProjection.project(v)
        assert xy_lambert.shape == (sx, sx, 2)
