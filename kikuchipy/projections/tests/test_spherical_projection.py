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
from orix.vector import Vector3d

from kikuchipy.projections.spherical_projection import (
    SphericalProjection,
    get_theta,
    get_phi,
    get_r,
)


def test_spherical_projection():
    """Compared against tests in orix."""
    n = 10
    v_arr = np.random.random_sample(n * 3).reshape((n, 3))
    v = Vector3d(v_arr)

    # Vector3d
    polar = SphericalProjection.project(v_arr)
    assert np.allclose(polar[..., 0], v.theta.data)
    assert np.allclose(polar[..., 1], v.phi.data)
    assert np.allclose(polar[..., 2], v.r.data)
    assert np.allclose(get_theta(v), v.theta.data)
    assert np.allclose(get_phi(v), v.phi.data)
    assert np.allclose(get_r(v), v.r.data)

    # NumPy array
    polar2 = SphericalProjection.project(v)
    assert np.allclose(polar2[..., 0], v.theta.data)
    assert np.allclose(polar2[..., 1], v.phi.data)
    assert np.allclose(polar2[..., 2], v.r.data)
    assert np.allclose(get_theta(v_arr), v.theta.data)
    assert np.allclose(get_phi(v_arr), v.phi.data)
    assert np.allclose(get_r(v_arr), v.r.data)
