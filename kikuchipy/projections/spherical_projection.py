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

from typing import Tuple, Union

import numpy as np
from orix.vector import Vector3d
from orix.vector.spherical_region import SphericalRegion


class SphericalProjection:
    """Spherical projection of a vector as implemented in MTEX."""

    spherical_region = SphericalRegion([0, 0, 1])

    def project(self, v: Vector3d) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert from cartesian to spherical coordinates."""
        phi, theta, r = get_polar(v)

        # Restrict to plotable domain
        is_antipodal = v < self.spherical_region
        phi[is_antipodal] += np.pi

        return theta, phi, r


def get_polar(
    v: Union[Vector3d, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert from cartesian to spherical coordinates."""
    if isinstance(v, Vector3d):
        x, y, z = v.xyz
    else:
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
    phi = np.arctan2(y, x)
    phi += (phi < 0) * 2 * np.pi
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    return theta, phi, r
