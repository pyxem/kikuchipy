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

from typing import Tuple

import numpy as np
from orix.vector import Vector3d

from kikuchipy.projections.spherical_projection import SphericalProjection


class GnomonicProjection(SphericalProjection):
    """Gnomonic projection of a vector as implemented in MTEX."""

    def project(self, v: Vector3d) -> Tuple[np.ndarray, np.ndarray]:
        theta, phi, _ = super().project(v)

        # Map to upper hemisphere
        is_upper = v.z > 0
        theta[is_upper] -= np.pi

        # Turn around antipodal vectors
        is_antipodal = v < self.spherical_region
        phi[is_antipodal] += np.pi

        # Formula for gnomonic projection
        r = np.tan(theta)

        # Compute coordinates
        x = np.cos(phi) * r
        y = np.sin(phi) * r

        return x, y

    @staticmethod
    def iproject(x: np.ndarray, y: np.ndarray) -> Vector3d:
        theta = np.arctan(np.sqrt(x ** 2 + y ** 2))
        phi = np.atan2(y, x)
        return Vector3d.from_polar(theta=theta, phi=phi)
