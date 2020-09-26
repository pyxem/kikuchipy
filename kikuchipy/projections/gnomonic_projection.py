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

from typing import Union

import numpy as np
from orix.vector import Vector3d

from kikuchipy.projections.spherical_projection import SphericalProjection


class GnomonicProjection(SphericalProjection):
    """Gnomonic projection of a vector as implemented in MTEX."""

    @classmethod
    def project(cls, v: Union[Vector3d, np.ndarray]) -> np.ndarray:
        """Convert from 3D cartesian coordinates (x, y, z) to 2D
        Gnomonic coordinates (x_g, y_g).

        Parameters
        ----------
        v
            3D vector(s) on the form [[x0, y0, z0], [x1, y1, z1], ...].

        Returns
        -------
        gnomonic_coordinates
            Gnomonic coordinates on the form [[x0, y0], [x1, y1], ...].

        Examples
        --------
        >>> import numpy as np
        >>> from kikuchipy.projections.gnomonic_projection import (
        ...     GnomonicProjection
        ... )
        >>> v = np.random.random_sample(30).reshape((10, 3))
        >>> xy = GnomonicProjection.project(v)
        """
        polar = super().project(v)
        theta, phi = polar[..., 0], polar[..., 1]

        # Map to upper hemisphere
        if isinstance(v, Vector3d):
            is_upper = v.z > 0
        else:
            is_upper = v[..., 2] > 0
        theta[is_upper] -= np.pi

        # Formula for gnomonic projection
        r = np.tan(theta)

        # Compute coordinates
        gnomonic = np.zeros(r.shape + (2,), dtype=r.dtype)
        gnomonic[..., 0] = np.cos(phi) * r
        gnomonic[..., 1] = np.sin(phi) * r

        return gnomonic

    @staticmethod
    def iproject(xy: np.ndarray) -> Vector3d:
        """Convert from 2D Gnomonic coordinates (x_g, y_g) to 3D
        cartesian coordiantes (x, y, z).

        Parameters
        ----------
        xy
            2D coordinates on the form
            [[x_g0, y_g0], [x_g1, y_g1], ...].

        Returns
        -------
        cartesian_coordinates
            Cartesian coordinates (x, y, z) on the form
            [[x0, y0, z0], [x1, y1, z1], ...].

        Examples
        --------
        >>> import numpy as np
        >>> from kikuchipy.projections.gnomonic_projection import (
        ...     GnomonicProjection
        ... )
        >>> xy_g = np.random.random_sample(20).reshape((10, 2))
        >>> xyz = GnomonicProjection.iproject(xy_g)
        """
        x, y = xy[..., 0], xy[..., 1]
        theta = np.arctan(np.sqrt(x ** 2 + y ** 2))
        phi = np.arctan2(y, x)
        return Vector3d.from_polar(theta=theta, phi=phi)
