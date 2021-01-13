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

"""Spherical projection of a cartesian vector according to the ISO
31-11 standard [SphericalWolfram]_.
"""

from typing import Union

import numpy as np
from orix.vector import Vector3d
from orix.vector.spherical_region import SphericalRegion


class SphericalProjection:
    """Spherical projection of a cartesian vector according to the ISO
    31-11 standard [SphericalWolfram]_.

    References
    ----------
    .. [SphericalWolfram] Weisstein, Eric W. "Spherical Coordinates,"
        *From MathWorld--A Wolfram Web Resource*,
        url: https://mathworld.wolfram.com/SphericalCoordinates.html
    """

    spherical_region = SphericalRegion([0, 0, 1])

    @classmethod
    def project(cls, v: Union[Vector3d, np.ndarray]) -> np.ndarray:
        """Convert from cartesian to spherical coordinates according to
        the ISO 31-11 standard [SphericalWolfram]_.

        Parameters
        ----------
        v
            3D vector(s) on the form [[x0, y0, z0], [x1, y1, z1], ...].

        Returns
        -------
        spherical_coordinates
            Spherical coordinates theta, phi and r on the form
            [[theta1, phi1, r1], [theta2, phi2, r2], ...].

        Examples
        --------
        >>> import numpy as np
        >>> from kikuchipy.projections.spherical_projection import (
        ...     SphericalProjection
        ... )
        >>> v = np.random.random_sample(30).reshape((10, 3))
        >>> theta, phi, r = SphericalProjection.project(v)
        >>> np.allclose(np.arccos(v[: 2] / r), theta)
        True
        >>> np.allclose(np.arctan2(v[:, 1], v[:, 2]), phi)
        True
        """
        return _get_polar(v)


def _get_polar(v: Union[Vector3d, np.ndarray]) -> np.ndarray:
    if isinstance(v, Vector3d):
        x, y, z = v.xyz
    else:
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
    polar = np.zeros(x.shape + (3,), dtype=x.dtype)
    polar[..., 1] = np.where(
        np.arctan2(y, x) < 0, np.arctan2(y, x) + 2 * np.pi, np.arctan2(y, x)
    )  # Phi
    polar[..., 2] = np.sqrt(x ** 2 + y ** 2 + z ** 2)  # r
    polar[..., 0] = np.arccos(z / polar[..., 2])  # Theta

    return polar


def get_theta(v: Union[Vector3d, np.ndarray]) -> np.ndarray:
    """Get spherical coordinate theta from cartesian according to the
    ISO 31-11 standard [SphericalWolfram]_.

    Parameters
    ----------
    v
        3D vector(s) on the form [[x0, y0, z0], [x1, y1, z1], ...].

    Returns
    -------
    theta
        Spherical coordinate theta.
    """
    if isinstance(v, Vector3d):
        x, y, z = v.xyz
    else:
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return np.arccos(z / r)


def get_phi(v: Union[Vector3d, np.ndarray]) -> np.ndarray:
    """Get spherical coordinate phi from cartesian according to the ISO
    31-11 standard [SphericalWolfram]_.

    Parameters
    ----------
    v
        3D vector(s) on the form [[x0, y0, z0], [x1, y1, z1], ...].

    Returns
    -------
    phi
        Spherical coordinate phi.
    """
    if isinstance(v, Vector3d):
        x, y, _ = v.xyz
    else:
        x, y = v[..., 0], v[..., 1]
    phi = np.arctan2(y, x)
    phi += (phi < 0) * 2 * np.pi
    return phi


def get_r(v: Union[Vector3d, np.ndarray]) -> np.ndarray:
    """Get radial spherical coordinate from cartesian coordinates.

    Parameters
    ----------
    v
        3D vector(s) on the form [[x0, y0, z0], [x1, y1, z1], ...].

    Returns
    -------
    phi
        Spherical coordinate phi.
    """
    if isinstance(v, Vector3d):
        x, y, z = v.xyz
    else:
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)
