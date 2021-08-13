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

"""Private tools for handling crystal orientations and directions.

The functions here is taken directly from orix, but modified to use
Numba. Ideally, these should be imported from orix instead.

This module and documentation is only relevant for kikuchipy developers,
not for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with no
    warning.
"""

# TODO: Implement these and similar functions in orix

import numba as nb
import numpy as np


@nb.jit("float64[:](float64, float64, float64)", nogil=True, nopython=True)
def _rotation_from_euler(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Convert three Euler angles (alpha, beta, gamma) to a unit
    quaternion.

    Taken from :meth:`orix.quaternion.Rotation.from_euler`.

    Parameters
    ----------
    alpha, beta, gamma
        Euler angles in the Bunge convention in radians.

    Returns
    -------
    rotation
        Unit quaternion.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    sigma = 0.5 * np.add(alpha, gamma)
    delta = 0.5 * np.subtract(alpha, gamma)
    c = np.cos(beta / 2)
    s = np.sin(beta / 2)

    rotation = np.zeros(4)
    rotation[0] = c * np.cos(sigma)
    rotation[1] = -s * np.cos(delta)
    rotation[2] = -s * np.sin(delta)
    rotation[3] = -c * np.sin(sigma)

    if rotation[0] < 0:
        rotation = -rotation

    return rotation


@nb.jit("float64[:, :](float64[:], float64[:, :])", nogil=True, nopython=True)
def _rotate_vector(rotation: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotation of vector(s) by a quaternion.

    Taken from :meth:`orix.quaternion.Quaternion.__mul__`.

    Parameters
    ----------
    rotation
        Quaternion rotation as an array of shape (4,) and data type
        64-bit floats.
    vector
        Vector(s) as an array of shape (n, 3) and data type 64-bit
        floats.

    Returns
    -------
    rotated_vector

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    a, b, c, d = rotation
    x = vector[:, 0]
    y = vector[:, 1]
    z = vector[:, 2]
    rotated_vector = np.zeros(vector.shape)
    aa = a ** 2
    bb = b ** 2
    cc = c ** 2
    dd = d ** 2
    ac = a * c
    ab = a * b
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d
    rotated_vector[:, 0] = (aa + bb - cc - dd) * x + 2 * ((ac + bd) * z + (bc - ad) * y)
    rotated_vector[:, 1] = (aa - bb + cc - dd) * y + 2 * ((ad + bc) * x + (cd - ab) * z)
    rotated_vector[:, 2] = (aa - bb - cc + dd) * z + 2 * ((ab + cd) * y + (bd - ac) * x)
    return rotated_vector
