# Copyright 2019-2022 The kikuchipy developers
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

The functions here are taken directly from orix, but modified to use
Numba.

This module and documentation is only relevant for kikuchipy developers,
not for users.

.. warning:

    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with no
    warning.
"""

from numba import njit
import numpy as np


@njit("float64[:](float64, float64, float64)", cache=True, nogil=True, fastmath=True)
def _rotation_from_rodrigues(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert a Rodrigues-Frank vector to a unit quaternion.

    Taken from :meth:`orix.quaternion.Rotation.from_neo_euler`.

    Parameters
    ----------
    rx
        X component.
    ry
        Y component.
    rz
        Z component.

    Returns
    -------
    rot
        Unit quaternion.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    rod = np.array([rx, ry, rz])
    norm = np.sqrt(np.sum(np.square(rod)))
    half_angle = np.arctan(norm)
    s = np.sin(half_angle)

    a = np.cos(half_angle)
    b = s * rx / norm
    c = s * ry / norm
    d = s * rz / norm
    rot = np.array([a, b, c, d], dtype="float64")

    if rot[0] < 0:  # pragma: no cover
        rot = -rot

    return rot


@njit("float64[:, :](float64[:], float64[:, :])", cache=True, nogil=True, fastmath=True)
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
        Rotated vector.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    a, b, c, d = rotation
    x = vector[:, 0]
    y = vector[:, 1]
    z = vector[:, 2]

    aa = a**2
    bb = b**2
    cc = c**2
    dd = d**2
    ac = a * c
    ab = a * b
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d

    rotated_vector = np.zeros(vector.shape)
    rotated_vector[:, 0] = (aa + bb - cc - dd) * x + 2 * ((ac + bd) * z + (bc - ad) * y)
    rotated_vector[:, 1] = (aa - bb + cc - dd) * y + 2 * ((ad + bc) * x + (cd - ab) * z)
    rotated_vector[:, 2] = (aa - bb - cc + dd) * z + 2 * ((ab + cd) * y + (bd - ac) * x)

    return rotated_vector
