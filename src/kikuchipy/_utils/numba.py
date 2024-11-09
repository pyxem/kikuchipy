# Copyright 2019-2024 The kikuchipy developers
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

"""Numba-accelerated functions useful across modules."""

import numba as nb
import numpy as np

# ---------------------------- Rotations ----------------------------- #


@nb.njit("float64[:](float64, float64, float64)", cache=True, fastmath=True, nogil=True)
def rotation_from_rodrigues(rx: float, ry: float, rz: float) -> np.ndarray:
    rod = np.array([rx, ry, rz])
    norm = np.sqrt(np.sum(np.square(rod)))
    half_angle = np.arctan(norm)
    s = np.sin(half_angle)
    a = np.cos(half_angle)
    b = s * rx / norm
    c = s * ry / norm
    d = s * rz / norm
    rot = np.array([a, b, c, d], dtype="float64")
    if rot[0] < 0:
        for i in range(4):
            rot[i] = -rot[i]
    return rot


@nb.njit("float64[:](float64, float64, float64)", cache=True, fastmath=True, nogil=True)
def rotation_from_euler(alpha: float, beta: float, gamma: float) -> np.ndarray:
    sigma = 0.5 * (alpha + gamma)
    delta = 0.5 * (alpha - gamma)
    c = np.cos(0.5 * beta)
    s = np.sin(0.5 * beta)
    rot = np.array(
        [c * np.cos(sigma), -s * np.cos(delta), -s * np.sin(delta), -c * np.sin(sigma)],
        dtype=np.float64,
    )
    if rot[0] < 0:
        for i in range(4):
            rot[i] = -rot[i]
    return rot


@nb.njit(
    "float64[:, :](float64[:], float64[:, :])", cache=True, fastmath=True, nogil=True
)
def rotate_vector(rotation: np.ndarray, vector: np.ndarray) -> np.ndarray:
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
    vector2 = np.zeros(vector.shape, dtype=np.float64)
    vector2[:, 0] = (aa + bb - cc - dd) * x + 2 * ((ac + bd) * z + (bc - ad) * y)
    vector2[:, 1] = (aa - bb + cc - dd) * y + 2 * ((ad + bc) * x + (cd - ab) * z)
    vector2[:, 2] = (aa - bb - cc + dd) * z + 2 * ((ab + cd) * y + (bd - ac) * x)
    return vector2


# ----------------------------- Vectors ------------------------------ #


@nb.njit("float64(float64[:], float64[:])", cache=True, fastmath=True, nogil=True)
def vec_dot(v1, v2):
    D = 0.0
    for i in range(3):
        D += v1[i] * v2[i]
    return D
