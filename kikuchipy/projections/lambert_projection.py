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

from typing import Union

import numba as nb
import numpy as np
from orix.vector import Vector3d

from kikuchipy.projections.gnomonic_projection import GnomonicProjection


# Reusable constants
SQRT_PI = np.sqrt(np.pi)
SQRT_PI_HALF = SQRT_PI / 2
TWO_OVER_SQRT_PI = 2 / SQRT_PI


class LambertProjection:
    """Lambert projection of a vector :cite:`callahan2013dynamical`."""

    @classmethod
    def vector2xy(cls, v: Union[Vector3d, np.ndarray]) -> np.ndarray:
        """Convert vector(s) from Cartesian to the Lambert projection.

        Parameters
        ----------
        v
            Vectors of any shape as long as the last dimension is (3,).

        Returns
        -------
        xy
            Vectors in the square Lambert projection, of the same shape
            as the input vectors, with the last dimension as (2,).
        """
        if isinstance(v, Vector3d):
            w = v.data
        else:
            w = v
        original_shape = w.shape[:-1]
        w = w.reshape((-1, 3)).astype(float)
        xy = _vector2xy(w)
        xy = xy.reshape(original_shape + (2,))
        return xy

    @staticmethod
    def xy2vector(xy: np.ndarray) -> Vector3d:
        """Convert (n, 2) array from Lambert to Cartesian coordinates."""
        X = xy[..., 0]
        Y = xy[..., 1]

        # Arrays used in setting x and y
        true_term = Y * np.pi / (4 * X)
        false_term = X * np.pi / (4 * Y)
        abs_yx = abs(Y) <= abs(X)
        c_x = _eq_c(X)
        c_y = _eq_c(Y)

        cart = np.zeros(X.shape + (3,), dtype=X.dtype)

        # Equations 8a and 8b from Callahan and De Graef (2013)
        cart[..., 0] = np.where(
            abs_yx, c_x * np.cos(true_term), c_y * np.sin(false_term)
        )
        cart[..., 1] = np.where(
            abs_yx, c_x * np.sin(true_term), c_y * np.cos(false_term)
        )
        cart[..., 2] = np.where(
            abs_yx, 1 - (2 * (X ** 2)) / np.pi, 1 - (2 * (Y ** 2)) / np.pi
        )

        return Vector3d(cart)

    @staticmethod
    def lambert_to_gnomonic(xy: np.ndarray) -> np.ndarray:
        """Convert (n,2) array from Lambert via Cartesian coordinates to
        Gnomonic."""
        # These two functions could probably be combined into 1 to decrease
        # runtime
        v = LambertProjection.xy2vector(xy)
        return GnomonicProjection.vector2xy(v)

    @staticmethod
    def gnomonic_to_lambert(xy: np.ndarray) -> np.ndarray:
        """Convert (n,2) array from Gnomonic via Cartesian coordinates to
        Lambert."""
        # These two functions could probably be combined into 1 to decrease
        # runtime
        v = GnomonicProjection.xy2vector(xy)
        return LambertProjection.vector2xy(v)


@nb.jit(nogil=True, nopython=True)
def _eq_c(p: Union[np.ndarray, float, int]) -> Union[np.ndarray, float, int]:
    """Private function used inside LambertProjection.xy2vector to
    increase readability.
    """
    return (2 * p / np.pi) * np.sqrt(np.pi - p ** 2)


@nb.jit("float64[:, :](float64[:, :])", nogil=True, cache=True, nopython=True)
def _vector2xy(v: np.ndarray) -> np.ndarray:
    """Lambert projection of vector(s) :cite:`callahan2013dynamical`.

    Parameters
    ----------
    v
        Vector(s) in an array of shape (n, 3) and 64-bit float data
        type in Cartesian coordinates.

    Returns
    -------
    lambert
        Square Lambert coordinates (X, Y) in array of shape (n, 2) and
        data type 64-bit float.
    """
    # Normalize vectors (vectorized operation is faster than per vector)
    norm = np.sqrt(np.sum(np.square(v), axis=1))
    norm = np.expand_dims(norm, axis=1)
    w = v / norm

    n_vectors = v.shape[0]
    lambert_xy = np.zeros((n_vectors, 2))
    for i in nb.prange(n_vectors):
        x, y, z = w[i]
        abs_z = np.abs(z)
        sqrt_z = np.sqrt(2 * (1 - abs_z))
        if abs_z == 1:  # (X, Y) = (0, 0)
            continue
        elif np.abs(y) <= np.abs(x):
            sign_x = np.sign(x)
            lambert_xy[i, 0] = sign_x * sqrt_z * SQRT_PI_HALF
            lambert_xy[i, 1] = sign_x * sqrt_z * TWO_OVER_SQRT_PI * np.arctan(y / x)
        else:
            sign_y = np.sign(y)
            lambert_xy[i, 0] = sign_y * sqrt_z * TWO_OVER_SQRT_PI * np.arctan(x / y)
            lambert_xy[i, 1] = sign_y * sqrt_z * SQRT_PI_HALF

    return lambert_xy
