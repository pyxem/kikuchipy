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

from kikuchipy.projections.gnomonic_projection import GnomonicProjection


class LambertProjection:
    """Lambert projection of a vector [Callahan2013]_."""

    @classmethod
    def project(cls, v: Union[Vector3d, np.ndarray]) -> np.ndarray:
        """Convert (n, 3) vector from Cartesian to the Lambert projection."""
        if isinstance(v, Vector3d):
            w = v.unit.data
            x = w[..., 0]
            y = w[..., 1]
            z = w[..., 2]
        else:
            norm = np.linalg.norm(v)
            v = v / norm
            x = v[..., 0]
            y = v[..., 1]
            z = v[..., 2]

        # Arrays used in both setting X and Y
        sqrt_z = np.sqrt(2 * (1 - abs(z)))
        sign_x = np.sign(x)
        sign_y = np.sign(y)
        abs_yx = abs(y) <= abs(x)

        # Reusable constants
        sqrt_pi = np.sqrt(np.pi)
        sqrt_pi_half = sqrt_pi / 2
        two_over_sqrt_pi = 2 / sqrt_pi

        lambert = np.zeros(x.shape + (2,), dtype=x.dtype)

        # Equations 10a and 10b from Callahan and De Graef (2013)
        lambert[..., 0] = np.where(
            abs_yx,
            sign_x * sqrt_z * sqrt_pi_half,
            sign_y * sqrt_z * (two_over_sqrt_pi * np.arctan(x / y)),
        )
        lambert[..., 1] = np.where(
            abs_yx,
            sign_x * sqrt_z * (two_over_sqrt_pi * np.arctan(y / x)),
            sign_y * sqrt_z * sqrt_pi_half,
        )

        return lambert

    @staticmethod
    def iproject(xy: np.ndarray) -> Vector3d:
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
            abs_yx,
            1 - (2 * (X ** 2)) / np.pi,
            1 - (2 * (Y ** 2)) / np.pi,
        )

        return Vector3d(cart)

    @staticmethod
    def lambert_to_gnomonic(xy: np.ndarray) -> np.ndarray:
        """Convert (n,2) array from Lambert via Cartesian coordinates to
        Gnomonic."""
        # These two functions could probably be combined into 1 to decrease
        # runtime
        vec = LambertProjection.iproject(xy)
        xy = GnomonicProjection.project(vec)
        return xy

    @staticmethod
    def gnomonic_to_lambert(xy: np.ndarray) -> np.ndarray:
        """Convert (n,2) array from Gnomonic via Cartesian coordinates to
        Lambert."""
        # These two functions could probably be combined into 1 to decrease
        # runtime
        vec = GnomonicProjection.iproject(xy)
        xy = LambertProjection.project(vec)
        return xy


def _eq_c(p: Union[np.ndarray, float, int]) -> Union[np.ndarray, float, int]:
    """Private function used inside LambertProjection.iproject to increase
    readability."""
    return (2 * p / np.pi) * np.sqrt(np.pi - p ** 2)
