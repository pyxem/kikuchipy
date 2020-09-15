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
            norm = np.sqrt(
                np.sum(np.square([v[..., 0], v[..., 1], v[..., 2]]), axis=0)
            )
            x = v[..., 0] / norm
            y = v[..., 1] / norm
            z = v[..., 2] / norm

        # Arrays used in both setting X and Y
        sqrt_z = np.sqrt(2 * (1 - z))
        sign_x = np.sign(x)
        sign_y = np.sign(y)
        # Reusable constants
        sqrt_pi = np.sqrt(np.pi)
        sqrt_pi_half = sqrt_pi / 2
        two_over_sqrt_pi = 2 / sqrt_pi

        X = np.where(
            abs(y) <= abs(x),
            sign_x * sqrt_z * sqrt_pi_half,
            sign_y * sqrt_z * (two_over_sqrt_pi * np.arctan2(x, y)),
        )
        Y = np.where(
            abs(y) <= abs(x),
            sign_x * sqrt_z * (two_over_sqrt_pi * np.arctan2(y, x)),
            sign_y * sqrt_z * sqrt_pi_half,
        )

        return np.column_stack((X, Y))

    @staticmethod
    def iproject(vec: np.ndarray) -> Vector3d:
        """Convert (n, 2) array from Lambert to Cartesian coordinates."""
        X = vec[..., 0]
        Y = vec[..., 1]

        x = np.where(
            abs(Y) <= abs(X),
            _eq_c(X) * np.cos((Y * np.pi) / (4 * X)),
            _eq_c(Y) * np.sin((X * np.pi) / (4 * Y)),
        )
        y = np.where(
            abs(Y) <= abs(X),
            _eq_c(X) * np.sin((Y * np.pi) / (4 * X)),
            _eq_c(Y) * np.cos((X * np.pi) / (4 * Y)),
        )
        z = np.where(
            abs(Y) <= abs(X),
            1 - (2 * (X ** 2)) / np.pi,
            1 - (2 * (Y ** 2)) / np.pi,
        )

        return Vector3d(np.column_stack((x, y, z)))

    @staticmethod
    def lambert_to_gnomonic(v: np.ndarray) -> np.ndarray:
        """Convert a (n,2) array from Lambert via Cartesian coordinates to
        Gnomonic."""
        # These two functions could probably be combined into 1 to decrease
        # runtime
        vec = LambertProjection.iproject(v)
        vec = GnomonicProjection.project(vec)
        return vec

    @staticmethod
    def gnomonic_to_lambert(v: np.ndarray) -> np.ndarray:
        """Convert a (n,2) array from Gnomonic via Cartesian coordinates to
        Lambert."""
        # These two functions could probably be combined into 1 to decrease
        # runtime
        vec = GnomonicProjection.iproject(v[..., 0], v[..., 1])
        vec = LambertProjection.project(vec)
        return vec


def _eq_c(p: Union[np.ndarray, float, int]) -> Union[np.ndarray, float, int]:
    """Private function used inside LambertProjection.iproject to increase
    readability."""
    return (2 * p / np.pi) * np.sqrt(np.pi - p ** 2)
