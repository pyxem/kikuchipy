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
    """Lambert Projection of a vector as implemented in MTEX"""

    @classmethod
    def project(self, v: Union[Vector3d, np.ndarray]) -> np.ndarray:
        """Convert from Cartesian to the Lambert projection"""
        if isinstance(v, Vector3d):
            x = v.x.data
            y = v.y.data
            z = v.z.data
        else:
            x = v[..., 0]
            y = v[..., 1]
            z = v[..., 2]

        # TODO: Should check if Northern or Southern hemisph and act accordingly

        # Lets try to use numpy arrays instead of lists!
        size = len(x)
        X = np.zeros(size)
        Y = np.zeros(size)

        for x_val, y_val, z_val in zip(x, y, z):
            index = 0
            # x^2 + y^2 + z^2 should equal 1 (Needs to lie on the unit sphere)
            # This checker might be too strict. maybe can format the LHS of the equation to 2 decimals or something

            if round(x_val ** 2 + y_val ** 2 + z_val ** 2, 1) != 1:
                raise ValueError(
                    "Vector is not on the unit sphere! Make sure x^2 + y^2 + z^2 = 1"
                )
            if abs(y_val) <= abs(
                x_val
            ):  # Equation 10a - Requirement: |y| <= |x|
                X[index] = (
                    np.sign(x_val)
                    * np.sqrt(2 * (1 - z_val))
                    * ((np.sqrt(np.pi)) / 2)
                )
                Y[index] = (
                    np.sign(x_val)
                    * np.sqrt(2 * (1 - z_val))
                    * ((2 / (np.sqrt(np.pi))) * np.arctan(y_val / x_val))
                )
            else:  # Equation 10b - Requirement: |x| <= |y|
                X[index] = (
                    np.sign(y_val)
                    * np.sqrt(2 * (1 - z_val))
                    * ((2 / (np.sqrt(np.pi))) * np.arctan(x_val / y_val))
                )
                Y[index] = (
                    np.sign(y_val)
                    * np.sqrt(2 * (1 - z_val))
                    * ((np.sqrt(np.pi)) / 2)
                )
            index += 1

        return np.column_stack((X, Y))

    @staticmethod
    def iproject(x: np.ndarray, y: np.ndarray) -> Vector3d:
        """Convert from Lambert to Cartesian coordinates."""
        X = x
        Y = y

        size = len(X)
        x = np.zeros(size)
        y = np.zeros(size)
        z = np.zeros(size)

        for x_val, y_val in zip(X, Y):
            index = 0
            if abs(x_val) > np.sqrt(np.pi / 2) or abs(y_val) > np.sqrt(
                np.pi / 2
            ):
                raise ValueError(
                    "The X and Y values MUST NOT be greater than L = sqrt(pi/2)"
                )
            if (
                x_val == 0 and y_val == 0
            ):  # This is probably covered in the equations below but it is here now :)
                x[index] = 0
                y[index] = 0
                z[index] = 1
            elif abs(y_val) <= abs(x_val):
                # 0 < |Y| <= |X| <= L - Equation 8a
                x[index] = _eq_c(x_val) * np.cos((y_val * np.pi) / (4 * x_val))
                y[index] = _eq_c(x_val) * np.sin((y_val * np.pi) / (4 * x_val))
                z[index] = 1 - (2 * (x_val ** 2)) / np.pi
            else:
                # 0 < |X| <= |Y| <= L - Equation 8b
                x[index] = _eq_c(y_val) * np.sin((x_val * np.pi) / (4 * y_val))
                y[index] = _eq_c(y_val) * np.cos((x_val * np.pi) / (4 * y_val))
                z[index] = 1 - (2 * (y_val ** 2)) / np.pi
            index += 1

        return Vector3d(np.column_stack((x, y, z)))


# I believe this method is implemented wrong. Why is it taking and returning np.ndarray?
def _eq_c(p: np.ndarray) -> np.ndarray:
    """Private function used inside LambertProjection.iproject to increase readability."""
    return 2 / np.pi * np.sqrt(np.pi - p ** 2)


def lambert_to_gnomonic(v: np.ndarray) -> np.ndarray:
    """Convert from Lambert via Cartesian coordinates to Gnomonic."""
    # These two functions could probably be combined into 1 to decrease runtime
    vec = LambertProjection.iproject(v[..., 0], v[..., 1])
    vec = GnomonicProjection.project(vec)
    return vec


def gnomonic_to_lambert(v: np.ndarray) -> np.ndarray:
    """Convert from Gnomonic via Cartesian coordinates to Lambert."""
    # These two functions could probably be combined into 1 to decrease runtime
    vec = GnomonicProjection.iproject(v[..., 0], v[..., 1])
    vec = LambertProjection.project(vec)
    return vec
