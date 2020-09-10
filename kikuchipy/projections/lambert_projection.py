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

        # TODO: The equations assume x^2 + y^2 + z^2 = 1, so we need to normalize the input
        # TODO: Should check if Northern or Southern hemisph and act accordingly

        # TODO: This should be converted into numpy arrays from the get-go using np.where() - But I am still looking for
        #  a way to make that work

        #  Use temporary lists for speed / memory
        li_X = [0]
        li_Y = [0]
        for x_val, y_val, z_val in zip(x, y, z):
            if abs(y_val) <= abs(
                x_val
            ):  # Equation 10a - Requirement: |y| <= |x|
                li_X.append(
                    np.sign(x_val)
                    * np.sqrt(2 * (1 - z_val))
                    * ((np.sqrt(np.pi)) / 2)
                )
                li_Y.append(
                    np.sign(x_val)
                    * np.sqrt(2 * (1 - z_val))
                    * ((2 / (np.sqrt(np.pi))) * np.arctan(y_val / x_val))
                )
            else:  # Equation 10b - Requirement: |x| <= |y|
                li_X.append(
                    np.sign(y_val)
                    * np.sqrt(2 * (1 - z_val))
                    * ((2 / (np.sqrt(np.pi))) * np.arctan(x_val / y_val))
                )
                li_Y.append(
                    np.sign(y_val)
                    * np.sqrt(2 * (1 - z_val))
                    * ((np.sqrt(np.pi)) / 2)
                )

        # Convert to numpy arrays when done appending and we remove initial element
        X = np.array(li_X[1::])
        Y = np.array(li_Y[1::])

        return np.column_stack((X, Y))

    @staticmethod
    def iproject(x: np.ndarray, y: np.ndarray) -> Vector3d:
        """Convert from Lambert to Cartesian coordinates."""
        X = x
        Y = y

        # TODO: This should be converted into numpy arrays from the get-go using np.where() - But I am still looking for
        #  a way to make that work

        #  Use temporary lists, ideally would be numpy arrays from the start. Still finding implementation
        li_x = [0]
        li_y = [0]
        li_z = [0]

        for x_val, y_val in zip(X, Y):
            # Edge cases - !! UNVERIFIED EQUATIONS !!
            if x_val == 0 and y_val == 0:
                li_x.append(0)
                li_y.append(0)
                li_z.append(1)
            elif x_val == 0 and y_val != 0:
                li_x.append(0)
                li_y.append(_eq_c(y_val))
                li_z.append(1 - (2 * (y_val ** 2)) / np.pi)
            elif y_val == 0 and x_val != 0:
                li_x.append(_eq_c(x_val))
                li_y.append(0)
                li_z.append(li_z.append(1 - (2 * (x_val ** 2)) / np.pi))
            elif abs(y_val) <= abs(x_val):
                # 0 < |Y| <= |X| <= L - Equation 8a
                li_x.append(
                    _eq_c(x_val) * np.cos((y_val * np.pi) / (4 * x_val))
                )
                li_y.append(
                    _eq_c(x_val) * np.sin((y_val * np.pi) / (4 * x_val))
                )
                li_z.append(1 - (2 * (x_val ** 2)) / np.pi)
            else:
                # 0 < |X| <= |Y| <= L - Equation 8b
                li_x.append(
                    _eq_c(y_val) * np.sin((x_val * np.pi) / (4 * y_val))
                )
                li_y.append(
                    _eq_c(y_val) * np.cos((x_val * np.pi) / (4 * y_val))
                )
                li_z.append(1 - (2 * (y_val ** 2)) / np.pi)

        # Done appending, convert to numpy array. Naive solution, probably a faster one out there
        x = np.array(li_x[1::])
        y = np.array(li_y[1::])
        z = np.array(li_z[1::])
        v = np.column_stack((x, y, z))

        return Vector3d(v)


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
