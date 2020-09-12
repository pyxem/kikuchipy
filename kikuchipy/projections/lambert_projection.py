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
            x = v.x.data / v.norm.data
            y = v.y.data / v.norm.data
            z = v.z.data / v.norm.data
        else:
            norm = np.sqrt(np.sum(np.square([v[..., 0], v[..., 1], v[..., 2]])))
            x = v[..., 0] / norm
            y = v[..., 1] / norm
            z = v[..., 2] / norm

        # TODO: Should check if Northern or Southern hemisph and act accordingly

        # Checks if multiple arrays/vectors or a single vector/array
        multi = False
        try:
            size = len(x)
            multi = True
        except TypeError:
            size = 1
        X = np.zeros(size)
        Y = np.zeros(size)

        if multi:
            index = 0
            for x_val, y_val, z_val in zip(x, y, z):
                X[index], Y[index] = _setLambert(x_val, y_val, z_val)
                index += 1
        else:
            X, Y = _setLambert(x, y, z)

        return np.column_stack((X, Y))

    @staticmethod
    def iproject(vec: np.ndarray) -> Vector3d:
        """Convert from Lambert to Cartesian coordinates."""
        X = vec[..., 0]
        Y = vec[..., 1]

        # Checks if multiple arrays/vectors or a single vector/array
        multi = False
        try:
            size = len(X)
            multi = True
        except TypeError:
            size = 1
        x = np.zeros(size)
        y = np.zeros(size)
        z = np.zeros(size)
        if multi:
            index = 0
            for x_val, y_val in zip(X, Y):
                x[index], y[index], z[index] = _setCartesian(x_val, y_val)
                index += 1
        else:
            x, y, z = _setCartesian(X, Y)
        return Vector3d(np.column_stack((x, y, z)))

    @staticmethod
    def lambert_to_gnomonic(v: np.ndarray) -> np.ndarray:
        """Convert a 2D array from Lambert via Cartesian coordinates to Gnomonic."""
        # These two functions could probably be combined into 1 to decrease runtime
        vec = LambertProjection.iproject(v)
        vec = GnomonicProjection.project(vec)
        return vec

    @staticmethod
    def gnomonic_to_lambert(v: np.ndarray) -> np.ndarray:
        """Convert a 2D array from Gnomonic via Cartesian coordinates to Lambert."""
        # These two functions could probably be combined into 1 to decrease runtime
        vec = GnomonicProjection.iproject(v[..., 0], v[..., 1])
        vec = LambertProjection.project(vec)
        return vec


def _eq_c(p: Union[np.ndarray, float, int]) -> Union[np.ndarray, float, int]:
    """Private function used inside LambertProjection.iproject to increase readability."""
    return (2 * p / np.pi) * np.sqrt(np.pi - p ** 2)


def _setLambert(
    x: Union[np.ndarray, float, int],
    y: Union[np.ndarray, float, int],
    z: Union[np.ndarray, float, int],
) -> (Union[np.ndarray, float, int], Union[np.ndarray, float, int]):
    """Takes the Cartesian coordinate x, y, z 1D arrays and returns the Lambert equivalent X and Y"""
    if abs(y) <= abs(x):  # Equation 10a - Requirement: |y| <= |x|
        X = np.sign(x) * np.sqrt(2 * (1 - z)) * ((np.sqrt(np.pi)) / 2)
        Y = (
            np.sign(x)
            * np.sqrt(2 * (1 - z))
            * ((2 / (np.sqrt(np.pi))) * np.arctan(y / x))
        )
    else:  # Equation 10b - Requirement: |x| <= |y|
        X = (
            np.sign(y)
            * np.sqrt(2 * (1 - z))
            * ((2 / (np.sqrt(np.pi))) * np.arctan(x / y))
        )
        Y = np.sign(y) * np.sqrt(2 * (1 - z)) * ((np.sqrt(np.pi)) / 2)
    return X, Y


def _setCartesian(
    X: Union[np.ndarray, float, int], Y: Union[np.ndarray, float, int]
) -> (
    Union[np.ndarray, float, int],
    Union[np.ndarray, float, int],
    Union[np.ndarray, float, int],
):
    """Takes Lambert X and Y coordinate arrays and returns the Cartesian equivalent x, y, z 1D arrays"""
    if abs(Y) <= abs(X):
        # 0 < |Y| <= |X| <= L - Equation 8a
        x = _eq_c(X) * np.cos((Y * np.pi) / (4 * X))
        y = _eq_c(X) * np.sin((Y * np.pi) / (4 * X))
        z = 1 - (2 * (X ** 2)) / np.pi
    else:
        # 0 < |X| <= |Y| <= L - Equation 8b
        x = _eq_c(Y) * np.sin((X * np.pi) / (4 * Y))
        y = _eq_c(Y) * np.cos((X * np.pi) / (4 * Y))
        z = 1 - (2 * (Y ** 2)) / np.pi
    return x, y, z
