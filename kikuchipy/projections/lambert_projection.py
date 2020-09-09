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


# Bunge Euler triplets
# phi1, Phi, phi2

# Pattern center
# 2D coordinates = xpc, ypc

# All coordinates in units of micrometers

from typing import Union
import numpy as np
from orix.vector import Vector3d

# These may not be needed?
# from kikuchipy.projections.gnomonic_projection import GnomonicProjection
from kikuchipy.projections.spherical_projection import SphericalProjection


class LambertProjection(SphericalProjection):
    """Lambert Projection of a vector (as implemented in MTEX?)"""

    def project(self, v: Union[Vector3d, np.ndarray]) -> np.ndarray:

        if isinstance(v, Vector3d):
            x = v.x
            y = v.y
            z = v.z
        else:
            x = v[..., 0]
            y = v[..., 1]
            z = v[..., 2]

        # TODO: Implement requirement checker for which equation to use
        something_important = True
        if something_important:
            # Equation 10a - Requirement: |y| <= |x|
            X = np.sign(x)*np.sqrt(2*(1-z))*((np.sqrt(np.pi))/2)
            Y = np.sign(x)*np.sqrt(2*(1-z))*((2/(np.sqrt(np.pi))) * np.arctan(y/x))
        else:
            # Equation 10b - Requirement: |x| <= |y|
            X = np.sign(y)*np.sqrt(2*(1-z))*((2/(np.sqrt(np.pi))) * np.arctan(x/y))
            Y = np.sign(y)*np.sqrt(2*(1-z))*((np.sqrt(np.pi))/2)

        return np.column_stack((X, Y))

    @staticmethod
    def iproject(x: np.ndarray, y: np.ndarray) -> Vector3d:
        # TODO: Implement equation 8a from Patrick G. Callahan and Marc De Graef
        # TODO: Implement equation 8b from Patrick G. Callahan and Marc De Graef
        pass
