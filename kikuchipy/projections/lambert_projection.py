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

    def project(self, v: Union[Vector3d, np.ndarray]) -> Vector3d:
        pass

    @staticmethod
    def iproject(x: np.ndarray, y: np.ndarray) -> Vector3d:
        pass