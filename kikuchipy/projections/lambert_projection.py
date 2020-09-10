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


# This may not be needed? Dont think the inherit is needed
from kikuchipy.projections.spherical_projection import SphericalProjection

# Temporary notes for myself :)
# Bunge Euler triplets
# phi1, Phi, phi2

# Pattern center
# 2D coordinates = xpc, ypc

# All coordinates in units of micrometers


class LambertProjection(SphericalProjection):
    """Lambert Projection of a vector (as implemented in MTEX?)"""

    @classmethod
    def project(self, v: Union[Vector3d, np.ndarray]) -> np.ndarray:

        if isinstance(v, Vector3d):
            x = v.x
            y = v.y
            z = v.z
        else:
            x = v[..., 0]
            y = v[..., 1]
            z = v[..., 2]

        #  Use temporary lists for speed / memory
        li_X = [0]
        li_Y = [0]
        for x_val, y_val, z_val in zip(x, y, z):
            if abs(y_val) <= abs(x_val): # Equation 10a - Requirement: |y| <= |x|
                li_X.append(np.sign(x_val)*np.sqrt(2*(1-z_val))*((np.sqrt(np.pi))/2))
                li_Y.append(np.sign(x_val)*np.sqrt(2*(1-z_val))*((2/(np.sqrt(np.pi))) * np.arctan(y_val/x_val)))
            else: # Equation 10b - Requirement: |x| <= |y|
                li_X.append(np.sign(y_val) * np.sqrt(2 * (1 - z_val)) * ((2 / (np.sqrt(np.pi))) * np.arctan(x_val / y_val)))
                li_Y.append(np.sign(y_val) * np.sqrt(2 * (1 - z_val)) * ((np.sqrt(np.pi)) / 2))

        # Convert to numpy arrays when done appending and we remove initial element
        X = np.array(li_X[1::])
        Y = np.array(li_Y[1::])

        return np.column_stack((X, Y))

    @staticmethod
    def iproject(x: np.ndarray, y: np.ndarray) -> Vector3d:
        X = x  # v[..., 0] The user needs to input this array?
        Y = y  # v[..., 1] The user needs to input this array?
        # A perhaps easier solution would be to take in **ONE** np.ndarray v and set X = v[..., 0], Y = v[..., 1]?

        #  Use temporary lists for speed / memory
        li_x = [0]
        li_y = [0]
        li_z = [0]

        for x_val, y_val in zip(X,Y):
            if abs(y_val) <= abs(x_val): # Equation requires values > 0, this is not accounted for and may lead to
                # errors in the code
                # 0 < |Y| <= |X| <= L - Equation 8a
                li_x.append(eq_c(x_val)*np.cos((y_val*np.pi)/(4*x_val)))
                li_y.append(eq_c(x_val) * np.sin((y_val * np.pi) / (4 * x_val)))
                li_z.append(1 - (2 * (x_val ** 2)) / np.pi)
            else:
                # 0 < |X| <= |X| <= L - Equation 8b
                li_x.append(eq_c(y_val)*np.sin((x_val*np.pi)/(4*y_val)))
                li_y.append(eq_c(y_val)*np.cos((x_val*np.pi)/(4*y_val)))
                li_z.append(1 - (2*(y_val**2))/np.pi)

        # Done appending, now we can use numpy arrays
        x = np.array(li_x[1::])
        y = np.array(li_y[1::])
        z = np.array(li_z[1::])

        v = np.column_stack((x, y, z))

        return Vector3d(v)


# This function is used inside of LambertProjection.iproject method to make code more readable
def _eq_c(p: np.ndarray) -> np.ndarray:
    return 2/(np.pi) * np.sqrt(np.pi - p**2)



def lambert_to_gnomonic(v: np.ndarray) -> np.ndarray:
    # Take vector from Lambert and convert it into an Orix Vector3d object
    vec = LambertProjection.iproject(v[..., 0], v[..., 1])
    # This requires GnomonicProjection.project to be a class method, was prev. instance method - is that ok?
    # Take Orix Vector 3d object and convert it to Gnomonic projection
    vec = GnomonicProjection.project(vec)
    return vec

def gnomonic_to_lambert(v: np.ndarray) -> np.ndarray:
    # Take vector from Gnomonic and convert it into an Orix Vector3d object
    vec = GnomonicProjection.iproject(v[..., 0], v[..., 1])
    # This requires LambertProjection.project to be a class method, was prev. instance methd - is that ok?
    # Take Orix Vector 3d object and convert it to Lambert projection
    vec = LambertProjection.project(vec)
    return vec

