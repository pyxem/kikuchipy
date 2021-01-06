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

"""Hessian normal form of a plane given by polar coordinates. (Not
currently used anywhere.)
"""

from typing import Optional

import numpy as np

from kikuchipy.projections.spherical_projection import get_theta


class HesseNormalForm:
    """Hessian normal form of a plane given by polar coordinates."""

    @staticmethod
    def project_polar(
        polar: np.ndarray, radius: Optional[float] = 10
    ) -> np.ndarray:
        """Return the Hesse normal form of plane(s) given by spherical
        coordinates.

        Parameters
        ----------
        polar
            Spherical coordinates theta, phi, r.
        radius
            Hesse radius. Default is 10.

        Returns
        -------
        hesse
            Hesse normal form coordinates distance and angle.
        """
        hesse = np.zeros(polar.shape[0:-1] + (2,))
        hesse_distance = np.tan(0.5 * np.pi - polar[..., 0])
        hesse[..., 0] = hesse_distance
        hesse[..., 1] = np.arccos(hesse_distance / radius)
        return hesse

    @staticmethod
    def project_cartesian(
        cartesian: np.ndarray, radius: Optional[float] = 10
    ) -> np.ndarray:
        """Return the Hesse normal form of plane(s) given by cartesian
        coordinates.

        Parameters
        ----------
        cartesian
            Cartesian coordinates x, y, z.
        radius
            Hesse radius. Default is 10.

        Returns
        -------
        hesse
            Hesse normal form coordinates distance and angle.
        """
        theta = get_theta(cartesian)
        hesse = np.zeros((theta.size, 2))
        hesse_distance = np.tan(0.5 * np.pi - theta)
        hesse[..., 0] = hesse_distance
        hesse[..., 1] = np.arccos(hesse_distance / radius)
        return hesse
