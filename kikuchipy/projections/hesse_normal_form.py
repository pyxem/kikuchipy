# Copyright 2019-2023 The kikuchipy developers
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

from kikuchipy.projections.spherical_projection import get_polar
from kikuchipy._util import deprecated


class HesseNormalForm:
    """[*Deprecated*] Hessian normal form of a plane given by polar
    coordinates.

    .. deprecated:: 0.8.0

        This class is deprecated and will be removed in 0.9.0, since it
        is not used internally. If you depend on this class, please open
        an issue at https://github.com/pyxem/kikuchipy/issues.
    """

    @staticmethod
    @deprecated(since="0.8.0", removal="0.9.0")
    def project_polar(polar: np.ndarray, radius: Optional[float] = 10) -> np.ndarray:
        """Return the Hesse normal form of plane(s) given by spherical
        coordinates.

        Parameters
        ----------
        polar
            The polar, azimuthal and radial spherical coordinates.
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
    @deprecated(since="0.8.0", removal="0.9.0")
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
        polar = get_polar(cartesian)
        hesse = np.zeros((polar.size, 2))
        hesse_distance = np.tan(0.5 * np.pi - polar)
        hesse[..., 0] = hesse_distance
        hesse[..., 1] = np.arccos(hesse_distance / radius)
        return hesse
