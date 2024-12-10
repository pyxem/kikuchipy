# Copyright 2019-2024 The kikuchipy developers
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

"""Function to calculate gnomonic bounds of an EBSDDetector"""

import numpy as np


def get_gnomonic_bounds(
    nrows: int, ncols: int, pcx: float, pcy: float, pcz: float
) -> np.ndarray:
    """
    Get a 1D array of gnomonic bounds for a single PC.

    This function is used in by the objective functions
    for refining orientations and PCs.

    Parameters
    ----------
    nrows
        Number of rows on the EBSD pattern.
        Same as EBSDDetector.nrows
    ncols
        Number of columns on the EBSD pattern.
        Same as EBSDDetector.ncols.
    pcx
        The x-coordinate of the pattern centre.
        Same as EBSDDetector.pcx
    pcy
        The y-coordinate of the pattern centre.
        Same as EBSDDetector.pcy
    pcz
        The z-coordinate of the pattern centre.
        Same as EBSDDetector.pcz

    Returns
    -------
    gnomonic_bounds
        Array of the gnomonic bounds
        [x_min, x_max, y_min, y_max].
    """
    aspect_ratio = ncols / nrows
    x_min = -aspect_ratio * (pcx / pcz)
    x_max = aspect_ratio * (1 - pcx) / pcz
    y_min = -(1 - pcy) / pcz
    y_max = pcy / pcz
    gnomonic_bounds = np.array([x_min, x_max, y_min, y_max])

    return gnomonic_bounds
