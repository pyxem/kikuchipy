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

"""Objective functions for the refinement of crystal orientations and
projection centers by optimizing the similarity between experimental
and simulated patterns.
"""

import numpy as np

from kikuchipy.indexing.similarity_metrics import _ncc_single_patterns_2d_float32
from kikuchipy._rotation import _rotation_from_euler
from kikuchipy.signals.util._master_pattern import (
    _project_single_pattern_from_master_pattern,
)


def _refine_orientation_objective_function(x: np.ndarray, *args: tuple) -> float:
    """Objective function to be minimized when optimizing an orientation
    (Euler angles).

    Parameters
    ----------
    x
        1D array containing the current Euler angles (phi1, Phi, phi2)
        in radians.
    args
        Tuple of fixed parameters needed to completely specify the
        function. The expected contents are:
            0. 2D experimental pattern of 32-bit floats
            1. 1D direction cosines
            2. 2D northern hemisphere of master pattern of 32-bit floats
            3. 2D southern hemisphere of master pattern of 32-bit floats
            4. Number of master pattern columns
            5. Number of master pattern rows
            6. Master pattern scale
            7. 1D signal mask
            8. Number of pattern pixels

    Returns
    -------
        Objective function value (normalized cross-correlation score).
    """
    simulated_pattern = _project_single_pattern_from_master_pattern(
        rotation=_rotation_from_euler(phi1=x[0], Phi=x[1], phi2=x[2]),
        direction_cosines=args[1],
        master_north=args[2],
        master_south=args[3],
        npx=args[4],
        npy=args[5],
        scale=args[6],
        n_pixels=args[8],
        rescale=False,
        out_min=0,  # Required, but not used here
        out_max=1,  # Required, but not used here
        dtype_out=np.float32,
    )
    simulated_pattern = simulated_pattern * args[7]  # Multiply by mask
    experimental = args[0]
    #    score = _ncc_single_patterns_1d_float32(exp=args[0], sim=simulated_pattern)
    simulated_pattern = simulated_pattern.reshape(experimental.shape)
    score = _ncc_single_patterns_2d_float32(exp=experimental, sim=simulated_pattern)
    return 1 - score
