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

"""Objective functions for the refinement of crystal orientations and
projection centers by optimizing the similarity between experimental
and simulated patterns.
"""

import numpy as np

from kikuchipy._utils._gnonomic_bounds import get_gnomonic_bounds
from kikuchipy._utils.numba import rotation_from_euler
from kikuchipy.indexing.similarity_metrics._normalized_cross_correlation import (
    _ncc_single_patterns_1d_float32_exp_centered,
)
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_for_fixed_pc,
    _project_single_pattern_from_master_pattern,
)


def _refine_orientation_objective_function(x: np.ndarray, *args) -> float:
    """Objective function to be minimized when optimizing an orientation
    (Euler angles).

    Parameters
    ----------
    x
        1D array containing the Euler angles (phi1, Phi, phi2).
    *args
        Tuple of fixed parameters needed to completely specify the
        function. The expected contents are:
            0. 1D centered experimental pattern of 32-bit floats
            1. 1D direction cosines
            2. 2D upper hemisphere of master pattern of 32-bit floats
            3. 2D lower hemisphere of master pattern of 32-bit floats
            4. Number of master pattern columns
            5. Number of master pattern rows
            6. Master pattern scale
            7. Squared norm of centered experimental pattern as 32-bit
               float

    Returns
    -------
        Normalized cross-correlation score.
    """
    simulated = _project_single_pattern_from_master_pattern(
        rotation=rotation_from_euler(*x),
        direction_cosines=args[1],
        master_upper=args[2],
        master_lower=args[3],
        npx=args[4],
        npy=args[5],
        scale=args[6],
        rescale=False,
        out_min=0,  # Required, but not used here
        out_max=1,  # Required, but not used here
        dtype_out=np.float32,
    )
    return 1 - _ncc_single_patterns_1d_float32_exp_centered(args[0], simulated, args[7])


def _refine_pc_objective_function(x: np.ndarray, *args) -> float:
    """Objective function to be minimized when optimizing projection
    center (PC) parameters PCx, PCy, and PCz.

    Parameters
    ----------
    x
        1D array containing the current PC parameters (PCx, PCy, PCz).
    *args
        Tuple of fixed parameters needed to completely specify the
        function. The expected contents are:
            0. 1D centered experimental pattern of 32-bit floats
            1. 1D quaternion
            2. 2D upper hemisphere of master pattern of 32-bit floats
            3. 2D lower hemisphere of master pattern of 32-bit floats
            4. Number of master pattern columns
            5. Number of master pattern rows
            6. Master pattern scale
            7. 1D signal mask
            8. Number of detector rows
            9. Number of detector columns
            10. Orientation matrix detector to sample coordinates
            11. Squared norm of centered experimental pattern as 32-bit
                float

    Returns
    -------
        Normalized cross-correlation score.
    """
    gn_bds = get_gnomonic_bounds(args[8], args[9], x[0], x[1], x[2])

    dc = _get_direction_cosines_for_fixed_pc(
        gnomonic_bounds=gn_bds,
        pcz=x[2],
        nrows=args[8],
        ncols=args[9],
        om_detector_to_sample=args[10],
        signal_mask=args[7],
    )

    simulated = _project_single_pattern_from_master_pattern(
        rotation=args[1],
        direction_cosines=dc,
        master_upper=args[2],
        master_lower=args[3],
        npx=args[4],
        npy=args[5],
        scale=args[6],
        rescale=False,
        out_min=0,  # Required, but not used here
        out_max=1,  # Required, but not used here
        dtype_out=np.float32,
    )
    return 1 - _ncc_single_patterns_1d_float32_exp_centered(
        args[0], simulated, args[11]
    )


def _refine_orientation_pc_objective_function(x: np.ndarray, *args) -> float:
    """Objective function to be minimized when optimizing orientations
    and projection center (PC) parameters PCx, PCy, and PCz.

    Parameters
    ----------
    x
        1D array containing the Euler angle triplet (phi1, Phi, phi2)
        and PC parameters (PCx, PCy, PCz).
    *args
        Tuple of fixed parameters needed to completely specify the
        function. The expected contents are:
            0. 1D experimental pattern of 32-bit floats
            1. 2D upper hemisphere of master pattern of 32-bit floats
            2. 2D lower hemisphere of master pattern of 32-bit floats
            3. Number of master pattern columns
            4. Number of master pattern rows
            5. Master pattern scale
            6. 1D signal mask
            7. Number of detector rows
            8. Number of detector columns
            9. Orientation matrix detector to sample coordinates
            10. Squared norm of centered experimental pattern as 32-bit
                float

    Returns
    -------
        normalized cross-correlation score.
    """
    gn_bds = get_gnomonic_bounds(args[7], args[8], x[3], x[4], x[5])

    dc = _get_direction_cosines_for_fixed_pc(
        gnomonic_bounds=gn_bds,
        pcz=x[5],
        nrows=args[7],
        ncols=args[8],
        om_detector_to_sample=args[9],
        signal_mask=args[6],
    )

    simulated = _project_single_pattern_from_master_pattern(
        rotation=rotation_from_euler(*x[:3]),
        direction_cosines=dc,
        master_upper=args[1],
        master_lower=args[2],
        npx=args[3],
        npy=args[4],
        scale=args[5],
        rescale=False,
        out_min=0,  # Required, but not used here
        out_max=1,  # Required, but not used here
        dtype_out=np.float32,
    )
    return 1 - _ncc_single_patterns_1d_float32_exp_centered(
        args[0], simulated, args[10]
    )
