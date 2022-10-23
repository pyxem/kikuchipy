# Copyright 2019-2022 The kikuchipy developers
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

from kikuchipy.indexing.similarity_metrics._normalized_cross_correlation import (
    _ncc_single_patterns_1d_float32_exp_centered,
)
from kikuchipy._rotation import _rotation_from_rodrigues
from kikuchipy.signals.util._master_pattern import (
    _project_single_pattern_from_master_pattern,
    _get_direction_cosines_for_fixed_pc,
)


def _refine_orientation_objective_function(x: np.ndarray, *args) -> float:
    """Objective function to be minimized when optimizing an orientation
    (Euler angles).

    Parameters
    ----------
    x
        1D array containing the Rodrigues-Frank vector components (Rx,
        Ry, Rz).
    *args
        Tuple of fixed parameters needed to completely specify the
        function. The expected contents are:
            0. 1D centered experimental pattern of 32-bit floats
            1. 1D direction cosines
            2. 2D northern hemisphere of master pattern of 32-bit floats
            3. 2D southern hemisphere of master pattern of 32-bit floats
            4. Number of master pattern columns
            5. Number of master pattern rows
            6. Master pattern scale
            7. Squared norm of centered experimental pattern as 32-bit
               float

    Returns
    -------
        Objective function value (normalized cross-correlation score).
    """
    simulated = _project_single_pattern_from_master_pattern(
        rotation=_rotation_from_rodrigues(*x),
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


def _refine_projection_center_objective_function(x: np.ndarray, *args) -> float:
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
            2. 2D northern hemisphere of master pattern of 32-bit floats
            3. 2D southern hemisphere of master pattern of 32-bit floats
            4. Number of master pattern columns
            5. Number of master pattern rows
            6. Master pattern scale
            7. 1D signal mask
            8. Number of detector rows
            9. Number of detector columns
            10. Detector tilt
            11. Detector azimuthal angle
            12. Sample tilt
            13. Squared norm of centered experimental pattern as 32-bit
                float

    Returns
    -------
        Objective function value (normalized cross-correlation score).
    """
    dc = _get_direction_cosines_for_fixed_pc(
        pcx=x[0],
        pcy=x[1],
        pcz=x[2],
        nrows=args[8],
        ncols=args[9],
        tilt=args[10],
        azimuthal=args[11],
        sample_tilt=args[12],
        mask=args[7],
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
        args[0], simulated, args[13]
    )


def _refine_orientation_projection_center_objective_function(
    x: np.ndarray, *args
) -> float:
    """Objective function to be minimized when optimizing orientations
    and projection center (PC) parameters PCx, PCy, and PCz.

    Parameters
    ----------
    x
        1D array containing the Rodrigues-Frank vector components (Rx,
        Ry, Rz) and PC parameters (PCx, PCy, PCz).
    *args
        Tuple of fixed parameters needed to completely specify the
        function. The expected contents are:
            0. 1D experimental pattern of 32-bit floats
            1. 2D northern hemisphere of master pattern of 32-bit floats
            2. 2D southern hemisphere of master pattern of 32-bit floats
            3. Number of master pattern columns
            4. Number of master pattern rows
            5. Master pattern scale
            6. 1D signal mask
            7. Number of pattern rows
            8. Number of pattern columns
            9. Detector tilt
            10. Detector azimuthal angle
            11. Sample tilt
            12. Squared norm of centered experimental pattern as 32-bit
                float

    Returns
    -------
        Objective function value (normalized cross-correlation score).
    """
    dc = _get_direction_cosines_for_fixed_pc(
        pcx=x[3],
        pcy=x[4],
        pcz=x[5],
        nrows=args[7],
        ncols=args[8],
        tilt=args[9],
        azimuthal=args[10],
        sample_tilt=args[11],
        mask=args[6],
    )
    simulated = _project_single_pattern_from_master_pattern(
        rotation=_rotation_from_rodrigues(*x[:3]),
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
        args[0], simulated, args[12]
    )
