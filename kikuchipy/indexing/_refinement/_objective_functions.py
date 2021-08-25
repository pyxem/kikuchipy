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

from kikuchipy.indexing.similarity_metrics._normalized_cross_correlation import (
    _ncc_single_patterns_2d_float32,
)
from kikuchipy._rotation import _rotation_from_euler
from kikuchipy.signals.util._master_pattern import (
    _project_single_pattern_from_master_pattern,
    _get_direction_cosines_for_single_pc,
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
        rotation=_rotation_from_euler(alpha=x[0], beta=x[1], gamma=x[2]),
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
    simulated_pattern = simulated_pattern.reshape(experimental.shape)
    score = _ncc_single_patterns_2d_float32(exp=experimental, sim=simulated_pattern)
    return 1 - score


def _refine_projection_center_objective_function(x: np.ndarray, *args: tuple) -> float:
    """Objective function to be minimized when optimizing projection
    center (PC) parameters PCx, PCy, and PCz.

    Parameters
    ----------
    x
        1D array containing the current PC parameters (PCx, PCy, PCz).
    args
        Tuple of fixed parameters needed to completely specify the
        function. The expected contents are:
            0. 2D experimental pattern of 32-bit floats
            1. 1D rotation as quaternion
            2. 2D northern hemisphere of master pattern of 32-bit floats
            3. 2D southern hemisphere of master pattern of 32-bit floats
            4. Number of master pattern columns
            5. Number of master pattern rows
            6. Master pattern scale
            7. 1D signal mask
            8. Number of pattern pixels
            9. Number of pattern rows
            10. Number of pattern columns
            11. Detector tilt
            12. Detector azimuthal angle
            13. Sample tilt

    Returns
    -------
        Objective function value (normalized cross-correlation score).
    """
    n_pixels = args[8]
    direction_cosines = _get_direction_cosines_for_single_pc(
        pcx=x[0],
        pcy=x[1],
        pcz=x[2],
        nrows=args[9],
        ncols=args[10],
        tilt=args[11],
        azimuthal=args[12],
        sample_tilt=args[13],
    ).reshape((n_pixels, 3))
    simulated_pattern = _project_single_pattern_from_master_pattern(
        rotation=args[1],
        direction_cosines=direction_cosines,
        master_north=args[2],
        master_south=args[3],
        npx=args[4],
        npy=args[5],
        scale=args[6],
        n_pixels=n_pixels,
        rescale=False,
        out_min=0,  # Required, but not used here
        out_max=1,  # Required, but not used here
        dtype_out=np.float32,
    )
    simulated_pattern = simulated_pattern * args[7]  # Multiply by mask
    experimental = args[0]
    simulated_pattern = simulated_pattern.reshape(experimental.shape)
    score = _ncc_single_patterns_2d_float32(exp=experimental, sim=simulated_pattern)
    return 1 - score


def _refine_orientation_projection_center_objective_function(
    x: np.ndarray, *args: tuple
) -> float:
    """Objective function to be minimized when optimizing orientations
    and projection center (PC) parameters PCx, PCy, and PCz.

    Parameters
    ----------
    x
        1D array containing the current Euler angles (phi1, Phi, phi2)
        and PC parameters (PCx, PCy, PCz).
    args
        Tuple of fixed parameters needed to completely specify the
        function. The expected contents are:
            0. 2D experimental pattern of 32-bit floats
            1. 2D northern hemisphere of master pattern of 32-bit floats
            2. 2D southern hemisphere of master pattern of 32-bit floats
            3. Number of master pattern columns
            4. Number of master pattern rows
            5. Master pattern scale
            6. 1D signal mask
            7. Number of pattern pixels
            8. Number of pattern rows
            9. Number of pattern columns
            10. Detector tilt
            11. Detector azimuthal angle
            12. Sample tilt

    Returns
    -------
        Objective function value (normalized cross-correlation score).
    """
    rotation = _rotation_from_euler(alpha=x[0], beta=x[1], gamma=x[2])
    n_pixels = args[7]
    direction_cosines = _get_direction_cosines_for_single_pc(
        pcx=x[3],
        pcy=x[4],
        pcz=x[5],
        nrows=args[8],
        ncols=args[9],
        tilt=args[10],
        azimuthal=args[11],
        sample_tilt=args[12],
    ).reshape((n_pixels, 3))
    simulated_pattern = _project_single_pattern_from_master_pattern(
        rotation=rotation,
        direction_cosines=direction_cosines,
        master_north=args[1],
        master_south=args[2],
        npx=args[3],
        npy=args[4],
        scale=args[5],
        n_pixels=n_pixels,
        rescale=False,
        out_min=0,  # Required, but not used here
        out_max=1,  # Required, but not used here
        dtype_out=np.float32,
    )
    simulated_pattern = simulated_pattern * args[6]  # Multiply by mask
    experimental = args[0]
    simulated_pattern = simulated_pattern.reshape(experimental.shape)
    score = _ncc_single_patterns_2d_float32(exp=experimental, sim=simulated_pattern)
    return 1 - score
