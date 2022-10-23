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

"""Solvers for the refinement of crystal orientations and projection
centers by optimizing the similarity between experimental and simulated
patterns.
"""

from typing import Callable, List, Optional, Tuple

from numba import njit
import numpy as np
from scipy.optimize import Bounds


from kikuchipy.indexing._refinement._objective_functions import (
    _refine_orientation_objective_function,
    _refine_orientation_projection_center_objective_function,
    _refine_projection_center_objective_function,
)
from kikuchipy.indexing._refinement import SUPPORTED_OPTIMIZATION_METHODS
from kikuchipy.pattern._pattern import (
    _rescale_without_min_max_1d_float32,
    _zero_mean_sum_square_1d_float32,
)
from kikuchipy.signals.util._master_pattern import _get_direction_cosines_for_fixed_pc


@njit("float32[:](float32[:],bool_[:])", cache=True, nogil=True, fastmath=True)
def _mask_pattern(pattern: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return pattern[mask].reshape(-1)


def _refine_orientation_solver(
    pattern: np.ndarray,
    rescale: bool,
    rotation: np.ndarray,
    method: Callable,
    method_kwargs: dict,
    fixed_parameters: Tuple[np.ndarray, np.ndarray, int, int, float],
    trust_region: List[int],
    trust_region_passed: bool,
    signal_mask: np.ndarray,
    direction_cosines: Optional[np.ndarray] = None,
    pcx: Optional[float] = None,
    pcy: Optional[float] = None,
    pcz: Optional[float] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    tilt: Optional[float] = None,
    azimuthal: Optional[float] = None,
    sample_tilt: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """Maximize the similarity between an experimental pattern and a
    projected simulated pattern by optimizing the orientation
    (Rodrigues-Frank vector) used in the projection.

    Parameters
    ----------
    pattern
        Flattened experimental pattern.
    rescale
        Whether pattern intensities must be rescaled to [-1, 1] and the
        data type set to 32-bit floats.
    rotation
        Rodrigues-Frank vector components (Rx, Ry, Rz), unscaled.
    method
        A supported :mod:`scipy.optimize` function. See ``method``
        parameter in :meth:`kikuchipy.signals.EBSD.refine_orientation`.
    method_kwargs
        Keyword arguments passed to the ``method`` function. For the
        list of possible arguments, see the SciPy documentation.
    fixed_parameters
        Fixed parameters used in the projection.
    trust_region
        List of +/- angular deviation in degrees as bound constraints on
        the three vector components.
    trust_region_passed
        Whether ``trust_region`` was passed to the public refinement
        method.
    signal_mask
        Boolean mask equal to the experimental patterns' detector shape
        ``(n rows, n columns)``, where only pixels equal to ``False``
        are matched.
    direction_cosines
        Vector array of shape (n pixels, 3) and data type 32-bit floats.
        If not given, ``pcx``, ``pcy``, ``pcz``, ``nrows``, ``ncols``,
        ``tilt``, ``azimuthal``, and ``sample_tilt`` must be passed.
    pcx
        Projection center (PC) x coordinate. Must be passed if
        ``direction_cosines`` is not given.
    pcy
        PC y coordinate. Must be passed if ``direction_cosines`` is not
        given.
    pcz
        PC z coordinate. Must be passed if ``direction_cosines`` is not
        given.
    nrows
        Number of detector rows. Must be passed if ``direction_cosines``
        is not given.
    ncols
        Number of detector columns. Must be passed if
        ``direction_cosines`` is not given.
    tilt
        Detector tilt from horizontal in degrees. Must be passed if
        ``direction_cosines`` is not given.
    azimuthal
        Sample tilt about the sample RD axis in degrees. Must be passed
        if ``direction_cosines`` is not given.
    sample_tilt
        Sample tilt from horizontal in degrees. Must be passed if
        ``direction_cosines`` is not given.

    Returns
    -------
    score
        Highest normalized cross-correlation score.
    phi1, Phi, phi2
        Optimized orientation (Euler angles) in radians.
    """
    # Apply mask
    pattern = _mask_pattern(pattern.astype("float32"), signal_mask)

    if rescale:
        pattern = _rescale_without_min_max_1d_float32(pattern)

    # Center intensities and get squared norm of centered pattern
    pattern, squared_norm = _zero_mean_sum_square_1d_float32(pattern)

    if direction_cosines is None:
        direction_cosines = _get_direction_cosines_for_fixed_pc(
            pcx=pcx,
            pcy=pcy,
            pcz=pcz,
            nrows=nrows,
            ncols=ncols,
            tilt=tilt,
            azimuthal=azimuthal,
            sample_tilt=sample_tilt,
            mask=signal_mask,
        )

    params = (pattern,) + (direction_cosines,) + fixed_parameters + (squared_norm,)
    method_name = method.__name__

    if method_name == "minimize":
        if trust_region_passed:
            method_kwargs["bounds"] = Bounds(
                rotation - trust_region,
                rotation + trust_region,
            )
        solution = method(
            fun=_refine_orientation_objective_function,
            x0=rotation,
            args=params,
            **method_kwargs,
        )
    elif SUPPORTED_OPTIMIZATION_METHODS[method_name]["supports_bounds"]:
        solution = method(
            func=_refine_orientation_objective_function,
            args=params,
            bounds=np.column_stack([rotation - trust_region, rotation + trust_region]),
            **method_kwargs,
        )
    else:  # Is always "basinhopping", due to prior check of method name
        method_kwargs["minimizer_kwargs"].update(args=params)
        solution = method(
            func=_refine_orientation_objective_function,
            x0=rotation,
            **method_kwargs,
        )

    x = solution.x

    return 1 - solution.fun, x[0], x[1], x[2]


def _refine_projection_center_solver(
    pattern: np.ndarray,
    rescale: bool,
    rotation: np.ndarray,
    method: Callable,
    method_kwargs: dict,
    fixed_parameters: tuple,
    trust_region: list,
    trust_region_passed: bool,
    pc: np.ndarray,
    signal_mask: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Maximize the similarity between an experimental pattern and a
    projected simulated pattern by optimizing the projection center (PC)
    parameters used in the projection.

    Parameters
    ----------
    pattern
        Experimental pattern of shape (nrows, ncols).
    rescale
        Whether pattern intensities must be rescaled to [-1, 1] and data
        type 32-bit floats.
    rotation
        Rotation as a quaternion array with shape (4,).
    method
        A supported :mod:`scipy.optimize` function. See `method`
        parameter in
        :meth:`kikuchipy.signals.EBSD.refine_projection_center`.
    method_kwargs
        Keyword arguments passed to the ``method`` function. For the
        list of possible arguments, see the SciPy documentation.
    fixed_parameters
        Fixed parameters used in the projection.
    trust_region
        List of +/- percentage deviations as bound constraints on
        the PC parameters in the Bruker convention. The parameter
        range is [0, 1].
    trust_region_passed
        Whether `trust_region` was passed to the public refinement
        method.
    pc
        Projection center (PC) coordinates (PCx, PCy, PCz).
    signal_mask
        Boolean mask equal to the experimental patterns' detector shape
        ``(n rows, n columns)``, where only pixels equal to ``False``
        are matched.

    Returns
    -------
    score
        Highest normalized cross-correlation score.
    pcx_refined, pcy_refined, pcz_refined
        Optimized PC parameters in the Bruker convention.
    """
    # Apply mask
    pattern = _mask_pattern(pattern.astype("float32"), signal_mask)

    if rescale:
        pattern = _rescale_without_min_max_1d_float32(pattern)

    # Center intensities and get squared norm of centered pattern
    pattern, squared_norm = _zero_mean_sum_square_1d_float32(pattern)

    params = (pattern,) + (rotation,) + fixed_parameters + (squared_norm,)
    method_name = method.__name__

    if method_name == "minimize":
        if trust_region_passed:
            method_kwargs["bounds"] = Bounds(pc - trust_region, pc + trust_region)
        solution = method(
            fun=_refine_projection_center_objective_function,
            x0=pc,
            args=params,
            **method_kwargs,
        )
    elif SUPPORTED_OPTIMIZATION_METHODS[method_name]["supports_bounds"]:
        solution = method(
            func=_refine_projection_center_objective_function,
            args=params,
            bounds=np.column_stack([pc - trust_region, pc + trust_region]),
            **method_kwargs,
        )
    else:  # Is always "basinhopping", due to prior check of method name
        method_kwargs["minimizer_kwargs"].update(args=params)
        solution = method(
            func=_refine_projection_center_objective_function,
            x0=pc,
            **method_kwargs,
        )

    x = solution.x

    return 1 - solution.fun, x[0], x[1], x[2]


def _refine_orientation_projection_center_solver(
    pattern: np.ndarray,
    rescale: bool,
    rot_pc: np.ndarray,
    method: Callable,
    method_kwargs: dict,
    fixed_parameters: tuple,
    trust_region: list,
    trust_region_passed: bool,
    signal_mask: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float]:
    """Maximize the similarity between an experimental pattern and a
    projected simulated pattern by optimizing the orientation and
    projection center (PC) parameters used in the projection.

    Parameters
    ----------
    pattern
        Experimental pattern of shape (nrows, ncols).
    rescale
        Whether pattern intensities must be rescaled to [-1, 1] and data
        type 32-bit floats.
    rot_pc
        Array with Euler angles (phi1, Phi, phi2) in radians and PC
        parameters (PCx, PCy, PCz) in range [0, 1].
    method
        A supported :mod:`scipy.optimize` function. See `method`
        parameter in
        :meth:`kikuchipy.signals.EBSD.refine_orientation_projection_center`.
    method_kwargs
        Keyword arguments passed to the `method` function. For the list
        of possible arguments, see the SciPy documentation.
    fixed_parameters
        Fixed parameters used in the projection.
    trust_region
        List of +/- angular deviations in degrees as bound
        constraints on the three Euler angles and +/- percentage
        deviations as bound constraints on the PC parameters in the
        Bruker convention.
    trust_region_passed
        Whether `trust_region` was passed to the public refinement
        method.
    signal_mask
        Boolean mask equal to the experimental patterns' detector shape
        ``(n rows, n columns)``, where only pixels equal to ``False``
        are matched.

    Returns
    -------
    score
        Highest normalized cross-correlation score.
    phi1, Phi, phi2
        Optimized orientation (Euler angles) in radians.
    pcx_refined, pcy_refined, pcz_refined
        Optimized PC parameters in the Bruker convention.
    """
    # Apply mask
    pattern = _mask_pattern(pattern.astype("float32"), signal_mask)

    if rescale:
        pattern = _rescale_without_min_max_1d_float32(pattern)

    # Center intensities and get squared norm of centered pattern
    pattern, squared_norm = _zero_mean_sum_square_1d_float32(pattern)

    params = (pattern,) + fixed_parameters + (squared_norm,)
    method_name = method.__name__

    if method_name == "minimize":
        if trust_region_passed:
            method_kwargs["bounds"] = Bounds(
                rot_pc - trust_region, rot_pc + trust_region
            )
        solution = method(
            fun=_refine_orientation_projection_center_objective_function,
            x0=rot_pc,
            args=params,
            **method_kwargs,
        )
    elif SUPPORTED_OPTIMIZATION_METHODS[method_name]["supports_bounds"]:
        solution = method(
            func=_refine_orientation_projection_center_objective_function,
            args=params,
            bounds=np.column_stack([rot_pc - trust_region, rot_pc + trust_region]),
            **method_kwargs,
        )
    else:  # Is always "basinhopping", due to prior check of method name
        method_kwargs["minimizer_kwargs"].update(args=params)
        solution = method(
            func=_refine_orientation_projection_center_objective_function,
            x0=rot_pc,
            **method_kwargs,
        )

    x = solution.x

    return 1 - solution.fun, x[0], x[1], x[2], x[3], x[4], x[5]
