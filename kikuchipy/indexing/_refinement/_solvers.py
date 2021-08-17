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

"""Solvers for the refinement of crystal orientations and projection
centers by optimizing the similarity between experimental and simulated
patterns.
"""

from typing import Callable, Optional, Tuple

import numba as nb
import numpy as np
from scipy.optimize import Bounds

from kikuchipy.indexing._refinement._objective_functions import (
    _refine_orientation_objective_function,
    _refine_projection_center_objective_function,
)
from kikuchipy.indexing._refinement import SUPPORTED_OPTIMIZATION_METHODS
from kikuchipy.pattern._pattern import _rescale
from kikuchipy.signals.util._master_pattern import _get_direction_cosines_for_single_pc


def _refine_orientation_solver(
    pattern: np.ndarray,
    rescale: bool,
    rotation: np.ndarray,
    method: Callable,
    method_kwargs: dict,
    fixed_parameters: tuple,
    trust_region: list,
    trust_region_passed: bool,
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
    re-projected simulated pattern by optimizing the orientation (Euler
    angles) used in the re-projection.

    Parameters
    ----------
    pattern
        Experimental pattern of shape (nrows, ncols).
    rescale
        Whether pattern intensities must be rescaled to [-1, 1] and data
        type 32-bit floats.
    rotation
        Euler angles (phi1, Phi, phi2) in radians.
    method
        A supported :mod:`scipy.optimize` function. See `method`
        parameter in :meth:`kikuchipy.signals.EBSD.refine_orientation`.
    method_kwargs
        Keyword arguments passed to the `method` function. For the list
        of possible arguments, see the SciPy documentation.
    fixed_parameters
        Fixed parameters used in the re-projection.
    trust_region
        List of +/- angular deviation in degrees as bound constraints on
        the three Euler angles.
    trust_region_passed
        Whether `trust_region` was passed to the public refinement
        method.
    direction_cosines
        Vector array of shape (n pixels, 3) and data type 32-bit floats.
        If not given, `pcx`, `pcy`, `pcz`, `nrows`, `ncols`, `tilt`,
        `azimuthal`, and `sample_tilt` must be passed.
    pcx
        Projection center (PC) x coordinate. Must be passed if
        `direction_cosines` is not given.
    pcy
        PC y coordinate. Must be passed if `direction_cosines` is not
        given.
    pcz
        PC z coordinate. Must be passed if `direction_cosines` is not
        given.
    nrows
        Number of detector rows. Must be passed if `direction_cosines`
        is not given.
    ncols
        Number of detector columns. Must be passed if
        `direction_cosines` is not given.
    tilt
        Detector tilt from horizontal in degrees. Must be passed if
        `direction_cosines` is not given.
    azimuthal
        Sample tilt about the sample RD axis in degrees. Must be passed
        if `direction_cosines` is not given.
    sample_tilt
        Sample tilt from horizontal in degrees. Must be passed if
        `direction_cosines` is not given.

    Returns
    -------
    score
        Highest normalized cross-correlation score.
    phi1, Phi, phi2
        Optimized orientation (Euler angles) in radians.
    """
    if rescale:
        pattern = _rescale_pattern(pattern.astype(np.float32))

    if direction_cosines is None:
        direction_cosines = _get_direction_cosines_for_single_pc(
            pcx=pcx,
            pcy=pcy,
            pcz=pcz,
            nrows=nrows,
            ncols=ncols,
            tilt=tilt,
            azimuthal=azimuthal,
            sample_tilt=sample_tilt,
        )
        direction_cosines = direction_cosines.reshape((nrows * ncols, 3))

    params = (pattern,) + (direction_cosines,) + fixed_parameters
    method_name = method.__name__

    if method_name == "minimize":
        if trust_region_passed:
            alpha_dev, beta_dev, gamma_dev = trust_region
            alpha, beta, gamma = rotation
            method_kwargs["bounds"] = Bounds(
                [alpha - alpha_dev, beta - beta_dev, gamma - gamma_dev],
                [alpha + alpha_dev, beta + beta_dev, gamma + gamma_dev],
            )
        solution = method(
            fun=_refine_orientation_objective_function,
            x0=rotation,
            args=params,
            **method_kwargs,
        )
    elif SUPPORTED_OPTIMIZATION_METHODS[method_name]["supports_bounds"]:
        alpha_dev, beta_dev, gamma_dev = trust_region
        alpha, beta, gamma = rotation
        solution = method(
            func=_refine_orientation_objective_function,
            args=params,
            bounds=[
                [alpha - alpha_dev, alpha + alpha_dev],
                [beta - beta_dev, beta + beta_dev],
                [gamma - gamma_dev, gamma + gamma_dev],
            ],
            **method_kwargs,
        )
    else:  # Is always "basinhopping", due to prior check of method name
        key_name = "minimizer_kwargs"
        if key_name not in method_kwargs:
            method_kwargs[key_name] = dict(args=params)
        else:
            method_kwargs[key_name].update(args=params)
        solution = method(
            func=_refine_orientation_objective_function,
            x0=rotation,
            **method_kwargs,
        )

    score = 1 - solution.fun
    alpha = solution.x[0]
    beta = solution.x[1]
    gamma = solution.x[2]

    return score, alpha, beta, gamma


def _refine_projection_center_solver(
    pattern: np.ndarray,
    rescale: bool,
    rotation: np.ndarray,
    method: Callable,
    method_kwargs: dict,
    fixed_parameters: tuple,
    trust_region: list,
    trust_region_passed: bool,
    pcx: float,
    pcy: float,
    pcz: float,
) -> Tuple[float, float, float, float]:
    """Maximize the similarity between an experimental pattern and a
    re-projected simulated pattern by optimizing the projection center
    (PC) parameters used in the re-projection.

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
        parameter in :meth:`kikuchipy.signals.EBSD.refine_orientation`.
    method_kwargs
        Keyword arguments passed to the `method` function. For the list
        of possible arguments, see the SciPy documentation.
    fixed_parameters
        Fixed parameters used in the re-projection.
    trust_region
        List of +/- angular deviation in degrees as bound constraints on
        the three Euler angles.
    trust_region_passed
        Whether `trust_region` was passed to the public refinement
        method.
    pcx
        Projection center (PC) x coordinate.
    pcy
        PC y coordinate.
    pcz
        PC z coordinate.

    Returns
    -------
    score
        Highest normalized cross-correlation score.
    pcx_refined, pcy_refined, pcz_refined
        Optimized PC parameters in the Bruker convention.
    """
    if rescale:
        pattern = _rescale_pattern(pattern.astype(np.float32))

    params = (pattern,) + (rotation,) + fixed_parameters
    method_name = method.__name__

    pc = [pcx, pcy, pcz]
    if method_name == "minimize":
        if trust_region_passed:
            pcx_dev, pcy_dev, pcz_dev = trust_region
            method_kwargs["bounds"] = Bounds(
                [pcx - pcx_dev, pcy - pcy_dev, pcz - pcz_dev],
                [pcx + pcx_dev, pcy + pcy_dev, pcz + pcz_dev],
            )
        solution = method(
            fun=_refine_projection_center_objective_function,
            x0=pc,
            args=params,
            **method_kwargs,
        )
    elif SUPPORTED_OPTIMIZATION_METHODS[method_name]["supports_bounds"]:
        pcx_dev, pcy_dev, pcz_dev = trust_region
        solution = method(
            func=_refine_projection_center_objective_function,
            args=params,
            bounds=[
                [pcx - pcx_dev, pcx + pcx_dev],
                [pcy - pcy_dev, pcy + pcy_dev],
                [pcz - pcz_dev, pcz + pcz_dev],
            ],
            **method_kwargs,
        )
    else:  # Is always "basinhopping", due to prior check of method name
        key_name = "minimizer_kwargs"
        if key_name not in method_kwargs:
            method_kwargs[key_name] = dict(args=params)
        else:
            method_kwargs[key_name].update(args=params)
        solution = method(
            func=_refine_projection_center_objective_function,
            x0=pc,
            **method_kwargs,
        )

    score = 1 - solution.fun
    pcx_refined = solution.x[0]
    pcy_refined = solution.x[1]
    pcz_refined = solution.x[2]

    return score, pcx_refined, pcy_refined, pcz_refined


@nb.jit("float32[:, :](float32[:, :])", cache=True, nopython=True, nogil=True)
def _rescale_pattern(pattern: np.ndarray) -> np.ndarray:
    imin = np.min(pattern)
    imax = np.max(pattern)
    return _rescale(pattern, imin=imin, imax=imax, omin=-1, omax=1)
