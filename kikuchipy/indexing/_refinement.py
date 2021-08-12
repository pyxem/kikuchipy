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

"""Private tools for refinement of crystal orientations and projection
centers by optimizing the similarity between experimental and simulated
patterns, using SciPy.
"""

import sys
from typing import Callable, Optional, Tuple, Union

import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import numba as nb
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation
import scipy.optimize

from kikuchipy.indexing.similarity_metrics import (
    _ncc_single_patterns_2d_float32,
    _ncc_single_patterns_1d_float32,
)
from kikuchipy.pattern import rescale_intensity
from kikuchipy.pattern._pattern import _rescale
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_for_multiple_pcs,
    _project_single_pattern_from_master_pattern,
)


def _refine_orientation(
    xmap: CrystalMap,
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns: Union[np.ndarray, da.Array],
    mask: Optional[np.ndarray] = None,
    method: Optional[str] = None,
    method_kwargs: Optional[dict] = None,
    trust_region: Union[None, np.ndarray, list] = None,
    compute: bool = True,
) -> CrystalMap:
    """See the docstring of
    :meth:`kikuchipy.signals.EBSD.refine_orientation`.
    """
    method, method_kwargs = _get_optimization_method_with_kwargs(method, method_kwargs)

    # Get rotations
    if xmap.rotations_per_point > 1:
        rot = xmap.rotations[:, 0]
    else:
        rot = xmap.rotations
    euler = rot.to_euler()

    # Detector parameters and other shapes
    n_patterns = xmap.size
    n_pixels = detector.size
    sig_shape = detector.shape
    pcx = detector.pcx
    pcy = detector.pcy
    pcz = detector.pcz
    if pcx.size == 1 and n_patterns > 1:
        pcx = np.full(n_patterns, pcx)
        pcy = np.full(n_patterns, pcy)
        pcz = np.full(n_patterns, pcz)

    # Do we have to flatten pattern array?
    patterns = patterns.reshape((-1,) + sig_shape)
    #    patterns = patterns.reshape((-1, n_pixels))

    # Determine whether pattern intensities must be rescaled
    dtype_in = patterns.dtype
    dtype_desired = np.float32
    if dtype_in != dtype_desired:
        rescale = True
    else:
        rescale = False

    if trust_region is None:
        trust_region = np.ones(3)
    trust_region = dask.delayed(np.deg2rad(trust_region))

    # Signal mask
    if mask is None:
        mask = 1
    else:
        mask = ~mask

    # Prepare parameters which are constant during optimization
    fixed_parameters = _check_master_pattern_and_return_data(master_pattern, energy)
    fixed_parameters += (mask,)
    fixed_parameters += (n_pixels,)
    fixed_parameters = dask.delayed(fixed_parameters)

    if isinstance(patterns, da.Array):
        refined_parameters = None
    else:  # NumPy array
        direction_cosines = _get_direction_cosines_for_multiple_pcs(
            pcx=pcx,
            pcy=pcy,
            pcz=pcz,
            nrows=sig_shape[0],
            ncols=sig_shape[1],
            tilt=detector.tilt,
            azimuthal=detector.azimuthal,
            sample_tilt=detector.sample_tilt,
        )
        refined_parameters = [
            dask.delayed(_refine_orientation_solver)(
                pattern=patterns[i],
                rescale=rescale,
                rotation=euler[i],
                direction_cosines=direction_cosines[i],
                method=method,
                method_kwargs=method_kwargs,
                fixed_parameters=fixed_parameters,
                trust_region=trust_region,
            )
            for i in range(n_patterns)
        ]

    if compute:
        with ProgressBar():
            print(f"Refining {n_patterns} orientations:", file=sys.stdout)
            results = dask.compute(*refined_parameters)
            results = np.array(results)
            output = CrystalMap(
                rotations=Rotation.from_euler(results[:, 1:]),
                phase_id=np.zeros(n_patterns),
                x=xmap.x,
                y=xmap.y,
                phase_list=PhaseList(phases=master_pattern.phase),
                prop=dict(scores=results[:, 0]),
                scan_unit=xmap.scan_unit,
            )
    else:
        output = refined_parameters

    return output


def _refine_projection_center(
    xmap: CrystalMap,
    master_pattern,
    signal,
    detector,
    energy: Union[int, float],
    mask: Optional[np.array] = None,
    method: Optional[str] = None,
    method_kwargs: Optional[dict] = None,
    trust_region: Optional[list] = None,
    compute: bool = True,
) -> tuple:
    """See the docstring of
    :meth:`kikuchipy.signals.EBSD.refine_projection_center`.
    """
    return


def _refine_orientation_projection_center(
    xmap: CrystalMap,
    master_pattern,
    signal,
    detector,
    energy: Union[int, float],
    mask: Optional[np.ndarray] = None,
    method: Optional[str] = None,
    method_kwargs: Optional[dict] = None,
    trust_region: Optional[list] = None,
    compute: bool = True,
) -> tuple:
    """See the docstring of
    :meth:`kikuchipy.signals.EBSD.refine_orientation_projection_center`.
    """
    return


def _get_optimization_method_with_kwargs(
    method: str = "minimize", method_kwargs: Optional[dict] = None
) -> Tuple[Callable, dict]:
    """Return correct optimization function and reasonable keyword
    arguments if not given.

    Parameters
    ----------
    method
        Name of a supported SciPy optimization method. See `method`
        parameter in :meth:`kikuchipy.signals.EBSD.refine_orientation`.
    method_kwargs : dict, optional
        Keyword arguments to pass to function.

    Returns
    -------
    method
        SciPy optimization function.
    method_kwargs
        Keyword arguments to pass to function.
    """
    if method is None:
        method = "minimize"
    if method == "minimize" and method_kwargs is None:
        method_kwargs = {"method": "Nelder-Mead"}
    elif method_kwargs is None:
        method_kwargs = {}
    method = getattr(scipy.optimize, method)
    return method, method_kwargs


def _check_master_pattern_and_return_data(
    master_pattern, energy: Union[int, float]
) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    """Check whether the master pattern is suitable for projection, and
    return the northern and southern hemispheres along with their shape.

    Parameters
    ----------
    master_pattern
      Master pattern in the square Lambert projection.
    energy
        Accelerating voltage of the electron beam in kV specifying which
        master pattern energy to use during projection of simulated
        patterns.

    Returns
    -------
    master_north, master_south
        Northern and southern hemispheres of master pattern of data type
        32-bit float.
    npx, npy
        Number of columns and rows of the master pattern.
    scale
        Factor to scale up from the square Lambert projection to the
        master pattern.
    """
    master_pattern._is_suitable_for_projection(raise_if_not=True)
    master_north, master_south = master_pattern._get_master_pattern_arrays_from_energy(
        energy=energy
    )
    npx, npy = master_pattern.axes_manager.signal_shape
    scale = (npx - 1) / 2
    dtype_desired = np.float32
    if master_north.dtype != dtype_desired:
        master_north = rescale_intensity(master_north, dtype_out=dtype_desired)
        master_south = rescale_intensity(master_south, dtype_out=dtype_desired)
    return master_north, master_south, npx, npy, scale


# ------------------------------- Solvers ---------------------------- #


def _refine_orientation_solver(
    pattern: np.ndarray,
    rescale: bool,
    rotation: np.ndarray,
    direction_cosines: np.ndarray,
    method: Callable,
    method_kwargs: dict,
    fixed_parameters: tuple,
    trust_region: list,
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
    direction_cosines
        Vector array of shape (nrows, ncols, 3) and data type 32-bit
        floats.
    method
        A supported :mod:`scipy.optimize` function. See `method`
        parameter in :meth:`kikuchipy.signals.EBSD.refine_orientation`.
    method_kwargs
        Keyword arguments passed to the `method` function. For the list
        of possible arguments, see the SciPy documentation.
    fixed_parameters
        Fixed parameters used in the re-projection.
    trust_region
        List of angular deviation in radians from the initial
        orientation for the three Euler angles. Only used for
        optimization methods that support bounds (excluding
        "Powell").

    Returns
    -------
    score
        Highest normalized cross-correlation score.
    phi1, Phi, phi2
        Optimized orientation (Euler angles) in radians.
    """
    if rescale:
        pattern = pattern.astype(np.float32)
        pattern = _rescale(
            pattern, imin=np.min(pattern), imax=np.max(pattern), omin=-1, omax=1
        )

    direction_cosines = direction_cosines.reshape((-1, 3))

    params = (pattern,) + (direction_cosines,) + fixed_parameters
    method_name = method.__name__

    if method_name == "minimize":
        solution = method(
            fun=_refine_orientation_objective_function,
            x0=rotation,
            args=params,
            **method_kwargs,
        )
    elif method_name == "differential_evolution":
        solution = None
    elif method_name == "dual_annealing":
        solution = None
    elif method_name == "basinhopping":
        solution = None

    score = 1 - solution.fun
    phi1 = solution.x[0]
    Phi = solution.x[1]
    phi2 = solution.x[2]

    return score, phi1, Phi, phi2


# ------------------------ Objective functions ----------------------- #


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
        function.

    Returns
    -------
        Objective function value (normalized cross-correlation score).
    """
    experimental = args[0]
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
    #    score = _ncc_single_patterns_1d_float32(exp=args[0], sim=simulated_pattern)

    simulated_pattern = simulated_pattern.reshape(experimental.shape)
    score = _ncc_single_patterns_2d_float32(exp=experimental, sim=simulated_pattern)
    return 1 - score


@nb.jit("float64[:](float64, float64, float64)", nogil=True, nopython=True)
def _rotation_from_euler(phi1: float, Phi: float, phi2: float) -> np.ndarray:
    """Convert three Euler angles (phi1, Phi, phi2) to a unit
    quaternion.

    Taken from :meth:`orix.quaternion.Rotation.from_euler`.

    Parameters
    ----------
    phi1, Phi, phi2
        Euler angles in radians.

    Returns
    -------
    rotation
        Unit quaternion.

    Notes
    -----
    This function is optimized with Numba, so care must be taken with
    array shapes and data types.
    """
    # TODO: Implement this and similar functions in orix
    sigma = 0.5 * np.add(phi1, phi2)
    delta = 0.5 * np.subtract(phi1, phi2)
    c = np.cos(Phi / 2)
    s = np.sin(Phi / 2)

    rotation = np.zeros(4)
    rotation[0] = c * np.cos(sigma)
    rotation[1] = -s * np.cos(delta)
    rotation[2] = -s * np.sin(delta)
    rotation[3] = -c * np.sin(sigma)

    if rotation[0] < 0:
        rotation = -rotation

    return rotation
