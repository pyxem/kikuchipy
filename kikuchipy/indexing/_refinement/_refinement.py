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

"""Setup of refinement refinement of crystal orientations and projection
centers by optimizing the similarity between experimental and simulated
patterns.
"""

import gc
import sys
from typing import Callable, Optional, Tuple, Union

import dask
from dask.diagnostics import ProgressBar
import dask.array as da
import numpy as np
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Rotation
import scipy.optimize

from kikuchipy.indexing._refinement._solvers import _refine_orientation_solver
from kikuchipy.pattern import rescale_intensity
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_for_single_pc_from_detector,
)


def compute_refine_orientation_results(
    results: list,
    xmap: CrystalMap,
    master_pattern,
) -> CrystalMap:
    """Compute the results from
    :meth:`~kikuchipy.signals.EBSD.refine_orientation` and return the
    :class:`~orix.crystal_map.CrystalMap`.

    Parameters
    ----------
    results
        Results returned from `refine_orientation()`, which is a list of
        :class:`~dask.delayed.Delayed`.
    xmap
        Crystal map passed to `refine_orientation()` to obtain
        `results`.
    master_pattern : ~kikuchipy.signals.EBSDMasterPattern
        Master pattern passed to `refine_orientation()` to obtain
        `results`.

    Returns
    -------
    refined_xmap
        Crystal map with refined orientations and scores.
    """
    n_patterns = len(results)
    with ProgressBar():
        print(f"Refining {n_patterns} orientations:", file=sys.stdout)
        results = dask.compute(*results)
        results = np.array(results)  # (n, score, phi1, Phi, phi2)
        xmap_refined = CrystalMap(
            rotations=Rotation.from_euler(results[:, 1:]),
            phase_id=np.zeros(n_patterns),
            x=xmap.x,
            y=xmap.y,
            phase_list=PhaseList(phases=master_pattern.phase),
            prop=dict(scores=results[:, 0]),
            scan_unit=xmap.scan_unit,
        )
    return xmap_refined


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

    # Get navigation shape and signal shape
    nav_shape = xmap.shape
    n_patterns = xmap.size
    sig_shape = detector.shape
    nrows, ncols = sig_shape
    n_pixels = detector.size

    # Get rotations in the correct shape
    if xmap.rotations_per_point > 1:
        rot = xmap.rotations[:, 0]
    else:
        rot = xmap.rotations
    euler = rot.to_euler()
    euler = euler.reshape(nav_shape + (3,))

    # Build up dictionary of keyword arguments to pass to the refinement
    # solver
    solver_kwargs = dict(method=method, method_kwargs=method_kwargs)

    # Determine whether pattern intensities must be rescaled
    dtype_in = patterns.dtype
    dtype_desired = np.float32
    if dtype_in != dtype_desired:
        solver_kwargs["rescale"] = True
    else:
        solver_kwargs["rescale"] = False

    if trust_region is None:
        trust_region = np.ones(3)
    solver_kwargs["trust_region"] = dask.delayed(np.deg2rad(trust_region))

    # Prepare parameters for the objective function which are constant
    # during optimization
    fixed_parameters = _check_master_pattern_and_get_data(master_pattern, energy)
    fixed_parameters += (_prepare_mask(mask),)
    fixed_parameters += (n_pixels,)
    solver_kwargs["fixed_parameters"] = dask.delayed(fixed_parameters)

    # Determine whether a new PC must be re-computed for every pattern
    new_pc = np.prod(detector.navigation_shape) != 1 and n_patterns > 1

    refined_parameters = []
    if new_pc:
        # Patterns have been indexed with varying PCs, so we
        # re-compute the PC for every pattern during refinement
        pcx = detector.pcx.astype(float).reshape(nav_shape)
        pcy = detector.pcy.astype(float).reshape(nav_shape)
        pcz = detector.pcz.astype(float).reshape(nav_shape)
        for idx in np.ndindex(nav_shape):
            delayed_solution = dask.delayed(_refine_orientation_solver)(
                pattern=patterns[idx],
                rotation=euler[idx],
                pcx=pcx[idx],
                pcy=pcy[idx],
                pcz=pcz[idx],
                nrows=nrows,
                ncols=ncols,
                tilt=detector.tilt,
                azimuthal=detector.azimuthal,
                sample_tilt=detector.sample_tilt,
                **solver_kwargs,
            )
            refined_parameters.append(delayed_solution)
    else:
        # All patterns have been indexed with the same PC, so we use
        # this during refinement of all patterns
        dc = _get_direction_cosines_for_single_pc_from_detector(detector).reshape(
            (n_pixels, 3)
        )
        for idx in np.ndindex(nav_shape):
            delayed_solution = dask.delayed(_refine_orientation_solver)(
                pattern=patterns[idx],
                rotation=euler[idx],
                direction_cosines=dc,
                **solver_kwargs,
            )
            refined_parameters.append(delayed_solution)

    print(_refinement_info_message(method, method_kwargs))

    if compute:
        output = compute_refine_orientation_results(
            results=refined_parameters, xmap=xmap, master_pattern=master_pattern
        )
    else:
        output = refined_parameters
    gc.collect()

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
        Default is "minimize".
    method_kwargs : dict, optional
        Keyword arguments to pass to function.

    Returns
    -------
    method
        SciPy optimization function.
    method_kwargs
        Keyword arguments to pass to function.
    """
    if method == "minimize" and method_kwargs is None:
        method_kwargs = {"method": "Nelder-Mead"}
    elif method_kwargs is None:
        method_kwargs = {}
    method = getattr(scipy.optimize, method)
    return method, method_kwargs


def _check_master_pattern_and_get_data(
    master_pattern, energy: Union[int, float]
) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    """Check whether the master pattern is suitable for projection, and
    return the northern and southern hemispheres along with their shape.

    Parameters
    ----------
    master_pattern : kikuchipy.signals.EBSDMasterPattern
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
    (
        master_north,
        master_south,
    ) = master_pattern._get_master_pattern_arrays_from_energy(energy=energy)
    npx, npy = master_pattern.axes_manager.signal_shape
    scale = (npx - 1) / 2
    dtype_desired = np.float32
    if master_north.dtype != dtype_desired:
        master_north = rescale_intensity(master_north, dtype_out=dtype_desired)
        master_south = rescale_intensity(master_south, dtype_out=dtype_desired)
    return master_north, master_south, npx, npy, scale


def _prepare_mask(mask: Optional[np.ndarray] = None) -> Union[int, np.ndarray]:
    """Ensure a valid mask or 1 is returned.

    Parameters
    ----------
    mask

    Returns
    -------
    mask
        Either a 1D array or 1.
    """
    if mask is None:
        mask = 1
    else:
        mask = ~mask.ravel()
    return mask


def _refinement_info_message(method: Callable, method_kwargs: dict) -> str:
    """Return a message with useful refinement information.

    Parameters
    ----------
    method
        SciPy optimization method.
    method_kwargs
        Keyword arguments to be passed to the optimization method.

    Returns
    -------
    msg
        Message with useful refinement information.
    """
    method_name = method.__name__
    if method_name == "minimize":
        method_type = "Local"
    else:
        method_type = "Global"
    msg = (
        "Refinement information:\n"
        f"\tMethod: {method.__name__}\n"
        f"\tMethod type: {method_type}\n"
        f"\tKeyword arguments passed to method: {method_kwargs}"
    )
    return msg
