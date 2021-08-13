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

import sys
from typing import Callable, Optional, Tuple, Union

import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Rotation
import scipy.optimize

from kikuchipy.indexing._refinement._solvers import _refine_orientation_solver
from kikuchipy.pattern import rescale_intensity
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_for_single_pc_from_detector,
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

    # Set the number of patterns and pattern rows, columns and pixels
    n_patterns = xmap.size
    n_pixels = detector.size
    nrows, ncols = detector.shape

    # Do we have to flatten pattern array?
    patterns = patterns.reshape((-1, nrows, ncols))
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

    # Prepare parameters for the objective function which are constant
    # during optimization
    fixed_parameters = _check_master_pattern_and_return_data(master_pattern, energy)
    fixed_parameters += (mask,)
    fixed_parameters += (n_pixels,)
    fixed_parameters = dask.delayed(fixed_parameters)

    # Set keyword arguments passed to the refinement solver which are
    # independent of the PC
    solver_kwargs = dict(
        rescale=rescale,
        method=method,
        method_kwargs=method_kwargs,
        fixed_parameters=fixed_parameters,
        trust_region=trust_region,
    )

    # Determine whether a new PC must be re-computed for every pattern
    new_pc = np.prod(detector.navigation_shape) != 1 and n_patterns > 1

    if isinstance(patterns, da.Array):
        refined_parameters = None
    else:  # NumPy array
        if new_pc:
            # Patterns have been indexed with varying PCs, so we
            # re-compute the PC for every pattern during refinement
            pcx = detector.pcx.astype(float)
            pcy = detector.pcy.astype(float)
            pcz = detector.pcz.astype(float)
            refined_parameters = [
                dask.delayed(_refine_orientation_solver)(
                    pattern=patterns[i],
                    rotation=euler[i],
                    pcx=pcx[i],
                    pcy=pcy[i],
                    pcz=pcz[i],
                    nrows=nrows,
                    ncols=ncols,
                    tilt=detector.tilt,
                    azimuthal=detector.azimuthal,
                    sample_tilt=detector.sample_tilt,
                    **solver_kwargs,
                )
                for i in range(n_patterns)
            ]
        else:
            # All patterns have been indexed with the same PC, so we use
            # this during refinement of all patterns
            dc = _get_direction_cosines_for_single_pc_from_detector(detector).reshape(
                (n_pixels, 3)
            )
            refined_parameters = [
                dask.delayed(_refine_orientation_solver)(
                    pattern=patterns[i],
                    rotation=euler[i],
                    direction_cosines=dc,
                    **solver_kwargs,
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
