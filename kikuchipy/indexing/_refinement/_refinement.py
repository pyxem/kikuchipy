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

from kikuchipy.indexing._refinement._solvers import (
    _refine_orientation_solver,
    _refine_orientation_projection_center_solver,
    _refine_projection_center_solver,
)
from kikuchipy.indexing._refinement import SUPPORTED_OPTIMIZATION_METHODS
from kikuchipy.pattern import rescale_intensity
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_for_single_pc_from_detector,
)


DEFAULT_TRUST_REGION_EULER_DEG = np.ones(3)
DEFAULT_TRUST_REGION_PC = np.full(3, 0.05)


# Note about use of dask.delayed
# ------------------------------
# Private refinement functions might call dask.delayed on a Dask array,
# specifically the experimental patterns. This is discouraged in the
# Dask documentation:
# https://docs.dask.org/en/latest/delayed-best-practices.html#don-t-call-dask-delayed-on-other-dask-collections
# However, since only one pattern is passed to dask.delayed at a time,
# this might be OK.


def _refine_setup(
    xmap: CrystalMap,
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns_dtype: np.dtype,
    mask: Optional[np.ndarray] = None,
    method: Optional[str] = None,
    method_kwargs: Optional[dict] = None,
) -> Tuple[Callable, dict, tuple, int, tuple, int, int, int, Rotation, dict, tuple]:
    """Set up and return everything that is common to all refinement
    functions.
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
        rotations = xmap.rotations[:, 0]
    else:
        rotations = xmap.rotations

    # Build up dictionary of keyword arguments to pass to the refinement
    # solver
    solver_kwargs = dict(method=method, method_kwargs=method_kwargs)

    # Determine whether pattern intensities must be rescaled
    dtype_desired = np.float32
    if patterns_dtype != dtype_desired:
        solver_kwargs["rescale"] = True
    else:
        solver_kwargs["rescale"] = False

    # Prepare parameters for the objective function which are constant
    # during optimization
    fixed_parameters = _check_master_pattern_and_get_data(master_pattern, energy)
    fixed_parameters += (_prepare_mask(mask),)
    fixed_parameters += (n_pixels,)

    return (
        method,
        method_kwargs,
        nav_shape,
        n_patterns,
        sig_shape,
        nrows,
        ncols,
        n_pixels,
        rotations,
        solver_kwargs,
        fixed_parameters,
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
    (
        method,
        method_kwargs,
        nav_shape,
        n_patterns,
        sig_shape,
        nrows,
        ncols,
        n_pixels,
        rotations,
        solver_kwargs,
        fixed_parameters,
    ) = _refine_setup(
        xmap=xmap,
        detector=detector,
        master_pattern=master_pattern,
        energy=energy,
        patterns_dtype=patterns.dtype,
        mask=mask,
        method=method,
        method_kwargs=method_kwargs,
    )

    # Get rotations in the correct shape
    euler = rotations.to_euler()
    euler = euler.reshape(nav_shape + (3,))

    if trust_region is None:
        trust_region = DEFAULT_TRUST_REGION_EULER_DEG
        solver_kwargs["trust_region_passed"] = False
    else:
        solver_kwargs["trust_region_passed"] = True
    solver_kwargs["trust_region"] = dask.delayed(np.deg2rad(trust_region))

    # Prepare parameters for the objective function which are constant
    # during optimization
    solver_kwargs["fixed_parameters"] = dask.delayed(fixed_parameters)

    # Determine whether a new PC is used for every pattern
    new_pc = np.prod(detector.navigation_shape) != 1 and n_patterns > 1

    # Delay data once
    patterns = dask.delayed(patterns)
    euler = dask.delayed(euler)

    refined_parameters = []
    if new_pc:
        # Patterns have been indexed with varying PCs, so we
        # re-compute the direction cosines for every pattern during
        # refinement
        pcx = detector.pcx.astype(float).reshape(nav_shape)
        pcy = detector.pcy.astype(float).reshape(nav_shape)
        pcz = detector.pcz.astype(float).reshape(nav_shape)
        pcx = dask.delayed(pcx)
        pcy = dask.delayed(pcy)
        pcz = dask.delayed(pcz)
        for idx in np.ndindex(*nav_shape):
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
        # Patterns have been indexed with the same PC, so we use the
        # same direction cosines during refinement of all patterns
        dc = _get_direction_cosines_for_single_pc_from_detector(detector).reshape(
            (n_pixels, 3)
        )
        for idx in np.ndindex(*nav_shape):
            delayed_solution = dask.delayed(_refine_orientation_solver)(
                pattern=patterns[idx],
                rotation=euler[idx],
                direction_cosines=dc,
                **solver_kwargs,
            )
            refined_parameters.append(delayed_solution)

    print(
        _refinement_info_message(
            method=method,
            method_kwargs=method_kwargs,
            trust_region=list(trust_region),
            trust_region_passed=solver_kwargs["trust_region_passed"],
        )
    )
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
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns: Union[np.ndarray, da.Array],
    mask: Optional[np.ndarray] = None,
    method: Optional[str] = None,
    method_kwargs: Optional[dict] = None,
    trust_region: Union[None, np.ndarray, list] = None,
    compute: bool = True,
) -> tuple:
    """See the docstring of
    :meth:`kikuchipy.signals.EBSD.refine_projection_center`.
    """
    (
        method,
        method_kwargs,
        nav_shape,
        n_patterns,
        sig_shape,
        nrows,
        ncols,
        n_pixels,
        rotations,
        solver_kwargs,
        fixed_parameters,
    ) = _refine_setup(
        xmap=xmap,
        detector=detector,
        master_pattern=master_pattern,
        energy=energy,
        patterns_dtype=patterns.dtype,
        mask=mask,
        method=method,
        method_kwargs=method_kwargs,
    )

    # Get rotations in the correct shape
    rotations = rotations.data
    rotations = rotations.reshape(nav_shape + (4,))

    if trust_region is None:
        trust_region = DEFAULT_TRUST_REGION_PC
        solver_kwargs["trust_region_passed"] = False
    else:
        solver_kwargs["trust_region_passed"] = True
    solver_kwargs["trust_region"] = dask.delayed(trust_region)

    # Prepare parameters for the objective function which are constant
    # during optimization
    fixed_parameters += (nrows,)
    fixed_parameters += (ncols,)
    fixed_parameters += (detector.tilt,)
    fixed_parameters += (detector.azimuthal,)
    fixed_parameters += (detector.sample_tilt,)
    solver_kwargs["fixed_parameters"] = dask.delayed(fixed_parameters)

    pc = _prepare_projection_centers(
        detector=detector, n_patterns=n_patterns, nav_shape=nav_shape
    )

    # Delay data once
    patterns = dask.delayed(patterns)
    rotations = dask.delayed(rotations)
    pc = dask.delayed(pc)

    refined_parameters = []
    for idx in np.ndindex(*nav_shape):
        delayed_solution = dask.delayed(_refine_projection_center_solver)(
            pattern=patterns[idx],
            rotation=rotations[idx],
            pc=pc[idx],
            **solver_kwargs,
        )
        refined_parameters.append(delayed_solution)

    print(
        _refinement_info_message(
            method=method,
            method_kwargs=method_kwargs,
            trust_region=list(trust_region),
            trust_region_passed=solver_kwargs["trust_region_passed"],
        )
    )
    if compute:
        output = compute_refine_projection_center_results(
            results=refined_parameters,
            detector=detector,
            xmap=xmap,
        )
    else:
        output = refined_parameters
    gc.collect()

    return output


def _refine_orientation_projection_center(
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
) -> tuple:
    """See the docstring of
    :meth:`kikuchipy.signals.EBSD.refine_orientation_projection_center`.
    """
    (
        method,
        method_kwargs,
        nav_shape,
        n_patterns,
        sig_shape,
        nrows,
        ncols,
        n_pixels,
        rotations,
        solver_kwargs,
        fixed_parameters,
    ) = _refine_setup(
        xmap=xmap,
        detector=detector,
        master_pattern=master_pattern,
        energy=energy,
        patterns_dtype=patterns.dtype,
        mask=mask,
        method=method,
        method_kwargs=method_kwargs,
    )

    # Get rotations in the correct shape
    euler = rotations.to_euler()
    euler = euler.reshape(nav_shape + (3,))

    if trust_region is None:
        trust_region = np.concatenate(
            [DEFAULT_TRUST_REGION_EULER_DEG, DEFAULT_TRUST_REGION_PC]
        )
        solver_kwargs["trust_region_passed"] = False
    else:
        solver_kwargs["trust_region_passed"] = True
    trust_region[:3] = np.deg2rad(trust_region[:3])
    solver_kwargs["trust_region"] = dask.delayed(trust_region)

    # Prepare parameters for the objective function which are constant
    # during optimization
    fixed_parameters += (nrows,)
    fixed_parameters += (ncols,)
    fixed_parameters += (detector.tilt,)
    fixed_parameters += (detector.azimuthal,)
    fixed_parameters += (detector.sample_tilt,)
    solver_kwargs["fixed_parameters"] = dask.delayed(fixed_parameters)

    pc = _prepare_projection_centers(
        detector=detector, n_patterns=n_patterns, nav_shape=nav_shape
    )

    # Stack Euler angles and PC parameters into one array of shape
    # `nav_shape` + (6,)
    euler_pc = np.dstack([euler, pc])

    # Delay data once
    patterns = dask.delayed(patterns)
    euler_pc = dask.delayed(euler_pc)

    refined_parameters = []
    for idx in np.ndindex(*nav_shape):
        delayed_solution = dask.delayed(_refine_orientation_projection_center_solver)(
            pattern=patterns[idx],
            euler_pc=euler_pc[idx],
            **solver_kwargs,
        )
        refined_parameters.append(delayed_solution)

    print(
        _refinement_info_message(
            method=method,
            method_kwargs=method_kwargs,
            trust_region=list(trust_region),
            trust_region_passed=solver_kwargs["trust_region_passed"],
        )
    )
    if compute:
        output = compute_refine_orientation_projection_center_results(
            results=refined_parameters,
            detector=detector,
            xmap=xmap,
            master_pattern=master_pattern,
        )
    else:
        output = refined_parameters
    gc.collect()

    return output


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
    supported_methods = list(SUPPORTED_OPTIMIZATION_METHODS)
    if method not in supported_methods:
        raise ValueError(
            f"Method {method} not in the list of supported methods {supported_methods}"
        )
    if method_kwargs is None:
        method_kwargs = {}
    if method == "minimize" and "method" not in method_kwargs:
        method_kwargs["method"] = "Nelder-Mead"
    if method == "basinhopping" and "minimizer_kwargs" not in method_kwargs:
        method_kwargs["minimizer_kwargs"] = {}
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


def _prepare_projection_centers(
    detector, n_patterns: int, nav_shape: tuple
) -> np.ndarray:
    """Return an array of projection center (PC) parameters of the
    appropriate shape and data type.

    Parameters
    ----------
    detector : ~kikuchipy.detectors.EBSDDetector
        Detector with PCs.
    n_patterns
        Number of patterns to use in refinement.
    nav_shape
        Navigation shape of EBSD signal to use in refinement.

    Returns
    -------
    pc
        PC array of shape nav_shape + (3,) of data type 32-bit float.
    """
    # Determine whether a new PC is used for every pattern
    new_pc = np.prod(detector.navigation_shape) != 1 and n_patterns > 1
    if new_pc:
        # Patterns have been indexed with varying PCs, so we use these
        # as the starting point for every pattern
        pc = detector.pc.astype(float).reshape(nav_shape + (3,))
    else:
        # Patterns have been indexed with the same PC, so we use this as
        # the starting point for every pattern
        pc = np.full(nav_shape + (3,), detector.pc[0], dtype=float)
    return pc


def _refinement_info_message(
    method: Callable,
    method_kwargs: dict,
    trust_region: list,
    trust_region_passed: bool,
) -> str:
    """Return a message with useful refinement information.

    Parameters
    ----------
    method
        SciPy optimization method.
    method_kwargs
        Keyword arguments to be passed to the optimization method.
    trust_region
        Trust region to use for bounds on parameters.
    trust_region_passed
        Whether a trust region was passed. Used when determining whether
        to print the trust region.

    Returns
    -------
    msg
        Message with useful refinement information.
    """
    method_name = method.__name__
    method_dict = SUPPORTED_OPTIMIZATION_METHODS[method_name]
    if method_name == "minimize":
        method_name = f"{method_kwargs['method']} (minimize)"
    optimization_type = method_dict["type"]
    msg = (
        "Refinement information:\n"
        f"\t{optimization_type.capitalize()} optimization method: {method_name}\n"
        f"\tKeyword arguments passed to method: {method_kwargs}"
    )
    if trust_region_passed or (
        optimization_type == "global" and method_dict["supports_bounds"]
    ):
        msg += f"\n\tTrust region: {trust_region}"
    return msg


def compute_refine_orientation_results(results: list, xmap: CrystalMap, master_pattern):
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
    refined_xmap : :class:`~orix.crystal_map.CrystalMap`
        Crystal map with refined orientations and scores.
    """
    n_patterns = len(results)
    with ProgressBar():
        print(f"Refining {n_patterns} orientation(s):", file=sys.stdout)
        computed_results = dask.compute(*results)
        # (n, score, phi1, Phi, phi2)
        computed_results = np.array(computed_results)
        xmap_refined = CrystalMap(
            rotations=Rotation.from_euler(computed_results[:, 1:]),
            phase_id=np.zeros(n_patterns),
            x=xmap.x,
            y=xmap.y,
            phase_list=PhaseList(phases=master_pattern.phase),
            prop=dict(scores=computed_results[:, 0]),
            scan_unit=xmap.scan_unit,
        )
    return xmap_refined


def compute_refine_projection_center_results(results: list, detector, xmap: CrystalMap):
    """Compute the results from
    :meth:`~kikuchipy.signals.EBSD.refine_projection_center` and return
    the score array and :class:`~kikuchipy.detectors.EBSDDetector`.

    Parameters
    ----------
    results
        Results returned from `refine_projection_center()`, which is a
        list of :class:`~dask.delayed.Delayed`.
    detector : ~kikuchipy.detectors.EBSDDetector
        Detector passed to `refine_projection_center()` to obtain
        `results`.
    xmap
        Crystal map passed to `refine_projection_center()` to obtain
        `results`.

    Returns
    -------
    new_scores : :class:`numpy.ndarray`
        Score array.
    new_detector : :class:`~kikuchipy.detectors.EBSDDetector`
        EBSD detector with refined projection center parameters.
    """
    n_patterns = len(results)
    nav_shape = xmap.shape
    with ProgressBar():
        print(f"Refining {n_patterns} projection center(s):", file=sys.stdout)
        computed_results = dask.compute(*results)
        # (n, score, PCx, PCy, PCz)
        computed_results = np.array(computed_results)
        new_detector = detector.deepcopy()
        new_detector.pc = computed_results[:, 1:].reshape(nav_shape + (3,))
    return computed_results[:, 0].reshape(nav_shape), new_detector


def compute_refine_orientation_projection_center_results(
    results: list,
    detector,
    xmap: CrystalMap,
    master_pattern,
):
    """Compute the results from
    :meth:`~kikuchipy.signals.EBSD.refine_orientation_projection_center`
    and return the :class:`~orix.crystal_map.CrystalMap` and
    :class:`~kikuchipy.detectors.EBSDDetector`.

    Parameters
    ----------
    results
        Results returned from `refine_orientation_projection_center()`,
        which is a list of :class:`~dask.delayed.Delayed`.
    detector : ~kikuchipy.detectors.EBSDDetector
        Detector passed to `refine_orientation_projection_center()` to
        obtain `results`.
    xmap
        Crystal map passed to `refine_orientation_projection_center()`
        to obtain `results`.
    master_pattern : ~kikuchipy.signals.EBSDMasterPattern
        Master pattern passed to
        `refine_orientation_projection_center()` to obtain `results`.

    Returns
    -------
    xmap_refined : :class:`~orix.crystal_map.CrystalMap`
        Crystal map with refined orientations and scores.
    new_detector : :class:`~kikuchipy.detectors.EBSDDetector`
        EBSD detector with refined projection center parameters.
    """
    n_patterns = len(results)
    nav_shape = xmap.shape
    with ProgressBar():
        print(
            f"Refining {n_patterns} orientation(s) and projection center(s):",
            file=sys.stdout,
        )
        computed_results = dask.compute(*results)
        computed_results = np.array(computed_results)
        # (n, score, phi1, Phi, phi2, PCx, PCy, PCz)
        xmap_refined = CrystalMap(
            rotations=Rotation.from_euler(computed_results[:, 1:4]),
            phase_id=np.zeros(n_patterns),
            x=xmap.x,
            y=xmap.y,
            phase_list=PhaseList(phases=master_pattern.phase),
            prop=dict(scores=computed_results[:, 0]),
            scan_unit=xmap.scan_unit,
        )
        new_detector = detector.deepcopy()
        new_detector.pc = computed_results[:, 4:].reshape(nav_shape + (3,))
    return xmap_refined, new_detector
