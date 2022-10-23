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

"""Setup of refinement refinement of crystal orientations and projection
centers by optimizing the similarity between experimental and simulated
patterns.
"""

import sys
from time import time
from typing import Callable, Optional, Tuple, Union

from dask.diagnostics import ProgressBar
import dask.array as da
import numpy as np
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Rotation
from orix.vector import Rodrigues
import scipy.optimize

from kikuchipy.indexing._refinement._solvers import (
    _refine_orientation_solver,
    _refine_orientation_projection_center_solver,
    _refine_projection_center_solver,
)
from kikuchipy.indexing._refinement import SUPPORTED_OPTIMIZATION_METHODS
from kikuchipy.pattern import rescale_intensity
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_from_detector,
)


DEFAULT_TRUST_REGION_ROT_DEG = np.ones(3, dtype=np.float64)
DEFAULT_TRUST_REGION_PC = np.full(3, 0.05, dtype=np.float64)


def _refine_setup(
    xmap: CrystalMap,
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns_dtype: np.dtype,
    method: Optional[str] = None,
    method_kwargs: Optional[dict] = None,
) -> Tuple[
    Callable,
    dict,
    tuple,
    int,
    int,
    int,
    Rotation,
    dict,
    Tuple[np.ndarray, np.ndarray, int, int, float],
]:
    """Set up and return everything that is common to all refinement
    functions.
    """
    # Build up dictionary of keyword arguments to pass to the refinement
    # solver
    method, method_kwargs = _get_optimization_method_with_kwargs(method, method_kwargs)
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

    # Get navigation shape and signal shape
    nav_shape = xmap.shape
    n_patterns = xmap.size
    nrows, ncols = detector.shape

    # Get rotations in the correct shape
    if xmap.rotations_per_point > 1:
        rotations = xmap.rotations[:, 0]
    else:
        rotations = xmap.rotations

    return (
        method,
        method_kwargs,
        nav_shape,
        n_patterns,
        nrows,
        ncols,
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
    signal_mask: np.ndarray,
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
        nrows,
        ncols,
        rotations,
        solver_kwargs,
        fixed_parameters,
    ) = _refine_setup(
        xmap=xmap,
        detector=detector,
        master_pattern=master_pattern,
        energy=energy,
        patterns_dtype=patterns.dtype,
        method=method,
        method_kwargs=method_kwargs,
    )

    chunks = patterns.chunksize[:-1] + (-1,)

    # Get Dask array of rotations as Rodrigues-Frank vectors
    rot = Rodrigues.from_rotation(rotations).data
    rot = rot.reshape(nav_shape + (3,))
    rot = da.from_array(rot, chunks=chunks)

    if trust_region is None:
        trust_region = DEFAULT_TRUST_REGION_ROT_DEG
        solver_kwargs["trust_region_passed"] = False
    else:
        solver_kwargs["trust_region_passed"] = True
    solver_kwargs["trust_region"] = np.deg2rad(trust_region)

    # Parameters for the objective function which are constant during
    # optimization
    solver_kwargs["fixed_parameters"] = fixed_parameters

    # Determine whether a new PC is used for every pattern
    new_pc = np.prod(detector.navigation_shape) != 1 and n_patterns > 1

    map_blocks_kwargs = dict(
        drop_axis=(patterns.ndim - 1,), new_axis=(len(nav_shape),), dtype=np.float64
    )

    if new_pc:
        # Patterns have been indexed with varying PCs, so we re-compute
        # the direction cosines for every pattern during refinement
        pc = _prepare_projection_centers(
            detector=detector, n_patterns=n_patterns, nav_shape=nav_shape, chunks=chunks
        )  # shape: nav_shape + (3,)
        nav_slices = (slice(None, None),) * len(nav_shape)
        pcx = pc[nav_slices + (slice(0, 1),)]
        pcy = pc[nav_slices + (slice(1, 2),)]
        pcz = pc[nav_slices + (slice(2, 3),)]

        output = da.map_blocks(
            _refine_orientation_chunk,
            patterns,
            rot,
            pcx,
            pcy,
            pcz,
            nrows=nrows,
            ncols=ncols,
            tilt=detector.tilt,
            azimuthal=detector.azimuthal,
            sample_tilt=detector.sample_tilt,
            signal_mask=signal_mask,
            solver_kwargs=solver_kwargs,
            **map_blocks_kwargs,
        )
    else:
        # Patterns have been indexed with the same PC, so we use the
        # same direction cosines during refinement of all patterns
        dc = _get_direction_cosines_from_detector(detector, signal_mask)

        output = da.map_blocks(
            _refine_orientation_chunk,
            patterns,
            rot,
            direction_cosines=dc,
            signal_mask=signal_mask,
            solver_kwargs=solver_kwargs,
            **map_blocks_kwargs,
        )

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
            results=output, xmap=xmap, master_pattern=master_pattern
        )

    return output


def _refine_orientation_chunk(
    patterns: np.ndarray,
    rotations: np.ndarray,
    pcx: Optional[np.ndarray] = None,
    pcy: Optional[np.ndarray] = None,
    pcz: Optional[np.ndarray] = None,
    signal_mask: Optional[np.ndarray] = None,
    solver_kwargs: Optional[dict] = None,
    direction_cosines: Optional[np.ndarray] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    tilt: Optional[float] = None,
    azimuthal: Optional[float] = None,
    sample_tilt: Optional[float] = None,
):
    """Refine orientations from patterns in one dask array chunk.

    Note that ``signal_mask`` and ``solver_kwargs`` are required. They
    are set to ``None`` to enable use of this function in
    :func:`~dask.array.Array.map_blocks`.
    """
    nav_shape = patterns.shape[:-1]
    results = np.empty(nav_shape + (4,), dtype=np.float64)
    rotations = rotations.reshape(nav_shape + (3,))
    if direction_cosines is None:
        # Remove extra dimensions necessary for use of dask's map_blocks
        pcx = pcx.reshape(nav_shape)
        pcy = pcy.reshape(nav_shape)
        pcz = pcz.reshape(nav_shape)
        for idx in np.ndindex(*nav_shape):
            results[idx] = _refine_orientation_solver(
                pattern=patterns[idx],
                rotation=rotations[idx],
                pcx=pcx[idx],
                pcy=pcy[idx],
                pcz=pcz[idx],
                nrows=nrows,
                ncols=ncols,
                tilt=tilt,
                azimuthal=azimuthal,
                sample_tilt=sample_tilt,
                signal_mask=signal_mask,
                **solver_kwargs,
            )
    else:
        for idx in np.ndindex(*nav_shape):
            results[idx] = _refine_orientation_solver(
                pattern=patterns[idx],
                rotation=rotations[idx],
                direction_cosines=direction_cosines,
                signal_mask=signal_mask,
                **solver_kwargs,
            )
    return results


def _refine_projection_center(
    xmap: CrystalMap,
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns: Union[np.ndarray, da.Array],
    signal_mask: np.ndarray,
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
        nrows,
        ncols,
        rotations,
        solver_kwargs,
        fixed_parameters,
    ) = _refine_setup(
        xmap=xmap,
        detector=detector,
        master_pattern=master_pattern,
        energy=energy,
        patterns_dtype=patterns.dtype,
        method=method,
        method_kwargs=method_kwargs,
    )

    chunks = patterns.chunksize[:-1] + (-1,)

    # Get rotations in the correct shape into a Dask array
    rot = rotations.data
    rot = rot.reshape(nav_shape + (4,))
    rot = da.from_array(rot, chunks=chunks)

    if trust_region is None:
        trust_region = DEFAULT_TRUST_REGION_PC
        solver_kwargs["trust_region_passed"] = False
    else:
        solver_kwargs["trust_region_passed"] = True
    solver_kwargs["trust_region"] = trust_region

    # Prepare parameters for the objective function which are constant
    # during optimization
    fixed_parameters += (signal_mask,)
    fixed_parameters += (nrows,)
    fixed_parameters += (ncols,)
    fixed_parameters += (detector.tilt,)
    fixed_parameters += (detector.azimuthal,)
    fixed_parameters += (detector.sample_tilt,)
    solver_kwargs["fixed_parameters"] = fixed_parameters

    pc = _prepare_projection_centers(
        detector=detector,
        n_patterns=n_patterns,
        nav_shape=nav_shape,
        chunks=chunks,
    )

    output = da.map_blocks(
        _refine_projection_center_chunk,
        patterns,
        rot,
        pc,
        signal_mask=signal_mask,
        solver_kwargs=solver_kwargs,
        drop_axis=(patterns.ndim - 1,),
        new_axis=(len(nav_shape),),
        dtype=np.float64,
    )

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
            results=output, detector=detector, xmap=xmap
        )

    return output


def _refine_projection_center_chunk(
    patterns: np.ndarray,
    rotations: np.ndarray,
    pc: np.ndarray,
    solver_kwargs: dict,
    signal_mask: np.ndarray,
):
    """Refine projection centers using patterns in one dask array chunk."""
    nav_shape = patterns.shape[:-1]
    results = np.empty(nav_shape + (4,), dtype=np.float64)
    pc = pc.reshape(nav_shape + (3,))
    rotations = rotations.reshape(nav_shape + (4,))
    for idx in np.ndindex(*nav_shape):
        results[idx] = _refine_projection_center_solver(
            pattern=patterns[idx],
            rotation=rotations[idx],
            pc=pc[idx],
            signal_mask=signal_mask,
            **solver_kwargs,
        )
    return results


def _refine_orientation_projection_center(
    xmap: CrystalMap,
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns: Union[np.ndarray, da.Array],
    signal_mask: Optional[np.ndarray] = None,
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
        nrows,
        ncols,
        rotations,
        solver_kwargs,
        fixed_parameters,
    ) = _refine_setup(
        xmap=xmap,
        detector=detector,
        master_pattern=master_pattern,
        energy=energy,
        patterns_dtype=patterns.dtype,
        method=method,
        method_kwargs=method_kwargs,
    )

    chunks = patterns.chunksize[:-1] + (-1,)

    # Get Dask array of rotations as Rodrigues-Frank vectors
    rot = Rodrigues.from_rotation(rotations).data
    rot = rot.reshape(nav_shape + (3,))
    rot = da.from_array(rot, chunks=chunks)

    if trust_region is None:
        trust_region = np.concatenate(
            [DEFAULT_TRUST_REGION_ROT_DEG, DEFAULT_TRUST_REGION_PC]
        )
        solver_kwargs["trust_region_passed"] = False
    else:
        solver_kwargs["trust_region_passed"] = True
    trust_region[:3] = np.deg2rad(trust_region[:3])
    solver_kwargs["trust_region"] = trust_region

    # Prepare parameters for the objective function which are constant
    # during optimization
    fixed_parameters += (signal_mask,)
    fixed_parameters += (nrows,)
    fixed_parameters += (ncols,)
    fixed_parameters += (detector.tilt,)
    fixed_parameters += (detector.azimuthal,)
    fixed_parameters += (detector.sample_tilt,)
    solver_kwargs["fixed_parameters"] = fixed_parameters

    pc = _prepare_projection_centers(
        detector=detector,
        n_patterns=n_patterns,
        nav_shape=nav_shape,
        chunks=chunks,
    )

    # Stack Euler angles and PC parameters into one array of shape
    # `nav_shape` + (6,)
    rot_pc = da.dstack((rot, pc))

    output = da.map_blocks(
        _refine_orientation_projection_center_chunk,
        patterns,
        rot_pc,
        signal_mask=signal_mask,
        solver_kwargs=solver_kwargs,
        drop_axis=(patterns.ndim - 1,),
        new_axis=(len(nav_shape),),
        dtype=np.float64,
    )

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
            results=output,
            detector=detector,
            xmap=xmap,
            master_pattern=master_pattern,
        )

    return output


def _refine_orientation_projection_center_chunk(
    patterns: np.ndarray,
    rot_pc: np.ndarray,
    signal_mask: Optional[np.ndarray] = None,
    solver_kwargs: Optional[dict] = None,
):
    """Refine orientations and projection centers using all patterns in
    one dask array chunk.
    """
    nav_shape = patterns.shape[:-1]
    results = np.empty(nav_shape + (7,), dtype=np.float64)
    rot_pc = rot_pc.reshape(nav_shape + (6,))
    for idx in np.ndindex(*nav_shape):
        results[idx] = _refine_orientation_projection_center_solver(
            pattern=patterns[idx],
            rot_pc=rot_pc[idx],
            signal_mask=signal_mask,
            **solver_kwargs,
        )
    return results


def _get_optimization_method_with_kwargs(
    method: str = "minimize", method_kwargs: Optional[dict] = None
) -> Tuple[Callable, dict]:
    """Return correct optimization function and reasonable keyword
    arguments if not given.

    Parameters
    ----------
    method
        Name of a supported SciPy optimization method. See ``method``
        parameter in :meth:`kikuchipy.signals.EBSD.refine_orientation`.
        Default is ``"minimize"``.
    method_kwargs
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
    master_pattern: "EBSDMasterPattern", energy: Union[int, float]
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


def _prepare_projection_centers(
    detector: "EBSDDetector",
    n_patterns: int,
    nav_shape: tuple,
    chunks: tuple,
) -> da.Array:
    """Return an array of projection center (PC) parameters of the
    appropriate shape and data type.

    Parameters
    ----------
    detector
        Detector with PCs.
    n_patterns
        Number of patterns to use in refinement.
    nav_shape
        Navigation shape of EBSD signal to use in refinement.
    chunks
        Chunk shape of the pattern dask array.

    Returns
    -------
    pc
        PC dask array of shape nav_shape + (3,) of data type 64-bit
        float.
    """
    # Determine whether a new PC is used for every pattern
    new_pc = np.prod(detector.navigation_shape) != 1 and n_patterns > 1

    dtype = np.float64
    shape = nav_shape + (3,)
    if new_pc:
        # Patterns have been indexed with varying PCs, so we use these
        # as the starting point for every pattern
        pc = detector.pc.astype(dtype).reshape(shape)
    else:
        # Patterns have been indexed with the same PC, so we use this as
        # the starting point for every pattern
        pc = np.full(shape, detector.pc[0], dtype=dtype)

    # Get dask array with proper chunks
    pc = da.from_array(pc, chunks=chunks)

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


def compute_refine_orientation_results(
    results: da.Array, xmap: CrystalMap, master_pattern: "EBSDMasterPattern"
) -> CrystalMap:
    """Compute the results from
    :meth:`~kikuchipy.signals.EBSD.refine_orientation` and return the
    :class:`~orix.crystal_map.CrystalMap`.

    Parameters
    ----------
    results
        Dask array returned from ``refine_orientation()``.
    xmap
        Crystal map passed to ``refine_orientation()`` to obtain
        ``results``.
    master_pattern
        Master pattern passed to ``refine_orientation()`` to obtain
        ``results``.

    Returns
    -------
    refined_xmap
        Crystal map with refined orientations and scores.
    """
    n_patterns = int(np.prod(results.shape[:-1]))
    with ProgressBar():
        print(f"Refining {n_patterns} orientation(s):", file=sys.stdout)
        time_start = time()
        computed_results = results.compute().reshape((-1, 4))
        total_time = time() - time_start
        patterns_per_second = int(np.floor(n_patterns / total_time))
        print(f"Refinement speed: {patterns_per_second} patterns/s", file=sys.stdout)
        # (n, score, phi1, Phi, phi2)
        computed_results = np.array(computed_results)
        xmap_refined = CrystalMap(
            rotations=Rotation.from_neo_euler(Rodrigues(computed_results[:, 1:])),
            phase_id=np.zeros(n_patterns),
            x=xmap.x,
            y=xmap.y,
            phase_list=PhaseList(phases=master_pattern.phase),
            prop=dict(scores=computed_results[:, 0]),
            scan_unit=xmap.scan_unit,
        )
    return xmap_refined


def compute_refine_projection_center_results(
    results: da.Array, detector: "EBSDDetector", xmap: CrystalMap
) -> Tuple[np.ndarray, "EBSDDetector"]:
    """Compute the results from
    :meth:`~kikuchipy.signals.EBSD.refine_projection_center` and return
    the score array and :class:`~kikuchipy.detectors.EBSDDetector`.

    Parameters
    ----------
    results
        Dask array returned from ``refine_projection_center()``.
    detector
        Detector passed to ``refine_projection_center()`` to obtain
        ``results``.
    xmap
        Crystal map passed to ``refine_projection_center()`` to obtain
        ``results``.

    Returns
    -------
    new_scores
        Score array.
    new_detector
        EBSD detector with refined projection center parameters.
    """
    n_patterns = int(np.prod(results.shape[:-1]))
    nav_shape = xmap.shape
    with ProgressBar():
        print(f"Refining {n_patterns} projection center(s):", file=sys.stdout)
        time_start = time()
        computed_results = results.compute().reshape((-1, 4))
        total_time = time() - time_start
        patterns_per_second = int(np.floor(n_patterns / total_time))
        print(f"Refinement speed: {patterns_per_second} patterns/s", file=sys.stdout)
        # (n, score, PCx, PCy, PCz)
        computed_results = np.array(computed_results)
        new_detector = detector.deepcopy()
        new_detector.pc = computed_results[:, 1:].reshape(nav_shape + (3,))
    return computed_results[:, 0].reshape(nav_shape), new_detector


def compute_refine_orientation_projection_center_results(
    results: da.Array,
    detector: "EBSDDetector",
    xmap: CrystalMap,
    master_pattern: "EBSDMasterPattern",
) -> Tuple[CrystalMap, "EBSDDetector"]:
    """Compute the results from
    :meth:`~kikuchipy.signals.EBSD.refine_orientation_projection_center`
    and return the :class:`~orix.crystal_map.CrystalMap` and
    :class:`~kikuchipy.detectors.EBSDDetector`.

    Parameters
    ----------
    results
        Dask array returned from
        ``refine_orientation_projection_center()``.
    detector
        Detector passed to ``refine_orientation_projection_center()`` to
        obtain ``results``.
    xmap
        Crystal map passed to ``refine_orientation_projection_center()``
        to obtain ``results``.
    master_pattern
        Master pattern passed to
        ``refine_orientation_projection_center()`` to obtain
        ``results``.

    Returns
    -------
    xmap_refined
        Crystal map with refined orientations and scores.
    new_detector
        EBSD detector with refined projection center parameters.

    See Also
    --------
    kikuchipy.signals.EBSD.refine_orientation_projection_center
    """
    n_patterns = int(np.prod(results.shape[:-1]))
    nav_shape = xmap.shape
    with ProgressBar():
        print(
            f"Refining {n_patterns} orientation(s) and projection center(s):",
            file=sys.stdout,
        )
        time_start = time()
        computed_results = results.compute().reshape((-1, 7))
        total_time = time() - time_start
        patterns_per_second = int(np.floor(n_patterns / total_time))
        print(f"Refinement speed: {patterns_per_second} patterns/s", file=sys.stdout)
        computed_results = np.array(computed_results)
        # (n, score, phi1, Phi, phi2, PCx, PCy, PCz)
        xmap_refined = CrystalMap(
            rotations=Rotation.from_neo_euler(Rodrigues(computed_results[:, 1:4])),
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
