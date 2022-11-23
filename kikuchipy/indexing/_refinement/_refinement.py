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
import scipy.optimize

from kikuchipy.indexing._refinement._solvers import (
    _refine_orientation_solver_nlopt,
    _refine_orientation_solver_scipy,
    _refine_orientation_pc_solver_nlopt,
    _refine_orientation_pc_solver_scipy,
    _refine_pc_solver_nlopt,
    _refine_pc_solver_scipy,
)
from kikuchipy.indexing._refinement import SUPPORTED_OPTIMIZATION_METHODS
from kikuchipy.indexing._refinement._setup import _RefinementSetup
from kikuchipy.pattern import rescale_intensity
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_from_detector,
)


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
            rotations=Rotation.from_euler(computed_results[:, 1:]),
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


# -------------------------- Setup functions ------------------------- #


def _refine_setup(
    xmap: CrystalMap,
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns_dtype: np.dtype,
    dimension: int,
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
    str,
]:
    """Set up and return everything that is common to all refinement
    functions.
    """
    # Build up dictionary of keyword arguments to pass to the refinement
    # solver
    method, method_kwargs, pkg = _get_optimization_method_with_kwargs(
        method, method_kwargs, dimension
    )
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
        pkg,
    )


def _get_optimization_method_with_kwargs(
    method: str = "minimize",
    method_kwargs: Optional[dict] = None,
    dimension: Optional[int] = None,
) -> Tuple[Callable, dict, str]:
    """Return correct optimization function and reasonable keyword
    arguments if not given.

    Parameters
    ----------
    method
        Name of a supported SciPy optimization method or NLopt
        algorithm. See ``method`` parameter in
        :meth:`kikuchipy.signals.EBSD.refine_orientation`. Default is
        ``"minimize"``.
    method_kwargs
        Keyword arguments to pass to optimization method. Only
        applicable if a SciPy function is requested.
    dimension
        Number of optimization parameters. Only applicable if a NLopt
        algorithm is requested.

    Returns
    -------
    method
        SciPy optimization function or NLopt optimizer.
    method_kwargs
        Keyword arguments to pass to function. Only applicable if a
        SciPy function is requested.
    pkg
        Optimization package, either ``"scipy"`` or ``"nlopt"``.
    """
    supported_methods = list(SUPPORTED_OPTIMIZATION_METHODS)
    method = method.lower()
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

    pkg = SUPPORTED_OPTIMIZATION_METHODS[method]["package"]
    if pkg == "scipy":
        method = getattr(scipy.optimize, method)
    else:
        method = _initialize_nlopt_optimizer(method, dimension)

    return method, method_kwargs, pkg


def _initialize_nlopt_optimizer(method: str, dimension: int) -> "nlopt.opt":
    """Initialize an NLopt optimizer.

    Parameters
    ----------
    method
        Supported algorithm, currently only ``"LN_NELDERMEAD"``.
    dimension
        Number of optimization parameters.

    Returns
    -------
    opt
        Optimizer instance.
    """
    from kikuchipy import _nlopt_installed

    if not _nlopt_installed:
        raise ImportError(
            f"Package `nlopt` required for method {method} is not installed"
        )
    import nlopt

    return nlopt.opt(method.upper(), dimension)


def _set_nlopt_parameters(
    opt: "nlopt.opt",
    rtol: float,
    initial_step: Union[float, np.ndarray, None] = None,
    maxeval: Optional[int] = None,
) -> "nlopt.opt":
    """Set NLopt optimization parameters.

    Parameters
    ----------
    opt
        Optimizer.
    rtol
        Relative tolerance stopping criterion.
    initial_step
        Initial parameter step(s).
    maxeval
        Maximum function evaluations stopping criterion.

    Returns
    -------
    opt
        Optimizer with parameters set.
    """
    opt.set_ftol_rel(rtol)
    if initial_step is not None:
        opt.set_initial_step(initial_step)
    if maxeval is not None:
        opt.set_maxeval(maxeval)
    return opt


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


def _prepare_pc(
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
    package: str,
    method: Union[Callable, "nlopt.opt"],
    method_kwargs: dict,
    trust_region: Optional[np.ndarray] = None,
    initial_step: Union[tuple, list, float, np.ndarray, None] = None,
    rtol: Optional[float] = None,
    maxeval: Optional[int] = None,
) -> str:
    """Return a message with useful refinement information.

    Parameters
    ----------
    package
        ``"scipy"`` or ``"nlopt"``.
    method
        *SciPy* optimization method or *NLopt* optimizer.
    method_kwargs
        Keyword arguments to be passed to the optimization method.
    trust_region
        Trust region to use for bounds on parameters.
    initial_step
        Initial step(s) for the parameters. Only displayed if given and
        ``package="nlopt"``.
    rtol
        Relative tolerance stopping criterion. Only displayed if given
        and ``package="nlopt"``.
    maxeval
        Maximum function evaluations. Only displayed if given and
        ``package="nlopt"``.

    Returns
    -------
    msg
        Message with useful refinement information.
    """
    if package == "scipy":
        method_name = method.__name__
        if method_name == "minimize":
            method_name = f"{method_kwargs['method']} (minimize)"
    else:
        method_name = "ln_neldermead"

    method_dict = SUPPORTED_OPTIMIZATION_METHODS[method_name]
    opt_type = method_dict["type"]

    msg = (
        "Refinement information:\n"
        f"  Method: {package} - {method_name} ({opt_type})\n"
    )

    if method_dict["supports_bounds"]:
        msg += f"\n\tTrust region: {np.array_str(trust_region, precision=5)}"

    if package == "scipy":
        msg += f"  Keyword arguments passed to method: {method_kwargs}"
    else:
        if rtol:
            msg += f"  Relative tolerance: {rtol}"
        if initial_step:
            msg += f"  Initial step: {initial_step}"
        if maxeval:
            msg += f"  Max. function evaulations: {maxeval}"

    return msg


# -------------------------- Refine orientation ---------------------- #


def _refine_orientation2(
    xmap: CrystalMap,
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns: Union[np.ndarray, da.Array],
    signal_mask: np.ndarray,
    trust_region: Union[tuple, list, np.ndarray],
    rtol: float,
    method: Optional[str] = None,
    method_kwargs: Optional[dict] = None,
    initial_step: Optional[float] = None,
    maxeval: Optional[int] = None,
    compute: bool = True,
):
    ref = _RefinementSetup(
        mode="ori",
        xmap=xmap,
        detector=detector,
        master_pattern=master_pattern,
        energy=energy,
        patterns=patterns,
        rtol=rtol,
        method=method,
        method_kwargs=method_kwargs,
        initial_step=initial_step,
        maxeval=maxeval,
    )

    if ref.package == "scipy":
        chunk_func = _refine_orientation_chunk_scipy
    else:
        chunk_func = _refine_orientation_chunk_nlopt

    rot = ref.rotations_array
    lower_bounds, upper_bounds = ref.get_bound_constraints(trust_region)

    if ref.unique_pc:
        # Patterns have been indexed with varying PCs, so we re-compute
        # the direction cosines for every pattern during refinement
        pc = ref.pc_array
        nav_slices = (slice(None, None),) * len(ref.nav_shape)
        pcx = pc[nav_slices + (slice(0, 1),)]
        pcy = pc[nav_slices + (slice(1, 2),)]
        pcz = pc[nav_slices + (slice(2, 3),)]

        res = da.map_blocks(
            chunk_func,
            patterns,
            rot,
            lower_bounds,
            upper_bounds,
            pcx,
            pcy,
            pcz,
            nrows=detector.nrows,
            ncols=detector.ncols,
            tilt=detector.tilt,
            azimuthal=detector.azimuthal,
            sample_tilt=detector.sample_tilt,
            signal_mask=signal_mask,
            solver_kwargs=ref.solver_kwargs,
            **ref.map_blocks_kwargs,
        )
    else:
        # Patterns have been indexed with the same PC, so we use the
        # same direction cosines during refinement of all patterns
        dc = _get_direction_cosines_from_detector(detector, signal_mask)

        res = da.map_blocks(
            chunk_func,
            patterns,
            rot,
            lower_bounds,
            upper_bounds,
            direction_cosines=dc,
            signal_mask=signal_mask,
            solver_kwargs=ref.solver_kwargs,
            **ref.map_blocks_kwargs,
        )

    msg = ref.get_info_message(trust_region)
    print(msg)

    if compute:
        res = compute_refine_orientation_results(
            results=res, xmap=xmap, master_pattern=master_pattern
        )

    return res


def _refine_orientation(
    xmap: CrystalMap,
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns: Union[np.ndarray, da.Array],
    signal_mask: np.ndarray,
    trust_region: Union[tuple, list, np.ndarray],
    rtol: float,
    method: Optional[str] = None,
    method_kwargs: Optional[dict] = None,
    initial_step: Optional[float] = None,
    maxeval: Optional[int] = None,
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
        pkg,
    ) = _refine_setup(
        xmap=xmap,
        detector=detector,
        master_pattern=master_pattern,
        energy=energy,
        patterns_dtype=patterns.dtype,
        dimension=3,
        method=method,
        method_kwargs=method_kwargs,
    )

    chunks = patterns.chunksize[:-1] + (-1,)

    # Get Dask array of rotations as Euler angles
    rot = rotations.to_euler()
    rot = rot.reshape(nav_shape + (3,))
    rot = da.from_array(rot, chunks=chunks)

    # Prepare bound constraints
    trust_region = np.deg2rad(trust_region)
    lower_bounds = da.fmax(rot - trust_region, 3 * [0])
    upper_bounds = da.fmin(rot + trust_region, [2 * np.pi, np.pi, 2 * np.pi])

    map_blocks_kwargs = dict(
        drop_axis=(patterns.ndim - 1,), new_axis=(len(nav_shape),), dtype=np.float64
    )

    if pkg == "nlopt":
        # Set up NLopt optimizer and prepare other parameters
        chunk_func = _refine_orientation_chunk_nlopt

        if initial_step is not None:
            initial_step = np.deg2rad(initial_step)

        # Add extra keyword arguments passed to the chunk function
        opt = solver_kwargs.pop("method")
        map_blocks_kwargs["opt"] = _set_nlopt_parameters(
            opt, rtol=rtol, initial_step=initial_step, maxeval=maxeval
        )

        # Remove remaining keyword arguments not accepted in the NLopt
        # solver function
        for k in ["method_kwargs"]:
            del solver_kwargs[k]
    else:
        # Prepare parameters to pass to the SciPy method
        chunk_func = _refine_orientation_chunk_scipy

    # Parameters for the objective function which are constant during
    # optimization
    solver_kwargs["fixed_parameters"] = fixed_parameters

    # Determine whether a new PC is used for every pattern
    new_pc = np.prod(detector.navigation_shape) != 1 and n_patterns > 1

    if new_pc:
        # Patterns have been indexed with varying PCs, so we re-compute
        # the direction cosines for every pattern during refinement
        pc = _prepare_pc(
            detector=detector, n_patterns=n_patterns, nav_shape=nav_shape, chunks=chunks
        )  # shape: nav_shape + (3,)
        nav_slices = (slice(None, None),) * len(nav_shape)
        pcx = pc[nav_slices + (slice(0, 1),)]
        pcy = pc[nav_slices + (slice(1, 2),)]
        pcz = pc[nav_slices + (slice(2, 3),)]

        output = da.map_blocks(
            chunk_func,
            patterns,
            rot,
            lower_bounds,
            upper_bounds,
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
            chunk_func,
            patterns,
            rot,
            lower_bounds,
            upper_bounds,
            direction_cosines=dc,
            signal_mask=signal_mask,
            solver_kwargs=solver_kwargs,
            **map_blocks_kwargs,
        )

    print(
        _refinement_info_message(
            package=pkg,
            method=method,
            method_kwargs=method_kwargs,
            trust_region=trust_region,
            initial_step=initial_step,
            rtol=rtol,
            maxeval=maxeval,
        )
    )
    if compute:
        output = compute_refine_orientation_results(
            results=output, xmap=xmap, master_pattern=master_pattern
        )

    return output


def _refine_orientation_chunk_scipy(
    patterns: np.ndarray,
    rotations: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
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
    """Refine orientations from patterns in one dask array chunk using
    *SciPy*.

    Note that ``signal_mask`` and ``solver_kwargs`` are required. They
    are set to ``None`` to enable use of this function in
    :func:`~dask.array.Array.map_blocks`.
    """
    nav_shape = patterns.shape[:-1]
    results = np.empty(nav_shape + (4,), dtype=np.float64)
    rotations = rotations.reshape(nav_shape + (3,))
    lower_bounds = lower_bounds.reshape(nav_shape + (3,))
    upper_bounds = upper_bounds.reshape(nav_shape + (3,))

    if direction_cosines is None:
        # Remove extra dimensions necessary for use of dask's map_blocks
        pcx = pcx.reshape(nav_shape)
        pcy = pcy.reshape(nav_shape)
        pcz = pcz.reshape(nav_shape)
        for idx in np.ndindex(*nav_shape):
            results[idx] = _refine_orientation_solver_scipy(
                pattern=patterns[idx],
                rotation=rotations[idx],
                lower_bounds=lower_bounds[idx],
                upper_bounds=upper_bounds[idx],
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
            results[idx] = _refine_orientation_solver_scipy(
                pattern=patterns[idx],
                rotation=rotations[idx],
                lower_bounds=lower_bounds[idx],
                upper_bounds=upper_bounds[idx],
                direction_cosines=direction_cosines,
                signal_mask=signal_mask,
                **solver_kwargs,
            )

    return results


def _refine_orientation_chunk_nlopt(
    patterns: np.ndarray,
    rotations: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    pcx: Optional[np.ndarray] = None,
    pcy: Optional[np.ndarray] = None,
    pcz: Optional[np.ndarray] = None,
    opt: "nlopt.opt" = None,
    signal_mask: Optional[np.ndarray] = None,
    solver_kwargs: Optional[dict] = None,
    direction_cosines: Optional[np.ndarray] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    tilt: Optional[float] = None,
    azimuthal: Optional[float] = None,
    sample_tilt: Optional[float] = None,
):
    """Refine orientations from patterns in one dask array chunk using
    *NLopt*.

    Note that ``signal_mask``, ``solver_kwargs`` and ``opt`` are
    required. They are set to ``None`` to enable use of this function in
    :func:`~dask.array.Array.map_blocks`.
    """
    # Copy optimizer
    import nlopt

    opt = nlopt.opt(opt)

    nav_shape = patterns.shape[:-1]
    results = np.empty(nav_shape + (4,), dtype=np.float64)
    rotations = rotations.reshape(nav_shape + (3,))
    lower_bounds = lower_bounds.reshape(nav_shape + (3,))
    upper_bounds = upper_bounds.reshape(nav_shape + (3,))

    if direction_cosines is None:
        # Remove extra dimensions necessary for use of Dask's map_blocks
        pcx = pcx.reshape(nav_shape)
        pcy = pcy.reshape(nav_shape)
        pcz = pcz.reshape(nav_shape)

        for idx in np.ndindex(*nav_shape):
            results[idx] = _refine_orientation_solver_nlopt(
                opt=opt,
                pattern=patterns[idx],
                rotation=rotations[idx],
                lower_bounds=lower_bounds[idx],
                upper_bounds=upper_bounds[idx],
                signal_mask=signal_mask,
                pcx=pcx[idx],
                pcy=pcy[idx],
                pcz=pcz[idx],
                nrows=nrows,
                ncols=ncols,
                tilt=tilt,
                azimuthal=azimuthal,
                sample_tilt=sample_tilt,
                **solver_kwargs,
            )
    else:
        for idx in np.ndindex(*nav_shape):
            results[idx] = _refine_orientation_solver_nlopt(
                opt=opt,
                pattern=patterns[idx],
                rotation=rotations[idx],
                lower_bounds=lower_bounds[idx],
                upper_bounds=upper_bounds[idx],
                signal_mask=signal_mask,
                direction_cosines=direction_cosines,
                **solver_kwargs,
            )

    return results


# ------------------------------ Refine PC --------------------------- #


def _refine_pc2(
    xmap: CrystalMap,
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns: Union[np.ndarray, da.Array],
    signal_mask: np.ndarray,
    trust_region: Union[tuple, list, np.ndarray],
    rtol: float,
    method: Optional[str] = None,
    method_kwargs: Optional[dict] = None,
    initial_step: Optional[float] = None,
    maxeval: Optional[int] = None,
    compute: bool = True,
):
    ref = _RefinementSetup(
        mode="pc",
        xmap=xmap,
        detector=detector,
        master_pattern=master_pattern,
        energy=energy,
        patterns=patterns,
        rtol=rtol,
        method=method,
        method_kwargs=method_kwargs,
        initial_step=initial_step,
        maxeval=maxeval,
        signal_mask=signal_mask,
    )

    if ref.package == "scipy":
        chunk_func = _refine_pc_chunk_scipy
    else:
        chunk_func = _refine_pc_chunk_nlopt

    rot = ref.rotations_array
    pc = ref.pc_array
    lower_bounds, upper_bounds = ref.get_bound_constraints(trust_region)

    res = da.map_blocks(
        chunk_func,
        patterns,
        rot,
        pc,
        lower_bounds,
        upper_bounds,
        signal_mask=signal_mask,
        solver_kwargs=ref.solver_kwargs,
        **ref.map_blocks_kwargs,
    )

    msg = ref.get_info_message(trust_region)
    print(msg)

    if compute:
        res = compute_refine_projection_center_results(
            results=res, detector=detector, xmap=xmap
        )

    return res


def _refine_pc(
    xmap: CrystalMap,
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns: Union[np.ndarray, da.Array],
    trust_region: Union[tuple, list, np.ndarray],
    rtol: float,
    signal_mask: np.ndarray,
    method: Optional[str] = None,
    method_kwargs: Optional[dict] = None,
    initial_step: Union[tuple, list, np.ndarray, None] = None,
    maxeval: Optional[int] = None,
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
        pkg,
    ) = _refine_setup(
        xmap=xmap,
        detector=detector,
        master_pattern=master_pattern,
        energy=energy,
        patterns_dtype=patterns.dtype,
        dimension=3,
        method=method,
        method_kwargs=method_kwargs,
    )

    chunks = patterns.chunksize[:-1] + (-1,)

    # Get rotations in the correct shape into a Dask array
    rot = rotations.data
    rot = rot.reshape(nav_shape + (4,))
    rot = da.from_array(rot, chunks=chunks)

    pc = _prepare_pc(
        detector=detector,
        n_patterns=n_patterns,
        nav_shape=nav_shape,
        chunks=chunks,
    )

    # Prepare bound constraints
    lower_bounds = da.fmax(pc - trust_region, 3 * [-3])
    upper_bounds = da.fmin(pc + trust_region, 3 * [3])

    map_blocks_kwargs = dict(
        drop_axis=(patterns.ndim - 1,), new_axis=(len(nav_shape),), dtype=np.float64
    )

    # Set up NLopt optimizer if requested
    if pkg == "nlopt":
        chunk_func = _refine_pc_chunk_nlopt

        # Add extra keyword arguments passed to the chunk function
        opt = solver_kwargs.pop("method")
        map_blocks_kwargs["opt"] = _set_nlopt_parameters(
            opt, rtol=rtol, initial_step=initial_step, maxeval=maxeval
        )

        # Remove remaining keyword arguments not accepted in the NLopt
        # solver function
        for k in ["method_kwargs"]:
            del solver_kwargs[k]
    else:
        chunk_func = _refine_pc_chunk_scipy

    # Prepare parameters for the objective function which are constant
    # during optimization
    fixed_parameters += (signal_mask,)
    fixed_parameters += (nrows,)
    fixed_parameters += (ncols,)
    fixed_parameters += (detector.tilt,)
    fixed_parameters += (detector.azimuthal,)
    fixed_parameters += (detector.sample_tilt,)
    solver_kwargs["fixed_parameters"] = fixed_parameters

    output = da.map_blocks(
        chunk_func,
        patterns,
        rot,
        pc,
        lower_bounds,
        upper_bounds,
        signal_mask=signal_mask,
        solver_kwargs=solver_kwargs,
        **map_blocks_kwargs,
    )

    print(
        _refinement_info_message(
            package=pkg,
            method=method,
            method_kwargs=method_kwargs,
            trust_region=trust_region,
            initial_step=initial_step,
            rtol=rtol,
            maxeval=maxeval,
        )
    )

    if compute:
        output = compute_refine_projection_center_results(
            results=output, detector=detector, xmap=xmap
        )

    return output


def _refine_pc_chunk_scipy(
    patterns: np.ndarray,
    rotations: np.ndarray,
    pc: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    solver_kwargs: dict,
    signal_mask: np.ndarray,
):
    """Refine projection centers using patterns in one dask array chunk."""
    nav_shape = patterns.shape[:-1]
    results = np.empty(nav_shape + (4,), dtype=np.float64)
    pc = pc.reshape(nav_shape + (3,))
    rotations = rotations.reshape(nav_shape + (4,))
    for idx in np.ndindex(*nav_shape):
        results[idx] = _refine_pc_solver_scipy(
            pattern=patterns[idx],
            rotation=rotations[idx],
            pc=pc[idx],
            signal_mask=signal_mask,
            **solver_kwargs,
        )
    return results


def _refine_pc_chunk_nlopt(
    patterns: np.ndarray,
    rotations: np.ndarray,
    pc: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    solver_kwargs: dict,
    signal_mask: np.ndarray,
    opt: "nlopt.opt" = None,
):
    """Refine projection centers using patterns in one dask array chunk."""
    # Copy optimizer
    import nlopt

    opt = nlopt.opt(opt)

    nav_shape = patterns.shape[:-1]
    results = np.empty(nav_shape + (4,), dtype=np.float64)
    pc = pc.reshape(nav_shape + (3,))
    rotations = rotations.reshape(nav_shape + (4,))
    lower_bounds = lower_bounds.reshape(nav_shape + (3,))
    upper_bounds = upper_bounds.reshape(nav_shape + (3,))

    for idx in np.ndindex(*nav_shape):
        results[idx] = _refine_pc_solver_nlopt(
            opt=opt,
            pattern=patterns[idx],
            pc=pc[idx],
            rotation=rotations[idx],
            lower_bounds=lower_bounds[idx],
            upper_bounds=upper_bounds[idx],
            signal_mask=signal_mask,
            **solver_kwargs,
        )

    return results


# ---------------------- Refine orientation and PC ------------------- #


def _refine_orientation_pc2(
    xmap: CrystalMap,
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns: Union[np.ndarray, da.Array],
    trust_region: Union[tuple, list, np.ndarray],
    rtol: float,
    signal_mask: Optional[np.ndarray] = None,
    method: Optional[str] = None,
    method_kwargs: Optional[dict] = None,
    initial_step: Union[tuple, list, np.ndarray, None] = None,
    maxeval: Optional[int] = None,
    compute: bool = True,
) -> tuple:
    """See the docstring of
    :meth:`kikuchipy.signals.EBSD.refine_orientation_projection_center`.
    """
    ref = _RefinementSetup(
        mode="ori_pc",
        xmap=xmap,
        detector=detector,
        master_pattern=master_pattern,
        energy=energy,
        patterns=patterns,
        rtol=rtol,
        method=method,
        method_kwargs=method_kwargs,
        initial_step=initial_step,
        maxeval=maxeval,
        signal_mask=signal_mask,
    )

    # Stack Euler angles and PC parameters into one array of shape
    # `nav_shape` + (6,)
    rot_pc = ref.rotations_pc_array

    # Prepare bound constraints
    lower_bounds, upper_bounds = ref.get_bound_constraints(trust_region)

    if ref.package == "nlopt":
        chunk_func = _refine_orientation_pc_chunk_nlopt
    else:
        chunk_func = _refine_orientation_pc_chunk_scipy

    res = da.map_blocks(
        chunk_func,
        patterns,
        rot_pc,
        lower_bounds,
        upper_bounds,
        signal_mask=signal_mask,
        solver_kwargs=ref.solver_kwargs,
        **ref.map_blocks_kwargs,
    )

    msg = ref.get_info_message(trust_region)
    print(msg)

    if compute:
        res = compute_refine_orientation_projection_center_results(
            results=res,
            detector=detector,
            xmap=xmap,
            master_pattern=master_pattern,
        )

    return res


def _refine_orientation_pc(
    xmap: CrystalMap,
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns: Union[np.ndarray, da.Array],
    trust_region: Union[tuple, list, np.ndarray],
    rtol: float,
    signal_mask: Optional[np.ndarray] = None,
    method: Optional[str] = None,
    method_kwargs: Optional[dict] = None,
    initial_step: Union[tuple, list, np.ndarray, None] = None,
    maxeval: Optional[int] = None,
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
        pkg,
    ) = _refine_setup(
        xmap=xmap,
        detector=detector,
        master_pattern=master_pattern,
        energy=energy,
        patterns_dtype=patterns.dtype,
        dimension=6,
        method=method,
        method_kwargs=method_kwargs,
    )

    chunks = patterns.chunksize[:-1] + (-1,)

    # Get Dask array of rotations as Euler angles
    rot = rotations.to_euler()
    rot = rot.reshape(nav_shape + (3,))
    rot = da.from_array(rot, chunks=chunks)

    # Get Dask array of PC values
    pc = _prepare_pc(
        detector=detector,
        n_patterns=n_patterns,
        nav_shape=nav_shape,
        chunks=chunks,
    )

    # Stack Euler angles and PC parameters into one array of shape
    # `nav_shape` + (6,)
    rot_pc = da.dstack((rot, pc))

    map_blocks_kwargs = dict(
        drop_axis=(patterns.ndim - 1,), new_axis=(len(nav_shape),), dtype=np.float64
    )

    # Prepare bound constraints
    trust_region = np.asarray(trust_region)
    trust_region[:3] = np.deg2rad(trust_region[:3])
    lower_bounds = da.fmax(rot_pc - trust_region, 3 * [0] + 3 * [-3])
    upper_bounds = da.fmax(
        rot_pc + trust_region, [2 * np.pi, np.pi, 2 * np.pi] + 3 * [3]
    )

    # Set up NLopt optimizer if requested
    if pkg == "nlopt":
        chunk_func = _refine_orientation_pc_chunk_nlopt

        if initial_step is not None:
            initial_step = np.array(
                3
                * [
                    initial_step[0],
                ]
                + 3
                * [
                    initial_step[1],
                ]
            )
            initial_step[:3] = np.deg2rad(initial_step[:3])

        # Add extra keyword arguments passed to the chunk function
        opt = solver_kwargs.pop("method")
        map_blocks_kwargs["opt"] = _set_nlopt_parameters(
            opt, rtol=rtol, initial_step=initial_step, maxeval=maxeval
        )

        # Remove remaining keyword arguments not accepted in the NLopt
        # solver function
        for k in ["method_kwargs"]:
            del solver_kwargs[k]
    else:
        chunk_func = _refine_orientation_pc_chunk_scipy

    # Prepare parameters for the objective function which are constant
    # during optimization
    fixed_parameters += (signal_mask,)
    fixed_parameters += (nrows,)
    fixed_parameters += (ncols,)
    fixed_parameters += (detector.tilt,)
    fixed_parameters += (detector.azimuthal,)
    fixed_parameters += (detector.sample_tilt,)
    solver_kwargs["fixed_parameters"] = fixed_parameters

    output = da.map_blocks(
        chunk_func,
        patterns,
        rot_pc,
        lower_bounds,
        upper_bounds,
        signal_mask=signal_mask,
        solver_kwargs=solver_kwargs,
        **map_blocks_kwargs,
    )

    print(
        _refinement_info_message(
            package=pkg,
            method=method,
            method_kwargs=method_kwargs,
            trust_region=trust_region,
            initial_step=initial_step,
            rtol=rtol,
            maxeval=maxeval,
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


def _refine_orientation_pc_chunk_scipy(
    patterns: np.ndarray,
    rot_pc: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    signal_mask: Optional[np.ndarray] = None,
    solver_kwargs: Optional[dict] = None,
):
    """Refine orientations and projection centers using all patterns in
    one dask array chunk using *SciPy*.
    """
    nav_shape = patterns.shape[:-1]
    results = np.empty(nav_shape + (7,), dtype=np.float64)
    rot_pc = rot_pc.reshape(nav_shape + (6,))
    for idx in np.ndindex(*nav_shape):
        results[idx] = _refine_orientation_pc_solver_scipy(
            pattern=patterns[idx],
            rot_pc=rot_pc[idx],
            signal_mask=signal_mask,
            **solver_kwargs,
        )
    return results


def _refine_orientation_pc_chunk_nlopt(
    patterns: np.ndarray,
    rot_pc: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    signal_mask: Optional[np.ndarray] = None,
    solver_kwargs: Optional[dict] = None,
    opt: "nlopt.opt" = None,
):
    """Refine orientations and projection centers using all patterns in
    one dask array chunk using *NLopt*.
    """
    # Copy optimizer
    import nlopt

    opt = nlopt.opt(opt)

    nav_shape = patterns.shape[:-1]
    results = np.empty(nav_shape + (7,), dtype=np.float64)
    rot_pc = rot_pc.reshape(nav_shape + (6,))
    lower_bounds = lower_bounds.reshape(nav_shape + (6,))
    upper_bounds = upper_bounds.reshape(nav_shape + (6,))

    for idx in np.ndindex(*nav_shape):
        results[idx] = _refine_orientation_pc_solver_nlopt(
            opt,
            pattern=patterns[idx],
            rot_pc=rot_pc[idx],
            lower_bounds=lower_bounds[idx],
            upper_bounds=upper_bounds[idx],
            signal_mask=signal_mask,
            **solver_kwargs,
        )

    return results
