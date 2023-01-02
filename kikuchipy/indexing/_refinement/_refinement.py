# Copyright 2019-2023 The kikuchipy developers
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


# -------------------------- Refine orientation ---------------------- #


def _refine_orientation(
    xmap: CrystalMap,
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns: Union[np.ndarray, da.Array],
    signal_mask: np.ndarray,
    trust_region: Union[tuple, list, np.ndarray, None],
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

    # Get bounds on control variables. If a trust region is not passed,
    # these arrays are all 0, and not used in the objective functions.
    lower_bounds, upper_bounds = ref.get_bound_constraints(trust_region)
    ref.solver_kwargs["trust_region_passed"] = trust_region is not None

    if ref.unique_pc:
        # Patterns have been indexed with varying PCs, so we re-compute
        # the direction cosines for every pattern during refinement
        pc = ref.pc_array
        nav_slices = (slice(None, None),) * len(ref.nav_shape)
        pcx = pc[nav_slices + (slice(0, 1),)]
        pcy = pc[nav_slices + (slice(1, 2),)]
        pcz = pc[nav_slices + (slice(2, 3),)]

        res = da.map_blocks(
            ref.chunk_func,
            patterns,
            ref.rotations_array,
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
            ref.chunk_func,
            patterns,
            ref.rotations_array,
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

    # SciPy requires a sequence of (min, max) for each control variable
    lower_bounds = lower_bounds.reshape(nav_shape + (3,))
    upper_bounds = upper_bounds.reshape(nav_shape + (3,))
    bounds = np.stack((lower_bounds, upper_bounds), axis=lower_bounds.ndim)

    if direction_cosines is None:
        # Remove extra dimensions that were necessary for use of Dask's
        # map_blocks()
        pcx = pcx.reshape(nav_shape)
        pcy = pcy.reshape(nav_shape)
        pcz = pcz.reshape(nav_shape)
        for idx in np.ndindex(*nav_shape):
            results[idx] = _refine_orientation_solver_scipy(
                pattern=patterns[idx],
                rotation=rotations[idx],
                bounds=bounds[idx],
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
                bounds=bounds[idx],
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


def _refine_pc(
    xmap: CrystalMap,
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns: Union[np.ndarray, da.Array],
    signal_mask: np.ndarray,
    trust_region: Union[tuple, list, np.ndarray, None],
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

    # Get bounds on control variables. If a trust region is not passed,
    # these arrays are all 0, and not used in the objective functions.
    lower_bounds, upper_bounds = ref.get_bound_constraints(trust_region)
    ref.solver_kwargs["trust_region_passed"] = trust_region is not None

    res = da.map_blocks(
        ref.chunk_func,
        patterns,
        ref.rotations_array,
        ref.pc_array,
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

    # SciPy requires a sequence of (min, max) for each control variable
    lower_bounds = lower_bounds.reshape(nav_shape + (3,))
    upper_bounds = upper_bounds.reshape(nav_shape + (3,))
    bounds = np.stack((lower_bounds, upper_bounds), axis=lower_bounds.ndim)

    for idx in np.ndindex(*nav_shape):
        results[idx] = _refine_pc_solver_scipy(
            pattern=patterns[idx],
            rotation=rotations[idx],
            pc=pc[idx],
            bounds=bounds[idx],
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

    # Get bounds on control variables. If a trust region is not passed,
    # these arrays are all 0, and not used in the objective functions.
    lower_bounds, upper_bounds = ref.get_bound_constraints(trust_region)
    ref.solver_kwargs["trust_region_passed"] = trust_region is not None

    res = da.map_blocks(
        ref.chunk_func,
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

    # SciPy requires a sequence of (min, max) for each control variable
    lower_bounds = lower_bounds.reshape(nav_shape + (6,))
    upper_bounds = upper_bounds.reshape(nav_shape + (6,))
    bounds = np.stack((lower_bounds, upper_bounds), axis=lower_bounds.ndim)

    for idx in np.ndindex(*nav_shape):
        results[idx] = _refine_orientation_pc_solver_scipy(
            pattern=patterns[idx],
            rot_pc=rot_pc[idx],
            bounds=bounds[idx],
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


# ------------------------- Refinement setup ------------------------- #


class _RefinementSetup:
    """Set up EBSD refinement.

    Parameters
    ----------
    mode
        Either ``"ori"``, ``"pc"`` or ``"ori_pc"``.
    xmap
        See docstring of e.g.
        :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
    detector
        See docstring of e.g.
        :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
    master_pattern
        See docstring of e.g.
        :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
    energy
        See docstring of e.g.
        :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
    patterns
        EBSD patterns in a 2D array with one navigation dimension and
        one signal dimension.
    rtol
        See docstring of e.g.
        :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
    method
        See docstring of e.g.
        :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
    method_kwargs
        See docstring of e.g.
        :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
    initial_step
        See docstring of e.g.
        :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
    maxeval
        See docstring of e.g.
        :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
    signal_mask
        See docstring of e.g.
        :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
    """

    mode: str
    # Data parameters
    data_shape: tuple
    nav_shape: tuple
    nav_size: int
    rotations_array: da.Array
    rotations_pc_array: Optional[da.Array] = None
    # Optimization parameters
    initial_step: Optional[list] = None
    maxeval: Optional[int] = None
    method_name: str
    optimization_type: str
    package: str
    supports_bounds: bool = False
    # Fixed parameters
    fixed_parameters: tuple
    # Arguments to pass to Dask and to solver functions
    chunk_func: Callable
    map_blocks_kwargs: dict = {}
    solver_kwargs: dict = {}

    def __init__(
        self,
        mode: str,
        xmap: CrystalMap,
        detector: "EBSDDetector",
        master_pattern: "EBSDMasterPattern",
        energy: Union[int, float],
        patterns: Union[np.ndarray, da.Array],
        rtol: float,
        method: str,
        method_kwargs: Optional[dict] = None,
        initial_step: Optional[float] = None,
        maxeval: Optional[int] = None,
        signal_mask: Optional[np.ndarray] = None,
    ):
        """Set up EBSD refinement."""
        self.mode = mode

        self.set_optimization_parameters(
            rtol=rtol,
            method=method,
            method_kwargs=method_kwargs,
            initial_step=initial_step,
            maxeval=maxeval,
        )

        self.set_fixed_parameters(
            master_pattern=master_pattern,
            energy=energy,
            signal_mask=signal_mask,
            detector=detector,
        )
        self.solver_kwargs["fixed_parameters"] = self.fixed_parameters

        # Relevant information from pattern array
        self.solver_kwargs["rescale"] = patterns.dtype == np.float32
        self.chunks = patterns.chunksize[:-1] + (-1,)

        self.nav_shape = xmap.shape
        self.nav_size = xmap.size

        # Relevant data from the crystal map
        if xmap.rotations_per_point > 1:
            rot = xmap.rotations[:, 0]
        else:
            rot = xmap.rotations
        if self.mode == "pc":
            rot = rot.data
            rot = rot.reshape(self.nav_shape + (4,))
        else:
            rot = rot.to_euler()
            rot = rot.reshape(self.nav_shape + (3,))
        self.rotations_array = da.from_array(rot, chunks=self.chunks)

        # Relevant data from the detector
        self.unique_pc = np.prod(detector.navigation_shape) != 1 and self.nav_size > 1
        dtype = np.float64
        pc_shape = self.nav_shape + (3,)
        if self.unique_pc:
            # Patterns have been initially indexed with varying PCs, so
            # we use these as the starting point for every pattern
            pc = detector.pc.astype(dtype).reshape(pc_shape)
        else:
            # Patterns have been initially indexed with the same PC, so
            # we use this as the starting point for every pattern
            pc = np.full(pc_shape, detector.pc[0], dtype=dtype)
        self.pc_array = da.from_array(pc, chunks=self.chunks)

        if mode == "ori_pc":
            self.rotations_pc_array = da.dstack((self.rotations_array, self.pc_array))

        # Keyword arguments passed to Dask when iterating over chunks
        self.map_blocks_kwargs.update(
            {
                "drop_axis": (patterns.ndim - 1,),
                "new_axis": (len(self.nav_shape),),
                "dtype": np.float64,
            }
        )

        # Have no idea how this can happen, but it does in tests...
        if self.package == "scipy" and "opt" in self.map_blocks_kwargs:
            del self.map_blocks_kwargs["opt"]

        chunk_funcs = {
            "ori": {
                "nlopt": _refine_orientation_chunk_nlopt,
                "scipy": _refine_orientation_chunk_scipy,
            },
            "pc": {
                "nlopt": _refine_pc_chunk_nlopt,
                "scipy": _refine_pc_chunk_scipy,
            },
            "ori_pc": {
                "nlopt": _refine_orientation_pc_chunk_nlopt,
                "scipy": _refine_orientation_pc_chunk_scipy,
            },
        }
        self.chunk_func = chunk_funcs[self.mode][self.package]

    @property
    def n_control_variables(self) -> int:
        return {"ori": 3, "pc": 3, "ori_pc": 6}[self.mode]

    def set_optimization_parameters(
        self,
        rtol: float,
        method: str,
        method_kwargs: Optional[dict] = None,
        initial_step: Union[float, int, Tuple[float, float], None] = None,
        maxeval: Optional[int] = None,
    ) -> None:
        """Set *NLopt* or *SciPy* optimization parameters.

        Parameters
        ----------
        rtol
            See docstring of e.g.
            :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
        method
            See docstring of e.g.
            :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
        method_kwargs
            See docstring of e.g.
            :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
        initial_step
            See docstring of e.g.
            :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
        maxeval
            See docstring of e.g.
            :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
        """
        method = method.lower()
        supported_methods = list(SUPPORTED_OPTIMIZATION_METHODS)

        if method not in supported_methods:
            raise ValueError(
                f"Method {method} not in the list of supported methods "
                f"{supported_methods}"
            )

        method_dict = SUPPORTED_OPTIMIZATION_METHODS[method]
        self.optimization_type = method_dict["type"]
        self.supports_bounds = method_dict["supports_bounds"]
        self.package = method_dict["package"]

        if self.package == "nlopt":
            from kikuchipy import _nlopt_installed

            method_upper = method.upper()
            if not _nlopt_installed:
                raise ImportError(
                    f"Package `nlopt`, required for method {method_upper}, is not "
                    "installed"
                )

            import nlopt

            opt = nlopt.opt(method_upper, self.n_control_variables)
            opt.set_ftol_rel(rtol)

            if initial_step is not None:
                initial_step = np.atleast_1d(initial_step)
                n_initial_steps = {"ori": 1, "pc": 1, "ori_pc": 2}
                if initial_step.size != n_initial_steps[self.mode]:
                    raise ValueError(
                        "`initial_step` must be a single number when refining "
                        "orientations or PCs and a list of two numbers when refining "
                        "both"
                    )
                initial_step = np.repeat(initial_step, 3)
                opt.set_initial_step(initial_step)
                self.initial_step = list(initial_step)

            if maxeval is not None:
                opt.set_maxeval(maxeval)
                self.maxeval = maxeval

            self.method_name = method_upper
            self.map_blocks_kwargs["opt"] = opt
        else:
            if method_kwargs is None:
                method_kwargs = {}

            if method == "minimize" and "method" not in method_kwargs:
                self.method_name = "Nelder-Mead"
                method_kwargs["method"] = self.method_name
            elif "method" in method_kwargs:
                self.method_name = method_kwargs["method"]
            else:
                self.method_name = method

            if method == "basinhopping" and "minimizer_kwargs" not in method_kwargs:
                method_kwargs["minimizer_kwargs"] = {}

            method = getattr(scipy.optimize, method)
            self.solver_kwargs = {"method": method, "method_kwargs": method_kwargs}

    def set_fixed_parameters(
        self,
        detector: "EBSDDetector",
        master_pattern: "EBSDMasterPattern",
        energy: Union[int, float],
        signal_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Set fixed parameters to pass to the objective function.

        Parameters
        ----------
        detector
            See docstring of e.g.
            :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
        master_pattern
            See docstring of e.g.
            :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
        energy
            See docstring of e.g.
            :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
        signal_mask
            See docstring of e.g.
            :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
        """
        params = _get_master_pattern_data(master_pattern, energy)

        if self.mode in ["pc", "ori_pc"]:
            nrows, ncols = detector.shape
            params += (
                signal_mask,
                nrows,
                ncols,
                detector.tilt,
                detector.azimuthal,
                detector.sample_tilt,
            )

        self.fixed_parameters = params

    def get_bound_constraints(
        self, trust_region: Union[tuple, list, np.ndarray, None]
    ) -> Tuple[da.Array, da.Array]:
        """Return the bound constraints on the control variables.

        Parameters
        ----------
        trust_region
            See docstring of e.g.
            :meth:`~kikuchipy.signals.EBSD.refine_orientation`.

        Returns
        -------
        lower_bounds
            Array of lower bounds for each control variable. If
            ``trust_region`` is ``None``, it is filled with zeros and is
            only meant to be iterated over in the chunking function, but
            not meant to be used.
        upper_bounds
            Array of upper bounds for each control variable. If
            ``trust_region`` is ``None``, it is filled with zeros and is
            only meant to be iterated over in the chunking function, but
            not meant to be used.
        """
        if trust_region is None:
            if self.mode == "ori":
                chunks = self.rotations_array.chunks
            elif self.mode == "pc":
                chunks = self.pc_array.chunks
            else:
                chunks = self.rotations_pc_array.chunks
            shape = self.nav_shape + (self.n_control_variables,)
            lower_bounds = da.zeros(shape, dtype="uint8", chunks=chunks)
            upper_bounds = lower_bounds.copy()
            return lower_bounds, upper_bounds

        eu_lower = 3 * [0]
        eu_upper = [2 * np.pi, np.pi, 2 * np.pi]
        pc_lower = 3 * [-2]
        pc_upper = 3 * [2]

        trust_region = np.asarray(trust_region)

        if self.mode == "ori":
            trust_region = np.deg2rad(trust_region)
            lower_abs = eu_lower
            upper_abs = eu_upper
            data_to_optimize = self.rotations_array
        elif self.mode == "pc":
            lower_abs = pc_lower
            upper_abs = pc_upper
            data_to_optimize = self.pc_array
        else:  # "ori_pc"
            trust_region[:3] = np.deg2rad(trust_region[:3])
            lower_abs = eu_lower + pc_lower
            upper_abs = eu_upper + pc_upper
            data_to_optimize = self.rotations_pc_array

        lower_bounds = da.fmax(data_to_optimize - trust_region, lower_abs)
        upper_bounds = da.fmin(data_to_optimize + trust_region, upper_abs)

        return lower_bounds, upper_bounds

    def get_info_message(self, trust_region) -> str:
        """Return a string with important refinement information to
        display to the user.

        Parameters
        ----------
        trust_region
            See docstring of e.g.
            :meth:`~kikuchipy.signals.EBSD.refine_orientation`.

        Returns
        -------
        info
            Important refinement impormation.
        """
        package_name = {"scipy": "SciPy", "nlopt": "NLopt"}[self.package]
        info = (
            "Refinement information:\n"
            f"  Method: {self.method_name} ({self.optimization_type}) from {package_name}"
        )

        if self.supports_bounds:
            tr_str = np.array_str(np.asarray(trust_region), precision=5)
            info += "\n  Trust region (+/-): " + tr_str

        if self.package == "scipy":
            info += f"\n  Keyword arguments passed to method: {self.solver_kwargs['method_kwargs']}"
        else:
            opt = self.map_blocks_kwargs["opt"]
            info += f"\n  Relative tolerance: {opt.get_ftol_rel()}"
            if self.initial_step:
                info += f"\n  Initial step(s): {self.initial_step}"
            if self.maxeval:
                info += f"\n  Max. function evaulations: {self.maxeval}"

        return info


def _get_master_pattern_data(
    master_pattern: "EBSDMasterPattern", energy: Union[int, float]
) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    """Return the upper and lower hemispheres along with their shape.

    Parameters
    ----------
    master_pattern
        See docstring of e.g.
        :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
    energy
        See docstring of e.g.
        :meth:`~kikuchipy.signals.EBSD.refine_orientation`.

    Returns
    -------
    mpu, mpl
        Upper and lower hemispheres of master pattern of data type
        32-bit float.
    npx, npy
        Number of columns and rows of the master pattern.
    scale
        Factor to scale up from the square Lambert projection to the
        master pattern.
    """
    mpu, mpl = master_pattern._get_master_pattern_arrays_from_energy(energy=energy)
    npx, npy = master_pattern.axes_manager.signal_shape
    scale = (npx - 1) / 2
    dtype_desired = np.float32
    if mpu.dtype != dtype_desired:
        mpu = rescale_intensity(mpu, dtype_out=dtype_desired)
        mpl = rescale_intensity(mpl, dtype_out=dtype_desired)
    return mpu, mpl, npx, npy, scale
