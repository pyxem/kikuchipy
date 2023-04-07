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
from orix.crystal_map import create_coordinate_arrays, CrystalMap, Phase, PhaseList
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
from kikuchipy.signals.util._crystal_map import _get_indexed_points_in_data_in_xmap
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_from_detector,
)


def compute_refine_orientation_results(
    results: da.Array,
    xmap: CrystalMap,
    master_pattern: "EBSDMasterPattern",
    navigation_mask: Optional[np.ndarray] = None,
    pseudo_symmetry_checked: bool = False,
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
    navigation_mask
        Navigation mask passed to ``refine_orientation()`` to obtain
        ``results``. If not given, it is assumed that it was not given
        to ``refine_orientation()`` either.
    pseudo_symmetry_checked
        Whether pseudo-symmetry operators were passed to
        ``refine_orientation()``. Default is ``False``.

    Returns
    -------
    xmap_refined
        Crystal map with refined orientations, scores, the number of
        function evaluations and the pseudo-symmetry index if
        ``pseudo_symmetry_checked=True``. See the docstring of
        ``refine_orientation()`` for details.
    """
    points_to_refine, is_in_data, phase_id, _ = _get_indexed_points_in_data_in_xmap(
        xmap, navigation_mask
    )

    nav_size = points_to_refine.size
    nav_size_in_data = points_to_refine.sum()

    xmap_kw = _get_crystal_map_parameters(
        xmap, nav_size, master_pattern.phase, phase_id, pseudo_symmetry_checked
    )
    xmap_kw["phase_id"][points_to_refine] = phase_id

    is_indexed = np.zeros_like(is_in_data)
    is_indexed[xmap.is_in_data] = xmap.is_indexed
    xmap_kw["phase_id"][~is_indexed] = -1

    print(f"Refining {nav_size_in_data} orientation(s):", file=sys.stdout)
    time_start = time()
    with ProgressBar():
        res = results.compute()
    total_time = time() - time_start
    patterns_per_second = nav_size_in_data / total_time
    print(f"Refinement speed: {patterns_per_second:.5f} patterns/s", file=sys.stdout)

    # Extract data: n x (score, number of evaluations, phi1, Phi, phi2,
    # [pseudo-symmetry index])
    res = np.array(res)
    xmap_kw["prop"]["scores"][points_to_refine] = res[:, 0]
    xmap_kw["prop"]["num_evals"][points_to_refine] = res[:, 1]
    xmap_kw["rotations"][points_to_refine] = Rotation.from_euler(res[:, 2:5]).data
    if pseudo_symmetry_checked:
        xmap_kw["prop"]["pseudo_symmetry_index"][points_to_refine] = res[:, 5]

    xmap_refined = CrystalMap(is_in_data=is_in_data, **xmap_kw)

    return xmap_refined


def compute_refine_projection_center_results(
    results: da.Array,
    detector: "EBSDDetector",
    xmap: CrystalMap,
    navigation_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, "EBSDDetector", np.ndarray]:
    """Compute the results from
    :meth:`~kikuchipy.signals.EBSD.refine_projection_center` and return
    the score array, :class:`~kikuchipy.detectors.EBSDDetector` and
    number of function evaluations per pattern.

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
    navigation_mask
        Navigation mask passed to ``refine_projection_center()`` to
        obtain ``results``. If not given, it is assumed that it was not
        given to ``refine_projection_center()`` either.

    Returns
    -------
    new_scores
        Score array.
    new_detector
        EBSD detector with refined projection center parameters.
    num_evals
        Number of function evaluations per pattern.
    """
    (points_to_refine, *_, mask_shape) = _get_indexed_points_in_data_in_xmap(
        xmap, navigation_mask
    )
    nav_size_in_data = points_to_refine.sum()

    new_detector = detector.deepcopy()

    print(f"Refining {nav_size_in_data} projection center(s):", file=sys.stdout)
    time_start = time()
    with ProgressBar():
        res = results.compute()
    total_time = time() - time_start
    patterns_per_second = nav_size_in_data / total_time
    print(f"Refinement speed: {patterns_per_second:.5f} patterns/s", file=sys.stdout)

    # Extract data: n x (score, number of evaluations, PCx, PCy, PCz)
    res = np.array(res)
    scores = res[:, 0]
    num_evals = res[:, 1].astype(np.int32)
    new_pc = res[:, 2:]

    if mask_shape is not None:
        scores = scores.reshape(mask_shape)
        num_evals = num_evals.reshape(mask_shape)
        new_pc = new_pc.reshape(mask_shape + (3,))

    new_detector.pc = new_pc

    return scores, new_detector, num_evals


def compute_refine_orientation_projection_center_results(
    results: da.Array,
    detector: "EBSDDetector",
    xmap: CrystalMap,
    master_pattern: "EBSDMasterPattern",
    navigation_mask: Optional[np.ndarray] = None,
    pseudo_symmetry_checked: bool = False,
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
    navigation_mask
        Navigation mask passed to
        ``refine_orientation_projection_center()`` to obtain
        ``results``. If not given, it is assumed that it was not given
        to ``refine_orientation_projection_center()`` either.
    pseudo_symmetry_checked
        Whether pseudo-symmetry operators were passed to
        ``refine_orientation_projection_center()``. Default is
        ``False``.

    Returns
    -------
    xmap_refined
        Crystal map with refined orientations, scores, the number of
        function evaluations and the pseudo-symmetry index if
        ``pseudo_symmetry_checked=True``. See the docstring of
        ``refine_orientation_projection_center()`` for details.
    new_detector
        EBSD detector with refined projection center parameters.

    See Also
    --------
    kikuchipy.signals.EBSD.refine_orientation_projection_center
    """
    (
        points_to_refine,
        is_in_data,
        phase_id,
        mask_shape,
    ) = _get_indexed_points_in_data_in_xmap(xmap, navigation_mask)

    nav_size = points_to_refine.size
    nav_size_in_data = points_to_refine.sum()

    xmap_kw = _get_crystal_map_parameters(
        xmap, nav_size, master_pattern.phase, phase_id, pseudo_symmetry_checked
    )
    xmap_kw["phase_id"][points_to_refine] = phase_id

    is_indexed = np.zeros_like(is_in_data)
    is_indexed[xmap.is_in_data] = xmap.is_indexed
    xmap_kw["phase_id"][~is_indexed] = -1

    new_detector = detector.deepcopy()

    print(
        f"Refining {nav_size_in_data} orientation(s) and projection center(s):",
        file=sys.stdout,
    )
    time_start = time()
    with ProgressBar():
        res = results.compute()
    total_time = time() - time_start
    patterns_per_second = nav_size_in_data / total_time
    print(f"Refinement speed: {patterns_per_second:.5f} patterns/s", file=sys.stdout)

    # Extract data: n x (score, number of evaluations, phi1, Phi, phi2,
    # PCx, PCy, PCz, [pseudo-symmetry index])
    res = np.array(res)
    xmap_kw["prop"]["scores"][points_to_refine] = res[:, 0]
    xmap_kw["prop"]["num_evals"][points_to_refine] = res[:, 1]
    xmap_kw["rotations"][points_to_refine] = Rotation.from_euler(res[:, 2:5]).data
    if pseudo_symmetry_checked:
        xmap_kw["prop"]["pseudo_symmetry_index"][points_to_refine] = res[:, 8]

    xmap_refined = CrystalMap(is_in_data=is_in_data, **xmap_kw)

    new_pc = res[:, 5:8]
    if mask_shape is not None:
        new_pc = new_pc.reshape(mask_shape + (3,))
    new_detector.pc = new_pc

    return xmap_refined, new_detector


def _get_crystal_map_parameters(
    xmap: CrystalMap,
    nav_size: int,
    master_pattern_phase: Phase,
    phase_id: int,
    pseudo_symmetry_checked: bool = False,
) -> dict:
    step_sizes = ()
    for step_size in [xmap.dy, xmap.dx]:
        if step_size != 0:
            step_sizes += (step_size,)

    xmap_dict, _ = create_coordinate_arrays(xmap.shape, step_sizes=step_sizes)
    xmap_dict.update(
        {
            "rotations": Rotation.identity((nav_size,)),
            "phase_id": np.zeros(nav_size, dtype=np.int32),
            "phase_list": PhaseList(phases=master_pattern_phase, ids=phase_id),
            "scan_unit": xmap.scan_unit,
            "prop": {
                "scores": np.zeros(nav_size, dtype=np.float64),
                "num_evals": np.zeros(nav_size, dtype=np.int32),
            },
        }
    )

    if pseudo_symmetry_checked:
        xmap_dict["prop"]["pseudo_symmetry_index"] = np.zeros(nav_size, dtype=np.int32)

    if "not_indexed" in xmap.phases.names:
        xmap_dict["phase_list"].add_not_indexed()

    return xmap_dict


# -------------------------- Refine orientation ---------------------- #


def _refine_orientation(
    xmap: CrystalMap,
    detector,
    master_pattern,
    energy: Union[int, float],
    patterns: Union[np.ndarray, da.Array],
    points_to_refine: np.ndarray,
    signal_mask: np.ndarray,
    trust_region: Union[tuple, list, np.ndarray, None],
    rtol: float,
    pseudo_symmetry_ops: Optional[Rotation] = None,
    method: Optional[str] = None,
    method_kwargs: Optional[dict] = None,
    initial_step: Optional[float] = None,
    maxeval: Optional[int] = None,
    compute: bool = True,
    navigation_mask: Optional[np.ndarray] = None,
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
        points_to_refine=points_to_refine,
        signal_mask=signal_mask,
        pseudo_symmetry_ops=pseudo_symmetry_ops,
    )

    # Get bounds on control variables. If a trust region is not passed,
    # these arrays are all 0, and not used in the objective functions.
    lower_bounds, upper_bounds = ref.get_bound_constraints(trust_region)
    ref.solver_kwargs["trust_region_passed"] = trust_region is not None

    if ref.unique_pc:
        # Patterns have been indexed with varying PCs, so we re-compute
        # the direction cosines for every pattern during refinement.
        # Since we're iterating over (n patterns, x parameters, 1) in
        # each Dask array, we need the PC arrays to stay 3D (hence the
        # 'weird' slicing).
        pc = ref.pc_array
        pcx = pc[:, :, 0:1]
        pcy = pc[:, :, 1:2]
        pcz = pc[:, :, 2:3]

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
            n_pseudo_symmetry_ops=ref.n_pseudo_symmetry_ops,
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
            n_pseudo_symmetry_ops=ref.n_pseudo_symmetry_ops,
            **ref.map_blocks_kwargs,
        )

    msg = ref.get_info_message(trust_region)
    print(msg)

    if compute:
        res = compute_refine_orientation_results(
            results=res,
            xmap=xmap,
            master_pattern=master_pattern,
            navigation_mask=navigation_mask,
            pseudo_symmetry_checked=pseudo_symmetry_ops is not None,
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
    n_pseudo_symmetry_ops: int = 0,
):
    """Refine orientations from patterns in one dask array chunk using
    *SciPy*.

    Note that ``signal_mask`` and ``solver_kwargs`` are required. They
    are set to ``None`` to enable use of this function in
    :func:`~dask.array.Array.map_blocks`.
    """
    nav_size = patterns.shape[0]
    value_size = 5
    if n_pseudo_symmetry_ops > 0:
        value_size += 1
    results = np.empty((nav_size, value_size), dtype=np.float64)

    # SciPy requires a sequence of (min, max) for each control variable
    bounds = np.stack((lower_bounds, upper_bounds), axis=lower_bounds.ndim)

    if direction_cosines is None:
        for i in range(nav_size):
            results[i] = _refine_orientation_solver_scipy(
                pattern=patterns[i, 0],
                rotation=rotations[i],
                bounds=bounds[i],
                pcx=float(pcx[i, 0, 0]),
                pcy=float(pcy[i, 0, 0]),
                pcz=float(pcz[i, 0, 0]),
                nrows=nrows,
                ncols=ncols,
                tilt=tilt,
                azimuthal=azimuthal,
                sample_tilt=sample_tilt,
                signal_mask=signal_mask,
                n_pseudo_symmetry_ops=n_pseudo_symmetry_ops,
                **solver_kwargs,
            )
    else:
        for i in range(nav_size):
            results[i] = _refine_orientation_solver_scipy(
                pattern=patterns[i, 0],
                rotation=rotations[i],
                bounds=bounds[i],
                direction_cosines=direction_cosines,
                signal_mask=signal_mask,
                n_pseudo_symmetry_ops=n_pseudo_symmetry_ops,
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
    n_pseudo_symmetry_ops: int = 0,
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

    nav_size = patterns.shape[0]
    value_size = 5
    if n_pseudo_symmetry_ops > 0:
        value_size += 1
    results = np.empty((nav_size, value_size), dtype=np.float64)

    if direction_cosines is None:
        for i in range(nav_size):
            results[i] = _refine_orientation_solver_nlopt(
                opt=opt,
                pattern=patterns[i, 0],
                rotation=rotations[i],
                lower_bounds=lower_bounds[i],
                upper_bounds=upper_bounds[i],
                signal_mask=signal_mask,
                pcx=float(pcx[i, 0, 0]),
                pcy=float(pcy[i, 0, 0]),
                pcz=float(pcz[i, 0, 0]),
                nrows=nrows,
                ncols=ncols,
                tilt=tilt,
                azimuthal=azimuthal,
                sample_tilt=sample_tilt,
                n_pseudo_symmetry_ops=n_pseudo_symmetry_ops,
                **solver_kwargs,
            )
    else:
        for i in range(nav_size):
            results[i] = _refine_orientation_solver_nlopt(
                opt=opt,
                pattern=patterns[i, 0],
                rotation=rotations[i],
                lower_bounds=lower_bounds[i],
                upper_bounds=upper_bounds[i],
                signal_mask=signal_mask,
                direction_cosines=direction_cosines,
                n_pseudo_symmetry_ops=n_pseudo_symmetry_ops,
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
    points_to_refine: np.ndarray,
    signal_mask: np.ndarray,
    trust_region: Union[tuple, list, np.ndarray, None],
    rtol: float,
    rotations_ps: Optional[Rotation] = None,
    method: Optional[str] = None,
    method_kwargs: Optional[dict] = None,
    initial_step: Optional[float] = None,
    maxeval: Optional[int] = None,
    compute: bool = True,
    navigation_mask: Optional[np.ndarray] = None,
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
        points_to_refine=points_to_refine,
        signal_mask=signal_mask,
        pseudo_symmetry_ops=rotations_ps,
    )

    # Get bounds on control variables. If a trust region is not passed,
    # these arrays are all 0, and not used in the objective functions.
    lower_bounds, upper_bounds = ref.get_bound_constraints(trust_region)
    ref.solver_kwargs["trust_region_passed"] = trust_region is not None

    res = da.map_blocks(
        ref.chunk_func,
        patterns[:, 0, :],
        ref.rotations_array[:, 0, :],
        ref.pc_array[:, 0, :],
        lower_bounds[:, 0, :],
        upper_bounds[:, 0, :],
        solver_kwargs=ref.solver_kwargs,
        **ref.map_blocks_kwargs,
    )

    msg = ref.get_info_message(trust_region)
    print(msg)

    if compute:
        res = compute_refine_projection_center_results(
            results=res,
            detector=detector,
            xmap=xmap,
            navigation_mask=navigation_mask,
        )

    return res


def _refine_pc_chunk_scipy(
    patterns: np.ndarray,
    rotations: np.ndarray,
    pc: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    solver_kwargs: dict,
):
    """Refine projection centers using patterns in one dask array chunk."""

    nav_size = patterns.shape[0]
    results = np.empty((nav_size, 5), dtype=np.float64)

    # SciPy requires a sequence of (min, max) for each control variable
    bounds = np.stack((lower_bounds, upper_bounds), axis=lower_bounds.ndim)

    for i in range(nav_size):
        results[i] = _refine_pc_solver_scipy(
            pattern=patterns[i],
            rotation=rotations[i],
            pc=pc[i],
            bounds=bounds[i],
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
    opt: "nlopt.opt" = None,
):
    """Refine projection centers using patterns in one dask array chunk."""
    # Copy optimizer
    import nlopt

    opt = nlopt.opt(opt)

    nav_size = patterns.shape[0]
    results = np.empty((nav_size, 5), dtype=np.float64)

    for i in range(nav_size):
        results[i] = _refine_pc_solver_nlopt(
            opt=opt,
            pattern=patterns[i],
            pc=pc[i],
            rotation=rotations[i],
            lower_bounds=lower_bounds[i],
            upper_bounds=upper_bounds[i],
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
    points_to_refine: np.ndarray,
    signal_mask: np.ndarray,
    trust_region: Union[tuple, list, np.ndarray, None],
    rtol: float,
    pseudo_symmetry_ops: Optional[Rotation] = None,
    method: Optional[str] = None,
    method_kwargs: Optional[dict] = None,
    initial_step: Union[tuple, list, np.ndarray, None] = None,
    maxeval: Optional[int] = None,
    compute: bool = True,
    navigation_mask: Optional[np.ndarray] = None,
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
        points_to_refine=points_to_refine,
        rtol=rtol,
        method=method,
        method_kwargs=method_kwargs,
        initial_step=initial_step,
        maxeval=maxeval,
        signal_mask=signal_mask,
        pseudo_symmetry_ops=pseudo_symmetry_ops,
    )

    # Stack Euler angles and PC parameters into one array of shape:
    # navigation shape + (6,)
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
        solver_kwargs=ref.solver_kwargs,
        n_pseudo_symmetry_ops=ref.n_pseudo_symmetry_ops,
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
            navigation_mask=navigation_mask,
            pseudo_symmetry_checked=pseudo_symmetry_ops is not None,
        )

    return res


def _refine_orientation_pc_chunk_scipy(
    patterns: np.ndarray,
    rot_pc: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    solver_kwargs: Optional[dict] = None,
    n_pseudo_symmetry_ops: int = 0,
):
    """Refine orientations and projection centers using all patterns in
    one dask array chunk using *SciPy*.
    """
    nav_size = patterns.shape[0]
    value_size = 8
    if n_pseudo_symmetry_ops > 0:
        value_size += 1
    results = np.empty((nav_size, value_size), dtype=np.float64)

    # SciPy requires a sequence of (min, max) for each control variable
    bounds = np.stack((lower_bounds, upper_bounds), axis=lower_bounds.ndim)

    for i in range(nav_size):
        results[i] = _refine_orientation_pc_solver_scipy(
            pattern=patterns[i, 0],
            rot_pc=rot_pc[i],
            bounds=bounds[i],
            n_pseudo_symmetry_ops=n_pseudo_symmetry_ops,
            **solver_kwargs,
        )

    return results


def _refine_orientation_pc_chunk_nlopt(
    patterns: np.ndarray,
    rot_pc: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    solver_kwargs: Optional[dict] = None,
    opt: "nlopt.opt" = None,
    n_pseudo_symmetry_ops: int = 0,
):
    """Refine orientations and projection centers using all patterns in
    one dask array chunk using *NLopt*.
    """
    # Copy optimizer
    import nlopt

    opt = nlopt.opt(opt)

    nav_size = patterns.shape[0]
    value_size = 8
    if n_pseudo_symmetry_ops > 0:
        value_size += 1
    results = np.empty((nav_size, value_size), dtype=np.float64)

    for i in range(nav_size):
        results[i] = _refine_orientation_pc_solver_nlopt(
            opt,
            pattern=patterns[i, 0],
            rot_pc=rot_pc[i],
            lower_bounds=lower_bounds[i],
            upper_bounds=upper_bounds[i],
            n_pseudo_symmetry_ops=n_pseudo_symmetry_ops,
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
    points_to_refine
        A 1D boolean array with points in the crystal map to refine. The
        number of ``True`` values is equal to the length of the
        pattern's navigation dimension.
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
        A 1D boolean array of equal size as the pattern signal
        dimension, with values equal to ``True`` to use in refinement.
    pseudo_symmetry_ops
        See docstring of e.g.
        :meth:`~kikuchipy.signals.EBSD.refine_orientation`.
    """

    mode: str
    # Data parameters
    data_shape: tuple
    nav_size: int
    n_pseudo_symmetry_ops: int = 0
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
        points_to_refine: np.ndarray,
        rtol: float,
        method: str,
        method_kwargs: Optional[dict] = None,
        initial_step: Optional[float] = None,
        maxeval: Optional[int] = None,
        signal_mask: Optional[np.ndarray] = None,
        pseudo_symmetry_ops: Optional[Rotation] = None,
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
        self.solver_kwargs["rescale"] = patterns.dtype == np.float32

        # Chunks for navigation size, pseudo-symmetry operators and
        # variables (e.g. detector pixels, control variables etc.)
        self.chunks = (patterns.chunksize[0], -1, -1)

        # Relevant rotations, potentially after applying pseudo-symmetry
        # operators, as a Dask array of shape (navigation size,
        # 1 + n pseudo-symmetry operators, n variables)
        self.nav_size = points_to_refine.sum()
        points_to_refine_in_data = points_to_refine[xmap.is_in_data]
        if xmap.rotations_per_point > 1:
            rot = xmap.rotations[points_to_refine_in_data, 0]
        else:
            rot = xmap.rotations[points_to_refine_in_data]
        if pseudo_symmetry_ops is not None:
            self.n_pseudo_symmetry_ops = pseudo_symmetry_ops.size
            rot_ps_data = pseudo_symmetry_ops.flatten().outer(rot).data
            rot_ps_data = rot_ps_data.transpose((1, 0, 2))
            rot = Rotation(np.hstack((rot.data[:, np.newaxis], rot_ps_data)))
        if self.mode == "pc":
            rot_data = rot.data
        else:
            rot_data = rot.to_euler()

        # Relevant projection centers as a Dask array of shape
        # (navigation size, 1, n variables)
        self.unique_pc = detector.navigation_size > 1
        if self.unique_pc:
            # Patterns have been initially indexed with varying PCs, so
            # we use these as the starting point for every pattern
            pc = detector.pc_flattened[points_to_refine]
        else:
            # Patterns have been initially indexed with the same PC, so
            # we use this as the starting point for every pattern
            pc = np.full((int(points_to_refine.sum()), 3), detector.pc[0])
        pc = pc.astype(np.float64)
        pc = np.expand_dims(pc, 1)  # Pseudo-symmetry operator axis
        self.pc_array = da.from_array(pc, chunks=self.chunks)

        if pseudo_symmetry_ops is None:
            rot_data = np.expand_dims(rot_data, 1)
        else:
            self.pc_array = da.repeat(
                self.pc_array, self.n_pseudo_symmetry_ops + 1, axis=1
            )

        self.rotations_array = da.from_array(rot_data, chunks=self.chunks)

        if mode == "ori_pc":
            self.rotations_pc_array = da.concatenate(
                [self.rotations_array, self.pc_array], axis=2
            )

        # Keyword arguments passed to Dask when iterating over chunks.
        # The axes of pseudo-symmetry operators and control variables is
        # dropped by Dask.
        drop_axis = (1,)
        if mode != "pc":
            drop_axis += (2,)
        self.map_blocks_kwargs.update(
            {"drop_axis": drop_axis, "new_axis": (1,), "dtype": np.float64}
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

    @property
    def bounds_chunks(self) -> tuple:
        if self.mode == "ori":
            return self.rotations_array.chunks
        elif self.mode == "pc":
            return self.pc_array.chunks
        else:
            return self.rotations_pc_array.chunks

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
                f"Method '{method}' not in the list of supported methods "
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
                raise ImportError(  # pragma: no cover
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
                        "The initial step must be a single number when refining "
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
        upper_bounds, lower_bounds
            Arrays of upper and lower bounds for each control variable.
            If ``trust_region`` is not given, they are filled with zeros
            and are only meant to be iterated over in the chunking
            function, but not meant to be used.
        """
        if trust_region is None:
            shape = (
                self.nav_size,
                self.n_pseudo_symmetry_ops + 1,
                self.n_control_variables,
            )
            lower_bounds = da.zeros(shape, dtype="uint8", chunks=self.bounds_chunks)
            upper_bounds = lower_bounds.copy()
            return lower_bounds, upper_bounds

        # Absolute constraints for Euler angles and PCs. Note that
        # constraints on the angles need some leeway since NLopt can
        # throw a RuntimeError when a starting Euler angle is too close
        # to the bounds.
        angle_leeway = np.deg2rad(5)
        eu_lower = 3 * [-angle_leeway]
        eu_upper = [
            2 * np.pi + angle_leeway,
            np.pi + angle_leeway,
            2 * np.pi + angle_leeway,
        ]
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
            info += (
                "\n  Keyword arguments passed to method: "
                f"{self.solver_kwargs['method_kwargs']}"
            )
        else:
            opt = self.map_blocks_kwargs["opt"]
            info += f"\n  Relative tolerance: {opt.get_ftol_rel()}"
            if self.initial_step:
                info += f"\n  Initial step(s): {self.initial_step}"
            if self.maxeval:
                info += f"\n  Max. function evaulations: {self.maxeval}"

        if self.n_pseudo_symmetry_ops > 0:
            info += f"\n  No. pseudo-symmetry operators: {self.n_pseudo_symmetry_ops}"

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
