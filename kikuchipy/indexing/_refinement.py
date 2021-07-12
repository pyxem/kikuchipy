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


import sys
from typing import Optional, Union

import dask
from dask.diagnostics import ProgressBar
import numba
import numpy as np
from orix.crystal_map import CrystalMap
from orix.quaternion import Rotation
import scipy.optimize

from kikuchipy.pattern import rescale_intensity


def full_refinement(
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
    """Performs an orientation and projection center refinement
    using the best orientation, for each point, from the initial indexing
    results stored in a single phase
    :class:`~orix.crystal_map.CrystalMap` and the projection center
    estimates stored in an
    :class:`~kikuchipy.detectors.EBSDDetector`.

    Parameters
    ----------
    xmap : CrystalMap
        A crystal map storing the results of the initial EBSD
        indexing.
    master_pattern : EBSDMasterPattern
        EBSDMasterPattern in the square Lambert projection.
    signal : EBSD
        Experimental EBSD data.
    detector : EBSDDetecor
        EBSD detector describing the detector dimensions and the
        detector-sample geometry with either a single, fixed
        projection/pattern center or a projection center for each
        scan point.
    energy : int
        Acceleration voltage, in kV, used to simulate the desired
        master pattern.
    mask : np.ndarray, optional
        Boolean mask to be applied to the simulated patterns. True values are
        masked away.
    method : str, optional
        Name of the scipy.optimize function to be used. Must be one
        of "minimize", "differential_evolution", "dual_annealing",
        or "basinhopping". If not specified, "minimize" is used.
    method_kwargs : dict, optional
        Keyword arguments passed to the scipy.optimize function
        specified above.
    trust_region : list, optional
        List of how wide the bounds, centered on the initial
        orientation indexing result and projection center,
        should be for (phi1, Phi, phi2) in degrees and
        (PCx, PCy, PCz) in the Bruker convention.
        Only used for methods that support bounds
        (excluding Powell). Defaults to [1, 1, 1, 0.05, 0.05, 0.05]
    compute : bool, optional
        Whether to return a computed result, by default True.
        For more information see :func:`~dask.array.Array.compute`.

    Returns
    -------
    CrystalMap, EBSDDetector
        A crystal map with the refined orientations and a new
        EBSD detector with refined projection centers.
    """
    method, method_kwargs = _get_method(method, method_kwargs)

    # Convert from Quaternions to Euler angles
    euler = Rotation.to_euler(xmap.rotations)

    # Extract best rotation from xmap if given more than 1
    if xmap.rotations_per_point > 2:
        euler = euler[:, 0, :]

    if not trust_region:
        trust_region = [
            0.0174532925,
            0.0174532925,
            0.0174532925,
            0.05,
            0.05,
            0.05,
        ]
    else:
        trust_region = np.deg2rad(trust_region[:3]).tolist() + trust_region[3:]

    if signal.data.dtype != np.float32:
        exp = signal.deepcopy()
        exp.rescale_intensity(dtype_out=np.float32)
    else:
        exp = signal

    exp_data = exp.data
    exp_shape = exp_data.shape

    pc = detector.pc

    # Set the PC equal across the scan if not given
    if len(pc) == 1:
        pc_val = pc[0]
        pc = np.full((exp_shape[0] * exp_shape[1], 3), pc_val)

    # 2D nav-dim
    if len(exp_shape) == 4:
        exp_data = exp_data.reshape(
            (exp_shape[0] * exp_shape[1], exp_shape[2], exp_shape[3])
        )
    elif len(exp_shape) == 2:  # 0D nav-dim
        exp_data = exp_data.reshape(((1,) + exp_data.shape))

    (
        master_north,
        master_south,
        npx,
        npy,
        scale,
    ) = _get_single_pattern_params(master_pattern, energy)

    theta_c = np.deg2rad(detector.tilt)
    sigma = np.deg2rad(detector.sample_tilt)
    alpha = (np.pi / 2) - sigma + theta_c

    detector_data = [detector.ncols, detector.nrows, detector.px_size, alpha]

    if mask is None:
        mask = 1
    else:
        mask = ~mask

    pre_args = (
        master_north,
        master_south,
        npx,
        npy,
        scale,
        detector_data,
        mask,
    )

    to_print = [
        "Refinement info:",
        f"method={method.__name__}",
        f"method kwargs={method_kwargs}",
    ]

    if method.__name__ not in ["minimize", "basinhopping"]:
        to_print.append(f"trust region={trust_region}")

    pre_args = dask.delayed(pre_args)
    trust_region = dask.delayed(trust_region)

    if isinstance(exp_data, dask.array.core.Array):
        patterns_in_chunk = exp_data.chunks[0]
        partitons = exp_data.to_delayed()  # List of delayed objects
        # equal to the number of chunks
        inner_index = 0
        refined_params = []
        for k, part in enumerate(partitons):
            data = part[0, 0]
            num_patterns = patterns_in_chunk[k]
            for i in range(num_patterns):
                res = dask.delayed(_full_refinement_solver)(
                    euler[i + inner_index],
                    pc[i + inner_index],
                    data[i],
                    pre_args,
                    method,
                    method_kwargs,
                    trust_region,
                )
                refined_params.append(res)

            inner_index += num_patterns  # Increase the index for
            # the next chunk
    else:  # NumPy array
        refined_params = [
            dask.delayed(_full_refinement_solver)(
                euler[i],
                pc[i],
                exp_data[i],
                pre_args,
                method,
                method_kwargs,
                trust_region,
            )
            for i in range(euler.shape[0])
        ]
    if compute:
        with ProgressBar():
            print(
                "\n".join([str(pr) for pr in to_print]),
                f"Refining {xmap.size} orientations and " f"projection centers:",
                file=sys.stdout,
            )
            results = dask.compute(*refined_params, scheduler="threads")
            refined_euler = np.empty((euler.shape[0], 3), dtype=np.float32)
            refined_pc = np.empty((euler.shape[0], 3), dtype=np.float32)
            refined_scores = np.empty((euler.shape[0]), dtype=np.float32)
            for i in range(euler.shape[0]):
                refined_scores[i] = results[i][0]

                refined_euler[i][0] = results[i][1]
                refined_euler[i][1] = results[i][2]
                refined_euler[i][2] = results[i][3]

                refined_pc[i][0] = results[i][4]
                refined_pc[i][1] = results[i][5]
                refined_pc[i][2] = results[i][6]

            new_det = detector.deepcopy()
            new_det.pc = refined_pc
            refined_rotations = Rotation.from_euler(refined_euler)
            xmap_dict = xmap.__dict__

            output = CrystalMap(
                rotations=refined_rotations,
                phase_id=xmap_dict["_phase_id"],
                x=xmap_dict["_x"],
                y=xmap_dict["_y"],
                phase_list=xmap_dict["phases"],
                prop={
                    "scores": refined_scores,
                },
                is_in_data=xmap_dict["is_in_data"],
                scan_unit=xmap_dict["scan_unit"],
            )
    else:
        output = dask.delayed(refined_params)
        new_det = -1
    return output, new_det


def refine_orientations(
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
) -> CrystalMap:
    """Performs an orientation refinement using the best orientation,
    for each point, from the initial indexing results stored in a single phase
    :class:`~orix.crystal_map.CrystalMap` and the fixed
    detector-sample geometry.

    Parameters
    ----------
    xmap : CrystalMap
        A crystal map storing the results of the initial EBSD
    master_pattern : EBSDMasterPattern
        EBSDMasterPattern in the square Lambert projection.
    signal : EBSD
        Experimental EBSD data.
    detector : EBSDDetector
        EBSD detector describing the detector dimensions and the
        detector-sample geometry with either a single, fixed
        projection/pattern center or a projection center for each
        scan point.
    energy : int
       Acceleration voltage, in kV, used to simulate the desired
        master pattern.
    mask : np.ndarray
        Boolean mask to be applied to the simulated patterns. True values are
        masked away.
    method : str, optional
        Name of the scipy.optimize function to be used. Must be one
        of "minimize", "differential_evolution", "dual_annealing",
        or "basinhopping". If not specified, "minimize" is used.
    method_kwargs : dict, optional
        Keyword arguments passed to the scipy.optimize function
        specified above.
    trust_region : list, optional
        List of how wide the bounds, centered on the initial
        orientation indexing result, should be for
        (phi1, Phi, phi2) in degrees. Only used for methods that
        support bounds (excluding Powell). Defaults to [1, 1, 1].
    compute : bool
        Whether to return a computed result, by default True.
        For more information see :func:`~dask.array.Array.compute`

    Returns
    -------
    CrystalMap
        A new crystal map where the orientations have been
        refined.
    """
    method, method_kwargs = _get_method(method, method_kwargs)

    # Convert from Quaternions to Euler angles
    euler = Rotation.to_euler(xmap.rotations)

    # Extract best rotation from xmap if given more than 1
    if xmap.rotations_per_point > 2:
        euler = euler[:, 0, :]

    if signal.data.dtype != np.float32:
        exp = signal.deepcopy()
        exp.rescale_intensity(dtype_out=np.float32)
    else:
        exp = signal
    exp_data = exp.data
    exp_shape = exp_data.shape

    if len(exp_shape) == 4:
        exp_data = exp_data.reshape(
            (exp_shape[0] * exp_shape[1], exp_shape[2], exp_shape[3])
        )
    elif len(exp_shape) == 2:  # 0D nav-dim
        exp_data = exp_data.reshape(((1,) + exp_data.shape))

    if not trust_region:
        trust_region = [0.0174532925, 0.0174532925, 0.0174532925]  # 1 deg
    else:
        trust_region = np.deg2rad(trust_region)

    scan_points = exp_data.shape[0]

    theta_c = np.deg2rad(detector.tilt)
    sigma = np.deg2rad(detector.sample_tilt)
    alpha = (np.pi / 2) - sigma + theta_c

    dncols = detector.ncols
    dnrows = detector.nrows
    px_size = detector.px_size

    pc_emsoft = detector.pc_emsoft()
    if len(pc_emsoft) == 1:
        xpc = np.full(scan_points, pc_emsoft[..., 0])
        ypc = np.full(scan_points, pc_emsoft[..., 1])
        L = np.full(scan_points, pc_emsoft[..., 2])
    else:
        xpc = pc_emsoft[..., 0]
        ypc = pc_emsoft[..., 1]
        L = pc_emsoft[..., 2]

    (
        master_north,
        master_south,
        npx,
        npy,
        scale,
    ) = _get_single_pattern_params(master_pattern, energy)

    if mask is None:
        mask = 1
    else:
        mask = ~mask

    pre_args = (
        master_north,
        master_south,
        npx,
        npy,
        scale,
        mask,
    )

    to_print = [
        "Refinement info:",
        f"method={method.__name__}",
        f"method kwargs={method_kwargs}",
    ]

    if method.__name__ not in ["minimize", "basinhopping"]:
        to_print.append(f"trust region={trust_region}")

    pre_args = dask.delayed(pre_args)
    trust_region = dask.delayed(trust_region)

    if isinstance(exp_data, dask.array.core.Array):
        patterns_in_chunk = exp_data.chunks[0]
        partitons = exp_data.to_delayed()  # List of delayed objects
        # equal to the number of chunks
        inner_index = 0
        refined_params = []
        for k, part in enumerate(partitons):
            data = part[0, 0]
            num_patterns = patterns_in_chunk[k]

            dc = dask.delayed(_fast_get_dc_multiple_pc)(
                xpc[inner_index : num_patterns + inner_index],
                ypc[inner_index : num_patterns + inner_index],
                L[inner_index : num_patterns + inner_index],
                num_patterns,
                dncols,
                dnrows,
                px_size,
                alpha,
            )

            for i in range(num_patterns):
                res = dask.delayed(_refine_orientations_solver)(
                    data[i],
                    euler[inner_index + i],
                    dc[i],
                    method,
                    method_kwargs,
                    pre_args,
                    trust_region,
                )
                refined_params.append(res)

            inner_index += num_patterns  # Increase the index for
            # the next chunk

    else:  # numpy array
        dc = _fast_get_dc_multiple_pc(
            xpc, ypc, L, scan_points, dncols, dnrows, px_size, alpha
        )

        refined_params = [
            dask.delayed(_refine_orientations_solver)(
                exp_data[i],
                euler[i],
                dc[i],
                method,
                method_kwargs,
                pre_args,
                trust_region,
            )
            for i in range(euler.shape[0])
        ]

    if compute:
        with ProgressBar():
            print(
                "\n".join([str(pr) for pr in to_print]),
                f"\nRefining {xmap.size} orientations:",
                file=sys.stdout,
            )
            results = dask.compute(*refined_params)
            refined_euler = np.empty((xmap.size, 3), dtype=np.float32)
            refined_scores = np.empty((xmap.size), dtype=np.float32)
            for i in range(xmap.size):
                refined_scores[i] = results[i][0]

                refined_euler[i][0] = results[i][1]
                refined_euler[i][1] = results[i][2]
                refined_euler[i][2] = results[i][3]

            refined_rotations = Rotation.from_euler(refined_euler)

            xmap_dict = xmap.__dict__

            output = CrystalMap(
                rotations=refined_rotations,
                phase_id=xmap_dict["_phase_id"],
                x=xmap_dict["_x"],
                y=xmap_dict["_y"],
                phase_list=xmap_dict["phases"],
                prop={
                    "scores": refined_scores,
                },
                is_in_data=xmap_dict["is_in_data"],
                scan_unit=xmap_dict["scan_unit"],
            )
    else:
        output = dask.delayed(refined_params)
    return output


def refine_projection_center(
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
    """Performs a projection center refinement using the
    fixed indexing results stored in a single phase
    :class:`~orix.crystal_map.CrystalMap` and the projection center
    estimates stored in an
    :class:`~kikuchipy.detectors.EBSDDetector`.

    Parameters
    ----------
    xmap : CrystalMap
        A crystal map storing the results of the initial EBSD
        indexing.
    master_pattern : EBSDMasterPattern
        EBSDMasterPattern in the square Lambert projection.
    signal : EBSD
        Experimental EBSD data.
    detector : EBSDDetector
        EBSD detector describing the detector dimensions and the
        detector-sample geometry with either a single, fixed
        projection/pattern center or a projection center for each
        scan point.
    energy : int
        Acceleration voltage, in kV, used to simulate the desired
        master pattern.
    mask : np.ndarray, optional
        Boolean mask to be applied to the simulated patterns. True values are
        masked away.
    method : str, optional
        Name of the scipy.optimize function to be used. Must be one
        of "minimize", "differential_evolution", "dual_annealing",
        or "basinhopping". If not specified, "minimize" is used.
    method_kwargs : dict, optional
        Keyword arguments passed to the scipy.optimize function
        specified above.
    trust_region : list, optional
        List of how wide the bounds, centered on the projection
        center, should be for (PCx, PCy, PCz) in Bruker convention.
        Only used for methods that support bounds
        (excluding Powell). Defaults to [0.05, 0.05, 0.05].
    compute : bool
        Whether to return a computed result, by default True.
        For more information see :func:`~dask.array.Array.compute`.

    Returns
    -------
    np.ndarray, EBSDDetector
        An array containing the similarity metric after refinement,
        and a new EBSD detector with refined projection centers.
    """
    method, method_kwargs = _get_method(method, method_kwargs)

    # Extract best rotation from xmap if given more than 1
    if len(xmap.rotations.shape) > 1:
        r = xmap.rotations[:, 0].data
    else:
        r = xmap.rotations.data

    if signal.data.dtype != np.float32:
        exp = signal.deepcopy()
        exp.rescale_intensity(dtype_out=np.float32)
    else:
        exp = signal
    exp_data = exp.data
    exp_shape = exp_data.shape

    pc = detector.pc

    # Set the PC equal across the scan if not given
    if len(pc) == 1:
        pc_val = pc[0]
        pc = np.full((exp_shape[0] * exp_shape[1], 3), pc_val)

    # 2D nav-dim
    if len(exp_shape) == 4:
        exp_data = exp_data.reshape(
            (exp_shape[0] * exp_shape[1], exp_shape[2], exp_shape[3])
        )
    elif len(exp_shape) == 2:  # 0D nav-dim
        exp_data = exp_data.reshape(((1,) + exp_data.shape))

    if not trust_region:
        trust_region = [0.05, 0.05, 0.05]

    (
        master_north,
        master_south,
        npx,
        npy,
        scale,
    ) = _get_single_pattern_params(master_pattern, energy)

    theta_c = np.deg2rad(detector.tilt)
    sigma = np.deg2rad(detector.sample_tilt)
    alpha = (np.pi / 2) - sigma + theta_c

    detector_data = [detector.ncols, detector.nrows, detector.px_size, alpha]

    if mask is None:
        mask = 1
    else:
        mask = ~mask

    pre_args = (
        master_north,
        master_south,
        npx,
        npy,
        scale,
        detector_data,
        mask,
    )
    to_print = [
        "Refinement info:",
        f"method={method.__name__}",
        f"method kwargs={method_kwargs}",
    ]

    if method.__name__ not in ["minimize", "basinhopping"]:
        to_print.append(f"trust region={trust_region}")

    pre_args = dask.delayed(pre_args)
    trust_region = dask.delayed(trust_region)

    if isinstance(exp_data, dask.array.core.Array):
        patterns_in_chunk = exp_data.chunks[0]
        partitons = exp_data.to_delayed()  # List of delayed objects
        # equal to the number of chunks
        inner_index = 0
        refined_params = []
        for k, part in enumerate(partitons):
            data = part[0, 0]
            num_patterns = patterns_in_chunk[k]
            for i in range(num_patterns):
                res = dask.delayed(_refine_pc_solver)(
                    data[i],
                    r[i + inner_index],
                    pc[i + inner_index],
                    method,
                    method_kwargs,
                    pre_args,
                    trust_region,
                )
                refined_params.append(res)

            inner_index += num_patterns  # Increase the index for
            # the next chunk
    else:  # NumPy array
        refined_params = [
            dask.delayed(_refine_pc_solver)(
                exp_data[i],
                r[i],
                pc[i],
                method,
                method_kwargs,
                pre_args,
                trust_region,
            )
            for i in range(xmap.size)
        ]

    output = refined_params
    if compute:
        with ProgressBar():
            print(
                "\n".join([str(pr) for pr in to_print]),
                f"\nRefining {xmap.size} projection centers:",
                file=sys.stdout,
            )
            results = dask.compute(*refined_params)

            refined_pc = np.empty((xmap.size, 3), dtype=np.float32)
            refined_scores = np.empty((xmap.size), dtype=np.float32)
            for i in range(xmap.size):
                refined_scores[i] = results[i][0]

                refined_pc[i][0] = results[i][1]
                refined_pc[i][1] = results[i][2]
                refined_pc[i][2] = results[i][3]

            new_det = detector.deepcopy()
            new_det.pc = refined_pc

            output = (refined_scores, new_det)

    return output


def _get_single_pattern_params(mp, energy):
    """Helper function that checks if the master pattern is in the
    square Lambert projection.

    Parameters
    ----------
    mp
      Master pattern in the square Lambert projection.
    energy
        Acceleration voltage, in kV, used to simulate the desired
        master pattern.

    Returns
    -------
    master_north, master_south
        Northern and southern hemisphere of the master pattern in
        the square Lambert projection.
    npx, npy
        Number of pixels on the master pattern in the x and y direction.
    scale
        Factor to scale up from the square Lambert projection to the
        master pattern.
    """

    if mp.projection != "lambert":
        raise NotImplementedError(
            "Master pattern must be in the square Lambert projection"
        )

    # Get the master pattern arrays created by a desired energy
    north_slice = ()
    if "energy" in [i.name for i in mp.axes_manager.navigation_axes]:
        energies = mp.axes_manager["energy"].axis
        north_slice += ((np.abs(energies - energy)).argmin(),)
    south_slice = north_slice
    if mp.hemisphere == "both":
        north_slice = (0,) + north_slice
        south_slice = (1,) + south_slice
    elif not mp.phase.point_group.contains_inversion:
        raise AttributeError(
            "For crystals of point groups without inversion symmetry, like "
            f"the current {mp.phase.point_group.name}, both hemispheres "
            "must be present in the master pattern signal"
        )
    master_north = mp.data[north_slice]
    master_south = mp.data[south_slice]
    npx, npy = mp.axes_manager.signal_shape
    scale = (npx - 1) / 2

    if master_north.dtype != np.float32:
        master_north = rescale_intensity(master_north, dtype_out=np.float32)
        master_south = rescale_intensity(master_south, dtype_out=np.float32)

    return master_north, master_south, npx, npy, scale


### NUMBA FUNCTIONS ###
@numba.njit(nogil=True)
def _fast_get_dc_multiple_pc(
    xpc: np.ndarray,
    ypc: np.ndarray,
    L: np.ndarray,
    scan_points: int,
    ncols: int,
    nrows: int,
    px_size: Union[int, float],
    alpha: float,
) -> np.ndarray:
    """Get the direction cosines between the detector and sample, with
     varying projection center.
     Based on :func:`~kikuchipy.indexing.refinement._fast_get_dc`.

    Parameters
    ----------
    xpc, ypc, L
        Projection center coordinates in the EMsoft convention for
        each scan point.
    scan_points
        Number of patterns in the scan.
    ncols, nrows
        Number of pixels in the x- and y-direction on the detector.
    px_size
        Pixel size in um.
    alpha
        Defined as (np.pi / 2) - sigma + theta_c, where sigma is the
        sample tilt and theta_c is the detector tilt.

    Returns
    -------
        Direction cosines unit vectors for each detector pixel.
    """
    nrows = int(nrows)
    ncols = int(ncols)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    # 1 DC per scan point
    r_g_array = np.zeros((scan_points, nrows, ncols, 3), dtype=np.float32)
    for k in range(scan_points):
        det_x = -1 * ((-xpc[k] - (1.0 - ncols) * 0.5) - np.arange(0, ncols)) * px_size
        det_y = ((ypc[k] - (1.0 - nrows) * 0.5) - np.arange(0, nrows)) * px_size
        L2 = L[k]
        for i in range(nrows):
            for j in range(ncols):
                x = det_y[nrows - i - 1] * ca + sa * L2
                y = det_x[j]
                z = -sa * det_y[nrows - i - 1] + ca * L2
                r_g_array[k][i][j][0] = x
                r_g_array[k][i][j][1] = y
                r_g_array[k][i][j][2] = z

    norm = np.sqrt(np.sum(np.square(r_g_array), axis=-1))
    norm = np.expand_dims(norm, axis=-1)
    r_g_array = r_g_array / norm

    return r_g_array


@numba.njit(nogil=True)
def _fast_get_dc(
    xpc: Union[int, float],
    ypc: Union[int, float],
    L: Union[int, float],
    ncols: int,
    nrows: int,
    px_size: Union[int, float],
    alpha: float,
) -> np.ndarray:
    """Get the direction cosines between the detector and sample, with
     a single, fixed projction center, as done in EMsoft
     and :cite:`callahan2013dynamical`.

    Parameters
    ----------
    xpc, ypc, L
        Projection center coordinates in the EMsoft convention.
    ncols, nrows
        Number of pixels in the x- and y-direction on the detector.
    px_size
        Pixel size in um.
    alpha
        Defined as (np.pi / 2) - sigma + theta_c, where sigma is the
        sample tilt and theta_c is the detector tilt.

    Returns
    -------
        Direction cosines unit vectors for each detector pixel.
    """
    # alpha: alpha = (np.pi / 2) - sigma + theta_c
    # Detector coordinates in microns
    nrows = int(nrows)
    ncols = int(ncols)
    det_x = -1 * ((-xpc - (1.0 - ncols) * 0.5) - np.arange(0, ncols)) * px_size
    det_y = ((ypc - (1.0 - nrows) * 0.5) - np.arange(0, nrows)) * px_size

    # Auxilliary angle to rotate between reference frames

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    r_g_array = np.zeros((nrows, ncols, 3), dtype=np.float32)

    for i in range(nrows):
        for j in range(ncols):
            x = det_y[nrows - i - 1] * ca + sa * L
            y = det_x[j]
            z = -sa * det_y[nrows - i - 1] + ca * L
            r_g_array[i][j][0] = x
            r_g_array[i][j][1] = y
            r_g_array[i][j][2] = z

    norm = np.sqrt(np.sum(np.square(r_g_array), axis=-1))
    norm = np.expand_dims(norm, axis=-1)
    r_g_array = r_g_array / norm

    return r_g_array


@numba.njit(nogil=True)
def _fast_simulate_single_pattern(
    r: np.ndarray,
    dc: np.ndarray,
    master_north: np.ndarray,
    master_south: np.ndarray,
    npx: int,
    npy: int,
    scale: Union[int, float],
) -> np.ndarray:
    """Simulates a single EBSD pattern for a given rotation. The pattern
    is found by bi-quadratic interpolation of the master pattern as
    described in EMsoft.

    Parameters
    ----------
    r
        Rotation represented by a unit quaternion.
    dc
        Direction cosines unit vector between detector and sample.
    master_north, master_south
        Northern and southern hemisphere of the master pattern in
        the square Lambert projection.
    npx, npy
        Number of pixels on the master pattern in the x and y direction.
    scale
        Factor to scale up from the square Lambert projection to the
        master pattern.

    Returns
    -------
    pattern
        Simulated EBSD pattern.
    """

    # From orix.quaternion.Quaternion.__mul__

    a = r[0]
    b = r[1]
    c = r[2]
    d = r[3]

    x = dc[..., 0]
    y = dc[..., 1]
    z = dc[..., 2]

    x_new = (a ** 2 + b ** 2 - c ** 2 - d ** 2) * x + 2 * (
        (a * c + b * d) * z + (b * c - a * d) * y
    )
    y_new = (a ** 2 - b ** 2 + c ** 2 - d ** 2) * y + 2 * (
        (a * d + b * c) * x + (c * d - a * b) * z
    )
    z_new = (a ** 2 - b ** 2 - c ** 2 + d ** 2) * z + 2 * (
        (a * b + c * d) * y + (b * d - a * c) * x
    )
    rotated_dc = np.stack((x_new, y_new, z_new), axis=-1)

    (
        nii,
        nij,
        niip,
        nijp,
        di,
        dj,
        dim,
        djm,
    ) = _fast_get_lambert_interpolation_parameters(
        rotated_direction_cosines=rotated_dc,
        npx=npx,
        npy=npy,
        scale=scale,
    )
    pattern = np.zeros(shape=rotated_dc.shape[0:-1], dtype=np.float32)
    for i in range(rotated_dc.shape[0]):
        for j in range(rotated_dc.shape[1]):
            _nii = nii[i][j]
            _nij = nij[i][j]
            _niip = niip[i][j]
            _nijp = nijp[i][j]
            _di = di[i][j]
            _dj = dj[i][j]
            _dim = dim[i][j]
            _djm = djm[i][j]
            if rotated_dc[..., 2][i][j] >= 0:
                pattern[i][j] = (
                    master_north[_nii, _nij] * _dim * _djm
                    + master_north[_niip, _nij] * _di * _djm
                    + master_north[_nii, _nijp] * _dim * _dj
                    + master_north[_niip, _nijp] * _di * _dj
                )
            else:
                pattern[i][j] = (
                    master_south[_nii, _nij] * _dim * _djm
                    + master_south[_niip, _nij] * _di * _djm
                    + master_south[_nii, _nijp] * _dim * _dj
                    + master_south[_niip, _nijp] * _di * _dj
                )
    return pattern


@numba.njit(nogil=True)
def _fast_lambert_projection(
    v: np.ndarray,
) -> np.ndarray:
    """Lambert projection of a vector as described in
    :cite:`callahan2013dynamical`.

    Parameters
    ----------
    v
        Rotated direction cosines with Cartesian coordinates.

    Returns
    -------
    lambert
        Rotated direction cosines in the square Lambert projection.
    """
    w = np.atleast_2d(v)
    norm = np.sqrt(np.sum(np.square(w), axis=-1))
    norm = np.expand_dims(norm, axis=-1)
    w = w / norm

    x = w[..., 0]
    y = w[..., 1]
    z = w[..., 2]

    # Arrays used in both setting X and Y
    sqrt_z = np.sqrt(2 * (1 - np.abs(z)))
    sign_x = np.sign(x)
    sign_y = np.sign(y)
    abs_yx = np.abs(y) <= np.abs(x)

    # Reusable constants
    sqrt_pi = np.sqrt(np.pi)
    sqrt_pi_half = sqrt_pi / 2
    two_over_sqrt_pi = 2 / sqrt_pi

    # Ensure (0, 0) is returned where |z| = 1
    lambert = np.zeros(x.shape + (2,), dtype=np.float32)
    # z_not_one = np.abs(z) != 1

    # I believe it currently returns invalid results for the
    # vector [0, 0, 1] as discussed in
    # https://github.com/pyxem/kikuchipy/issues/272

    # Numba does not support the fix implemented in the main code,
    # one workaround could be to implement a standard loop setting
    # the values

    # Equations (10a) and (10b) from Callahan and De Graef (2013)
    lambert[..., 0] = np.where(
        abs_yx,
        sign_x * sqrt_z * sqrt_pi_half,
        sign_y * sqrt_z * (two_over_sqrt_pi * np.arctan(x / y)),
    )
    lambert[..., 1] = np.where(
        abs_yx,
        sign_x * sqrt_z * (two_over_sqrt_pi * np.arctan(y / x)),
        sign_y * sqrt_z * sqrt_pi_half,
    )
    return lambert


@numba.njit(nogil=True)
def _fast_get_lambert_interpolation_parameters(
    rotated_direction_cosines: np.ndarray,
    npx: int,
    npy: int,
    scale: Union[int, float],
) -> tuple:
    """Get interpolation parameters in the square Lambert projection, as
    implemented in EMsoft.

    Parameters
    ----------
    rotated_direction_cosines
        Rotated direction cosines vector.
    npx, npy
        Number of pixels on the master pattern in the x and y direction.
    scale
        Factor to scale up from the square Lambert projection to the
        master pattern.

    Returns
    -------
    nii, nij
        Row and column coordinate of a point.
    niip, nijp
        Row and column coordinate of neighboring point.
    di, dim, dj, djm
        Row and column interpolation weight factors.
    """

    xy = (
        scale
        * _fast_lambert_projection(rotated_direction_cosines)
        / (np.sqrt(np.pi / 2))
    )

    i = xy[..., 1]
    j = xy[..., 0]
    nii = (i + scale).astype(np.int32)
    nij = (j + scale).astype(np.int32)
    niip = nii + 1
    nijp = nij + 1
    niip = np.where(niip < npx, niip, nii).astype(np.int32)
    nijp = np.where(nijp < npy, nijp, nij).astype(np.int32)
    nii = np.where(nii < 0, niip, nii).astype(np.int32)
    nij = np.where(nij < 0, nijp, nij).astype(np.int32)
    di = i - nii + scale
    dj = j - nij + scale
    dim = 1.0 - di
    djm = 1.0 - dj

    return nii, nij, niip, nijp, di, dj, dim, djm


# OBJECTIVE FUNCTIONS
def _orientation_objective_function_euler(
    x: np.ndarray,
    *args: tuple,
) -> float:
    """Objective function to be minimized when optimizing euler angles
    (phi1, Phi, phi2).

    Parameters
    ----------
    x
        1-D array containing the current (phi1, Phi, phi2)
        in radians.
    args
        Tuple of fixed parameters needed completely specify
        the function.

    Returns
    -------
        Objective function value.
    """
    experimental = args[0]
    master_north = args[1]
    master_south = args[2]
    npx = args[3]
    npy = args[4]
    scale = args[5]
    mask = args[6]
    dc = args[7]

    rotation = _rotation_from_euler(x[0], x[1], x[2])

    sim_pattern = _fast_simulate_single_pattern(
        rotation,
        dc,
        master_north,
        master_south,
        npx,
        npy,
        scale,
    )

    sim_pattern = sim_pattern * mask

    result = _py_ncc(experimental, sim_pattern)
    return 1 - result


def _projection_center_objective_function(
    x: np.ndarray,
    *args: tuple,
) -> float:
    """Objective function to be minimized when optimizing the projection
    center coordinates (PCx, PCy, PCz).

    Parameters
    ----------
    x
        1-D array containing the current (PCx, PCy, PCz) in the
        Bruker convention.
    args
        Tuple of fixed parameters needed completely specify
        the function.

    Returns
    -------
        Objective function value.
    """
    x_star = x[0]
    y_star = x[1]
    z_star = x[2]

    experimental = args[0]
    master_north = args[1]
    master_south = args[2]
    npx = args[3]
    npy = args[4]
    scale = args[5]
    detector_data = args[6]
    mask = args[7]
    rotation = args[8]

    detector_ncols = detector_data[0]
    detector_nrows = detector_data[1]
    detector_px_size = detector_data[2]
    alpha = detector_data[3]

    xpc = detector_ncols * (x_star - 0.5)  # Might be sign issue here?
    xpc = -xpc
    ypc = detector_nrows * (0.5 - y_star)
    L = detector_nrows * detector_px_size * z_star

    dc = _fast_get_dc(
        xpc, ypc, L, detector_ncols, detector_nrows, detector_px_size, alpha
    )

    sim_pattern = _fast_simulate_single_pattern(
        rotation,
        dc,
        master_north,
        master_south,
        npx,
        npy,
        scale,
    )

    sim_pattern = sim_pattern * mask

    result = _py_ncc(experimental, sim_pattern)
    return 1 - result


def _full_objective_function_euler(x: np.ndarray, *args: tuple) -> float:
    """Objective function to be minimized when optimizing euler angles
    (phi1, Phi, phi2) and projection center coordinates (PCx, PCy, PCz).

    Parameters
    ----------
    x
        1-D array containing the current (phi1, Phi, phi2) in radians
        and (PCx, PCy, PCz) in Bruker convention.
    args
        Tuple of fixed parameters needed completely specify
        the function.

    Returns
    -------
        Objective function value.
    """
    experimental = args[0]
    master_north = args[1]
    master_south = args[2]
    npx = args[3]
    npy = args[4]
    scale = args[5]
    detector_data = args[6]
    mask = args[7]

    detector_ncols = detector_data[0]
    detector_nrows = detector_data[1]
    detector_px_size = detector_data[2]

    rotation = _rotation_from_euler(x[0], x[1], x[2])

    x_star = x[3]
    y_star = x[4]
    z_star = x[5]

    xpc = detector_ncols * (x_star - 0.5)  # Might be sign issue here?
    xpc = -xpc
    ypc = detector_nrows * (0.5 - y_star)
    L = detector_nrows * detector_px_size * z_star

    alpha2 = detector_data[3]  # Different alpha
    dc = _fast_get_dc(
        xpc, ypc, L, detector_ncols, detector_nrows, detector_px_size, alpha2
    )

    sim_pattern = _fast_simulate_single_pattern(
        rotation,
        dc,
        master_north,
        master_south,
        npx,
        npy,
        scale,
    )
    sim_pattern = sim_pattern * mask

    result = _py_ncc(experimental, sim_pattern)
    return 1 - result


# SOLVERS
def _full_refinement_solver(
    r: np.ndarray,
    pc: np.ndarray,
    exp: np.ndarray,
    pre_args: tuple,
    method: callable,
    method_kwargs: dict,
    trust_region: list,
) -> tuple:
    """Maximizes the similarity between an experimental pattern and
    a simulated pattern by optimizing the euler angles (phi1, Phi, phi2)
    and the projection center coordinates (PCx, PCy, PCz).

    Parameters
    ----------
    r
        Euler angles (phi1, Phi, phi2) in radians.
    pc
        Projection center coordinates in Bruker convention.
    exp
        One experimental pattern with shape (n x m) and data type
        float 32.
    pre_args
        Tuple of fixed parameters used for single pattern
        simulations.
    method
        scipy.optimize function.
    method_kwargs
        Keyword arguments for the specific scipy.optimize function.
        For the list of possible keyword arguments, see
        the scipy documentation.
    trust_region
        List of how wide the bounds, centered on r and pc,
        should be for (phi1, Phi, phi2, PCx, PCy, PCz).
        Only used for methods that support bounds (excluding Powell).

    Returns
    -------
        score
            Highest NCC value.
        phi1, Phi , phi2
            The euler angles which gave the highest score, in radians.
        pcx, pxy, pxz
            The projection center coordinates which gave the highest
            score, in the Bruker convention.
    """
    phi1_0 = r[..., 0]
    Phi_0 = r[..., 1]
    phi2_0 = r[..., 2]
    eu_x0 = np.array((phi1_0, Phi_0, phi2_0))

    args = (exp,) + pre_args

    full_x0 = np.concatenate((eu_x0, pc), axis=None)

    if method.__name__ == "minimize":
        soln = method(
            _full_objective_function_euler,
            x0=full_x0,
            args=args,
            **method_kwargs,
        )
    elif method.__name__ == "differential_evolution":
        soln = method(
            _full_objective_function_euler,
            bounds=[
                (full_x0[0] - trust_region[0], full_x0[0] + trust_region[0]),
                (full_x0[1] - trust_region[1], full_x0[1] + trust_region[1]),
                (full_x0[2] - trust_region[2], full_x0[2] + trust_region[2]),
                (full_x0[3] - trust_region[3], full_x0[3] + trust_region[3]),
                (full_x0[4] - trust_region[4], full_x0[4] + trust_region[4]),
                (full_x0[5] - trust_region[5], full_x0[5] + trust_region[5]),
            ],
            args=args,
            **method_kwargs,
        )
    elif method.__name__ == "dual_annealing":
        soln = method(
            _full_objective_function_euler,
            bounds=[
                (full_x0[0] - trust_region[0], full_x0[0] + trust_region[0]),
                (full_x0[1] - trust_region[1], full_x0[1] + trust_region[1]),
                (full_x0[2] - trust_region[2], full_x0[2] + trust_region[2]),
                (full_x0[3] - trust_region[3], full_x0[3] + trust_region[3]),
                (full_x0[4] - trust_region[4], full_x0[4] + trust_region[4]),
                (full_x0[5] - trust_region[5], full_x0[5] + trust_region[5]),
            ],
            args=args,
            **method_kwargs,
        )
    elif method.__name__ == "basinhopping":
        method_kwargs["minimizer_kwargs"]["args"] = args
        soln = method(
            _full_objective_function_euler,
            x0=full_x0,
            **method_kwargs,
        )

    score = 1 - soln.fun
    phi1 = soln.x[0]
    Phi = soln.x[1]
    phi2 = soln.x[2]
    pcx = soln.x[3]
    pxy = soln.x[4]
    pxz = soln.x[5]

    return (score, phi1, Phi, phi2, pcx, pxy, pxz)


def _refine_pc_solver(
    exp: np.ndarray,
    r: np.ndarray,
    pc: np.ndarray,
    method: callable,
    method_kwargs: dict,
    pre_args: tuple,
    trust_region: list,
) -> tuple:
    """Maximizes the similarity between an experimental pattern and a
        simulated pattern by optimizing the projection center
        coordinates (PCx, PCy, PCz) used in the simulation.

    Parameters
    ----------
    exp
        One experimental pattern with shape (n x m) and data type
        float 32.
    r
        Euler angles (phi1, Phi, phi2) in radians.
    pc
        Projection center coordinates in Bruker convention.
    method
        scipy.optimize function.
    method_kwargs
        Keyword arguments for the specific scipy.optimize function.
        For the list of possible keyword arguments, see
        the scipy documentation.
    pre_args
        Tuple of fixed parameters used for single pattern
        simulations.
    trust_region
        List of how wide the bounds, centered pc, should be for
        (PCx, PCy, PCz). Only used for methods that support bounds
        (excluding Powell).

    Returns
    -------
        score
            Highest NCC value.
        pcx, pxy, pxz
            The projection center coordinates which gave the highest
            score, in the Bruker convention.
    """
    args = (exp,) + pre_args + (r,)
    pc_x0 = pc

    if method.__name__ == "minimize":
        soln = method(
            _projection_center_objective_function,
            x0=pc_x0,
            args=args,
            **method_kwargs,
        )
    elif method.__name__ == "differential_evolution":
        soln = method(
            _projection_center_objective_function,
            bounds=[
                (pc_x0[0] - trust_region[0], pc_x0[0] + trust_region[0]),
                (pc_x0[1] - trust_region[1], pc_x0[1] + trust_region[1]),
                (pc_x0[2] - trust_region[2], pc_x0[2] + trust_region[2]),
            ],
            args=args,
            **method_kwargs,
        )
    elif method.__name__ == "dual_annealing":
        soln = method(
            _projection_center_objective_function,
            bounds=[
                (pc_x0[0] - trust_region[0], pc_x0[0] + trust_region[0]),
                (pc_x0[1] - trust_region[1], pc_x0[1] + trust_region[1]),
                (pc_x0[2] - trust_region[2], pc_x0[2] + trust_region[2]),
            ],
            args=args,
            **method_kwargs,
        )
    elif method.__name__ == "basinhopping":
        method_kwargs["minimizer_kwargs"]["args"] = args
        soln = method(
            _projection_center_objective_function,
            x0=pc_x0,
            **method_kwargs,
        )

    score = 1 - soln.fun
    pcx = soln.x[0]
    pcy = soln.x[1]
    pcz = soln.x[2]
    return (score, pcx, pcy, pcz)


def _refine_orientations_solver(
    exp: np.ndarray,
    r: np.ndarray,
    dc: np.ndarray,
    method: callable,
    method_kwargs: dict,
    pre_args: tuple,
    trust_region: list,
) -> tuple:
    """Maximizes the similarity between an experimental pattern and a
    simulated pattern by optimizing the euler angles (phi1, Phi, phi2)
    used in the simulation.

    Parameters
    ----------
    exp
        One experimental pattern with shape (n x m) and data type
        float 32.
    r
        Euler angles (phi1, Phi, phi2) in radians.
    dc
        Direction cosines with shape (n x m) and data type float 32.
    method
        scipy.optimize function.
    method_kwargs
        Keyword arguments for the specific scipy.optimize function.
        For the list of possible keyword arguments,
        see the scipy documentation.
    pre_args
        Tuple of fixed parameters used for single pattern
        simulations.
    trust_region
        List of how wide the bounds, centered on r, should be for
        (phi1, Phi, phi2). Only used for methods that
        support bounds (excluding Powell).

    Returns
    -------
        score
            Highest NCC value.
        phi1, Phi , phi2
            The euler angles which gave the highest score, in radians.
    """

    phi1 = r[..., 0]
    Phi = r[..., 1]
    phi2 = r[..., 2]

    args = (exp,) + pre_args + (dc,)

    r_x0 = np.array((phi1, Phi, phi2), dtype=np.float32)

    if method.__name__ == "minimize":
        soln = method(
            _orientation_objective_function_euler,
            x0=r_x0,
            args=args,
            **method_kwargs,
        )
    elif method.__name__ == "differential_evolution":
        soln = method(
            _orientation_objective_function_euler,
            bounds=[
                (r_x0[0] - trust_region[0], r_x0[0] + trust_region[0]),
                (r_x0[1] - trust_region[1], r_x0[1] + trust_region[1]),
                (r_x0[2] - trust_region[2], r_x0[2] + trust_region[2]),
            ],
            args=args,
            **method_kwargs,
        )
    elif method.__name__ == "dual_annealing":
        soln = method(
            _orientation_objective_function_euler,
            bounds=[
                (r_x0[0] - trust_region[0], r_x0[0] + trust_region[0]),
                (r_x0[1] - trust_region[1], r_x0[1] + trust_region[1]),
                (r_x0[2] - trust_region[2], r_x0[2] + trust_region[2]),
            ],
            args=args,
            **method_kwargs,
        )
    elif method.__name__ == "basinhopping":
        method_kwargs["minimizer_kwargs"]["args"] = args
        soln = method(
            _orientation_objective_function_euler,
            x0=r_x0,
            **method_kwargs,
        )

    score = 1 - soln.fun
    refined_phi1 = soln.x[0]
    refined_Phi = soln.x[1]
    refined_phi2 = soln.x[2]

    return (score, refined_phi1, refined_Phi, refined_phi2)


# SIMILARITY METRICS
@numba.njit(fastmath=True)
def _py_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates the normalized cross-correlation coefficient (NCC)
    between two patterns of the same size and with data type float32.

    Parameters
    ----------
    a, b
        np.ndarray with shape (n x m) and with data type np.float32.

    Returns
    -------
        float representing the NCC between a and b.
    """
    abar = np.mean(a)
    bbar = np.mean(b)
    astar = a - abar
    bstar = b - bbar
    return np.sum(astar * bstar) / np.sqrt(
        np.sum(np.square(astar)) * np.sum(np.square(bstar))
    )


def _refinement_parameter_check(exp, xmap, detector, method, mask):
    """Helper function to determine if the input to the different EBSD
    refinement methods are valid.

    Parameters
    ----------
    exp
        Experimental data
    xmap
        CrystalMap containing the indexing result.
    detector
        EBSDDetector describing the experimental geometry.
    method
        Name of the scipy optimization method.
    mask
        Mask applied to the patterns.

    Raises
    ------
    ValueError
        If one of the parameters are invalid.
    """
    # Signal and Detector must have same shape
    if exp.axes_manager.signal_shape[::-1] != detector.shape:
        raise ValueError("Detector must have same shape as the signal shape")

    # Minimization strategy must be supported
    # Invalid local minimizers will be caught by the scipy functions
    if method is not None:
        supported_methods = [
            "minimize",
            "differential_evolution",
            "dual_annealing",
            "basinhopping",
        ]
        if method not in supported_methods:
            raise ValueError("Method not supported")

    scan_points = exp.axes_manager.navigation_size

    # Must have 1 PC or n x m PCs
    if len(detector.pc) != 1 and len(detector.pc) != scan_points:
        raise ValueError(
            "Detector must have exactly one projection center, or one projection center per scan point"
        )

    # xmap must be single phase
    if len(xmap.phases.ids) != 1:
        raise ValueError("Crystal map must have exactly one phase")

    # Same number of rotations in xmap as scan points
    if xmap.size != scan_points:
        raise ValueError(
            "Number of rotations in crystal map must be equal to the number of experimental patterns"
        )

    # mask must fit pattern
    if mask is not None:
        if exp.axes_manager.signal_shape != mask.shape:
            raise ValueError("Mask and signal must have the same shape")


def _get_method(method, method_kwargs):
    """Helper function that gets the correct optimization function,
    and sets reasonable keyword arguments if not specified.

    Parameters
    ----------
    method : str, optional
        Name of the scipy optimization method
    method_kwargs : dict, optional
        Keyword arguments for function

    Returns
    -------
    method : function
        Scipy optimization function.
    method_kwargs : dict
        Keyword arguments for function.
    """
    if not method:
        method = "minimize"
    if method == "minimize" and not method_kwargs:
        method_kwargs = {"method": "Nelder-Mead"}
    elif not method_kwargs:
        method_kwargs = {}
    method = getattr(scipy.optimize, method)

    return method, method_kwargs


@numba.njit(fastmath=True)
def _rotation_from_euler(phi1, Phi, phi2):
    """Converts the three Euler angles (phi1, Phi, phi2), in radians, to a
    unit quaternion.

    Parameters
    ----------
    phi1, Phi, phi2
        Euler angles in radians.

    Returns
    -------
    q
        Unit quaternion representing the same rotation.
    """
    # From Orix.rotation.from_euler()
    alpha = phi1
    beta = Phi
    gamma = phi2

    sigma = 0.5 * np.add(alpha, gamma)
    delta = 0.5 * np.subtract(alpha, gamma)
    c = np.cos(beta / 2)
    s = np.sin(beta / 2)

    # Using P = 1 from A.6
    q = np.zeros(4, dtype=np.float32)
    q[0] = c * np.cos(sigma)
    q[1] = -s * np.cos(delta)
    q[2] = -s * np.sin(delta)
    q[3] = -c * np.sin(sigma)

    condition = q[0] < 0

    if condition:
        q[1] = -q[1]
        q[2] = -q[2]
        q[3] = -q[3]
        q[0] = -q[0]

    return q
