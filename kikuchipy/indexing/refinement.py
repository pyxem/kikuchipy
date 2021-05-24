import sys
from typing import Optional, Union

import cv2
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import numba
import numpy as np
from orix.crystal_map import CrystalMap
from orix.vector import Vector3d, Rodrigues
from orix.quaternion import Rotation
import scipy.optimize

from kikuchipy.detectors import EBSDDetector
from kikuchipy.pattern import rescale_intensity
from kikuchipy.signals import EBSD, LazyEBSD
from kikuchipy.indexing.similarity_metrics import ncc


class Refinement:
    @staticmethod
    def refine_xmap(
        xmap,
        mp,
        exp,
        det,
        mask,
        energy,
        method="minimize",
        method_kwargs=None,
        compute=True,
    ):
        if method == "minimize" and not method_kwargs:
            method_kwargs = {"method": "Nelder-Mead"}
        elif not method_kwargs:
            method_kwargs = {}
        method = getattr(scipy.optimize, method)

        # Convert from Quaternions to Euler angles
        with np.errstate(divide="ignore", invalid="ignore"):
            euler = Rotation.to_euler(xmap.rotations)

        # Extract best rotation from xmap if given more than 1
        if len(euler.shape) > 2:
            euler = euler[:, 0, :]

        exp_data = exp.data
        exp_data = rescale_intensity(exp_data, dtype_out=np.float32)
        # exp_data = exp_data * mask # The mask should already have been
        # applied during DI
        exp_shape = exp_data.shape

        pc = det.pc

        # Set the PC equal across the scan if not given
        if len(pc) == 1:
            pc_val = pc[0]
            pc = np.full((exp_shape[0] * exp_shape[1], 3), pc_val)
        # Should raise error here if len pc not equal to scan size

        # 2D nav-dim
        if len(exp_shape) == 4:
            exp_data = exp_data.reshape(
                (exp_shape[0] * exp_shape[1], exp_shape[2], exp_shape[3])
            )
        elif len(exp_shape) == 2:  # 0D nav-dim
            exp_data = np.expand_dims(exp_data, axis=0)

        (
            master_north,
            master_south,
            npx,
            npy,
            scale,
        ) = _get_single_pattern_params(mp, det, energy)

        theta_c = np.deg2rad(det.tilt)
        sigma = np.deg2rad(det.sample_tilt)
        alpha = (np.pi / 2) - sigma + theta_c

        detector_data = [det.ncols, det.nrows, det.px_size, alpha]

        pre_args = (
            master_north,
            master_south,
            npx,
            npy,
            scale,
            detector_data,
            mask,
        )

        pre_args = dask.delayed(pre_args)
        exp_data = dask.delayed(exp_data)
        refined_params = [
            dask.delayed(_refine_xmap_solver)(
                euler[i], pc[i], exp_data[i], pre_args, method, method_kwargs
            )
            for i in range(euler.shape[0])
        ]
        if compute:
            with ProgressBar():
                print(
                    f"Refining {xmap.rotations.shape[0]} orientations and "
                    f"projection centers:",
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

                new_det = det.deepcopy()
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

    @staticmethod
    def refine_orientations(
        xmap,
        mp,
        exp,
        det,
        energy,
        mask=1,
        method="minimize",
        method_kwargs=None,
        compute=True,
    ):

        if method not in ["minimize", "dual_annealing"]:
            raise NotImplementedError
        if method == "minimize" and not method_kwargs:
            method_kwargs = {"method": "Nelder-Mead"}
        elif not method_kwargs:
            method_kwargs = {}
        method = getattr(scipy.optimize, method)

        # Convert from Quaternions to Euler angles
        with np.errstate(divide="ignore", invalid="ignore"):
            euler = Rotation.to_euler(xmap.rotations)

        # Extract best rotation from xmap if given more than 1
        if len(euler.shape) > 2:
            euler = euler[:, 0, :]

        exp_data = exp.data
        exp_data = rescale_intensity(exp_data, dtype_out=np.float32)
        exp_shape = exp_data.shape

        if len(exp_shape) == 4:
            exp_data = exp_data.reshape(
                (exp_shape[0] * exp_shape[1], exp_shape[2], exp_shape[3])
            )
        elif len(exp_shape) == 2:  # 0D nav-dim
            exp_data = np.expand_dims(exp_data, axis=0)

        scan_points = exp_data.shape[0]

        theta_c = np.deg2rad(det.tilt)
        sigma = np.deg2rad(det.sample_tilt)
        alpha = (np.pi / 2) - sigma + theta_c

        dncols = det.ncols
        dnrows = det.nrows
        px_size = det.px_size

        # TODO: Make this work with
        pc_emsoft = det.pc_emsoft()
        if len(pc_emsoft) == 1:
            xpc = np.full(scan_points, pc_emsoft[..., 0])
            ypc = np.full(scan_points, pc_emsoft[..., 1])
            L = np.full(scan_points, pc_emsoft[..., 2])
        else:  # Should raise error here if shape mismatch with exp!!
            xpc = pc_emsoft[..., 0]
            ypc = pc_emsoft[..., 1]
            L = pc_emsoft[..., 2]

        dc = _fast_get_dc_multiple_pc(
            xpc, ypc, L, scan_points, dncols, dnrows, px_size, alpha
        )

        (
            master_north,
            master_south,
            npx,
            npy,
            scale,
        ) = _get_single_pattern_params(mp, det, energy)

        pre_args = (
            master_north,
            master_south,
            npx,
            npy,
            scale,
            mask,
        )

        pre_args = dask.delayed(pre_args)
        exp_data = dask.delayed(exp_data)
        refined_params = [
            dask.delayed(_refine_orientations_solver)(
                exp_data[i], euler[i], dc[i], method, method_kwargs, pre_args
            )
            for i in range(euler.shape[0])
        ]

        if compute:
            with ProgressBar():
                print(
                    f"Refining {xmap.rotations.shape[0]} orientations:",
                    file=sys.stdout,
                )
                results = dask.compute(*refined_params)
                refined_euler = np.empty(
                    (xmap.rotations.shape[0], 3), dtype=np.float32
                )
                refined_scores = np.empty(
                    (xmap.rotations.shape[0]), dtype=np.float32
                )
                for i in range(xmap.rotations.shape[0]):
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

    @staticmethod
    def refine_projection_center(
        xmap,
        mp,
        exp,
        det,
        energy,
        mask=1,
        method="minimize",
        method_kwargs=None,
        compute=True,
    ):
        if method == "minimize" and not method_kwargs:
            method_kwargs = {"method": "Nelder-Mead"}
        elif not method_kwargs:
            method_kwargs = {}
        method = getattr(scipy.optimize, method)

        # Extract best rotation from xmap if given more than 1
        if len(xmap.rotations.shape) > 1:
            r = xmap.rotations[:, 0].data
        else:
            r = xmap.rotations.data

        exp_data = exp.data
        exp_data = rescale_intensity(exp_data, dtype_out=np.float32)
        exp_shape = exp_data.shape

        pc = det.pc

        # Set the PC equal across the scan if not given
        if len(pc) == 1:
            pc_val = pc[0]
            pc = np.full((exp_shape[0] * exp_shape[1], 3), pc_val)
        # Should raise error here if len pc not equal to scan size

        # 2D nav-dim
        if len(exp_shape) == 4:
            exp_data = exp_data.reshape(
                (exp_shape[0] * exp_shape[1], exp_shape[2], exp_shape[3])
            )
        elif len(exp_shape) == 2:  # 0D nav-dim
            exp_data = np.expand_dims(exp_data, axis=0)

        (
            master_north,
            master_south,
            npx,
            npy,
            scale,
        ) = _get_single_pattern_params(mp, det, energy)

        theta_c = np.deg2rad(det.tilt)
        sigma = np.deg2rad(det.sample_tilt)
        alpha = (np.pi / 2) - sigma + theta_c

        detector_data = [det.ncols, det.nrows, det.px_size, alpha]

        pre_args = (
            master_north,
            master_south,
            npx,
            npy,
            scale,
            detector_data,
            mask,
        )

        pre_args = dask.delayed(pre_args)
        exp_data = dask.delayed(exp_data)
        refined_params = [
            dask.delayed(_refine_pc_solver)(
                exp_data[i], r[i], pc[i], method, method_kwargs, pre_args
            )
            for i in range(xmap.rotations.shape[0])
        ]

        output = refined_params
        if compute:
            with ProgressBar():
                print(
                    f"Refining {xmap.rotations.shape[0]} projection centers:",
                    file=sys.stdout,
                )
                results = dask.compute(*refined_params)

                refined_pc = np.empty(
                    (xmap.rotations.shape[0], 3), dtype=np.float32
                )
                refined_scores = np.empty(
                    (xmap.rotations.shape[0]), dtype=np.float32
                )
                for i in range(xmap.rotations.shape[0]):
                    refined_scores[i] = results[i][0]

                    refined_pc[i][0] = results[i][1]
                    refined_pc[i][1] = results[i][2]
                    refined_pc[i][2] = results[i][3]

                new_det = det.deepcopy()
                new_det.pc = refined_pc

                output = (refined_scores, new_det)

        return output


def _get_direction_cosines_lean(detector: EBSDDetector) -> Vector3d:
    """Get the direction cosines between the detector and sample as done
    in EMsoft and :cite:`callahan2013dynamical`.

    Parameters
    ----------
    detector : EBSDDetector
        EBSDDetector object with a certain detector geometry and one
        projection center.

    Returns
    -------
    Vector3d
        Direction cosines for each detector pixel.
    """

    # TODO: Make even leaner

    pc = detector.pc_emsoft()
    xpc = pc[..., 0]
    ypc = pc[..., 1]
    L = pc[..., 2]

    # Detector coordinates in microns
    det_x = (
        -1
        * ((-xpc - (1.0 - detector.ncols) * 0.5) - np.arange(0, detector.ncols))
        * detector.px_size
    )
    det_y = (
        (ypc - (1.0 - detector.nrows) * 0.5) - np.arange(0, detector.nrows)
    ) * detector.px_size

    # Auxilliary angle to rotate between reference frames
    theta_c = np.radians(detector.tilt)
    sigma = np.radians(detector.sample_tilt)

    alpha = (np.pi / 2) - sigma + theta_c
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    r_g_array = np.zeros((detector.nrows, detector.ncols, 3))

    i, j = np.meshgrid(
        np.arange(detector.nrows - 1, -1, -1),
        np.arange(detector.ncols),
        indexing="ij",
    )

    r_g_array[..., 0] = det_y[i] * ca + sa * L
    r_g_array[..., 1] = det_x[j]
    r_g_array[..., 2] = -sa * det_y[i] + ca * L
    r_g = Vector3d(r_g_array)

    return r_g.unit


def _get_single_pattern_params(mp, detector, energy):
    # This method is already a part of the EBSDMasterPattern.get_patterns so
    # it could probably replace it?
    if mp.projection != "lambert":
        raise NotImplementedError(
            "Master pattern must be in the square Lambert projection"
        )
    # if len(detector.pc) > 1:
    #     raise NotImplementedError(
    #         "Detector must have exactly one projection center"
    #     )

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

    return master_north, master_south, npx, npy, scale


### NUMBA FUNCTIONS ###
@numba.njit(nogil=True)
def _fast_get_dc_multiple_pc(
    xpc, ypc, L, scan_points, ncols, nrows, px_size, alpha
):
    nrows = int(nrows)
    ncols = int(ncols)

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    # 1 DC per scan point
    r_g_array = np.zeros((scan_points, nrows, ncols, 3), dtype=np.float32)
    for k in range(scan_points):
        det_x = (
            -1
            * ((-xpc[k] - (1.0 - ncols) * 0.5) - np.arange(0, ncols))
            * px_size
        )
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
def _fast_get_dc(xpc, ypc, L, ncols, nrows, px_size, alpha):
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
def _fast_norm_dc(r_g_array):
    norm = np.sqrt(np.sum(np.square(r_g_array), axis=-1))
    norm = np.expand_dims(norm, axis=-1)
    r_g_array = r_g_array / norm

    return r_g_array


@numba.njit(nogil=True)
def _fast_simulate_single_pattern(
    r,
    dc,
    master_north,
    master_south,
    npx,
    npy,
    scale,
):

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
def _fast_lambert_projection(v):
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

    # I believe it currently returns invalid results for the vector [0, 0, 1]
    # as discussed in https://github.com/pyxem/kikuchipy/issues/272

    # Numba does not support the fix implemented in the main code
    # one workaround could be to implement a standard loop setting the values

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


### OBJECTIVE FUNCTIONS ###


def _orientation_objective_function_euler(x, *args):
    experimental = args[0]
    master_north = args[1]
    master_south = args[2]
    npx = args[3]
    npy = args[4]
    scale = args[5]
    mask = args[6]
    dc = args[7]

    # From Orix.rotation.from_euler()
    alpha = x[0]  # psi1
    beta = x[1]  # Psi
    gamma = x[2]  # psi3

    sigma = 0.5 * np.add(alpha, gamma)
    delta = 0.5 * np.subtract(alpha, gamma)
    c = np.cos(beta / 2)
    s = np.sin(beta / 2)

    # Using P = 1 from A.6
    q = np.zeros((4,))
    q[..., 0] = c * np.cos(sigma)
    q[..., 1] = -s * np.cos(delta)
    q[..., 2] = -s * np.sin(delta)
    q[..., 3] = -c * np.sin(sigma)

    for i in [1, 2, 3, 0]:  # flip the zero element last
        q[..., i] = np.where(q[..., 0] < 0, -q[..., i], q[..., i])

    r = q

    sim_pattern = _fast_simulate_single_pattern(
        r,
        dc,
        master_north,
        master_south,
        npx,
        npy,
        scale,
    )

    sim_pattern = sim_pattern * mask

    result = py_ncc(experimental, sim_pattern)
    return 1 - result


def _projection_center_objective_function(x, *args):
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

    result = py_ncc(experimental, sim_pattern)
    return 1 - result


def _full_objective_function_euler(x, *args):
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

    # From Orix.rotation.from_euler()
    phi1 = x[0]
    Phi = x[1]
    phi2 = x[2]

    alpha = phi1
    beta = Phi
    gamma = phi2

    sigma = 0.5 * np.add(alpha, gamma)
    delta = 0.5 * np.subtract(alpha, gamma)
    c = np.cos(beta / 2)
    s = np.sin(beta / 2)

    # Using P = 1 from A.6
    q = np.zeros((4,))
    q[..., 0] = c * np.cos(sigma)
    q[..., 1] = -s * np.cos(delta)
    q[..., 2] = -s * np.sin(delta)
    q[..., 3] = -c * np.sin(sigma)

    for i in [1, 2, 3, 0]:  # flip the zero element last
        q[..., i] = np.where(q[..., 0] < 0, -q[..., i], q[..., i])

    rotation = q

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

    result = py_ncc(experimental, sim_pattern)
    return 1 - result


### Callback ###


class MinimizeStopper(object):
    def __init__(self, max_score):
        self.max_score = max_score

    def __call__(self, x, f, context):
        score = 1 - f
        if score > self.max_score:
            return True
        else:
            return False


def _refine_xmap_solver(r, pc, exp, pre_args, method, method_kwargs):
    phi1_0 = r[..., 0]
    Phi_0 = r[..., 1]
    phi2_0 = r[..., 2]
    eu_x0 = np.array((phi1_0, Phi_0, phi2_0))

    args = (exp,) + pre_args

    soln = method(
        _full_objective_function_euler,
        x0=np.concatenate((eu_x0, pc), axis=None),
        args=args,
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


def _refine_pc_solver(exp, r, pc, method, method_kwargs, pre_args):
    args = (exp,) + pre_args + (r,)
    pc_x0 = pc
    soln = method(
        _projection_center_objective_function,
        x0=pc_x0,
        args=args,
        **method_kwargs,
    )

    score = 1 - soln.fun
    pcx = soln.x[0]
    pcy = soln.x[1]
    pcz = soln.x[2]
    return (score, pcx, pcy, pcz)


def _refine_orientations_solver(exp, r, dc, method, method_kwargs, pre_args):

    phi1 = r[..., 0]
    Phi = r[..., 1]
    phi2 = r[..., 2]

    args = (exp,) + pre_args + (dc,)

    r_x0 = np.array((phi1, Phi, phi2), dtype=np.float32)

    soln = method(
        _orientation_objective_function_euler,
        x0=r_x0,
        args=args,
        **method_kwargs,
    )
    score = 1 - soln.fun
    refined_phi1 = soln.x[0]
    refined_Phi = soln.x[1]
    refined_phi2 = soln.x[2]

    return (score, refined_phi1, refined_Phi, refined_phi2)


#### Custom Similarity Metrics ####


@numba.njit(fastmath=True)
def py_ncc(a, b):
    # Input should already be np.float32
    # a = a.astype(np.float32)
    # b = b.astype(np.float32)
    abar = np.mean(a)
    bbar = np.mean(b)
    astar = a - abar
    bstar = b - bbar
    return np.sum(astar * bstar) / np.sqrt(
        np.sum(np.square(astar)) * np.sum(np.square(bstar))
    )


@numba.njit(fastmath=True)
def py_ndp(a, b):
    # Input should already be np.float32
    # a = a.astype(np.float32)
    # b = b.astype(np.float32)
    return np.sum(a * b) / np.sqrt(np.sum(np.square(a)) * np.sum(np.square(b)))
