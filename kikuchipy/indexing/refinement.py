import sys
from typing import Optional, Union

import cv2
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


class Refinement:
    @staticmethod
    def refine_orientations(
        xmap,
        mp,
        exp,
        det,
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

        ncols = exp.data.shape[1]

        # Convert from Quaternions to Rodrigues vector
        rodrigues = Rodrigues.from_rotation(xmap.rotations)
        rdata = rodrigues.data
        chunk_size = ("auto",) + (-1,) * (len(rdata.shape) - 1)

        r_da = da.from_array(rdata, chunks=chunk_size)

        (
            master_north,
            master_south,
            npx,
            npy,
            scale,
        ) = _get_single_pattern_params(mp, det, energy)

        # TODO: Replace this method with njit rework, needed for pc and 6D
        dc = _get_direction_cosines_lean(det)
        dc = dc.data
        # dc = _get_direction_cosines(det)

        exp_data = exp.data
        exp_data = rescale_intensity(exp_data, dtype_out=np.float32)
        refined_params = r_da.map_blocks(
            _refine_orientations_chunk,
            method=method,
            method_kwargs=method_kwargs,
            master_north=master_north,
            master_south=master_south,
            npx=npx,
            npy=npy,
            scale=scale,
            exp=exp_data,
            dc=dc,
            ncols=ncols,
            dtype_out=np.float32,
            dtype=np.float32,
        )
        if compute:
            with ProgressBar():
                print(
                    f"Refining {xmap.rotations.shape[0]} orientations:",
                    file=sys.stdout,
                )
                output_params = refined_params.compute()
                rodrigues_params = np.column_stack(
                    (
                        output_params[..., 1],
                        output_params[..., 2],
                        output_params[..., 3],
                    )
                )
                rod = Rodrigues(rodrigues_params)
                output_rotation = Rotation.from_neo_euler(rod)
                xmap_dict = xmap.__dict__
                output_scores = output_params[..., 0]
                # TODO: Needs vast improvements!
                output = CrystalMap(
                    rotations=output_rotation,
                    phase_id=xmap_dict["_phase_id"],
                    x=xmap_dict["_x"],
                    y=xmap_dict["_y"],
                    phase_list=xmap_dict["phases"],
                    prop={
                        "simulation_indices": xmap_dict["_prop"][
                            "simulation_indices"
                        ][..., 0],
                        "scores": output_scores,
                    },
                    is_in_data=xmap_dict["is_in_data"],
                    scan_unit=xmap_dict["scan_unit"],
                )
        else:
            output = refined_params
        return output


# TODO: Make a function that just returns the data
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
    if len(detector.pc) > 1:
        raise NotImplementedError(
            "Detector must have exactly one projection center"
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

    return master_north, master_south, npx, npy, scale


### NUMBA FUNCTIONS ###
@numba.njit()
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


@numba.njit()
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


@numba.njit()
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


### CHUNK FUNCTIONS ###


def _refine_orientations_chunk(
    r,
    method,
    method_kwargs,
    master_north,
    master_south,
    npx,
    npy,
    scale,
    exp,
    dc,
    ncols,
    dtype_out=np.float32,
):
    # rotations = Rotation(r)
    rotations_shape = r.shape
    refined_params = np.empty(
        shape=(rotations_shape[0],) + (4,), dtype=dtype_out
    )
    # If xmap were to store unique PCs the following line should be placed inside the for-loop

    for i in np.ndindex(rotations_shape[0]):
        index = i[0]
        # TODO: Fix this mess, used here for single pattern refinement
        if len(exp.shape) == 2:  # Single Experimental pattern
            exp_data = exp
        elif len(exp.shape) == 3:  # Experimental Pattern 1D nav shape
            exp_data = exp[index]
        else:  # Experimental Patterns 2D nav shape
            row = index // ncols
            col = index % ncols
            exp_data = exp[row, col]

        rotation = r[index]
        if len(rotation.shape) > 1:
            best_rotation = rotation[0]
        else:
            best_rotation = rotation

        # Initial Rodrigues vector params
        rx = best_rotation[..., 0]
        ry = best_rotation[..., 1]
        rz = best_rotation[..., 2]
        rod_x0 = np.array((rx, ry, rz))

        args = (
            exp_data,
            master_north,
            master_south,
            npx,
            npy,
            scale,
            dc,
        )
        soln = method(
            _orientation_objective_function,
            x0=rod_x0,
            args=args,
            **method_kwargs,
        )
        score = 1 - soln.fun
        rx = soln.x[0]
        ry = soln.x[1]
        rz = soln.x[2]

        refined_params[index] = np.array((score, rx, ry, rz))

    return refined_params


### OBJECTIVE FUNCTIONS ###

# @numba.njit
def _orientation_objective_function(x, *args):
    experimental = args[0]
    master_north = args[1]
    master_south = args[2]
    npx = args[3]
    npy = args[4]
    scale = args[5]
    dc = args[6]

    # rx = x[0]
    # ry = x[1]
    # rz = x[2]
    #
    # v = np.array((rx, ry, rz))
    v = np.array((x[0], x[1], x[2]), dtype=np.float32)
    norm = np.sqrt(np.sum(np.square(v), axis=-1))
    v = v / norm
    x = v[0]
    y = v[1]
    z = v[2]
    # angle = np.arctan(norm) * 2
    # half_angle = angle / 2  # Here we are multiplying by 2 just to divide by 2 :)
    half_angle = np.arctan(norm)
    # Adapted from Orix.quaternion.rotation.Rotation.from_neo_euler
    s = np.sin(half_angle)
    a = np.cos(half_angle)
    b = s * x
    c = s * y
    d = s * z
    r = np.array((a, b, c, d))

    sim_pattern = _fast_simulate_single_pattern(
        r,
        dc,
        master_north,
        master_south,
        npx,
        npy,
        scale,
    )

    result = cv2.matchTemplate(experimental, sim_pattern, cv2.TM_CCOEFF_NORMED)
    return 1 - result[0][0]
