import sys

import cv2
import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np

from orix.vector import Vector3d, Rodrigues
from orix.quaternion import Rotation
import scipy.optimize
from kikuchipy.detectors import EBSDDetector
from kikuchipy.signals import (
    _get_direction_cosines,
    _get_lambert_interpolation_parameters,
)
from kikuchipy.indexing.similarity_metrics import ncc, ndp

import pybobyqa

from kikuchipy.pattern import rescale_intensity


class Refinement:
    @staticmethod
    def refine_orientations_projection_center_3_step():
        pass

    @staticmethod
    def refine_orientations(
        xmap,
        mp,
        exp,
        det,
        energy,
        bounds=1,
        methodname="Nelder-Mead",
        tol=0.01,
        compute=True,
    ):
        bounds = bounds * (np.pi / 180)
        ncols = exp.data.shape[1]

        rdata = xmap.rotations.data
        # rdata = xmap.rotations.data[:10]  # Smaller dataset for testing
        dtype = rdata.dtype
        # TODO: Optimize chunks, currently hardcoded for SDSS dataset
        # r_da = da.from_array(rdata, chunks=(11700, 1, 4))
        r_da = da.from_array(rdata, chunks=(1000, 10, 4))
        # r_da = da.from_array(rdata, chunks=(1, 4))  # single Si mvp test

        (
            master_north,
            master_south,
            npx,
            npy,
            scale,
        ) = _get_single_pattern_params(mp, det, energy)

        dc = _get_direction_cosines_lean(det)
        # dc = _get_direction_cosines(det)

        exp_data = exp.data
        exp_data = rescale_intensity(exp_data, dtype_out=np.float32)
        # TODO: When Chunks is optimzied I think we need to add drop axis etc.
        refined_params = r_da.map_blocks(
            _refine_orientations_chunk,
            master_north=master_north,
            master_south=master_south,
            npx=npx,
            npy=npy,
            scale=scale,
            exp=exp_data,
            dc=dc,
            rads=bounds,
            ncols=ncols,
            methodname=methodname,
            tol=tol,
            dtype_out=np.float32,
            dtype=np.float32,
        )
        if compute:
            with ProgressBar():
                print(
                    f"Refining {xmap.rotations.shape[0]} orientations:",
                    file=sys.stdout,
                )
                output = refined_params.compute()
        else:
            # output = refined_params.visualize(
            #     filename="refinement_test.svg", rankdir="LR"
            # )
            output = refined_params
        return output

    @staticmethod
    def refine_projection_center(
        xmap,
        mp,
        exp,
        det,
        energy,
        bounds=None,
        methodname="Nelder-Mead",
        tol=0.001,
        compute=True,
    ):

        ncols = exp.data.shape[1]

        rdata = xmap.rotations.data
        # rdata = xmap.rotations.data[:10]  # Smaller dataset for testing
        dtype = rdata.dtype
        # TODO: Optimize chunks, currently hardcoded for SDSS dataset
        # r_da = da.from_array(rdata, chunks=(11700, 1, 4))
        # r_da = da.from_array(rdata, chunks=(10, 1, 4))
        r_da = da.from_array(rdata, chunks=(1, 4))  # single Si mvp test

        (
            master_north,
            master_south,
            npx,
            npy,
            scale,
        ) = _get_single_pattern_params(mp, det, energy)

        pc = det.pc
        xpc = det.pcx[0]
        ypc = det.pcy[0]
        zpc = det.pcz[0]

        if bounds is None and methodname != "Nelder-Mead":
            bounds = [
                (max(0, xpc - 0.1), min(xpc + 0.1, 1)),
                (max(0, ypc - 0.1), min(ypc + 0.1, 1)),
                (max(0, zpc - 0.1), zpc + 0.1),
            ]
        else:
            bounds = None

        exp_data = exp.data
        exp_data = rescale_intensity(exp_data, dtype_out=np.float32)
        refined_params = r_da.map_blocks(
            _refine_projection_center_chunk,
            master_north=master_north,
            master_south=master_south,
            npx=npx,
            npy=npy,
            scale=scale,
            exp=exp_data,
            det=det,
            pc=pc,
            ncols=ncols,
            methodname=methodname,
            tol=tol,
            bounds=bounds,
            dtype_out=np.float32,
            dtype=np.float32,
        )
        if compute:
            with ProgressBar():
                print(
                    f"Refining {xmap.rotations.shape[0]} PCs:",
                    file=sys.stdout,
                )
                output = refined_params.compute()
        else:
            # output = refined_params.visualize(
            #     filename="refinement_test.svg", rankdir="LR"
            # )
            output = refined_params
        return output


def _refine_orientations_chunk(
    r,
    master_north,
    master_south,
    npx,
    npy,
    scale,
    exp,
    dc,
    rads,
    ncols,
    methodname,
    tol,
    dtype_out=np.float32,
):
    rotations = Rotation(r)
    rotations_shape = r.shape
    refined_params = np.empty(
        shape=(rotations_shape[0],) + (4,), dtype=dtype_out
    )
    # If xmap were to store unique PCs the following line should be placed inside the for-loop

    for i in np.ndindex(rotations_shape[0]):
        index = i[0]
        # TODO: Fix this mess, used here for single pattern refinement
        if len(exp.shape) == 2:
            exp_data = exp
        else:
            # print(exp.shape)
            # row = index // ncols
            # col = index % ncols
            # exp_data = exp[row, col]
            exp_data = exp[index]
            # print(exp_data.shape)

        rotation = rotations[index]
        best_rotation = rotation[0]
        # rodrigues = Rodrigues.from_rotation(best_rotation)
        # r1 = rodrigues.data[..., 0][0]
        # r2 = rodrigues.data[..., 1][0]
        # r3 = rodrigues.data[..., 2][0]

        # x0 = rodrigues.data[0]

        q1 = best_rotation.data[..., 0][0]
        q2 = best_rotation.data[..., 1][0]
        q3 = best_rotation.data[..., 2][0]
        q4 = best_rotation.data[..., 3][0]

        x0 = best_rotation.data[0]

        args = (exp_data, master_north, master_south, npx, npy, scale, dc)

        # if methodname == "Nelder-Mead":
        #     bounds = None
        #     options = {"adaptive": False}
        # else:
        #     bounds = None
        #     options = None
        #
        # optimized = scipy.optimize.minimize(
        #     _orientation_objective_function,
        #     tol=tol,
        #     x0=x0,
        #     bounds=bounds,
        #     args=args,
        #     method=methodname,
        #     options=options,
        # )
        # score = -optimized.fun
        # #r1 = optimized.x[0]
        # #r2 = optimized.x[1]
        # #r3 = optimized.x[2]
        # q1 = optimized.x[0]
        # q2 = optimized.x[1]
        # q3 = optimized.x[2]
        # q4 = optimized.x[3]
        # refined_params[index] = np.array((score, q1, q2, q3, q4))

        # Py-BOBYQA Test

        # Euler angles
        r_euler = best_rotation.to_euler()
        phi1 = r_euler[..., 0][0]
        Phi = r_euler[..., 1][0]
        phi2 = r_euler[..., 2][0]
        x0 = np.array((phi1, Phi, phi2))

        # Rodrigues vector changing axis and angle
        r_rod = Rodrigues.from_rotation(best_rotation)
        rx = r_rod.data[..., 0][0]
        ry = r_rod.data[..., 1][0]
        rz = r_rod.data[..., 2][0]
        x0 = np.array((rx, ry, rz))
        # ==> Perfect:  0.988  Good:  0.0  OK:  0.0  Poor:  0.012

        # Rodrigues Vector changing only axis
        alpha = r_rod.angle.data[0]
        tan_alpha = np.tan(alpha / 2)

        rxa = r_rod.axis.data[..., 0][0]
        ryb = r_rod.axis.data[..., 1][0]
        rzc = r_rod.axis.data[..., 2][0]
        x0 = np.array((rxa, ryb, rzc))
        # ==> Perfect:  0.988  Good:  0.0  OK:  0.0  Poor:  0.012

        args = (
            exp_data,
            master_north,
            master_south,
            npx,
            npy,
            scale,
            dc,
            tan_alpha,
        )

        # print(x0)

        lower = np.array([-1.1, -1.1, -1.1])
        upper = np.array([1.1, 1.1, 1.1])
        soln = pybobyqa.solve(
            _orientation_objective_function,
            x0,
            args=args,
            bounds=(lower, upper),
            do_logging=False,
            # scaling_within_bounds=True,
            rhobeg=1,
            rhoend=0.000000001,
            seek_global_minimum=True,
            user_params={"model.abs_tol": -0.98},
        )
        # print(soln)
        score = -soln.f
        q1 = soln.x[0]
        q2 = soln.x[1]
        q3 = soln.x[2]
        #   q4 = soln.x[0]
        refined_params[index] = np.array((score, q1, q2, q3))

        # params = []
        # for r in rotation:
        # #best_rotation = rotation[0]
        #     best_rotation = r
        #     r_euler = best_rotation.to_euler()
        #     phi1 = r_euler[..., 0][0]
        #     Phi = r_euler[..., 1][0]
        #     phi2 = r_euler[..., 2][0]
        #     x0 = np.array((phi1, Phi, phi2))
        # # x0 = [phi1, Phi, phi2]
        # # print("\n", rotation, "\n", best_rotation, "\n", r_euler, "\n", phi1, Phi, phi2,"\n", x0.shape)
        # # print("\n", best_rotation, "\n")
        #
        #     args = (exp_data, master_north, master_south, npx, npy, scale, dc)
        #
        #     if methodname == "Nelder-Mead":
        #         bounds = None
        #     else:
        #         # The z* bound is a bit sketchy and could probably be made cleaner
        #         bounds = [
        #             (phi1 - rads, phi1 + rads),
        #             (Phi - rads, Phi + rads),
        #             (phi2 - rads, phi2 + rads),
        #         ]
        #
        #     optimized = scipy.optimize.minimize(
        #         _orientation_objective_function,
        #         tol=tol,
        #         x0=x0,
        #         bounds=bounds,
        #         args=args,
        #         method=methodname,
        #         options={"adaptive": True},
        #     )
        #     score = -optimized.fun
        #     phi1 = optimized.x[0]
        #     Phi = optimized.x[1]
        #     phi2 = optimized.x[2]
        #     params.append((score, phi1, Phi, phi2))
        #     if score >= 0.98:
        #         break
        # sorted_params = sorted(params, key=lambda x: x[0])
        # (score, phi1, Phi, phi2) = sorted_params[-1]
        # refined_params[index] = np.array((score, r1, r2, r3))
        # Just for testing
        # return optimized
    return refined_params


def _refine_projection_center_chunk(
    r,
    master_north,
    master_south,
    npx,
    npy,
    scale,
    exp,
    det,
    pc,
    ncols,
    methodname,
    tol,
    bounds,
    dtype_out=np.float32,
):
    rotations = Rotation(r)
    rotations_shape = r.shape
    refined_params = np.empty(
        shape=(rotations_shape[0],) + (3,), dtype=dtype_out
    )

    # If xmap were to store unique PCs the following line should be placed inside the for-loop
    x0 = pc

    for i in np.ndindex(rotations_shape[0]):
        index = i[0]
        # TODO: Fix this mess
        if len(exp.data.shape) == 2:
            exp_data = exp
        else:
            row = index // ncols
            col = index % ncols
            exp_data = exp[row, col]

        rotation = rotations[index]
        best_rotation = rotation[0]

        detector_data = [
            det.ncols,
            det.nrows,
            det.px_size,
            det.tilt,
            det.sample_tilt,
        ]
        args = (
            exp_data,
            master_north,
            master_south,
            npx,
            npy,
            scale,
            detector_data,
            best_rotation,
        )

        optimized = scipy.optimize.minimize(
            _projection_center_objective_function,
            tol=tol,
            x0=x0,
            bounds=bounds,
            args=args,
            method=methodname,
            options={"adaptive": False},
        )
        x_star = optimized.x[0]
        y_star = optimized.x[1]
        z_star = optimized.x[2]

        refined_params[index] = np.array((x_star, y_star, z_star))
        # Just for testing
        return optimized
    return refined_params


def _orientation_objective_function(x, *args):
    # phi1 = x[0]
    # Phi = x[1]
    # phi2 = x[2]
    # rotation = Rotation.from_euler((phi1, Phi, phi2))

    # rotation = Rotation((x[0], x[1], x[2], x[3]))

    experimental = args[0]
    master_north = args[1]
    master_south = args[2]
    npx = args[3]
    npy = args[4]
    scale = args[5]
    dc = args[6]
    tan_alpha = args[7]

    rx = x[0]
    ry = x[1]
    rz = x[2]

    rod = Rodrigues((tan_alpha * rx, tan_alpha * ry, tan_alpha * rz))
    rotation = Rotation.from_neo_euler(rod)

    # print("\n", rotation,"\n", )
    sim_pattern = _simulate_single_pattern(
        rotation,
        dc,
        master_north,
        master_south,
        npx,
        npy,
        scale,
    )
    # print(sim_pattern.shape)
    # return -ncc(experimental, sim_pattern)
    result = cv2.matchTemplate(experimental, sim_pattern, cv2.TM_CCOEFF_NORMED)
    # print("\n", -result, "\n")
    return -result[0][0]


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
    rotation = args[7]

    detector_ncols = detector_data[0]
    detector_nrows = detector_data[1]
    detector_px_size = detector_data[2]
    detector_tilt = detector_data[3]
    sample_tilt = detector_data[4]

    detector = EBSDDetector(
        shape=(detector_nrows, detector_ncols),
        pc=(x_star, y_star, z_star),
        tilt=detector_tilt,
        sample_tilt=sample_tilt,
        px_size=detector_px_size,
        convention="bruker",
    )

    dc = _get_direction_cosines_lean(detector)

    sim_pattern = _simulate_single_pattern(
        rotation,
        dc,
        master_north,
        master_south,
        npx,
        npy,
        scale,
    )
    result = cv2.matchTemplate(experimental, sim_pattern, cv2.TM_CCOEFF_NORMED)
    return -result[0][0]


def _simulate_single_pattern(
    rotation,
    dc,
    master_north,
    master_south,
    npx,
    npy,
    scale,
):
    rotated_dc = rotation * dc
    (
        nii,
        nij,
        niip,
        nijp,
        di,
        dj,
        dim,
        djm,
    ) = _get_lambert_interpolation_parameters(
        rotated_direction_cosines=rotated_dc,
        npx=npx,
        npy=npy,
        scale=scale,
    )
    pattern = np.where(
        rotated_dc.z >= 0,
        (
            master_north[nii, nij] * dim * djm
            + master_north[niip, nij] * di * djm
            + master_north[nii, nijp] * dim * dj
            + master_north[niip, nijp] * di * dj
        ),
        (
            master_south[nii, nij] * dim * djm
            + master_south[niip, nij] * di * djm
            + master_south[nii, nijp] * dim * dj
            + master_south[niip, nijp] * di * dj
        ),
    )
    # return pattern
    return pattern.astype(np.float32)


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
        -((-xpc - (1.0 - detector.ncols) * 0.5) - np.arange(0, detector.ncols))
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
