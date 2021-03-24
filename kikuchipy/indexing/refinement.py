import sys

import cv2
import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np

from orix.crystal_map import CrystalMap
from orix.vector import Vector3d, Rodrigues
from orix.quaternion import Rotation
import scipy.optimize
from kikuchipy.detectors import EBSDDetector
from kikuchipy.signals import (
    _get_direction_cosines,
    _get_lambert_interpolation_parameters,
)
from kikuchipy.indexing.similarity_metrics import ncc, ndp

import gradient_free_optimizers

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
        method="BOBYQA",
        compute=True,
    ):
        ncols = exp.data.shape[1]

        # Convert from Quaternions to Rodrigues vector
        rodrigues = Rodrigues.from_rotation(xmap.rotations)
        rdata = rodrigues.data
        # dtype = rdata.dtype

        r_da = da.from_array(rdata, chunks=("auto", -1, -1))

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
        refined_params = r_da.map_blocks(
            _refine_orientations_chunk,
            meth=method,
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
                output_scores = output_params[..., 0]
                # TODO: Needs vast improvements!
                output = CrystalMap(
                    rotations=output_rotation,
                    prop={
                        "scores": output_scores,
                    },
                )
        else:
            # output = refined_params.visualize(
            #     filename="refinement_test.svg", rankdir="LR"
            # )
            output = refined_params
        return output

    @staticmethod
    def refine_orientations2(
        xmap,
        mp,
        exp,
        det,
        energy,
        compute=True,
    ):
        ncols = exp.data.shape[1]

        # Convert from Quaternions to Rodrigues vector
        test_data = xmap.rotations.data[0:10]
        test_rotation = Rotation(test_data)
        rodrigues = Rodrigues.from_rotation(test_rotation)
        # rodrigues = Rodrigues.from_rotation(xmap.rotations)
        rdata = rodrigues.data
        # dtype = rdata.dtype

        r_da = da.from_array(rdata, chunks=("auto", -1, -1))

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
            _refine_orientations_chunk2,
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
    meth,
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
        best_rotation = rotation[0]

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
        if meth == "BOBYQA":
            soln = pybobyqa.solve(
                _orientation_objective_function,
                x0=rod_x0,
                args=args,
                # bounds=(lower, upper),
                do_logging=False,
                user_params={"model.abs_tol": 0.01},
            )
            # ==> R-DI Avg score:  0.9686444236040115  Perfect:  0.973  Good:  0.0  OK:  0.0  Poor:  0.027
            # print(soln)
            score = 1 - soln.f
            r1 = soln.x[0]
            r2 = soln.x[1]
            r3 = soln.x[2]
        elif meth == "Nelder-Mead":
            soln = scipy.optimize.minimize(
                _orientation_objective_function,
                x0=rod_x0,
                args=args,
                method="Nelder-Mead",
            )
            score = 1 - soln.fun
            r1 = soln.x[0]
            r2 = soln.x[1]
            r3 = soln.x[2]
        else:
            raise NotImplementedError(
                "Invalid Solver! Only BOBYQA and Nelder-Mead currently supported."
            )

        refined_params[index] = np.array((score, r1, r2, r3))
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
    experimental = args[0]
    master_north = args[1]
    master_south = args[2]
    npx = args[3]
    npy = args[4]
    scale = args[5]
    dc = args[6]

    rx = x[0]
    ry = x[1]
    rz = x[2]

    rod = Rodrigues((rx, ry, rz))
    rotation = Rotation.from_neo_euler(rod)

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
    return 1 - result[0][0]


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


def _extra_orientation_refinement_chunk(
    r,
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
    rotations_shape = r.shape
    refined_params = np.empty(
        shape=(rotations_shape[0],) + (4,), dtype=dtype_out
    )
    sufficient_score = np.mean(r[..., 0]) - 2 * np.std(r[..., 0])
    for i in np.ndindex(rotations_shape[0]):
        index = i[0]
        if len(exp.shape) == 2:  # Single Experimental pattern
            exp_data = exp
        elif len(exp.shape) == 3:  # Experimental Pattern 1D nav shape
            exp_data = exp[index]
        else:  # Experimental Patterns 2D nav shape
            row = index // ncols
            col = index % ncols
            exp_data = exp[row, col]

        rotation = r[index]

        rx = rotation[1]
        ry = rotation[2]
        rz = rotation[3]
        rod = Rodrigues((rx, ry, rz))
        rot = Rotation.from_neo_euler(rod)
        rot_data = np.rad2deg((rot.to_euler()[0]))
        phi1 = rot_data[0]
        Phi = rot_data[1]
        phi2 = rot_data[2]

        if rotation[0] >= sufficient_score:
            refined_params[index] = np.array((rotation[0], phi1, Phi, phi2))
        else:

            x0 = np.rad2deg((phi1, Phi, phi2))

            args = (
                exp_data,
                master_north,
                master_south,
                npx,
                npy,
                scale,
                dc,
            )

            soln = scipy.optimize.dual_annealing(
                _orientation_objective_function_euler,
                bounds=[(0, 360), (0, 360), (0, 360)],
                args=args,
                local_search_options={"method": "Nelder-Mead"},
            )
            # print(soln)
            score = 1 - soln.fun
            r1 = soln.x[0]
            r2 = soln.x[1]
            r3 = soln.x[2]

            refined_params[index] = np.array((score, r1, r2, r3))
    return refined_params


def _orientation_objective_function_euler(x, *args):
    experimental = args[0]
    master_north = args[1]
    master_south = args[2]
    npx = args[3]
    npy = args[4]
    scale = args[5]
    dc = args[6]

    phi1 = x[0]
    Phi = x[1]
    phi2 = x[2]

    rotation = Rotation.from_euler(np.deg2rad((phi1, Phi, phi2)))

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
    return 1 - result[0][0]


def _refine_orientations_chunk2(
    r,
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
        best_rotation = rotation[0]
        # Initial Rodrigues vector params
        rx = best_rotation[..., 0]
        ry = best_rotation[..., 1]
        rz = best_rotation[..., 2]

        rod = Rodrigues((rx, ry, rz))
        rot = Rotation.from_neo_euler(rod)

        euler = np.rad2deg((rot.to_euler()[0]))
        phi1 = euler[..., 0]
        Phi = euler[..., 1]
        phi2 = euler[..., 2]

        args = (
            exp_data,
            master_north,
            master_south,
            npx,
            npy,
            scale,
            dc,
        )

        def _objective_function_2(x):
            _phi1 = x["x1"]
            _Phi = x["x2"]
            _phi2 = x["x3"]

            _rotation = Rotation.from_euler(np.deg2rad((_phi1, _Phi, _phi2)))

            _sim_pattern = _simulate_single_pattern(
                _rotation,
                dc,
                master_north,
                master_south,
                npx,
                npy,
                scale,
            )

            _result = cv2.matchTemplate(
                exp_data, _sim_pattern, cv2.TM_CCOEFF_NORMED
            )
            return _result[0][0]

        p1l = phi1 - 1
        p1u = phi1 + 1

        search_space = {
            "x1": np.arange(phi1 - 1, phi1 + 1, 0.1),
            "x2": np.arange(Phi - 1, Phi + 1, 0.1),
            "x3": np.arange(phi2 - 1, phi2 - 1, 0.1),
        }
        print(search_space["x3"])
        # print("hello")
        opt = gradient_free_optimizers.RepulsingHillClimbingOptimizer(
            search_space
        )
        opt.search(
            _objective_function_2, max_score=0.98, n_iter=100, verbosity=False
        )
        score = opt.best_score
        print("\n", score, "\n")
        r1 = opt.best_para["x1"]
        r2 = opt.best_para["x2"]
        r3 = opt.best_para["x3"]

        # refined_params[index] = np.array((score, r1, r2, r3))
    return refined_params
