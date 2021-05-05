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
    def refine_xmap3(
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
                        "simulation_indices": xmap_dict["_prop"][
                            "simulation_indices"
                        ][..., 0],
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
    def refine_xmap2(
        xmap,
        mp,
        exp,
        det,
        energy,
        mode="both",
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

        ncols = exp.data.shape[1]

        # Convert from Quaternions to Euler angles
        with np.errstate(divide="ignore", invalid="ignore"):
            euler = Rotation.to_euler(xmap.rotations)

        # Extract best rotation from xmap if given more than 1
        if len(euler.shape) > 2:
            euler = euler[:, 0, :]

        exp_data = exp.data
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
                exp_shape[0] * exp_shape[1], exp_shape[2], exp_shape[3]
            )
            chunk_shape = ("auto", -1, -1)
        elif len(exp_shape) == 3:  # 1D nav-dim
            chunk_shape = ("auto", -1, -1)
        else:  # 0D nav-dim
            # Will this work if it is a Dask Array ??
            exp_data = np.expand_dims(exp_data, axis=0)
            chunk_shape = (-1, -1, -1)

        exp_da = da.from_array(exp_data, chunks=chunk_shape)
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
        refined_params = exp_da.map_blocks(
            _refine_xmap_chunk2,
            r=euler,
            pc=pc,
            method=method,
            method_kwargs=method_kwargs,
            master_north=master_north,
            master_south=master_south,
            npx=npx,
            npy=npy,
            scale=scale,
            detector_data=detector_data,
            drop_axis=(1, 2),
            new_axis=1,
            dtype=np.float32,
        )
        new_det = -1
        if compute:
            with ProgressBar():
                print(
                    f"Refining {xmap.rotations.shape[0]} orientations and "
                    f"projcetion centers:",
                    file=sys.stdout,
                )
                print("9")
                output_params = refined_params.compute()
                euler_params = np.column_stack(
                    (
                        output_params[..., 1],
                        output_params[..., 2],
                        output_params[..., 3],
                    )
                )
                pcs = np.column_stack(
                    (
                        output_params[..., 4],
                        output_params[..., 5],
                        output_params[..., 6],
                    )
                )
                new_det = det.deepcopy()
                new_det.pc = pcs
                rot = Rotation.from_euler(euler_params)
                output_rotation = rot
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
        return output, new_det

    @staticmethod
    def refine_orientations2(
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
    def refine_orientations(
        xmap,
        mp,
        exp,
        det,
        energy,
        deep_refine=False,
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

        dc = _get_direction_cosines_lean(det)
        dc = dc.data
        # dc = _get_direction_cosines(det)

        scores = xmap.scores
        if deep_refine:
            accept = np.mean(scores) - 2 * np.std(scores)
        else:
            accept = None

        exp_data = exp.data
        exp_data = rescale_intensity(exp_data, dtype_out=np.float32)
        refined_params = r_da.map_blocks(
            _refine_orientations_chunk,
            scores=scores,
            accept=accept,
            method=method,
            method_kwargs=method_kwargs,
            deep_refine=deep_refine,
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

    @staticmethod
    def refine_projection_center2(
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

    @staticmethod
    def refine_projection_center(
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

        rdata = xmap.rotations.data
        chunk_size = ("auto",) + (-1,) * (len(rdata.shape) - 1)

        r_da = da.from_array(rdata, chunks=chunk_size)

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

        pc = det.pc

        exp_data = exp.data
        exp_data = rescale_intensity(exp_data, dtype_out=np.float32)
        refined_params = r_da.map_blocks(
            _refine_pc_chunk,
            pc=pc,
            method=method,
            method_kwargs=method_kwargs,
            master_north=master_north,
            master_south=master_south,
            npx=npx,
            npy=npy,
            scale=scale,
            exp=exp_data,
            ncols=ncols,
            detector_data=detector_data,
            dtype_out=np.float32,
            dtype=np.float32,
        )
        if compute:
            with ProgressBar():
                print(
                    f"Refining {xmap.rotations.shape[0]} projection centers:",
                    file=sys.stdout,
                )
                output = refined_params.compute()
        else:
            output = refined_params
        return output

    @staticmethod
    def refine_xmap(
        xmap,
        mp,
        exp,
        det,
        energy,
        deep_refine=False,
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
        print("UPDATED1!")

        # Convert from Quaternions to Rodrigues vector
        euler = Rotation.to_euler(xmap.rotations)
        rdata = euler.data
        chunk_size = ("auto",) + (-1,) * (len(rdata.shape) - 1)
        pc = det.pc
        ncols = exp.data.shape[1]

        r_da = da.from_array(rdata, chunks=chunk_size)

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

        scores = xmap.scores
        if deep_refine:
            accept = np.mean(scores) - 2 * np.std(scores)
        else:
            accept = None

        exp_data = exp.data
        exp_data = rescale_intensity(exp_data, dtype_out=np.float32)
        refined_params = r_da.map_blocks(
            _refine_xmap_chunk,
            scores=scores,
            accept=accept,
            method=method,
            method_kwargs=method_kwargs,
            deep_refine=deep_refine,
            master_north=master_north,
            master_south=master_south,
            npx=npx,
            npy=npy,
            scale=scale,
            exp=exp_data,
            pc=pc,
            ncols=ncols,
            detector_data=detector_data,
            dtype_out=np.float32,
            dtype=np.float32,
        )
        if compute:
            with ProgressBar():
                print(
                    f"Refining {xmap.rotations.shape[0]} orientations and "
                    f"projcetion centers:",
                    file=sys.stdout,
                )
                output_params = refined_params.compute()
                euler_params = np.column_stack(
                    (
                        output_params[..., 1],
                        output_params[..., 2],
                        output_params[..., 3],
                    )
                )
                pcs = np.column_stack(
                    (
                        output_params[..., 4],
                        output_params[..., 5],
                        output_params[..., 6],
                    )
                )
                new_det = det.deepcopy()
                new_det.pc = pcs
                rot = Rotation.from_euler(euler_params)
                output_rotation = rot
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
        return output, new_det


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


### CHUNK FUNCTIONS ###


def _refine_orientations_chunk(
    r,
    scores,
    accept,
    method,
    method_kwargs,
    deep_refine,
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

        if deep_refine:
            if scores[index] >= accept:
                refined_params[index] = np.array((scores[index], rx, ry, rz))
                continue
            soln = scipy.optimize.dual_annealing(
                _orientation_objective_function_euler,
                args=args,
                x0=np.array((np.pi, np.pi / 2, np.pi)),
                bounds=((0, 2 * np.pi), (0, np.pi), (0, 2 * np.pi)),
                callback=MinimizeStopper(accept),
                **method_kwargs,
            )
            score = 1 - soln.fun
            rx = soln.x[0]
            ry = soln.x[1]
            rz = soln.x[2]
        else:
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


def _refine_pc_chunk(
    r,
    pc,
    method,
    method_kwargs,
    master_north,
    master_south,
    npx,
    npy,
    scale,
    exp,
    detector_data,
    ncols,
    dtype_out=np.float32,
):
    rotations_shape = r.shape
    refined_params = np.empty(
        shape=(rotations_shape[0],) + (4,), dtype=dtype_out
    )

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
        if len(rotation.shape) > 1:
            best_rotation = rotation[0]
        else:
            best_rotation = rotation
        if len(pc) > 1:
            pc_x0 = pc[..., index]
        else:
            pc_x0 = pc[0]

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

        refined_params[index] = np.array((score, pcx, pcy, pcz))

    return refined_params


def _refine_xmap_chunk(
    r,
    scores,
    accept,
    method,
    method_kwargs,
    deep_refine,
    master_north,
    master_south,
    npx,
    npy,
    scale,
    exp,
    pc,
    detector_data,
    ncols,
    dtype_out=np.float32,
):
    rotations_shape = r.shape
    refined_params = np.empty(
        shape=(rotations_shape[0],) + (7,), dtype=dtype_out
    )
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
        if len(rotation.shape) > 1:
            best_rotation = rotation[0]
        else:
            best_rotation = rotation
        if len(pc) > 1:
            pc_x0 = pc[..., index]
        else:
            pc_x0 = pc[0]

        phi1_0 = best_rotation[..., 0]
        Phi_0 = best_rotation[..., 1]
        phi2_0 = best_rotation[..., 2]
        eu_x0 = np.array((phi1_0, Phi_0, phi2_0))

        args = (
            exp_data,
            master_north,
            master_south,
            npx,
            npy,
            scale,
            detector_data,
        )

        if deep_refine:
            if scores[index] >= accept:
                refined_params[index] = np.array(
                    (
                        scores[index],
                        phi1_0,
                        Phi_0,
                        phi2_0,
                        pc_x0[0],
                        pc_x0[1],
                        pc_x0[2],
                    )
                )
                continue
            soln = scipy.optimize.dual_annealing(
                _full_objective_function_euler,
                args=args,
                x0=np.array(
                    (np.pi, np.pi / 2, np.pi, pc_x0[0], pc_x0[1], pc_x0[2])
                ),
                bounds=(
                    (0, 2 * np.pi),
                    (0, np.pi),
                    (0, 2 * np.pi),
                    (0, 1),
                    (0, 1),
                    (0.1, 1),
                ),
                callback=MinimizeStopper(accept),
                **method_kwargs,
            )
            score = 1 - soln.fun
            phi1 = soln.x[0]
            Phi = soln.x[1]
            phi2 = soln.x[2]
            pcx = soln.x[3]
            pxy = soln.x[4]
            pxz = soln.x[5]
        else:
            soln = method(
                _full_objective_function_euler,
                x0=np.concatenate((eu_x0, pc_x0), axis=None),
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

        refined_params[index] = np.array(
            (score, phi1, Phi, phi2, pcx, pxy, pxz)
        )
    return refined_params


def _refine_xmap_chunk2(
    exp,
    r,
    method,
    method_kwargs,
    master_north,
    master_south,
    npx,
    npy,
    scale,
    pc,
    detector_data,
):
    rotations_shape = r.shape
    refined_params = np.empty(
        shape=(rotations_shape[0],) + (7,), dtype=np.float32
    )

    for i in np.arange(rotations_shape[0]):
        exp_data = exp[i]
        pc_x0 = pc[i]
        best_rotation = r[i]

        phi1_0 = best_rotation[..., 0]
        Phi_0 = best_rotation[..., 1]
        phi2_0 = best_rotation[..., 2]
        eu_x0 = np.array((phi1_0, Phi_0, phi2_0))

        args = (
            exp_data,
            master_north,
            master_south,
            npx,
            npy,
            scale,
            detector_data,
        )
        soln = method(
            _full_objective_function_euler,
            x0=np.concatenate((eu_x0, pc_x0), axis=None),
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

        refined_params[i] = np.array((score, phi1, Phi, phi2, pcx, pxy, pxz))
    return refined_params


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


def _orientation_objective_function(x, *args):
    experimental = args[0]
    master_north = args[1]
    master_south = args[2]
    npx = args[3]
    npy = args[4]
    scale = args[5]
    mask = args[6]
    dc = args[7]

    # rx = x[0]
    # ry = x[1]
    # rz = x[2]
    #
    # v = np.array((rx, ry, rz))
    v = np.array((x[0], x[1], x[2]), dtype=np.float32)
    norm = np.sqrt(np.sum(np.square(v), axis=-1))
    if norm != 0:
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


def _full_objective_function(x, *args):
    pass


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


def _full_objective_function_euler2(x, *args):
    experimental = args[0]
    master_north = args[1]
    master_south = args[2]
    npx = args[3]
    npy = args[4]
    scale = args[5]
    detector_data = args[6]

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

    alpha = detector_data[3]  # Different alpha
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

    result = ncc(sim_pattern, experimental)
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


### KAM ###

from orix.quaternion import Misorientation


def KAM(xmap, y_shape, x_shape, order, threshold, symmetry_list):
    threshold = np.deg2rad(threshold)

    # Get the cutoff value from the order
    # , probably a formula for this somewhere... :)
    # 0th order ==> 0
    # 1st order ==> 1
    # 2nd order ==> 4
    # 3rd order ==> 5
    # 4th ordr ==> 8

    if order < 0:
        raise ValueError("Order must be a positive integer!")
    if order > 4:
        raise NotImplementedError("Order must be between 0 and 4.")

    order_list = [0, 1, 4, 5, 8]
    max_distance = order_list[int(order)]

    # Reshape the 1D rotation array to 2D grid with same shape as scan
    r = Rotation(xmap.rotations.data.reshape(y_shape, x_shape, 4))
    # Convert to Misorientations with given symmetries
    m = Misorientation(r).set_symmetry(*symmetry_list)

    kam_map = np.zeros((y_shape, x_shape), dtype=np.float32)

    for i in range(y_shape):
        for j in range(x_shape):
            N = 0
            total_misorientation = 0
            for k in range(i - 2, i + 3):
                for l in range(j - 2, j + 3):
                    # Do we consider ourself?
                    # if i == k and j == l:
                    # 	continue
                    if k ** 2 + l ** 2 >= max_distance:
                        continue
                    else:  # Can we index misorientation object?
                        m_kl = m[i, j].angle_with(m[k, l])
                        if m_kl <= threshold:
                            N += 1
                            total_misorientation += m_kl
            kam_value = total_misorientation / N
            kam_map[i, j] = kam_value
    return kam_map


# Assumes entire exp data fits into memory
def _refine_xmap_fast(xmap, mp, exp, det, energy, method, method_kwargs):
    if method == "minimize" and not method_kwargs:
        method_kwargs = {"method": "Nelder-Mead"}
    elif not method_kwargs:
        method_kwargs = {}
    method = getattr(scipy.optimize, method)

    euler = Rotation.to_euler(xmap.rotations)
    if len(euler.shape) > 2:
        euler = euler[:, 0, :]  # Extract best rotation
    pc = det.pc
    # ncols = exp.data.shape[1]

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

    detector_data = np.array(
        (det.ncols, det.nrows, det.px_size, alpha), dtype=np.float32
    )

    exp_data = exp.data
    exp_data = rescale_intensity(exp_data, dtype_out=np.float32)

    # Make exp_data 1D
    if len(exp_data.shape) > 3:
        exp_data = exp_data.reshape(
            (
                exp_data.shape[0] * exp_data.shape[1],
                exp_data.shape[2],
                exp_data.shape[3],
            )
        )

    n_rotations = euler.shape[0]
    refined_params = np.zeros(shape=(n_rotations, 7), dtype=np.float32)
    i = np.arange(n_rotations)

    if len(pc) > 1:
        refined_params[i] = 0
    else:
        pc_x0 = pc[0]
        refined_params[i] = _compute_params_xmap(
            euler[i],
            pc_x0,
            exp_data[i],
            master_north,
            master_south,
            npx,
            npy,
            scale,
            detector_data,
            method_kwargs,
        )
    return refined_params


def _compute_params_xmap(
    rotation,
    pc,
    exp,
    master_north,
    master_south,
    npx,
    npy,
    scale,
    detector_data,
    method_kwargs,
):
    args = (
        exp,
        master_north,
        master_south,
        npx,
        npy,
        scale,
        detector_data,
    )
    phi1_0 = rotation[..., 0]
    Phi_0 = rotation[..., 1]
    phi2_0 = rotation[..., 2]

    eu_x0 = np.array((phi1_0, Phi_0, phi2_0))
    print(eu_x0.shape)

    # TODO: Implement the other methods if needed
    soln = scipy.optimize.minimize(
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
    pcy = soln.x[4]
    pcz = soln.x[5]
    return np.array((score, phi1, Phi, phi2, pcx, pcy, pcz))


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
