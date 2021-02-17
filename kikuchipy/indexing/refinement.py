import sys

import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
from orix.quaternion import Rotation
import scipy.optimize
from kikuchipy.indexing.similarity_metrics import ncc
from kikuchipy.signals import (
    EBSDMasterPattern,
    _get_direction_cosines,
    _get_lambert_interpolation_parameters,
)


class Refinement:
    @staticmethod
    def refine_xmap(
        xmap, mp, exp, det, energy, xy_px=5, L_px=500, degs=0.5, compute=True
    ):
        pc = det.pc_emsoft()
        xpc_guess = pc[0][0]
        ypc_guess = pc[0][1]
        L_guess = pc[0][2]
        pc_bounds = [
            (xpc_guess - xy_px, xpc_guess + xy_px),
            (ypc_guess - xy_px, ypc_guess + xy_px),
            (L_guess - L_px, L_guess + L_px),
        ]
        ncols = exp.data.shape[1]
        rdata = xmap.rotations.data
        rdata = xmap.rotations.data[:10]  # Smaller dataset for testing
        dtype = rdata.dtype
        # TODO: Optimize chunks, currently hardcoded for SDSS dataset
        # r_da = da.from_array(rdata, chunks=(11700, 1, 4))
        r_da = da.from_array(rdata, chunks=(10, 1, 4))

        (
            master_north,
            master_south,
            dc,
            npx,
            npy,
            scale,
        ) = _get_single_pattern_params(mp, det, energy)

        exp_data = exp.data
        refined_params = r_da.map_blocks(
            _get_refined_params_chunk,
            master_north=master_north,
            master_south=master_south,
            dc=dc,
            npx=npx,
            npy=npy,
            scale=scale,
            exp=exp_data,
            det=det,
            pc_bounds=pc_bounds,
            degs=degs,
            ncols=ncols,
            dtype_out=dtype,
            dtype=dtype,
        )
        if compute:
            with ProgressBar():
                print(
                    f"Refining {xmap.rotations.shape[0]} rotations and PCs:",
                    file=sys.stdout,
                )
                output = refined_params.compute()
        else:
            output = refined_params.visualize(
                filename="refinement_test.svg", rankdir="LR"
            )
        return output


def _shgo_objective_function(x, *args):
    xpc = x[0]
    ypc = x[1]
    L = x[2]
    phi1 = x[3]
    Phi = x[4]
    phi2 = x[5]
    experimental = args[0]
    master_north = args[1]
    master_south = args[2]
    dc = args[3]
    npx = args[4]
    npy = args[5]
    scale = args[6]
    detector = args[7]
    # We could probably speed this up quite a bit by using Bruker PC all the way
    # TODO: Use Bruker PC instead of EMsoft PC
    detector.pcx = 0.5 + (
        -xpc / (detector.ncols * detector.binning)
    )  # Assumes EMsoft v5 PC
    detector.pcy = 0.5 - (ypc / (detector.nrows * detector.binning))
    detector.pcz = L / (detector.nrows * detector.px_size * detector.binning)
    r = Rotation.from_euler(np.radians(np.array((phi1, Phi, phi2))))
    sim_pattern = _simulate_single_pattern(
        r, dc, master_north, master_south, npx, npy, scale
    )
    return -ncc(experimental, sim_pattern)


def _get_refined_params_chunk(
    r,
    master_north,
    master_south,
    dc,
    npx,
    npy,
    scale,
    exp,
    det,
    pc_bounds,
    degs,
    ncols,
    dtype_out=np.float32,
):
    rotations = Rotation(r)
    rotations_shape = r.shape
    refined_params = np.empty(
        shape=(rotations_shape[0],) + (6,), dtype=dtype_out
    )
    for i in np.ndindex(rotations_shape[0]):
        index = i[0]
        row = index // ncols
        col = index % ncols
        exp_data = exp[row, col]
        rotation = rotations[index]
        best = rotation[0]
        r_euler = best.to_euler()
        phi1_guess = np.rad2deg(r_euler[..., 0])
        Phi_guess = np.rad2deg(r_euler[..., 1])
        phi2_guess = np.rad2deg(r_euler[..., 2])
        rotation_bounds = [
            (phi1_guess[0] - degs, phi1_guess[0] + degs),
            (Phi_guess[0] - degs, Phi_guess[0] + degs),
            (phi2_guess[0] - degs, phi2_guess[0] + degs),
        ]
        # Could probably speed up here as well
        bounds = pc_bounds + rotation_bounds
        args = (exp_data, master_north, master_south, dc, npx, npy, scale, det)
        optimized = scipy.optimize.shgo(
            _shgo_objective_function, bounds, args=args
        )
        xpc = optimized.x[0]
        ypc = optimized.x[1]
        zpc = optimized.x[2]
        phi1 = optimized.x[3]
        Phi = optimized.x[4]
        phi2 = optimized.x[5]
        refined_params[index] = np.array((xpc, ypc, zpc, phi1, Phi, phi2))
    return refined_params


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
    dc = _get_direction_cosines(detector)
    npx, npy = mp.axes_manager.signal_shape
    scale = (npx - 1) / 2

    return master_north, master_south, dc, npx, npy, scale


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
    return pattern
