import sys

import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
from orix.quaternion import Rotation
import scipy.optimize
from kikuchipy.indexing.similarity_metrics import ncc
from kikuchipy.signals import EBSDMasterPattern
import graphviz


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
        r_da = da.from_array(rdata, chunks=(11700, 1, 4))
        mp_data = mp.data
        exp_data = exp.data
        refined_params = r_da.map_blocks(
            _get_refined_params_chunk,
            mp=mp_data,
            exp=exp_data,
            energy=energy,
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
    master_pattern_data = args[1]
    detector = args[2]
    energy = args[3]
    # We could probably speed this up quite a bit by using Bruker PC all the way
    # TODO: Use Bruker PC instead of EMsoft PC
    detector.pcx = 0.5 + (
        -xpc / (detector.ncols * detector.binning)
    )  # Assumes EMsoft v5 PC
    detector.pcy = 0.5 - (ypc / (detector.nrows * detector.binning))
    detector.pcz = L / (detector.nrows * detector.px_size * detector.binning)
    r = Rotation.from_euler(np.radians(np.array((phi1, Phi, phi2))))
    master_pattern = EBSDMasterPattern(
        master_pattern_data,
        projection="lambert",
        hemisphere="both",
        energy=energy,
    )
    # TODO: Create single pattern function that works on just arrays
    sim_pattern = master_pattern.get_patterns(r, detector, energy, compute=True)
    return -ncc(experimental, sim_pattern.inav[0].data)


def _get_refined_params_chunk(
    r, mp, exp, energy, det, pc_bounds, degs, ncols, dtype_out=np.float32
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
        args = (exp_data, mp, det, energy)
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
