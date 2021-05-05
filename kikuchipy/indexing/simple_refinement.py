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
from kikuchipy.signals import EBSD, LazyEBSD
from kikuchipy.indexing.similarity_metrics import ncc


class Refinement2:
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
    ):
        pass


###########################################################################################################################################################################################
###########################################################################################################################################################################################
# OLD OLD OLD OLD OLD OLD OLD OLD OLD
###########################################################################################################################################################################################
###########################################################################################################################################################################################

# import sys
#
# import cv2
# import dask.array as da
# from dask.diagnostics import ProgressBar
# import numpy as np
# from orix.quaternion import Rotation
# import scipy.optimize
# from kikuchipy.signals import (
#     _get_direction_cosines,
#     _get_lambert_interpolation_parameters,
# )
# from kikuchipy.indexing.similarity_metrics import ncc
#
# from kikuchipy.pattern import rescale_intensity
#
#
# class Refinement:
#     @staticmethod
#     def global_refine_test(
#         xmap, mp, exp, det, energy, xy_px=5, L_px=500, degs=0.5, compute=True
#     ):
#         # Convert to radians
#         rads = degs * (np.pi / 180)
#         ny = det.ncols
#         nx = det.nrows
#         pc = det.pc_emsoft()
#         xpc = pc[0][0]
#         ypc = pc[0][1]
#         L = pc[0][2]
#
#         # Can maybe do something like this down the road?
#         # x_star_lower = max(0, det.pcx - xy_px/nx)
#         # x_star_upper = min(1, det.pcx + xy_px/nx)
#         #
#         # y_star_lower = max(0, det.pcy - xy_px/ny)
#         # y_star_upper = min(1, det.pcy - xy_px/ny)
#
#         x_star_lower = ((xpc - xy_px) / nx) + 0.5
#         x_star_upper = ((xpc + xy_px) / nx) + 0.5
#
#         y_star_lower = -((ypc + xy_px) / ny) + 0.5
#         y_star_upper = -((ypc - xy_px) / ny) + 0.5
#
#         #       nydelta = ny * 90.55 # should be px_size
#         nydelta = (
#             ny * det.px_size
#         )  # px_size is 1 since Bruker so we can remove this
#         z_star_lower = (L - L_px) / nydelta
#         z_star_upper = (L + L_px) / nydelta
#
#         pc_bounds = [
#             (x_star_lower, x_star_upper),
#             (y_star_lower, y_star_upper),
#             (z_star_lower, z_star_upper),
#         ]
#         #        print(pc_bounds)
#         ncols = exp.data.shape[1]
#         rdata = xmap.rotations.data
#         # rdata = xmap.rotations.data[:10]  # Smaller dataset for testing
#         dtype = rdata.dtype
#         # TODO: Optimize chunks, currently hardcoded for SDSS dataset
#         # r_da = da.from_array(rdata, chunks=(11700, 1, 4))
#         # r_da = da.from_array(rdata, chunks=(10, 1, 4))
#         r_da = da.from_array(rdata, chunks=(1, 4))  # single Si mvp test
#
#         (
#             master_north,
#             master_south,
#             npx,
#             npy,
#             scale,
#         ) = _get_single_pattern_params(mp, det, energy)
#         exp_data = exp.data
#         exp_data = rescale_intensity(exp_data, dtype_out=np.float32)
#         refined_params = r_da.map_blocks(
#             _get_global_refined_params_chunk,
#             master_north=master_north,
#             master_south=master_south,
#             npx=npx,
#             npy=npy,
#             scale=scale,
#             exp=exp_data,
#             det=det,
#             pc_bounds=pc_bounds,
#             rads=rads,
#             ncols=ncols,
#             dtype_out=np.float32,
#             dtype=np.float32,
#         )
#         if compute:
#             with ProgressBar():
#                 print(
#                     f"Refining {xmap.rotations.shape[0]} rotations and PCs:",
#                     file=sys.stdout,
#                 )
#                 output = refined_params.compute()
#         else:
#             output = refined_params.visualize(
#                 filename="refinement_test.svg", rankdir="LR"
#             )
#         return output
#
#     @staticmethod
#     def refine_test(
#         method_name,
#         xmap,
#         mp,
#         exp,
#         det,
#         energy,
#         xy_px=5,
#         L_px=500,
#         degs=0.5,
#         compute=True,
#     ):
#         # Convert to radians
#         rads = degs * (np.pi / 180)
#         ny = det.ncols
#         nx = det.nrows
#         pc = det.pc_emsoft()
#         xpc = pc[0][0]
#         ypc = pc[0][1]
#         L = pc[0][2]
#
#         # Can maybe do something like this down the road?
#         # x_star_lower = max(0, det.pcx - xy_px/nx)
#         # x_star_upper = min(1, det.pcx + xy_px/nx)
#         #
#         # y_star_lower = max(0, det.pcy - xy_px/ny)
#         # y_star_upper = min(1, det.pcy - xy_px/ny)
#
#         x_star_lower = ((xpc - xy_px) / nx) + 0.5
#         x_star_upper = ((xpc + xy_px) / nx) + 0.5
#
#         y_star_lower = -((ypc + xy_px) / ny) + 0.5
#         y_star_upper = -((ypc - xy_px) / ny) + 0.5
#
#         #       nydelta = ny * 90.55 # should be px_size
#         nydelta = (
#             ny * det.px_size
#         )  # px_size is 1 since Bruker so we can remove this
#         z_star_lower = (L - L_px) / nydelta
#         z_star_upper = (L + L_px) / nydelta
#
#         pc_bounds = [
#             (x_star_lower, x_star_upper),
#             (y_star_lower, y_star_upper),
#             (z_star_lower, z_star_upper),
#         ]
#         #        print(pc_bounds)
#         ncols = exp.data.shape[1]
#         rdata = xmap.rotations.data
#         # rdata = xmap.rotations.data[:10]  # Smaller dataset for testing
#         dtype = rdata.dtype
#         # TODO: Optimize chunks, currently hardcoded for SDSS dataset
#         # r_da = da.from_array(rdata, chunks=(11700, 1, 4))
#         # r_da = da.from_array(rdata, chunks=(10, 1, 4))
#         r_da = da.from_array(rdata, chunks=(1, 4))  # single Si mvp test
#
#         (
#             master_north,
#             master_south,
#             npx,
#             npy,
#             scale,
#         ) = _get_single_pattern_params(mp, det, energy)
#         exp_data = exp.data
#         exp_data = rescale_intensity(exp_data, dtype_out=np.float32)
#         refined_params = r_da.map_blocks(
#             _get_local_refined_params_chunk,
#             master_north=master_north,
#             master_south=master_south,
#             npx=npx,
#             npy=npy,
#             scale=scale,
#             exp=exp_data,
#             det=det,
#             pc_bounds=pc_bounds,
#             rads=rads,
#             ncols=ncols,
#             methodname=method_name,
#             dtype_out=np.float32,
#             dtype=np.float32,
#         )
#         if compute:
#             with ProgressBar():
#                 print(
#                     f"Refining {xmap.rotations.shape[0]} rotations and PCs:",
#                     file=sys.stdout,
#                 )
#                 output = refined_params.compute()
#         else:
#             output = refined_params.visualize(
#                 filename="refinement_test.svg", rankdir="LR"
#             )
#         return output
#
#     @staticmethod
#     def refine_xmap(
#         xmap, mp, exp, det, energy, xy_px=5, L_px=500, degs=0.5, compute=True
#     ):
#         # Convert to radians
#         rads = degs * (np.pi / 180)
#         ny = det.ncols
#         nx = det.nrows
#         pc = det.pc_emsoft()
#         xpc = pc[0][0]
#         ypc = pc[0][1]
#         L = pc[0][2]
#
#         # Can maybe do something like this down the road?
#         # x_star_lower = max(0, det.pcx - xy_px/nx)
#         # x_star_upper = min(1, det.pcx + xy_px/nx)
#         #
#         # y_star_lower = max(0, det.pcy - xy_px/ny)
#         # y_star_upper = min(1, det.pcy - xy_px/ny)
#
#         x_star_lower = ((xpc - xy_px) / nx) + 0.5
#         x_star_upper = ((xpc + xy_px) / nx) + 0.5
#
#         y_star_lower = -((ypc + xy_px) / ny) + 0.5
#         y_star_upper = -((ypc - xy_px) / ny) + 0.5
#
#         #       nydelta = ny * 90.55 # should be px_size
#         nydelta = (
#             ny * det.px_size
#         )  # px_size is 1 since Bruker so we can remove this
#         z_star_lower = (L - L_px) / nydelta
#         z_star_upper = (L + L_px) / nydelta
#
#         pc_bounds = [
#             (x_star_lower, x_star_upper),
#             (y_star_lower, y_star_upper),
#             (z_star_lower, z_star_upper),
#         ]
#         #        print(pc_bounds)
#         ncols = exp.data.shape[1]
#         rdata = xmap.rotations.data
#         # rdata = xmap.rotations.data[:10]  # Smaller dataset for testing
#         dtype = rdata.dtype
#         # TODO: Optimize chunks, currently hardcoded for SDSS dataset
#         # r_da = da.from_array(rdata, chunks=(11700, 1, 4))
#         # r_da = da.from_array(rdata, chunks=(10, 1, 4))
#         r_da = da.from_array(rdata, chunks=(1, 4))  # single Si mvp test
#
#         (
#             master_north,
#             master_south,
#             npx,
#             npy,
#             scale,
#         ) = _get_single_pattern_params(mp, det, energy)
#         exp_data = exp.data
#         exp_data = rescale_intensity(exp_data, dtype_out=np.float32)
#         refined_params = r_da.map_blocks(
#             _get_refined_params_chunk,
#             master_north=master_north,
#             master_south=master_south,
#             npx=npx,
#             npy=npy,
#             scale=scale,
#             exp=exp_data,
#             det=det,
#             pc_bounds=pc_bounds,
#             rads=rads,
#             ncols=ncols,
#             dtype_out=np.float32,
#             dtype=np.float32,
#         )
#         if compute:
#             with ProgressBar():
#                 print(
#                     f"Refining {xmap.rotations.shape[0]} rotations and PCs:",
#                     file=sys.stdout,
#                 )
#                 output = refined_params.compute()
#         else:
#             output = refined_params.visualize(
#                 filename="refinement_test.svg", rankdir="LR"
#             )
#         return output
#
#
# def _shgo_objective_function(x, *args):
#     x_star = x[0]
#     y_star = x[1]
#     z_star = x[2]
#
#     # Passed in radians
#     # phi1 = x[3]
#     # Phi = x[4]
#     # phi2 = x[5]
#
#     experimental = args[0]
#     master_north = args[1]
#     master_south = args[2]
#     npx = args[3]
#     npy = args[4]
#     scale = args[5]
#     detector_data = args[6]
#
#     detector_ncols = detector_data[0]
#     detector_nrows = detector_data[1]
#     detector_px_size = detector_data[2]
#     detector_tilt = detector_data[3]
#     sample_tilt = detector_data[4]
#     from kikuchipy.detectors import EBSDDetector
#
#     detector = EBSDDetector(
#         shape=(detector_nrows, detector_ncols),
#         pc=(x_star, y_star, z_star),
#         tilt=detector_tilt,
#         sample_tilt=sample_tilt,
#         px_size=detector_px_size,
#         convention="bruker",
#     )
#
#     dc = _get_direction_cosines(detector)
#
#     # r = Rotation.from_euler((phi1, Phi, phi2))
#     r = Rotation.from_euler(
#         (2.32652389, 1.54810705, 3.10319541), convention="bunge"
#     )
#     sim_pattern = _simulate_single_pattern(
#         r,
#         dc,
#         master_north,
#         master_south,
#         npx,
#         npy,
#         scale,
#     )
#
#     result = cv2.matchTemplate(experimental, sim_pattern, cv2.TM_CCOEFF_NORMED)
#     # print(result)
#     return -result[0][0]
#     return -ncc(experimental, sim_pattern)
#
#
# def _get_refined_params_chunk(
#     r,
#     master_north,
#     master_south,
#     npx,
#     npy,
#     scale,
#     exp,
#     det,
#     pc_bounds,
#     rads,
#     ncols,
#     dtype_out=np.float32,
# ):
#     rotations = Rotation(r)
#     rotations_shape = r.shape
#     refined_params = np.empty(
#         shape=(rotations_shape[0],) + (3,), dtype=dtype_out
#     )
#     for i in np.ndindex(rotations_shape[0]):
#         index = i[0]
#         # TODO: Fix this mess
#         if len(exp.data.shape) == 2:
#             exp_data = exp
#         else:
#             row = index // ncols
#             col = index % ncols
#             exp_data = exp[row, col]
#         rotation = rotations[index]
#         best = rotation[0]
#         r_euler = best.to_euler()
#         phi1_guess = r_euler[..., 0]
#         Phi_guess = r_euler[..., 1]
#         phi2_guess = r_euler[..., 2]
#         rotation_bounds = [
#             (phi1_guess[0] - rads, phi1_guess[0] + rads),
#             (Phi_guess[0] - rads, Phi_guess[0] + rads),
#             (phi2_guess[0] - rads, phi2_guess[0] + rads),
#         ]
#         # Could probably speed up here as well
#         bounds = pc_bounds + rotation_bounds
#         #       print(bounds)
#         bounds = pc_bounds
#         bounds = [
#             (0.523, 0.525),
#             (0.154, 0.157),
#             (0.5050, 0.5150),
#         ]  # Very good result
#         bounds = [(0.52, 0.53), (0.15, 0.16), (0.5, 0.52)]  # Very good result
#         bounds = [(0.515, 0.535), (0.145, 0.165), (0.45, 0.525)]  # Little worse
#         bounds = [(0.51, 0.54), (0.14, 0.17), (0.4, 0.53)]  # Little worse
#         bounds = [(0.50, 0.55), (0.13, 0.18), (0.3, 0.63)]  # Little worse
#         bounds = [
#             (0.50, 0.55),
#             (0.13, 0.18),
#             (0.5, 0.52),
#         ]  # Pretty good result! Assume chaning z* matters a lot!
#         bounds = [
#             (0, 1),
#             (0, 1),
#             (0.5, 0.52),
#         ]  # Too large bounds places them in the middle by default?
#         bounds = [
#             (0.5249 * 0.9, 0.5249 * 1.1),
#             (0.156 * 0.9, 0.156 * 1.1),
#             (0.51 * 0.9, 0.51 * 1.1),
#         ]  # Very Good result nvm, just middle which is key
#         bounds = [
#             (0.31, 0.73),
#             (0.106, 0.196),
#             (0.48, 0.52),
#         ]  # Very Good result
#         bounds = [(0.11, 0.73), (0.106, 0.196), (0.48, 0.52)]  # Awful
#         detector_data = [
#             det.ncols,
#             det.nrows,
#             det.px_size,
#             det.tilt,
#             det.sample_tilt,
#         ]
#         args = (
#             exp_data,
#             master_north,
#             master_south,
#             npx,
#             npy,
#             scale,
#             detector_data,
#         )
#         optimized = scipy.optimize.shgo(
#             _shgo_objective_function, bounds, args=args
#         )
#         x_star = optimized.x[0]
#         y_star = optimized.x[1]
#         z_star = optimized.x[2]
#
#         refined_params[index] = np.array((x_star, y_star, z_star))
#         return optimized
#     return refined_params
#
#
# def _get_single_pattern_params(mp, detector, energy):
#     # This method is already a part of the EBSDMasterPattern.get_patterns so
#     # it could probably replace it?
#     if mp.projection != "lambert":
#         raise NotImplementedError(
#             "Master pattern must be in the square Lambert projection"
#         )
#     if len(detector.pc) > 1:
#         raise NotImplementedError(
#             "Detector must have exactly one projection center"
#         )
#
#     # Get the master pattern arrays created by a desired energy
#     north_slice = ()
#     if "energy" in [i.name for i in mp.axes_manager.navigation_axes]:
#         energies = mp.axes_manager["energy"].axis
#         north_slice += ((np.abs(energies - energy)).argmin(),)
#     south_slice = north_slice
#     if mp.hemisphere == "both":
#         north_slice = (0,) + north_slice
#         south_slice = (1,) + south_slice
#     elif not mp.phase.point_group.contains_inversion:
#         raise AttributeError(
#             "For crystals of point groups without inversion symmetry, like "
#             f"the current {mp.phase.point_group.name}, both hemispheres "
#             "must be present in the master pattern signal"
#         )
#     master_north = mp.data[north_slice]
#     master_south = mp.data[south_slice]
#     npx, npy = mp.axes_manager.signal_shape
#     scale = (npx - 1) / 2
#
#     return master_north, master_south, npx, npy, scale
#
#
# def _simulate_single_pattern(
#     rotation,
#     dc,
#     master_north,
#     master_south,
#     npx,
#     npy,
#     scale,
# ):
#     rotated_dc = rotation * dc
#     (
#         nii,
#         nij,
#         niip,
#         nijp,
#         di,
#         dj,
#         dim,
#         djm,
#     ) = _get_lambert_interpolation_parameters(
#         rotated_direction_cosines=rotated_dc,
#         npx=npx,
#         npy=npy,
#         scale=scale,
#     )
#     pattern = np.where(
#         rotated_dc.z >= 0,
#         (
#             master_north[nii, nij] * dim * djm
#             + master_north[niip, nij] * di * djm
#             + master_north[nii, nijp] * dim * dj
#             + master_north[niip, nijp] * di * dj
#         ),
#         (
#             master_south[nii, nij] * dim * djm
#             + master_south[niip, nij] * di * djm
#             + master_south[nii, nijp] * dim * dj
#             + master_south[niip, nijp] * di * dj
#         ),
#     )
#     return pattern.astype(np.float32)
#
#
# def local_objective_func(x, *args):
#     x_star = x[0]
#     y_star = x[1]
#     z_star = x[2]
#
#     # Passed in radians
#     # phi1 = x[3]
#     # Phi = x[4]
#     # phi2 = x[5]
#
#     experimental = args[0]
#     master_north = args[1]
#     master_south = args[2]
#     npx = args[3]
#     npy = args[4]
#     scale = args[5]
#     detector_data = args[6]
#
#     detector_ncols = detector_data[0]
#     detector_nrows = detector_data[1]
#     detector_px_size = detector_data[2]
#     detector_tilt = detector_data[3]
#     sample_tilt = detector_data[4]
#     from kikuchipy.detectors import EBSDDetector
#
#     detector = EBSDDetector(
#         shape=(detector_nrows, detector_ncols),
#         pc=(x_star, y_star, z_star),
#         tilt=detector_tilt,
#         sample_tilt=sample_tilt,
#         px_size=detector_px_size,
#         convention="bruker",
#     )
#
#     dc = _get_direction_cosines(detector)
#
#     # r = Rotation.from_euler((phi1, Phi, phi2))
#     r = Rotation.from_euler(
#         (2.32652389, 1.54810705, 3.10319541), convention="bunge"
#     )
#     sim_pattern = _simulate_single_pattern(
#         r,
#         dc,
#         master_north,
#         master_south,
#         npx,
#         npy,
#         scale,
#     )
#
#     result = cv2.matchTemplate(experimental, sim_pattern, cv2.TM_CCOEFF_NORMED)
#     return -result[0][0]
#     return -ncc(experimental, sim_pattern)
#
#
# def _get_local_refined_params_chunk(
#     r,
#     master_north,
#     master_south,
#     npx,
#     npy,
#     scale,
#     exp,
#     det,
#     pc_bounds,
#     rads,
#     ncols,
#     methodname,
#     dtype_out=np.float32,
# ):
#     rotations = Rotation(r)
#     rotations_shape = r.shape
#     refined_params = np.empty(
#         shape=(rotations_shape[0],) + (3,), dtype=dtype_out
#     )
#     for i in np.ndindex(rotations_shape[0]):
#         index = i[0]
#         # TODO: Fix this mess
#         if len(exp.data.shape) == 2:
#             exp_data = exp
#         else:
#             row = index // ncols
#             col = index % ncols
#             exp_data = exp[row, col]
#         rotation = rotations[index]
#         best = rotation[0]
#         r_euler = best.to_euler()
#         phi1_guess = r_euler[..., 0]
#         Phi_guess = r_euler[..., 1]
#         phi2_guess = r_euler[..., 2]
#         xa = pc_bounds[0]
#         xb = pc_bounds[1]
#         xc = pc_bounds[2]
#         x0 = np.array((xa, xb, xc))
#         rotation_bounds = [
#             (phi1_guess[0] - rads, phi1_guess[0] + rads),
#             (Phi_guess[0] - rads, Phi_guess[0] + rads),
#             (phi2_guess[0] - rads, phi2_guess[0] + rads),
#         ]
#         # Could probably speed up here as well
#         # bounds = pc_bounds + rotation_bounds
#         #       print(bounds)
#         # bounds = pc_bounds
#         # bounds = [(0.523, 0.525), (0.154, 0.157), (0.5050, 0.5150)]  # Very good result
#         # bounds = [(0.52, 0.53), (0.15, 0.16), (0.5, 0.52)] # Very good result
#         # bounds = [(0.515, 0.535), (0.145, 0.165), (0.45, 0.525)] # Little worse
#         # bounds = [(0.51, 0.54), (0.14, 0.17), (0.4, 0.53)] # Little worse
#         # bounds = [(0.50, 0.55), (0.13, 0.18), (0.3, 0.63)] # Little worse
#         # bounds = [(0.50, 0.55), (0.13, 0.18), (0.5, 0.52)] # Pretty good result! Assume chaning z* matters a lot!
#         # bounds = [(0, 1), (0, 1), (0.5, 0.52)] # Too large bounds places them in the middle by default?
#         # bounds = [(0.5249*0.9, 0.5249*1.1), (0.156*0.9, 0.156*1.1), (0.51*0.9, 0.51*1.1)] # Very Good result nvm, just middle which is key
#         # Initial guess: 0.512, 0.139, 0.498
#         x0 = [0.512, 0.139, 0.498]
#         bounds = [
#             (0.31, 0.73),
#             (0.106, 0.196),
#             (0.48, 0.52),
#         ]  # Very Good result
#         # bounds = [(0.11, 0.73), (0.106, 0.196), (0.48, 0.52)] # Awful
#         detector_data = [
#             det.ncols,
#             det.nrows,
#             det.px_size,
#             det.tilt,
#             det.sample_tilt,
#         ]
#         args = (
#             exp_data,
#             master_north,
#             master_south,
#             npx,
#             npy,
#             scale,
#             detector_data,
#         )
#         optimized = scipy.optimize.minimize(
#             local_objective_func,
#             x0=x0,
#             bounds=bounds,
#             args=args,
#             method=methodname,
#         )
#         x_star = optimized.x[0]
#         y_star = optimized.x[1]
#         z_star = optimized.x[2]
#
#         refined_params[index] = np.array((x_star, y_star, z_star))
#         return optimized
#     return refined_params
#
#
# def _get_global_refined_params_chunk(
#     r,
#     master_north,
#     master_south,
#     npx,
#     npy,
#     scale,
#     exp,
#     det,
#     pc_bounds,
#     rads,
#     ncols,
#     dtype_out=np.float32,
# ):
#     rotations = Rotation(r)
#     rotations_shape = r.shape
#     refined_params = np.empty(
#         shape=(rotations_shape[0],) + (3,), dtype=dtype_out
#     )
#     for i in np.ndindex(rotations_shape[0]):
#         index = i[0]
#         # TODO: Fix this mess
#         if len(exp.data.shape) == 2:
#             exp_data = exp
#         else:
#             row = index // ncols
#             col = index % ncols
#             exp_data = exp[row, col]
#         rotation = rotations[index]
#         best = rotation[0]
#         r_euler = best.to_euler()
#         phi1_guess = r_euler[..., 0]
#         Phi_guess = r_euler[..., 1]
#         phi2_guess = r_euler[..., 2]
#         xa = pc_bounds[0]
#         xb = pc_bounds[1]
#         xc = pc_bounds[2]
#         x0 = np.array((xa, xb, xc))
#         rotation_bounds = [
#             (phi1_guess[0] - rads, phi1_guess[0] + rads),
#             (Phi_guess[0] - rads, Phi_guess[0] + rads),
#             (phi2_guess[0] - rads, phi2_guess[0] + rads),
#         ]
#         # Could probably speed up here as well
#         # bounds = pc_bounds + rotation_bounds
#         #       print(bounds)
#         # bounds = pc_bounds
#         # bounds = [(0.523, 0.525), (0.154, 0.157), (0.5050, 0.5150)]  # Very good result
#         # bounds = [(0.52, 0.53), (0.15, 0.16), (0.5, 0.52)] # Very good result
#         # bounds = [(0.515, 0.535), (0.145, 0.165), (0.45, 0.525)] # Little worse
#         # bounds = [(0.51, 0.54), (0.14, 0.17), (0.4, 0.53)] # Little worse
#         # bounds = [(0.50, 0.55), (0.13, 0.18), (0.3, 0.63)] # Little worse
#         # bounds = [(0.50, 0.55), (0.13, 0.18), (0.5, 0.52)] # Pretty good result! Assume chaning z* matters a lot!
#         # bounds = [(0, 1), (0, 1), (0.5, 0.52)] # Too large bounds places them in the middle by default?
#         # bounds = [(0.5249*0.9, 0.5249*1.1), (0.156*0.9, 0.156*1.1), (0.51*0.9, 0.51*1.1)] # Very Good result nvm, just middle which is key
#         bounds = [(0.31, 0.73), (0.100, 0.3), (0.48, 0.52)]  # Very Good result
#         # bounds = [(0.11, 0.73), (0.106, 0.196), (0.48, 0.52)] # Awful
#         detector_data = [
#             det.ncols,
#             det.nrows,
#             det.px_size,
#             det.tilt,
#             det.sample_tilt,
#         ]
#         args = (
#             exp_data,
#             master_north,
#             master_south,
#             npx,
#             npy,
#             scale,
#             detector_data,
#         )
#         optimized = scipy.optimize.minimize(
#             _shgo_objective_function,
#             bounds=bounds,
#             args=args,
#         )
#         x_star = optimized.x[0]
#         y_star = optimized.x[1]
#         z_star = optimized.x[2]
#
#         refined_params[index] = np.array((x_star, y_star, z_star))
#         return optimized
#     return refined_params
