# Copyright 2019-2022 The kikuchipy developers
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

from typing import Optional, Tuple, Union

import dask.array as da
from orix.crystal_map import CrystalMap
from orix.quaternion import Rotation
import numpy as np
import scipy

from kikuchipy.pattern import rescale_intensity


SUPPORTED_OPTIMIZATION_METHODS = {
    # Local
    "minimize": {
        "type": "local",
        "supports_bounds": True,
        "package": "scipy",
    },
    "ln_neldermead": {
        "type": "local",
        "supports_bounds": True,
        "package": "nlopt",
    },
    # Global
    "basinhopping": {
        "type": "global",
        "supports_bounds": False,
        "package": "scipy",
    },
    "differential_evolution": {
        "type": "global",
        "supports_bounds": True,
        "package": "scipy",
    },
    "dual_annealing": {
        "type": "global",
        "supports_bounds": True,
        "package": "scipy",
    },
    "shgo": {
        "type": "global",
        "supports_bounds": True,
        "package": "scipy",
    },
}


class _RefinementSetup:
    mode: str
    # Data parameters
    data_shape: tuple
    nav_shape: tuple
    nav_size: int
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
        rtol: float,
        method: Optional[str] = None,
        method_kwargs: Optional[dict] = None,
        initial_step: Optional[float] = None,
        maxeval: Optional[int] = None,
        signal_mask: Optional[np.ndarray] = None,
    ):
        # "ori", "pc" or "ori_pc"
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

        # Extra relevant information from pattern array
        self.solver_kwargs["rescale"] = patterns.dtype == np.float32
        self.chunks = patterns.chunksize[:-1] + (-1,)

        self.nav_shape = xmap.shape
        self.nav_size = xmap.size

        # Extract relevant data from the crystal map
        if xmap.rotations_per_point > 1:
            rot = xmap.rotations[:, 0]
        else:
            rot = xmap.rotations
        if self.mode == "pc":
            rot = rot.data
            rot = rot.reshape(self.nav_shape + (4,))
        else:
            rot = rot.to_euler()
            rot = rot.reshape(self.nav_shape + (3,))
        self.rotations_array = da.from_array(rot, chunks=self.chunks)

        # Extract relevant data from the detector
        pc = detector.pc
        self.unique_pc = np.prod(detector.navigation_shape) != 1 and self.nav_size > 1
        dtype = np.float64
        pc_shape = self.nav_shape + (3,)
        if self.unique_pc:
            # Patterns have been indexed with varying PCs, so we use
            # these as the starting point for every pattern
            pc = pc.astype(dtype).reshape(pc_shape)
        else:
            # Patterns have been indexed with the same PC, so we use
            # this as the starting point for every pattern
            pc = np.full(pc_shape, pc[0], dtype=dtype)
        self.pc_array = da.from_array(pc, chunks=self.chunks)

        if mode == "ori_pc":
            self.rotations_pc_array = da.dstack((self.rotations_array, self.pc_array))

        # Set keyword arguments passed to Dask
        self.map_blocks_kwargs.update(
            {
                "drop_axis": (patterns.ndim - 1,),
                "new_axis": (len(self.nav_shape),),
                "dtype": np.float64,
            }
        )

    def set_optimization_parameters(
        self,
        rtol: float,
        method: Optional[str] = None,
        method_kwargs: Optional[dict] = None,
        initial_step: Optional[float] = None,
        maxeval: Optional[int] = None,
    ) -> None:
        method = method.lower()
        supported_methods = list(SUPPORTED_OPTIMIZATION_METHODS)

        if method not in supported_methods:
            raise ValueError(
                f"Method {method} not in the list of supported methods "
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
                raise ImportError(
                    f"Package `nlopt`, required for method {method_upper}, is not "
                    "installed"
                )

            import nlopt

            dimensions = {"ori": 3, "pc": 3, "ori_pc": 6}
            opt = nlopt.opt(method_upper, dimensions[self.mode])

            opt.set_ftol_rel(rtol)

            if initial_step is not None:
                initial_step = np.asarray(initial_step)
                n_initial_steps = {"ori": 1, "pc": 1, "ori_pc": 2}
                if initial_step.size != n_initial_steps[self.mode]:
                    raise ValueError(
                        "`initial_step` must be a single number when refining "
                        "orientations or PCs and a list of two numbers when refining "
                        "both"
                    )
                if self.mode == "ori_pc":
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
                method_kwargs["method"] = self.method_name = "Nelder-Mead"
            else:
                self.method_name = method_kwargs["method"]

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
        self, trust_region: Union[tuple, list, np.ndarray]
    ) -> Tuple[da.Array, da.Array]:
        eu_lower = 3 * [0]
        eu_upper = [2 * np.pi, np.pi, 2 * np.pi]
        pc_lower = 3 * [-3]
        pc_upper = 3 * [3]

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
        upper_bounds = da.fmax(data_to_optimize + trust_region, upper_abs)

        return lower_bounds, upper_bounds

    def get_info_message(self, trust_region) -> str:
        msg = (
            "Refinement information:\n"
            f"  Method: {self.package} ({self.optimization_type}) - {self.method_name}"
        )

        if self.supports_bounds:
            tr_str = np.array_str(np.asarray(trust_region), precision=5)
            msg += "\n  Trust region: " + tr_str

        if self.package == "scipy":
            msg += f"\n Keyword arguments passed to method: {self.solver_kwargs['method_kwargs']}"
        else:
            opt = self.map_blocks_kwargs["opt"]
            msg += f"\n  Relative tolerance: {opt.get_ftol_rel()}"
            if self.initial_step:
                msg += f"\n  Initial step(s): {self.initial_step}"
            if self.maxeval:
                msg += f"\n  Max. function evaulations: {self.maxeval}"

        return msg


def _get_master_pattern_data(
    master_pattern: "EBSDMasterPattern", energy: Union[int, float]
) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    """Return the upper and lower hemispheres along with their shape.

    Parameters
    ----------
    master_pattern
      Master pattern in the square Lambert projection.
    energy
        Accelerating voltage of the electron beam in kV specifying which
        master pattern energy to use during projection of simulated
        patterns.

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
