# Copyright 2019-2023 The kikuchipy developers
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

"""Solvers for the refinement of crystal orientations and projection
centers by optimizing the similarity between experimental and simulated
patterns.
"""

from typing import Callable, Optional, Tuple, Union

from numba import njit
import numpy as np

from kikuchipy.indexing._refinement._objective_functions import (
    _refine_orientation_objective_function,
    _refine_orientation_pc_objective_function,
    _refine_pc_objective_function,
)
from kikuchipy.indexing._refinement import SUPPORTED_OPTIMIZATION_METHODS
from kikuchipy.pattern._pattern import (
    _rescale_without_min_max_1d_float32,
    _zero_mean_sum_square_1d_float32,
)
from kikuchipy.signals.util._master_pattern import _get_direction_cosines_for_fixed_pc


@njit(cache=True, nogil=True, fastmath=True)
def _prepare_pattern(pattern: np.ndarray, rescale: bool) -> Tuple[np.ndarray, float]:
    """Prepare experimental pattern.

    Parameters
    ----------
    pattern
        Experimental pattern.
    rescale
        Whether to rescale pattern.

    Returns
    -------
    prepared_pattern
        Prepared pattern.
    squared_norm
        Squared norm of the centered pattern.
    """
    pattern = pattern.astype(np.float32)
    if rescale:
        pattern = _rescale_without_min_max_1d_float32(pattern)
    prepared_pattern, squared_norm = _zero_mean_sum_square_1d_float32(pattern)
    return prepared_pattern, squared_norm


# --------------------------- SciPy solvers -------------------------- #


def _refine_orientation_solver_scipy(
    pattern: np.ndarray,
    rotation: np.ndarray,
    bounds: np.ndarray,
    signal_mask: np.ndarray,
    rescale: bool,
    method: Callable,
    method_kwargs: dict,
    trust_region_passed: bool,
    fixed_parameters: Tuple[np.ndarray, np.ndarray, int, int, float],
    direction_cosines: Optional[np.ndarray] = None,
    pcx: Optional[float] = None,
    pcy: Optional[float] = None,
    pcz: Optional[float] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    tilt: Optional[float] = None,
    azimuthal: Optional[float] = None,
    sample_tilt: Optional[float] = None,
    n_pseudo_symmetry_ops: int = 0,
) -> Union[
    Tuple[float, int, float, float, float],
    Tuple[float, int, int, float, float, float],
]:
    """Maximize the similarity between an experimental pattern and a
    projected simulated pattern by optimizing the orientation
    (Rodrigues-Frank vector) used in the projection.

    Parameters
    ----------
    pattern
        Flattened experimental pattern.
    rotation
        Rodrigues-Frank vector components (Rx, Ry, Rz), unscaled.
    signal_mask
        Boolean mask equal to the experimental patterns' detector shape
        ``(n rows, n columns)``, where only pixels equal to ``False``
        are matched.
    rescale
        Whether pattern intensities must be rescaled to [-1, 1] and the
        data type set to 32-bit floats.
    method
        A supported :mod:`scipy.optimize` function. See ``method``
        parameter in :meth:`kikuchipy.signals.EBSD.refine_orientation`.
    method_kwargs
        Keyword arguments passed to the ``method`` function. For the
        list of possible arguments, see the SciPy documentation.
    fixed_parameters
        Fixed parameters used in the projection.
    direction_cosines
        Vector array of shape (n pixels, 3) and data type 32-bit floats.
        If not given, ``pcx``, ``pcy``, ``pcz``, ``nrows``, ``ncols``,
        ``tilt``, ``azimuthal``, and ``sample_tilt`` must be passed.
    pcx
        Projection center (PC) x coordinate. Must be passed if
        ``direction_cosines`` is not given.
    pcy
        PC y coordinate. Must be passed if ``direction_cosines`` is not
        given.
    pcz
        PC z coordinate. Must be passed if ``direction_cosines`` is not
        given.
    nrows
        Number of detector rows. Must be passed if ``direction_cosines``
        is not given.
    ncols
        Number of detector columns. Must be passed if
        ``direction_cosines`` is not given.
    tilt
        Detector tilt from horizontal in degrees. Must be passed if
        ``direction_cosines`` is not given.
    azimuthal
        Sample tilt about the sample RD axis in degrees. Must be passed
        if ``direction_cosines`` is not given.
    sample_tilt
        Sample tilt from horizontal in degrees. Must be passed if
        ``direction_cosines`` is not given.
    n_pseudo_symmetry_ops
        Number of pseudo-symmetry operators. Default is 0.

    Returns
    -------
    score
        Highest normalized cross-correlation score.
    num_evals
        Number of optimization evaluations.
    phi1, Phi, phi2
        Optimized orientation (Euler angles) in radians.
    """
    pattern, squared_norm = _prepare_pattern(pattern, rescale)

    if direction_cosines is None:
        direction_cosines = _get_direction_cosines_for_fixed_pc(
            pcx=pcx,
            pcy=pcy,
            pcz=pcz,
            nrows=nrows,
            ncols=ncols,
            tilt=tilt,
            azimuthal=azimuthal,
            sample_tilt=sample_tilt,
            signal_mask=signal_mask,
        )

    params = (pattern,) + (direction_cosines,) + fixed_parameters + (squared_norm,)
    method_name = method.__name__

    if n_pseudo_symmetry_ops == 0:
        if method_name == "minimize":
            if trust_region_passed:
                method_kwargs["bounds"] = bounds[0]
            res = method(
                fun=_refine_orientation_objective_function,
                x0=rotation[0],
                args=params,
                **method_kwargs,
            )
        elif SUPPORTED_OPTIMIZATION_METHODS[method_name]["supports_bounds"]:
            res = method(
                func=_refine_orientation_objective_function,
                args=params,
                bounds=bounds[0],
                **method_kwargs,
            )
        else:  # Is always "basinhopping", due to prior check of method name
            method_kwargs["minimizer_kwargs"].update(args=params)
            res = method(
                func=_refine_orientation_objective_function,
                x0=rotation[0],
                **method_kwargs,
            )

        ncc = 1 - res.fun
        num_evals = res.nfev
        phi1, Phi, phi2 = res.x

        return ncc, num_evals, phi1, Phi, phi2
    else:
        res_list = []
        for i in range(n_pseudo_symmetry_ops + 1):
            if method_name == "minimize":
                if trust_region_passed:
                    method_kwargs["bounds"] = bounds[i]
                res = method(
                    fun=_refine_orientation_objective_function,
                    x0=rotation[i],
                    args=params,
                    **method_kwargs,
                )
            elif SUPPORTED_OPTIMIZATION_METHODS[method_name]["supports_bounds"]:
                res = method(
                    func=_refine_orientation_objective_function,
                    args=params,
                    bounds=bounds[i],
                    **method_kwargs,
                )
            else:  # Is always "basinhopping", due to prior check of method name
                method_kwargs["minimizer_kwargs"].update(args=params)
                res = method(
                    func=_refine_orientation_objective_function,
                    x0=rotation[i],
                    **method_kwargs,
                )

            res_list.append(res)

        ncc_all = [1 - res.fun for res in res_list]
        best_idx = int(np.argmax(ncc_all))

        best_res = res_list[best_idx]
        ncc = ncc_all[best_idx]
        num_evals = best_res.nfev
        phi1, Phi, phi2 = best_res.x

        return ncc, num_evals, phi1, Phi, phi2, best_idx


def _refine_pc_solver_scipy(
    pattern: np.ndarray,
    rotation: np.ndarray,
    pc: np.ndarray,
    bounds: np.ndarray,
    rescale: bool,
    method: Callable,
    method_kwargs: dict,
    fixed_parameters: tuple,
    trust_region_passed: bool,
) -> Tuple[float, int, float, float, float]:
    """Maximize the similarity between an experimental pattern and a
    projected simulated pattern by optimizing the projection center (PC)
    parameters used in the projection.

    Parameters
    ----------
    pattern
        Experimental pattern of shape (nrows, ncols).
    rotation
        Rotation as a quaternion array with shape (4,).
    pc
        Projection center (PC) coordinates (PCx, PCy, PCz).
    rescale
        Whether pattern intensities must be rescaled to [-1, 1] and data
        type 32-bit floats.
    method
        A supported :mod:`scipy.optimize` function. See `method`
        parameter in
        :meth:`kikuchipy.signals.EBSD.refine_projection_center`.
    method_kwargs
        Keyword arguments passed to the ``method`` function. For the
        list of possible arguments, see the SciPy documentation.
    fixed_parameters
        Fixed parameters used in the projection.
    trust_region_passed
        Whether ``trust_region`` was passed.

    Returns
    -------
    score
        Highest normalized cross-correlation score.
    num_evals
        Number of optimization evaluations.
    pcx_refined, pcy_refined, pcz_refined
        Optimized PC parameters in the Bruker convention.
    """
    pattern, squared_norm = _prepare_pattern(pattern, rescale)

    params = (pattern,) + (rotation,) + fixed_parameters + (squared_norm,)
    method_name = method.__name__

    if method_name == "minimize":
        if trust_region_passed:
            method_kwargs["bounds"] = bounds
        res = method(
            fun=_refine_pc_objective_function,
            x0=pc,
            args=params,
            **method_kwargs,
        )
    elif SUPPORTED_OPTIMIZATION_METHODS[method_name]["supports_bounds"]:
        res = method(
            func=_refine_pc_objective_function,
            args=params,
            bounds=bounds,
            **method_kwargs,
        )
    else:  # Is always "basinhopping", due to prior check of method name
        method_kwargs["minimizer_kwargs"].update(args=params)
        res = method(
            func=_refine_pc_objective_function,
            x0=pc,
            **method_kwargs,
        )

    ncc = 1 - res.fun
    num_evals = res.nfev
    pcx, pcy, pcz = res.x

    return ncc, num_evals, pcx, pcy, pcz


def _refine_orientation_pc_solver_scipy(
    pattern: np.ndarray,
    rot_pc: np.ndarray,
    bounds: np.ndarray,
    rescale: bool,
    method: Callable,
    method_kwargs: dict,
    fixed_parameters: tuple,
    trust_region_passed: bool,
    n_pseudo_symmetry_ops: int = 0,
) -> Union[
    Tuple[float, int, float, float, float, float, float, float],
    Tuple[float, int, int, float, float, float, float, float, float],
]:
    """Maximize the similarity between an experimental pattern and a
    projected simulated pattern by optimizing the orientation and
    projection center (PC) parameters used in the projection.

    Parameters
    ----------
    pattern
        Experimental pattern of shape (nrows, ncols).
    rot_pc
        Array with Euler angles (phi1, Phi, phi2) in radians and PC
        parameters (PCx, PCy, PCz) in range [0, 1].
    rescale
        Whether pattern intensities must be rescaled to [-1, 1] and data
        type 32-bit floats.
    method
        A supported :mod:`scipy.optimize` function. See `method`
        parameter in
        :meth:`kikuchipy.signals.EBSD.refine_orientation_projection_center`.
    method_kwargs
        Keyword arguments passed to the `method` function. For the list
        of possible arguments, see the SciPy documentation.
    fixed_parameters
        Fixed parameters used in the projection.
    trust_region_passed
        Whether `trust_region` was passed to the public refinement
        method.

    Returns
    -------
    score
        Highest normalized cross-correlation score.
    num_evals
        Number of optimization evaluations.
    phi1, Phi, phi2
        Optimized orientation (Euler angles) in radians.
    pcx_refined, pcy_refined, pcz_refined
        Optimized PC parameters in the Bruker convention.
    best_idx
    """
    pattern, squared_norm = _prepare_pattern(pattern, rescale)

    params = (pattern,) + fixed_parameters + (squared_norm,)
    method_name = method.__name__

    if n_pseudo_symmetry_ops == 0:
        if method_name == "minimize":
            if trust_region_passed:
                method_kwargs["bounds"] = bounds[0]
            res = method(
                fun=_refine_orientation_pc_objective_function,
                x0=rot_pc[0],
                args=params,
                **method_kwargs,
            )
        elif SUPPORTED_OPTIMIZATION_METHODS[method_name]["supports_bounds"]:
            res = method(
                func=_refine_orientation_pc_objective_function,
                args=params,
                bounds=bounds[0],
                **method_kwargs,
            )
        else:  # Is always "basinhopping", due to prior check of method name
            method_kwargs["minimizer_kwargs"].update(args=params)
            res = method(
                func=_refine_orientation_pc_objective_function,
                x0=rot_pc[0],
                **method_kwargs,
            )

        ncc = 1 - res.fun
        phi1, Phi, phi2, pcx, pcy, pcz = res.x
        num_evals = res.nfev

        return ncc, num_evals, phi1, Phi, phi2, pcx, pcy, pcz
    else:
        res_list = []
        for i in range(n_pseudo_symmetry_ops + 1):
            if method_name == "minimize":
                if trust_region_passed:
                    method_kwargs["bounds"] = bounds[i]
                res = method(
                    fun=_refine_orientation_pc_objective_function,
                    x0=rot_pc[i],
                    args=params,
                    **method_kwargs,
                )
            elif SUPPORTED_OPTIMIZATION_METHODS[method_name]["supports_bounds"]:
                res = method(
                    func=_refine_orientation_pc_objective_function,
                    args=params,
                    bounds=bounds[i],
                    **method_kwargs,
                )
            else:  # Is always "basinhopping", due to prior check of method name
                method_kwargs["minimizer_kwargs"].update(args=params)
                res = method(
                    func=_refine_orientation_pc_objective_function,
                    x0=rot_pc[i],
                    **method_kwargs,
                )

            res_list.append(res)

        ncc_all = [1 - res.fun for res in res_list]
        best_idx = int(np.argmax(ncc_all))

        best_res = res_list[best_idx]
        ncc = ncc_all[best_idx]
        num_evals = best_res.nfev
        phi1, Phi, phi2, pcx, pcy, pcz = best_res.x

        return ncc, num_evals, phi1, Phi, phi2, pcx, pcy, pcz, best_idx


# --------------------------- NLopt solvers -------------------------- #


def _refine_orientation_solver_nlopt(
    opt: "nlopt.opt",
    pattern: np.ndarray,
    rotation: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    signal_mask: np.ndarray,
    rescale: bool,
    trust_region_passed: bool,
    fixed_parameters: Tuple[np.ndarray, np.ndarray, int, int, float],
    direction_cosines: Optional[np.ndarray] = None,
    pcx: Optional[float] = None,
    pcy: Optional[float] = None,
    pcz: Optional[float] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    tilt: Optional[float] = None,
    azimuthal: Optional[float] = None,
    sample_tilt: Optional[float] = None,
    n_pseudo_symmetry_ops: int = 0,
) -> Union[
    Tuple[float, int, float, float, float],
    Tuple[float, int, int, float, float, float],
]:
    pattern, squared_norm = _prepare_pattern(pattern, rescale)

    # Get direction cosines if a unique PC per pattern is used
    if direction_cosines is None:
        direction_cosines = _get_direction_cosines_for_fixed_pc(
            pcx=pcx,
            pcy=pcy,
            pcz=pcz,
            nrows=nrows,
            ncols=ncols,
            tilt=tilt,
            azimuthal=azimuthal,
            sample_tilt=sample_tilt,
            signal_mask=signal_mask,
        )

    # Combine tuple of fixed parameters passed to the objective function
    params = (pattern,) + (direction_cosines,) + fixed_parameters + (squared_norm,)

    opt.set_min_objective(
        lambda x, grad: _refine_orientation_objective_function(x, *params)
    )

    if n_pseudo_symmetry_ops == 0:
        if trust_region_passed:
            opt.set_lower_bounds(lower_bounds[0])
            opt.set_upper_bounds(upper_bounds[0])

        phi1, Phi, phi2 = opt.optimize(rotation[0])
        ncc = 1 - opt.last_optimum_value()
        num_evals = opt.get_numevals()

        return ncc, num_evals, phi1, Phi, phi2
    else:
        n_rot = n_pseudo_symmetry_ops + 1
        eu_all = np.zeros((n_rot, 3), dtype=np.float64)
        ncc_inv_all = np.zeros(n_rot, dtype=np.float64)
        num_evals_all = np.zeros(n_rot, dtype=np.int32)
        for i in range(n_rot):
            if trust_region_passed:
                opt.set_lower_bounds(lower_bounds[i])
                opt.set_upper_bounds(upper_bounds[i])

            eu_all[i] = opt.optimize(rotation[i])
            ncc_inv_all[i] = opt.last_optimum_value()
            num_evals_all[i] = opt.get_numevals()

        best_idx = int(np.argmin(ncc_inv_all))
        ncc = 1 - ncc_inv_all[best_idx]
        num_evals = num_evals_all[best_idx]
        phi1, Phi, phi2 = eu_all[best_idx]

        return ncc, num_evals, phi1, Phi, phi2, best_idx


def _refine_pc_solver_nlopt(
    opt: "nlopt.opt",
    pattern: np.ndarray,
    pc: np.ndarray,
    rotation: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    rescale: bool,
    fixed_parameters: tuple,
    trust_region_passed: bool,
) -> Tuple[float, int, float, float, float]:
    pattern, squared_norm = _prepare_pattern(pattern, rescale)

    # Combine tuple of fixed parameters passed to the objective function
    params = (pattern,) + (rotation,) + fixed_parameters + (squared_norm,)

    # Prepare NLopt optimizer
    if trust_region_passed:
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
    opt.set_min_objective(lambda x, grad: _refine_pc_objective_function(x, *params))

    # Run optimization and extract optimized Euler angles and PC values
    # and the optimized normalized cross-correlation (NCC) score
    pcx, pcy, pcz = opt.optimize(pc)
    ncc = 1 - opt.last_optimum_value()
    num_evals = opt.get_numevals()

    return ncc, num_evals, pcx, pcy, pcz


def _refine_orientation_pc_solver_nlopt(
    opt: "nlopt.opt",
    pattern: np.ndarray,
    rot_pc: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    rescale: bool,
    fixed_parameters: tuple,
    trust_region_passed: bool,
    n_pseudo_symmetry_ops: int = 0,
) -> Union[
    Tuple[float, int, float, float, float, float, float, float],
    Tuple[float, int, int, float, float, float, float, float, float],
]:
    pattern, squared_norm = _prepare_pattern(pattern, rescale)

    # Combine tuple of fixed parameters passed to the objective function
    params = (pattern,) + fixed_parameters + (squared_norm,)

    opt.set_min_objective(
        lambda x, grad: _refine_orientation_pc_objective_function(x, *params)
    )

    if n_pseudo_symmetry_ops == 0:
        if trust_region_passed:
            opt.set_lower_bounds(lower_bounds[0])
            opt.set_upper_bounds(upper_bounds[0])

        phi1, Phi, phi2, pcx, pcy, pcz = opt.optimize(rot_pc[0])
        ncc = 1 - opt.last_optimum_value()
        num_evals = opt.get_numevals()

        return ncc, num_evals, phi1, Phi, phi2, pcx, pcy, pcz
    else:
        n_rot = n_pseudo_symmetry_ops + 1
        eu_pc_all = np.zeros((n_rot, 6), dtype=np.float64)
        ncc_inv_all = np.zeros(n_rot, dtype=np.float64)
        num_evals_all = np.zeros(n_rot, dtype=np.int32)
        for i in range(n_rot):
            if trust_region_passed:
                opt.set_lower_bounds(lower_bounds[i])
                opt.set_upper_bounds(upper_bounds[i])

            eu_pc_all[i] = opt.optimize(rot_pc[i])
            ncc_inv_all[i] = opt.last_optimum_value()
            num_evals_all[i] = opt.get_numevals()

        best_idx = int(np.argmin(ncc_inv_all))
        ncc = 1 - ncc_inv_all[best_idx]
        num_evals = num_evals_all[best_idx]
        phi1, Phi, phi2, pcx, pcy, pcz = eu_pc_all[best_idx]

        return ncc, num_evals, phi1, Phi, phi2, pcx, pcy, pcz, best_idx
