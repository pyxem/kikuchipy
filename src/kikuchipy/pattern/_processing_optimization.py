#
# Copyright 2019-2026 the kikuchipy developers
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
#

"""Optimization of processing parameters of a single EBSD pattern
against a reference pattern using global Bayesian optimization or local
optimization with Nelder-Mead.
"""

from collections.abc import Callable, Sequence
import inspect
import logging
from time import time
from typing import Any, Literal, NamedTuple, get_args
import warnings

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from kikuchipy._constants import verify_dependency_or_raise
from kikuchipy.filters.window import bandpass_fft_filter
from kikuchipy.indexing.similarity_metrics._normalized_cross_correlation import (
    _ncc_single_patterns_2d_ref_normalized,
    _zero_mean_normalize_pattern_numpy,
)
from kikuchipy.pattern._pattern import (
    FILTER_DOMAIN,
    REMOVAL_OPERATION,
    _adaptive_histogram_equalization,
    fft_filter,
    get_image_quality,
    remove_dynamic_background,
    rescale_intensity,
)

_logger = logging.getLogger(__name__)

OPTIMIZATION_STEPS = Literal[
    "remove_dynamic_background",
    "adaptive_histogram_equalization",
    "bandpass_filter",
]
OPTIMIZATION_METHODS = Literal["bayesian", "nelder-mead"]


# --------------------- Single processing steps ---------------------- #
# Not available as functions in the single pattern processing module.


def _equalize_adaptive_histogram(
    pattern: np.ndarray, kernel_size: int, clip_limit: float, nbins: int
) -> np.ndarray:
    return _adaptive_histogram_equalization(
        pattern,
        kernel_size=(kernel_size, kernel_size),
        clip_limit=clip_limit,
        nbins=nbins,
    )


def _bandpass_filter(
    pattern: np.ndarray,
    highpass_cutoff: float,
    lowpass_cutoff: float,
    highpass_cutoff_width: float,
    lowpass_cutoff_width: float,
) -> np.ndarray:
    transfer_function = bandpass_fft_filter(
        pattern.shape,
        highpass_cutoff=highpass_cutoff,
        lowpass_cutoff=lowpass_cutoff,
        highpass_cutoff_width=highpass_cutoff_width,
        lowpass_cutoff_width=lowpass_cutoff_width,
    )
    filtered = fft_filter(
        pattern.astype(np.float32), transfer_function=transfer_function, shift=True
    )
    return rescale_intensity(filtered, dtype_out=pattern.dtype.type)


# ------------------------ Public optimizers ------------------------- #


def optimize_remove_dynamic_background(
    pattern: np.ndarray,
    reference: np.ndarray,
    std: float | tuple | list | None = None,
    truncate: float | tuple | list = (2.0, 10.0),
    operation: REMOVAL_OPERATION | list[REMOVAL_OPERATION] = "subtract",
    filter_domain: FILTER_DOMAIN | list[FILTER_DOMAIN] = "frequency",
    signal_mask: np.ndarray | None = None,
    method: OPTIMIZATION_METHODS = "bayesian",
    n_calls: int = 50,
    n_initial_points: int = 10,
    n_restarts: int = 2,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Find dynamic background removal parameters that maximize the
    similarity between a single EBSD pattern and a reference pattern,
    using Bayesian global or Nelder-Mead local optimization.

    See :func:`~kikuchipy.pattern.remove_dynamic_background` for a
    description of the processing parameters. Each parameter is either
    fixed or searched, as described in
    :func:`~kikuchipy.pattern.optimize_pattern_processing`.

    This Bayesian optimization of pattern processing parameters was
    first presented in :cite:`griesbach2026ferroelectric` and
    :cite:`griesbach2026global` to map ferroelectric polarization
    domains.

    Parameters
    ----------
    pattern
        Experimental EBSD pattern.
    reference
        Simulated (or otherwise trusted) reference pattern of the same
        shape as *pattern*.
    std
        Standard deviation of the Gaussian window. If not given, the
        range from width/32 to width/4 is searched, where width is the
        pattern width. The default in
        :func:`~kikuchipy.pattern.remove_dynamic_background` is width/8.
    truncate
        Truncation of the Gaussian window in standard deviations, so
        that the window size scales with *std*. Default is to search the
        range [2, 10].
    operation
        Whether to "subtract" (default) or "divide" by the dynamic
        background.
    filter_domain
        Whether to obtain the dynamic background in the "frequency"
        (default) or "spatial" domain.
    signal_mask
        A boolean mask of the same shape as *pattern*, where only pixels
        equal to False are used in the similarity score. If not given,
        all pixels are used.
    method
        Optimization method, either "bayesian" (default) or
        "nelder-mead". See
        :func:`~kikuchipy.pattern.optimize_pattern_processing` for
        details.
    n_calls
        Maximum number of calls to the objective function. Default is
        50.
    n_initial_points
        Number of random evaluations before the Bayesian surrogate
        model takes over if *method* is "bayesian". Default is 10.
    n_restarts
        Number of restarts of the local optimization if *method* is
        "nelder-mead". Default is 2.
    random_state
        Seed for reproducible optimization.

    Returns
    -------
    result
        Optimization result, see
        :func:`~kikuchipy.pattern.optimize_pattern_processing`.

    See Also
    --------
    optimize_adaptive_histogram_equalization,
    optimize_bandpass_filter,
    optimize_pattern_processing
    """
    return optimize_pattern_processing(
        pattern,
        reference,
        steps=("remove_dynamic_background",),
        signal_mask=signal_mask,
        method=method,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        n_restarts=n_restarts,
        random_state=random_state,
        remove_dynamic_background={
            "std": std,
            "truncate": truncate,
            "operation": operation,
            "filter_domain": filter_domain,
        },
    )


def optimize_adaptive_histogram_equalization(
    pattern: np.ndarray,
    reference: np.ndarray,
    kernel_size: int | tuple | list | None = None,
    clip_limit: float | tuple | list = (1e-4, 5e-3, "log"),
    nbins: int | tuple | list = [128, 256, 512],
    signal_mask: np.ndarray | None = None,
    method: OPTIMIZATION_METHODS = "bayesian",
    n_calls: int = 50,
    n_initial_points: int = 10,
    n_restarts: int = 2,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Find adaptive histogram equalization parameters that maximize
    the similarity between a single EBSD pattern and a reference
    pattern, using Bayesian global or Nelder-Mead local optimization.

    See :meth:`~kikuchipy.signals.EBSD.adaptive_histogram_equalization`
    for a description of the processing parameters. Each parameter is
    either fixed or searched, as described in
    :func:`~kikuchipy.pattern.optimize_pattern_processing`.

    This Bayesian optimization of pattern processing parameters was
    first presented in :cite:`griesbach2026ferroelectric` and
    :cite:`griesbach2026global` to map ferroelectric polarization
    domains.

    Parameters
    ----------
    pattern
        Experimental EBSD pattern. Should have an integer data type.
    reference
        Simulated (or otherwise trusted) reference pattern of the same
        shape as *pattern*.
    kernel_size
        Size of the square contextual regions. If not given, the integer
        range from width//8 to width//2 is searched, where width is the
        pattern width. The default in
        :meth:`~kikuchipy.signals.EBSD.adaptive_histogram_equalization`
        is width//4.
    clip_limit
        Clipping limit. Default is to search the range [1e-4, 5e-3] on a
        logarithmic scale.
    nbins
        Number of gray bins for the histogram. Default is to search the
        candidates ``[128, 256, 512]``.
    signal_mask
        A boolean mask of the same shape as *pattern*, where only
        pixels equal to ``False`` are used in the similarity score. If
        not given, all pixels are used.
    method
        Optimization method, either "bayesian" (default) or
        "nelder-mead". See
        :func:`~kikuchipy.pattern.optimize_pattern_processing` for
        details.
    n_calls
        Maximum number of calls to the objective function. Default is
        50.
    n_initial_points
        Number of random evaluations before the Bayesian surrogate
        model takes over if ``method="bayesian". Default is 10.
    n_restarts
        Number of restarts of the local optimization if
        ``method="nelder-mead". Default is 2.
    random_state
        Seed for reproducible optimization.

    Returns
    -------
    result
        Optimization result, see
        :func:`~kikuchipy.pattern.optimize_pattern_processing`.

    See Also
    --------
    optimize_remove_dynamic_background,
    optimize_bandpass_filter,
    optimize_pattern_processing
    """
    return optimize_pattern_processing(
        pattern,
        reference,
        steps=("adaptive_histogram_equalization",),
        signal_mask=signal_mask,
        method=method,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        n_restarts=n_restarts,
        random_state=random_state,
        adaptive_histogram_equalization={
            "kernel_size": kernel_size,
            "clip_limit": clip_limit,
            "nbins": nbins,
        },
    )


def optimize_bandpass_filter(
    pattern: np.ndarray,
    reference: np.ndarray,
    highpass_cutoff: float | tuple | list | None = None,
    lowpass_cutoff: float | tuple | list | None = None,
    highpass_cutoff_width: float | tuple | list | None = None,
    lowpass_cutoff_width: float | tuple | list | None = None,
    signal_mask: np.ndarray | None = None,
    method: OPTIMIZATION_METHODS = "bayesian",
    n_calls: int = 50,
    n_initial_points: int = 10,
    n_restarts: int = 2,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Find FFT bandpass filter parameters that maximize the similarity
    between a single EBSD pattern and a reference pattern, using
    Bayesian global or Nelder-Mead local optimization.

    The bandpass filter is created with
    :func:`~kikuchipy.filters.bandpass_fft_filter` and applied with
    :func:`~kikuchipy.pattern.fft_filter`. Each parameter is either
    fixed or searched, as described in
    :func:`~kikuchipy.pattern.optimize_pattern_processing`.

    This Bayesian optimization of pattern processing parameters was
    first presented in :cite:`griesbach2026ferroelectric` and
    :cite:`griesbach2026global` to map ferroelectric polarization
    domains.

    Parameters
    ----------
    pattern
        Experimental EBSD pattern.
    reference
        Simulated (or otherwise trusted) reference pattern of the same
        shape as *pattern*.
    highpass_cutoff
        Cutoff frequency of the highpass filter. If not given, the range
        from width/60 to width/10 is searched, where width is the
        pattern width.
    lowpass_cutoff
        Cutoff frequency of the lowpass filter. If not given, the range
        from width/4 to width/2 is searched.
    highpass_cutoff_width
        Width of the highpass filter transition. If not given, it is
        fixed to width/30.
    lowpass_cutoff_width
        Width of the lowpass filter transition. If not given, it is
        fixed to width/6.
    signal_mask
        A boolean mask of the same shape as *pattern*, where only
        pixels equal to ``False`` are used in the similarity score. If
        not given, all pixels are used.
    method
        Optimization method, either "bayesian" (default) or
        "nelder-mead". See
        :func:`~kikuchipy.pattern.optimize_pattern_processing` for
        details.
    n_calls
        Maximum number of calls to the objective function. Default is
        50.
    n_initial_points
        Number of random evaluations before the Bayesian surrogate
        model takes over if ``method="bayesian". Default is 10.
    n_restarts
        Number of restarts of the local optimization if
        ``method="nelder-mead". Default is 2.
    random_state
        Seed for reproducible optimization.

    Returns
    -------
    result
        Optimization result, see
        :func:`~kikuchipy.pattern.optimize_pattern_processing`.

    See Also
    --------
    optimize_remove_dynamic_background,
    optimize_adaptive_histogram_equalization,
    optimize_pattern_processing,
    kikuchipy.filters.bandpass_fft_filter

    Notes
    -----
    To apply the optimized filter to all patterns in a signal ``s``,
    create the filter with
    ``kp.filters.bandpass_fft_filter(s.detector.shape, **result["parameters"]["bandpass_filter"])``
    and pass it to :meth:`~kikuchipy.signals.EBSD.fft_filter` with
    ``function_domain="frequency" and ``shift=True``.
    """
    return optimize_pattern_processing(
        pattern,
        reference,
        steps=("bandpass_filter",),
        signal_mask=signal_mask,
        method=method,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        n_restarts=n_restarts,
        random_state=random_state,
        bandpass_filter={
            "highpass_cutoff": highpass_cutoff,
            "lowpass_cutoff": lowpass_cutoff,
            "highpass_cutoff_width": highpass_cutoff_width,
            "lowpass_cutoff_width": lowpass_cutoff_width,
        },
    )


def optimize_pattern_processing(
    pattern: np.ndarray,
    reference: np.ndarray,
    steps: Sequence[OPTIMIZATION_STEPS] | None = None,
    signal_mask: np.ndarray | None = None,
    method: OPTIMIZATION_METHODS = "bayesian",
    n_calls: int = 150,
    n_initial_points: int = 12,
    n_restarts: int = 2,
    random_state: int | None = None,
    **kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Find processing parameters that maximize the similarity between
    a single EBSD pattern and a reference pattern, using Bayesian global
    or Nelder-Mead local optimization.

    The processing steps are applied in the order given by *steps*,
    and the parameters of all steps are searched simultaneously. Each
    candidate is scored by the normalized cross-correlation (NCC)
    between the processed *pattern* and *reference*.

    This Bayesian optimization of pattern processing parameters was
    first presented in :cite:`griesbach2026ferroelectric` and
    :cite:`griesbach2026global` to map ferroelectric polarization
    domains.

    Parameters
    ----------
    pattern
        Experimental EBSD pattern.
    reference
        Reference pattern of the same shape as *pattern*. This should
        be a trusted reference, such as a simulated pattern from
        :meth:`~kikuchipy.signals.EBSDMasterPattern.get_patterns`.
    steps
        Processing steps to apply, in order. Available steps are:

        - "remove_dynamic_background"
        - "adaptive_histogram_equalization"
        - "bandpass_filter"

        Default steps are "remove_dynamic_background" and
        "bandpass_filter", in that order.
    signal_mask
        A boolean mask of the same shape as *pattern*, where only pixels
        equal to False are used in the similarity score. If not given,
        all pixels are used.
    method
        Optimization method, either "bayesian" (default) or
        "nelder-mead". See the notes for details.
    n_calls
        Maximum number of calls to the objective function. Default is
        150.
    n_initial_points
        Number of random evaluations before the Bayesian surrogate model
        takes over if *method* is "bayesian". Must be at least 1 and at
        most *n_calls*. Default is 12.
    n_restarts
        Number of restarts of the local optimization if *method* is
        "nelder-mead". Default is 2.
    random_state
        Seed for reproducible optimization.
    **kwargs
        Fixed values or search spaces of the parameters of a step,
        passed as a dictionary with the step name as keyword, e.g.
        ``bandpass_filter={"lowpass_cutoff": (40, 80)}``. Parameters and
        their defaults are those of
        :func:`~kikuchipy.pattern.optimize_remove_dynamic_background`,
        :func:`~kikuchipy.pattern.optimize_adaptive_histogram_equalization`,
        and :func:`~kikuchipy.pattern.optimize_bandpass_filter`.

    Returns
    -------
    result
        Dictionary with the following keys:

        - "steps": processing steps, in order.
        - "parameters": dictionary with the best parameters per step,
          including fixed ones.
        - "score": NCC of the processed pattern using the best
          parameters.
        - "patterns": dictionary with the pattern before processing
          ("raw") and after each step, using the best parameters.
        - "image_quality": image quality of each pattern in "patterns".
        - "normalized_cross_correlation": NCC of each pattern in
          "patterns".
        - "signal_mask": the signal mask used, or None.
        - "optimizer": if *method* is "bayesian", the
          :class:`bayes_opt.BayesianOptimization` instance used, with
          all evaluated parameters and scores. If *method* is
          "nelder-mead", a list with the
          :class:`scipy.optimize.OptimizeResult` of each start, with
          parameters scaled to the range [0, 1] of the search space.

    See Also
    --------
    optimize_remove_dynamic_background,
    optimize_adaptive_histogram_equalization,
    optimize_bandpass_filter,
    kikuchipy.draw.plot_pattern_processing_result

    Notes
    -----
    Two optimization methods are available:

    - "bayesian": Bayesian optimization based on Gaussian process
      regression, using :class:`bayes_opt.BayesianOptimization` with
      the expected improvement acquisition function. After
      *n_initial_points* random evaluations, each new candidate is
      chosen from a surrogate model of all evaluations so far.

      This requires the optional dependency
      :mod:`bayesian-optimization`.
    - "nelder-mead": local optimization with the Nelder-Mead
      method, using :func:`scipy.optimize.minimize`. The first start is
      the center of the search space, and each of the *n_restarts*
      restarts is from the best parameters so far, slightly perturbed.
      The *n_calls* evaluations are shared between the starts, with
      evaluations left by a start which converges early passed on to
      the next. This method is typically several times faster than
      Bayesian optimization, since choosing each new candidate is
      cheaper, while it might converge to a local maximum.

    Each processing parameter is specified as either:

    - A scalar or string: the value is fixed and not searched.
    - A tuple (low, high): an integer range if both values are integers,
      otherwise a real range. An optional third element sets the scale
      the range is searched on, either "linear" (default) or "log", e.g.
      (1e-4, 5e-3, "log"). A logarithmic scale gives each order of
      magnitude equal weight, which suits ranges spanning several
      orders of magnitude.
    - A list: at least two candidate values to choose between.
    - None: the default of the single step function, which for some
      parameters depends on the pattern size.

    Processing steps preserve the pattern's data type. Adaptive
    histogram equalization gives bad results for patterns with a
    floating point data type.

    To process a full pattern stack with the best parameters, pass
    ``result["parameters"]["remove_dynamic_background"]`` and
    ``result["parameters"]["adaptive_histogram_equalization"]`` to
    :meth:`~kikuchipy.signals.EBSD.remove_dynamic_background` and
    :meth:`~kikuchipy.signals.EBSD.adaptive_histogram_equalization`,
    respectively, and pass
    ``result["parameters"]["bandpass_filter"],`` to
    :func:`~kikuchipy.filters.bandpass_fft_filter` to create the filter
    for :meth:`~kikuchipy.signals.EBSD.fft_filter`. In the latter case,
    remember to pass ``function_domain="frequency", shift=True`` to
    ``fft_filter()``.
    """
    # Validate method
    optimization_methods = get_args(OPTIMIZATION_METHODS)
    if method not in optimization_methods:
        raise ValueError(
            f"Unknown optimization method {method!r}, must be among "
            f"{optimization_methods}"
        )
    if method == "bayesian":
        verify_dependency_or_raise(
            "bayesian-optimization", "Pattern processing Bayesian global optimization"
        )
    if method == "bayesian" and not 1 <= n_initial_points <= n_calls:
        raise ValueError(
            f"Number of initial points {n_initial_points} must be at least 1 and at "
            f"most the number of calls {n_calls}"
        )
    if method == "nelder-mead" and n_restarts < 0:
        raise ValueError(f"Number of restarts {n_restarts} must be at least 0")

    # Validate patterns and mask
    pattern = np.asarray(pattern)
    reference = np.asarray(reference)
    if pattern.ndim != 2 or pattern.shape != reference.shape:
        raise ValueError(
            f"Pattern shape {pattern.shape} and reference shape {reference.shape} must "
            "be equal and 2D"
        )
    if signal_mask is not None:
        signal_mask = np.asarray(signal_mask, dtype=bool)
        if signal_mask.shape != pattern.shape:
            raise ValueError(
                f"Signal mask shape {signal_mask.shape} and pattern shape "
                f"{pattern.shape} must be equal"
            )

    # Validate steps and parameters
    if steps is None:
        steps = ("remove_dynamic_background", "bandpass_filter")
    else:
        steps = tuple(steps)
    parameters = _get_parameters(steps, kwargs, pattern.shape[1])
    if "adaptive_histogram_equalization" in steps and np.issubdtype(
        pattern.dtype, np.floating
    ):
        warnings.warn(
            "Equalization of patterns with floating point data type has been shown to "
            "give bad results. Rescaling intensities to integer intensities is "
            "recommended."
        )

    # Parameters with a search space are searched, the rest stay fixed
    searched = []
    for step, step_params in parameters.items():
        for name, spec in step_params.items():
            searched_param = _get_searched_parameter(step, name, spec)
            if searched_param is not None:
                searched.append(searched_param)
    if not searched:
        raise ValueError("No parameters to optimize, as all are fixed")

    reference_normalized = _zero_mean_normalize_pattern_numpy(reference, signal_mask)
    progress_logger = _ProcessingOptimizationProgressLogger(n_calls)

    def objective(values: dict[str, float]) -> float:
        params = _set_parameters(parameters, searched, values)
        processed = _process_pattern(pattern, steps, params)
        ncc = _ncc_single_patterns_2d_ref_normalized(
            processed, reference_normalized, signal_mask
        )
        _logger.debug(f"NCC {ncc:.4f} with parameters {params}")
        progress_logger(ncc)
        return ncc

    _logger.info(
        f"Optimize {len(searched)} parameters of steps {steps} with method "
        f"{method!r} and up to {n_calls} calls"
    )
    if method == "bayesian":
        best_values, best_score, optimizer = _maximize_bayesian(
            objective, searched, n_calls, n_initial_points, random_state
        )
    else:
        best_values, best_score, optimizer = _maximize_nelder_mead(
            objective, searched, n_calls, n_restarts, random_state
        )

    best_parameters = _set_parameters(parameters, searched, best_values)
    _logger.info(f"Best NCC {best_score:.4f} with parameters {best_parameters}")

    patterns = _process_pattern(
        pattern, steps, best_parameters, return_intermediate=True
    )
    image_quality = []
    ncc = []
    for processed in patterns.values():
        image_quality.append(get_image_quality(processed))
        ncc.append(
            _ncc_single_patterns_2d_ref_normalized(
                processed, reference_normalized, signal_mask
            )
        )
    image_quality = np.array(image_quality)
    ncc = np.array(ncc)

    return {
        "steps": steps,
        "parameters": best_parameters,
        "score": ncc[-1],
        "patterns": patterns,
        "image_quality": image_quality,
        "normalized_cross_correlation": ncc,
        "signal_mask": signal_mask,
        "optimizer": optimizer,
    }


# ----------------------------- Helpers ------------------------------ #


class _Step(NamedTuple):
    func: Callable
    optimize_func: Callable
    parameters: tuple[str, ...]
    label: str
    # Defaults of parameters given as None, from the pattern width
    size_defaults: Callable[[int], dict[str, Any]]


_STEPS = {
    "remove_dynamic_background": _Step(
        func=remove_dynamic_background,
        optimize_func=optimize_remove_dynamic_background,
        parameters=("std", "truncate", "operation", "filter_domain"),
        label="DBR",
        size_defaults=lambda width: {"std": (width / 32, width / 4)},
    ),
    "adaptive_histogram_equalization": _Step(
        func=_equalize_adaptive_histogram,
        optimize_func=optimize_adaptive_histogram_equalization,
        parameters=("kernel_size", "clip_limit", "nbins"),
        label="AHE",
        size_defaults=lambda width: {"kernel_size": (max(width // 8, 1), width // 2)},
    ),
    "bandpass_filter": _Step(
        func=_bandpass_filter,
        optimize_func=optimize_bandpass_filter,
        parameters=(
            "highpass_cutoff",
            "lowpass_cutoff",
            "highpass_cutoff_width",
            "lowpass_cutoff_width",
        ),
        label="Bandpass",
        size_defaults=lambda width: {
            "highpass_cutoff": (width / 60, width / 10),
            "lowpass_cutoff": (width / 4, width / 2),
            "highpass_cutoff_width": width / 30,
            "lowpass_cutoff_width": width / 6,
        },
    ),
}


def _get_parameters(
    steps: tuple[str, ...], kwargs: dict[str, dict[str, Any]], width: int
) -> dict[str, dict[str, Any]]:
    """Return the parameter specifications per step, with user given
    values replacing the defaults of the step's optimize function, and
    parameters given as None set from the pattern *width*.
    """
    for step in steps:
        if step not in _STEPS:
            raise ValueError(
                f"Unknown processing step {step!r}, must be among {list(_STEPS)}"
            )
    if len(set(steps)) != len(steps):
        raise ValueError(f"Processing steps {steps} cannot contain duplicates")
    for step in kwargs:
        if step not in steps:
            raise ValueError(
                f"Parameters given for step {step!r}, which is not among {steps}"
            )

    parameters: dict[str, dict[str, Any]] = {}
    for step in steps:
        step_params = dict(kwargs.get(step, {}))

        # Raise if unknown parameters are given
        allowed = _STEPS[step].parameters
        unknown = set(step_params) - set(allowed)
        if unknown:
            raise ValueError(
                f"Unknown parameters {sorted(unknown)} for step {step!r}, must be among"
                f" {list(allowed)}"
            )

        # Get parameters in this order:
        # 1. User-provided
        # 2. Default in function to optimize
        # 3. Default provided in this module
        signature = inspect.signature(_STEPS[step].optimize_func)
        size_defaults = _STEPS[step].size_defaults(width)
        parameters[step] = {}
        for name in allowed:
            value = step_params.get(name, signature.parameters[name].default)
            if value is None:
                value = size_defaults[name]
            parameters[step][name] = value

    return parameters


class _SearchedParameter(NamedTuple):
    """Searched parameter of a processing step.

    All parameters are searched as floats by bayes_opt: integers are
    rounded, candidates are indexed, and ranges on a logarithmic scale
    are searched by their exponent. bayes_opt's own integer and categorical
    parameters are avoided, as optimizing the acquisition function over
    them is about ten times slower.
    """

    step: str
    name: str
    low: float = 0
    high: float = 0
    log: bool = False
    integer: bool = False
    candidates: tuple | None = None

    @property
    def key(self) -> str:
        return f"{self.step}.{self.name}"

    @property
    def pbound(self) -> tuple[float, float]:
        """Bounds of the float searched by bayes_opt."""
        if self.candidates is not None:
            return 0.0, float(len(self.candidates))
        elif self.log:
            return float(np.log10(self.low)), float(np.log10(self.high))
        elif self.integer:
            return self.low - 0.5, self.high + 0.5
        else:  # Float
            return float(self.low), float(self.high)

    def to_value(self, value: float) -> Any:
        """Return the parameter value from the float searched by
        bayes_opt.
        """
        if self.candidates is not None:
            i = int(np.clip(np.floor(value), a_min=0, a_max=len(self.candidates) - 1))
            return self.candidates[i]

        if self.log:
            value = 10**value
        if self.integer:
            value = np.clip(np.round(value), self.low, self.high)

        # Plain Python types, since e.g. scikit-image's adaptive
        # histogram equalization function fails with NumPy integer bins
        if self.integer:
            value = int(value)
        else:
            value = float(value)

        return value


def _get_searched_parameter(
    step: str, name: str, spec: Any
) -> _SearchedParameter | None:
    """Return a searched parameter from a parameter specification, or
    None if the parameter is fixed.
    """
    key = f"{step}.{name}"

    if isinstance(spec, (list, np.ndarray)):
        candidates_list = []
        for v in spec:
            if isinstance(v, np.generic):
                v = v.item()
            candidates_list.append(v)
        candidates = tuple(candidates_list)

        if len(set(candidates)) != len(candidates) or len(candidates) < 2:
            raise ValueError(
                f"Candidates of {key!r} must be at least two unique values, not {spec}"
            )

        return _SearchedParameter(step, name, candidates=candidates)
    elif isinstance(spec, tuple):
        if len(spec) not in (2, 3):
            raise ValueError(
                f"Range of {key!r} must be (low, high) or (low, high, scale), not "
                f"{spec}"
            )
        low, high = spec[:2]
        if len(spec) == 3:
            scale = spec[2]
        else:
            scale = "linear"
        if scale not in ("linear", "log"):
            raise ValueError(
                f"Scale of {key!r} must be 'linear' or 'log', not {scale!r}"
            )
        if scale == "log" and low <= 0:
            raise ValueError(f"Range of {key!r} on a log scale must be positive")
        integer = isinstance(low, (int, np.integer)) and isinstance(
            high, (int, np.integer)
        )
        return _SearchedParameter(
            step, name, low, high, log=scale == "log", integer=integer
        )
    else:
        return


def _set_parameters(
    parameters: dict[str, dict[str, Any]],
    searched: list[_SearchedParameter],
    values: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return a copy of *parameters* with searched parameters set to
    *values* from bayes_opt.
    """
    new_parameters = {}
    for step, params in parameters.items():
        new_parameters[step] = dict(params)
    for param in searched:
        new_parameters[param.step][param.name] = param.to_value(values[param.key])
    return new_parameters


def _maximize_bayesian(
    objective: Callable[[dict[str, float]], float],
    searched: list[_SearchedParameter],
    n_calls: int,
    n_initial_points: int,
    random_state: int | None,
) -> tuple[dict[str, float], float, Any]:
    """Maximize *objective* with Bayesian optimization, returning the
    best searched values, the best score, and the optimizer.
    """
    from bayes_opt import BayesianOptimization
    from bayes_opt.acquisition import ExpectedImprovement

    pbounds = {}
    for param in searched:
        pbounds[param.key] = param.pbound

    optimizer = BayesianOptimization(
        f=lambda **values: objective(values),
        pbounds=pbounds,
        acquisition_function=ExpectedImprovement(xi=0.01),
        random_state=random_state,
        verbose=0,
        # Suggestions at the bounds of the search space may repeat
        allow_duplicate_points=True,
    )
    optimizer.maximize(init_points=n_initial_points, n_iter=n_calls - n_initial_points)

    return optimizer.max["params"], float(optimizer.max["target"]), optimizer


def _maximize_nelder_mead(
    objective: Callable[[dict[str, float]], float],
    searched: list[_SearchedParameter],
    n_calls: int,
    n_restarts: int,
    random_state: int | None,
    initial_step: float = 0.25,
    perturbation: float = 0.1,
) -> tuple[dict[str, float], float, list[OptimizeResult]]:
    """Maximize *objective* with the Nelder-Mead method from the center
    of the search space, restarting *n_restarts* times from the best
    point so far, perturbed.

    The search space is scaled to the range [0, 1] in each dimension.
    The initial simplex has steps of *initial_step* from the start,
    which is large enough to not stall on integer and candidate
    parameters. Restarts are perturbed with normally distributed steps
    with a standard deviation of *perturbation*.

    Returns the best searched values, the best score, and the result of
    each start.
    """
    low = []
    high = []
    for param in searched:
        low.append(param.pbound[0])
        high.append(param.pbound[1])
    low = np.array(low)
    high = np.array(high)
    n_dim = low.size
    rng = np.random.default_rng(random_state)

    n_starts = n_restarts + 1
    # Each start needs evaluations to create and update its simplex
    n_calls_min = n_starts * (n_dim + 2)
    if n_calls < n_calls_min:
        raise ValueError(
            f"Number of calls {n_calls} must be at least {n_calls_min} to optimize "
            f"{n_dim} parameters with {n_restarts} restarts"
        )

    def to_values(x: np.ndarray) -> dict[str, float]:
        """Return searched values from coordinates in the range [0, 1]."""
        values = {}
        for param, value in zip(searched, low + x * (high - low)):
            values[param.key] = value
        return values

    n_evaluations = 0
    best_score = -np.inf
    best_x = np.full(n_dim, 0.5)

    def negative_objective(x: np.ndarray) -> float:
        nonlocal n_evaluations, best_score, best_x
        if n_evaluations >= n_calls:
            # SciPy's Nelder-Mead may exceed the maximum number of
            # evaluations within its last iteration
            return np.inf
        n_evaluations += 1
        x = np.clip(x, 0, 1)
        score = objective(to_values(x))
        if score > best_score:
            best_score, best_x = score, x
        return -score

    results = []
    for i in range(n_starts):
        n_remaining = n_calls - n_evaluations
        if i == 0:
            x0 = np.full(n_dim, 0.5)
        else:
            x0 = np.clip(best_x + rng.normal(scale=perturbation, size=n_dim), 0, 1)
            _logger.debug(f"Restart {i}/{n_restarts} after {n_evaluations} calls")
        # Step towards the center, to stay within the bounds
        steps = np.where(x0 < 0.5, initial_step, -initial_step)
        initial_simplex = np.vstack([x0, x0 + np.diag(steps)])
        result = minimize(
            negative_objective,
            x0,
            method="Nelder-Mead",
            bounds=[(0, 1)] * n_dim,
            options={
                # Share remaining evaluations with the remaining starts
                "maxfev": n_remaining // (n_starts - i),
                "initial_simplex": initial_simplex,
                "xatol": 1e-3,
                "fatol": 1e-4,
            },
        )
        results.append(result)

    return to_values(best_x), float(best_score), results


def _process_pattern(
    pattern: np.ndarray,
    steps: tuple[str, ...],
    parameters: dict[str, dict[str, Any]],
    return_intermediate: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
    """Apply processing steps in order to a pattern, returning either
    the final pattern or a dictionary with the pattern before and after
    each step.
    """
    patterns = {"raw": pattern}
    for step in steps:
        pattern = _STEPS[step].func(pattern, **parameters[step])
        patterns[step] = pattern
    if return_intermediate:
        return patterns
    else:
        return pattern


class _ProcessingOptimizationProgressLogger:
    """Log the best score of a pattern processing optimization about
    every tenth of the calls to the objective function.
    """

    def __init__(self, n_calls: int):
        self.n_calls = n_calls
        self.log_every = max(1, n_calls // 10)
        self.i = 0
        self.best_score = -np.inf
        self.time_start = time()

    def __call__(self, score: float) -> None:
        self.i += 1
        self.best_score = max(self.best_score, score)
        if self.i % self.log_every == 0 or self.i == self.n_calls:
            _logger.info(
                f"Call {self.i}/{self.n_calls}: best NCC {self.best_score:.4f} "
                f"({time() - self.time_start:.1f} s)"
            )
