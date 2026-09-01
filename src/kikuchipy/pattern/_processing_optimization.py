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

"""Bayesian optimization of a single pattern's processing pipeline
against a reference pattern.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from kikuchipy._constants import verify_dependency_or_raise
from kikuchipy.filters.window import Window
from kikuchipy.pattern._pattern import (
    _adaptive_histogram_equalization,
    fft_filter,
    get_image_quality,
    remove_dynamic_background,
)


def _normalized_cross_correlation(
    experimental: np.ndarray, simulated: np.ndarray
) -> float:
    """Return the normalized cross-correlation coefficient between an
    experimental and a simulated pattern.

    Parameters
    ----------
    experimental
        Experimental EBSD pattern.
    simulated
        Simulated EBSD pattern of the same shape as *experimental*,
        used as the reference to match against.

    Returns
    -------
    ncc
        Normalized cross-correlation coefficient in the range
        ``[-1, 1]``. Returns ``0.0`` if either pattern has zero
        variance (e.g. a flat pattern), since the coefficient is
        undefined in that case.
    """
    experimental = np.asarray(experimental, dtype=np.float32)
    simulated = np.asarray(simulated, dtype=np.float32)
    a = experimental - np.mean(experimental)
    b = simulated - np.mean(simulated)
    denominator = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denominator == 0:
        return 0.0
    return float(np.sum(a * b) / denominator)


def _process_pattern_pipeline(
    pattern: np.ndarray,
    reference: np.ndarray,
    dynamic_background_std: float,
    dynamic_background_truncate: float,
    fft_highpass_cutoff: float,
    fft_lowpass_cutoff: float,
    ahe_kernel_size: int,
    ahe_clip_limit: float,
    ahe_nbins: int,
    ahe_on: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Run the three-stage processing pipeline once on a single
    pattern with a given set of parameters.

    The stages are, in order, dynamic background subtraction,
    (optional) adaptive histogram equalization, and FFT bandpass
    filtering. All three are already available as public
    :class:`~kikuchipy.signals.EBSD` methods; this function chains
    their array-level counterparts so a single set of parameters can
    be scored quickly without the overhead of a full signal object.

    Parameters
    ----------
    pattern
        Experimental EBSD pattern to process.
    reference
        Simulated (or otherwise trusted) reference pattern of the same
        shape as *pattern*, used to score each processing stage.
    dynamic_background_std
        Standard deviation passed to
        :func:`~kikuchipy.pattern.remove_dynamic_background`.
    dynamic_background_truncate
        Truncation passed to
        :func:`~kikuchipy.pattern.remove_dynamic_background`.
    fft_highpass_cutoff
        Cutoff of the FFT highpass filter.
    fft_lowpass_cutoff
        Cutoff of the FFT lowpass filter.
    ahe_kernel_size
        Kernel size (in both dimensions) for adaptive histogram
        equalization.
    ahe_clip_limit
        Clip limit for adaptive histogram equalization.
    ahe_nbins
        Number of histogram bins for adaptive histogram equalization.
    ahe_on
        Whether to apply adaptive histogram equalization at all. If
        ``False``, the output of the dynamic background subtraction
        stage is passed through unchanged.

    Returns
    -------
    patterns
        Dictionary with the pattern after each stage, with keys
        ``"raw"``, ``"dynamic_background"``, ``"ahe"``, and ``"fft"``.
    image_quality
        Image quality (see
        :func:`~kikuchipy.pattern.get_image_quality`) after each of
        the four stages above, in that order.
    normalized_cross_correlation
        Normalized cross-correlation against *reference* after each of
        the four stages above, in that order.
    """
    image_quality = np.zeros(4)
    ncc = np.zeros(4)

    image_quality[0] = get_image_quality(pattern, normalize=True)
    ncc[0] = _normalized_cross_correlation(pattern, reference)

    # 1) Dynamic background subtraction
    pattern_dbs = remove_dynamic_background(
        pattern,
        operation="subtract",
        filter_domain="frequency",
        std=int(dynamic_background_std),
        truncate=int(dynamic_background_truncate),
    )
    image_quality[1] = get_image_quality(pattern_dbs, normalize=True)
    ncc[1] = _normalized_cross_correlation(pattern_dbs, reference)

    # 2) Adaptive histogram equalization (optional)
    if ahe_on:
        # scikit-optimize's internal transform/inverse-transform round-trip can
        # hand back numpy scalar types (e.g. np.int64) instead of plain Python
        # ints/floats. scikit-image's equalize_adapthist() does an in-place
        # floor-division on a fixed-width lookup table internally, which
        # raises under NumPy >= 2.0's stricter same-kind casting rules if
        # nbins isn't a plain Python int. Coerce defensively here.
        pattern_ahe = _adaptive_histogram_equalization(
            pattern_dbs,
            kernel_size=(int(ahe_kernel_size), int(ahe_kernel_size)),
            clip_limit=float(ahe_clip_limit),
            nbins=int(ahe_nbins),
        )
    else:
        pattern_ahe = pattern_dbs
    image_quality[2] = get_image_quality(pattern_ahe, normalize=True)
    ncc[2] = _normalized_cross_correlation(pattern_ahe, reference)

    # 3) FFT bandpass filter (lowpass to filter noise, highpass to
    # filter large variations across the detector)
    pattern_shape = pattern.shape
    w_low = Window(
        window="lowpass",
        cutoff=int(fft_lowpass_cutoff),
        cutoff_width=10,
        shape=pattern_shape,
    )
    w_high = Window(
        window="highpass",
        cutoff=int(fft_highpass_cutoff),
        cutoff_width=2,
        shape=pattern_shape,
    )
    pattern_fft = fft_filter(
        pattern_ahe,
        transfer_function=w_low * w_high,
        shift=True,
    )
    image_quality[3] = get_image_quality(pattern_fft, normalize=True)
    ncc[3] = _normalized_cross_correlation(pattern_fft, reference)

    patterns = {
        "raw": pattern,
        "dynamic_background": pattern_dbs,
        "ahe": pattern_ahe,
        "fft": pattern_fft,
    }

    return patterns, image_quality, ncc


def optimize_pattern_processing(
    pattern: np.ndarray,
    reference: np.ndarray,
    dynamic_background_std: tuple[int, int] = (8, 40),
    dynamic_background_truncate: tuple[int, int] = (2, 10),
    fft_highpass_cutoff: tuple[int, int] = (1, 7),
    fft_lowpass_cutoff: tuple[int, int] = (50, 100),
    ahe_kernel_sizes: Sequence[int] = (48, 64, 80, 96, 112, 128, 256),
    ahe_clip_limits: Sequence[float] | None = None,
    ahe_nbins: Sequence[int] = (128, 256, 512),
    n_calls: int = 150,
    n_initial_points: int = 12,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Find pattern processing parameters that maximize the similarity
    between a single EBSD pattern and a reference pattern, using
    Bayesian optimization.

    This searches over dynamic background subtraction, (optional)
    adaptive histogram equalization, and FFT bandpass filtering
    parameters (in that order) via :func:`skopt.gp_minimize`, scoring
    each candidate pipeline by its normalized cross-correlation with
    *reference* after all three stages. The processing steps
    themselves are existing public parts of kikuchipy; only the search
    over their parameters is new here.

    This function operates on a single pattern and does not require
    Dask or MPI. To apply the resulting parameters to a full pattern
    stack, pass ``result["best_parameters"]`` to the corresponding
    :class:`~kikuchipy.signals.EBSD` methods
    (:meth:`~kikuchipy.signals.EBSD.remove_dynamic_background`,
    :meth:`~kikuchipy.signals.EBSD.adaptive_histogram_equalization`,
    :meth:`~kikuchipy.signals.EBSD.fft_filter`).

    Parameters
    ----------
    pattern
        Experimental EBSD pattern to optimize the processing of.
    reference
        Simulated (or otherwise trusted) reference pattern of the same
        shape as *pattern* to match against, e.g. from
        :meth:`~kikuchipy.signals.EBSDMasterPattern.get_patterns`.
    dynamic_background_std
        ``(low, high)`` bounds for the dynamic background subtraction
        standard deviation search space.
    dynamic_background_truncate
        ``(low, high)`` bounds for the dynamic background subtraction
        truncation search space.
    fft_highpass_cutoff
        ``(low, high)`` bounds for the FFT highpass filter cutoff
        search space.
    fft_lowpass_cutoff
        ``(low, high)`` bounds for the FFT lowpass filter cutoff
        search space.
    ahe_kernel_sizes
        Candidate kernel sizes for adaptive histogram equalization.
    ahe_clip_limits
        Candidate clip limits for adaptive histogram equalization. If
        not given, seven values log-spaced between 1e-4 and 5e-3 are
        used.
    ahe_nbins
        Candidate histogram bin counts for adaptive histogram
        equalization.
    n_calls
        Number of calls to the objective function during optimization.
        Default is 150.
    n_initial_points
        Number of initial random evaluations before the Bayesian
        surrogate model takes over. Default is 12.
    random_state
        Seed for reproducible optimization.

    Returns
    -------
    result
        Dictionary with the following keys:

        - ``"best_parameters"``: dictionary with the best-scoring
          parameters found.
        - ``"best_score"``: the normalized cross-correlation of the
          best-scoring parameters.
        - ``"optimize_result"``: the raw :class:`scipy.optimize.OptimizeResult`-like
          object returned by :func:`skopt.gp_minimize`.
        - ``"patterns"``: dictionary with the pattern after each
          processing stage, using the best-scoring parameters. See
          :func:`_process_pattern_pipeline` for the keys.
        - ``"image_quality"``: image quality after each processing
          stage, using the best-scoring parameters.
        - ``"normalized_cross_correlation"``: normalized
          cross-correlation with *reference* after each processing
          stage, using the best-scoring parameters.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.nickel_ebsd_small(allow_download=True).inav[0, 0]  # doctest: +SKIP
    >>> mp = kp.data.nickel_ebsd_master_pattern_small()  # doctest: +SKIP
    >>> simulated = mp.get_patterns(...)  # doctest: +SKIP
    >>> result = kp.pattern.optimize_pattern_processing(
    ...     s.data, simulated.data
    ... )  # doctest: +SKIP
    >>> result["best_parameters"]  # doctest: +SKIP
    """
    verify_dependency_or_raise("scikit-optimize", "Pattern processing optimization")

    from skopt import gp_minimize
    from skopt.space import Categorical, Integer
    from skopt.utils import use_named_args

    if ahe_clip_limits is None:
        ahe_clip_limits = [
            float(f"{v:.6f}") for v in np.logspace(np.log10(1e-4), np.log10(5e-3), 7)
        ]

    dimensions = [
        Integer(*dynamic_background_std, name="dynamic_background_std"),
        Integer(*dynamic_background_truncate, name="dynamic_background_truncate"),
        Integer(*fft_highpass_cutoff, name="fft_highpass_cutoff"),
        Integer(*fft_lowpass_cutoff, name="fft_lowpass_cutoff"),
        Categorical(list(ahe_kernel_sizes), name="ahe_kernel_size"),
        Categorical(list(ahe_clip_limits), name="ahe_clip_limit"),
        Categorical(list(ahe_nbins), name="ahe_nbins"),
        Categorical([False, True], name="ahe_on"),
    ]

    @use_named_args(dimensions)
    def objective(**params: Any) -> float:
        _, _, ncc = _process_pattern_pipeline(pattern, reference, **params)
        # gp_minimize() minimizes, so maximize the NCC by minimizing its negative
        return -ncc[-1]

    optimize_result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_state,
    )

    best_parameters = dict(zip([d.name for d in dimensions], optimize_result.x))
    best_patterns, best_image_quality, best_ncc = _process_pattern_pipeline(
        pattern, reference, **best_parameters
    )

    return {
        "best_parameters": best_parameters,
        "best_score": -optimize_result.fun,
        "optimize_result": optimize_result,
        "patterns": best_patterns,
        "image_quality": best_image_quality,
        "normalized_cross_correlation": best_ncc,
    }
