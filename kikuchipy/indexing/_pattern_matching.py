# -*- coding: utf-8 -*-
# Copyright 2019-2020 The kikuchipy developers
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

"""Matching of experimental to simulated gray-tone patterns."""

from typing import Optional, Tuple, Union

import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
import psutil
from tqdm import tqdm

from kikuchipy.indexing.similarity_metrics import (
    _SIMILARITY_METRICS,
    SimilarityMetric,
    _get_nav_shape,
    _get_number_of_simulated,
)


# TODO: Support masking signal space
# TODO: Support masking navigation space


def _pattern_match(
    experimental: Union[da.Array, np.ndarray],
    simulated: Union[da.Array, np.ndarray],
    keep_n: int = 50,
    metric: Union[str, SimilarityMetric] = "ncc",
    compute: bool = True,
    n_slices: int = 1,
    phase_name: Optional[str] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[da.Array, da.Array]]:
    """Find the best matching simulations to experimental data based on
    given `metric`.

    Function is primarily for use in
    :class:`~kikuchipy.indexing.StaticPatternMatching`.

    Parameters
    ----------
    experimental : numpy.ndarray or dask.array.Array
        Experimental patterns.
    simulated : numpy.ndarray or dask.array.Array
        Simulated patterns.
    keep_n : int, optional
        Number of match results to keep for each pattern, by default 50.
    metric : str or SimilarityMetric
        Similarity metric, by default "ncc".
    compute : bool, optional
        Whether to compute dask arrays before returning, by default
        True.
    n_slices : int, optional
        Number of simulated slices to process sequentially. Default is
        1, i.e. the simulated pattern array is not sliced.
    phase_name : str, optional
        Simulated patterns phase name, shown in the progressbar if
        `n_slices` > 1.

    Returns
    -------
    simulation_indices : numpy.ndarray or dask.array.Array
        Simulation indices corresponding with metric results.
    scores : numpy.ndarray or dask.array.Array
        Metric results with data shapes (ny*nx, keep_n). Sorted along
        `keep_n` axis according to the metric used.
    """
    metric = _SIMILARITY_METRICS.get(metric, metric)
    if not isinstance(metric, SimilarityMetric):
        raise ValueError(
            f"{metric} must be either of {list(_SIMILARITY_METRICS.keys())} "
            "or an instance of SimilarityMetric. See "
            "kikuchipy.indexing.similarity_metrics.make_similarity_metric()"
        )

    # Expects signal data to be located on the two last axis for all scopes
    sig_data_shape = experimental.shape[-2:]
    t_sig_shape = simulated.shape[-2:]
    if sig_data_shape != t_sig_shape:
        raise OSError(
            f"The experimental {sig_data_shape} and simulated {t_sig_shape} "
            "signal shapes are not identical"
        )

    if not metric._is_compatible(experimental.ndim, simulated.ndim):
        raise OSError(
            f"The shape of experimental {experimental.shape} and simulated "
            f"{simulated.shape} are not compatible with the scope "
            f"{metric.scope} of {type(metric).__name__}"
        )

    keep_n = min(keep_n, _get_number_of_simulated(simulated))

    if n_slices == 1:
        return _pattern_match_single_slice(
            experimental,
            simulated,
            keep_n=keep_n,
            metric=metric,
            compute=compute,
        )
    else:
        if not compute:
            raise NotImplementedError(
                "Slicing simulations and returning dask arrays is not "
                "implemented"
            )
        return _pattern_match_slice_simulated(
            experimental,
            simulated,
            keep_n=keep_n,
            metric=metric,
            n_slices=n_slices,
            phase_name=phase_name,
        )


def _pattern_match_slice_simulated(
    experimental: Union[np.ndarray, da.Array],
    simulated: Union[np.ndarray, da.Array],
    keep_n: int,
    metric: SimilarityMetric,
    n_slices: int = 1,
    phase_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """See :func:`_pattern_match`.

    Parameters
    ----------
    experimental : numpy.ndarray or dask.array.Array
        Experimental patterns.
    simulated : numpy.ndarray or dask.array.Array
        Simulated patterns.
    keep_n : int
        Number of results to keep.
    metric : SimilarityMetric
        Similarity metric.
    n_slices : int, optional
        Number of simulation slices to process sequentially. Default is
        1 (no slicing).
    phase_name : str, optional
        Simulated patterns phase name, shown in the progressbar.

    Returns
    -------
    simulation_indices : numpy.ndarray or dask.array.Array
        Ranked simulation indices corresponding to metric results.
    scores : numpy.ndarray or dask.array.Array
        Ranked metric results with data shapes (ny*nx, keep_n).
    """
    # TODO: Try to respect/utilize chunks when slicing

    nav_shape = _get_nav_shape(experimental)
    nav_size = int(np.prod(nav_shape))
    num_simulated = _get_number_of_simulated(simulated)
    slice_size = num_simulated // n_slices

    n = min(keep_n, num_simulated)
    aggregate_shape = (nav_size, n_slices * n)
    result_shape = (nav_size, n)
    simulated_indices_aggregate = np.zeros(aggregate_shape, np.int32)
    scores_aggregate = np.zeros(aggregate_shape, metric._dtype_out)

    if phase_name is None or phase_name == "":
        desc = "Matching patterns"
    else:
        desc = f"Matching {phase_name[:10]} patterns"

    with tqdm(
        iterable=range(n_slices), desc=desc, unit="slice", total=n_slices
    ) as t:
        start = 0
        for i in range(n_slices):
            t.set_postfix_str(f"mem={psutil.virtual_memory().percent}%")

            if i != n_slices - 1:
                end = start + slice_size
            else:  # Last iteration
                end = num_simulated

            simulated_indices, scores = _pattern_match_single_slice(
                experimental,
                simulated[start:end],
                keep_n=keep_n,
                metric=metric,
                compute=False,
            )

            # Adjust simulation indices matches to correspond with
            # original simulated
            simulated_indices += start

            da.store(
                sources=[simulated_indices, scores],
                targets=[simulated_indices_aggregate, scores_aggregate],
                regions=[np.s_[:, i * n : (i + 1) * n]] * 2,
            )

            start += slice_size

            # Update progressbar
            t.update()

            # TODO: Perform a test to see if memory use in this loop
            #  would benefit from garbage collection
            # gc.collect()

    simulated_indices = np.zeros(result_shape, np.int32)
    scores = np.zeros(result_shape, metric._dtype_out)
    for i in range(nav_size):
        indices = (metric.sign * -scores_aggregate[i]).argsort(
            kind="mergesort"
        )[:keep_n]
        simulated_indices[i] = simulated_indices_aggregate[i][indices]
        scores[i] = scores_aggregate[i][indices]

    return simulated_indices, scores


def _pattern_match_single_slice(
    experimental: Union[np.ndarray, da.Array],
    simulated: Union[np.ndarray, da.Array],
    keep_n: int,
    metric: SimilarityMetric,
    compute: bool,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[da.Array, da.Array]]:
    """See :func:`_pattern_match`.

    Parameters
    ----------
    experimental : numpy.ndarray or dask.array.Array
        Experimental patterns.
    simulated : numpy.ndarray or dask.array.Array
        Simulated patterns.
    keep_n : int
        Number of results to keep.
    metric : SimilarityMetric
        Similarity metric.
    compute : bool
        Whether to compute dask arrays before returning the results.

    Returns
    -------
    simulation_indices : numpy.ndarray or dask.array.Array
        Ranked simulation indices corresponding to metric results.
    scores : numpy.ndarray or dask.array.Array
        Ranked metric results with data shapes (ny*nx, keep_n).
    """
    similarities = metric(experimental, simulated)
    similarities = da.asarray(similarities)

    # If MetricScope.ONE_TO_ONE
    if similarities.shape == ():
        similarity = (
            np.array([similarities.compute()]) if compute else similarities
        )
        return np.array([0]), similarity

    # keep_n_aggregate: If N is < keep_n => keep_n = N
    keep_n = min(keep_n, len(simulated))

    simulated_indices = similarities.argtopk(metric.sign * keep_n, axis=-1)
    scores = similarities.topk(metric.sign * keep_n, axis=-1)

    if compute:
        with ProgressBar():
            simulated_indices, scores = da.compute(simulated_indices, scores)

    # Flattens the signal axis if not already flat. This is foremost a
    # design choice for returning standard outputs.
    if not metric.flat:
        simulated_indices = simulated_indices.reshape(-1, keep_n)
        scores = scores.reshape(-1, keep_n)

    return simulated_indices, scores
