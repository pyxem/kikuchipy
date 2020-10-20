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

from typing import Union, Tuple
import sys

import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np

from kikuchipy.indexing.similarity_metrics import (
    SIMILARITY_METRICS,
    SimilarityMetric,
    FlatSimilarityMetric,
    MetricScope,
    make_similarity_metric,
)

from kikuchipy.indexing.similarity_metrics import (
    _get_nav_shape,
    _get_number_of_simulated,
)

# Future Work: mask -> maskedarrays


def pattern_match(
    experimental: Union[da.Array, np.ndarray],
    simulated: Union[da.Array, np.ndarray],
    keep_n: int = 1,
    metric: Union[str, SimilarityMetric] = "zncc",
    compute: bool = True,
    n_slices: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the best matching simulations to experimental data based on given metric.

    Function is primarily for use in
    :class:`~kikuchipy.indexing.StaticDictionaryIndexing` and
    :class:`~kikuchipy.indexing.DynamicDictionaryIndexing`.

    Parameters
    ----------
    experimental : da.Array or np.ndarray
        experimental
    simulated : da.Array or np.ndarray
        simulated
    keep_n : int, optional
        Number of match results to keep for each pattern, by default 1
    metric : str or SimilarityMetric, optional
        Similarity metric, by default "zncc".
    compute : bool, optional
        Whether to compute dask arrays before returning, by default True.
    n_slices : int, optional
        Number of simulated slices to process sequentially, by default None.

    Returns
    -------
    simulation_indices, metric_result : Tuple[np.ndarray, np.ndarray]
        Simulation indices and corresponding metric results with data shapes (ny*nx,keep_n).
        Both arrays are sorted along keep_n axis according to metric used.
    """
    metric = SIMILARITY_METRICS.get(metric, metric)
    if n_slices is not None:
        if not compute:
            raise NotImplementedError(
                "Slicing simulations and returning dask arrays is not implemented."
            )

        return _pattern_match_slice_simulated(
            experimental,
            simulated,
            keep_n=keep_n,
            metric=metric,
            n_slices=n_slices,
        )
    if not isinstance(metric, SimilarityMetric):
        raise ValueError(
            f"{metric} must be either of {list(SIMILARITY_METRICS.keys())} "
            "or an instance of SimilarityMetric. See make_similarity_metric."
        )

    # Expects signal data to be located on the two last axis for all scopes
    sig_data_shape = experimental.shape[-2:]
    t_sig_shape = simulated.shape[-2:]
    if sig_data_shape != t_sig_shape:
        raise OSError(
            f"The experimental {sig_data_shape} and simulated {t_sig_shape} "
            "signal shapes are not identical."
        )

    if not metric._is_compatible(experimental.ndim, simulated.ndim):
        raise OSError(
            f"The shape of experimental {experimental.shape} and simulated {simulated.shape} "
            f"are not compatible with the scope {metric.scope} of {type(metric).__name__}"
        )

    similarities = metric(experimental, simulated)
    similarities = da.asarray(similarities)

    # ONE_TO_ONE
    if similarities.shape == ():
        similarity = (
            np.array([similarities.compute()]) if compute else similarities
        )
        return np.array([0]), similarity

    # If N is < keep_n => keep_n = N
    keep_n = min(keep_n, len(simulated))

    match_result = (
        similarities.argtopk(metric.sign * keep_n, axis=-1),
        similarities.topk(metric.sign * keep_n, axis=-1),
    )
    if compute:
        with ProgressBar():
            match_result = da.compute(*match_result)

    # Flattens the signal axis if not already flat
    # This is foremost a design choice for returning standard outputs
    if not metric.flat:
        match_result = (
            match_result[0].reshape(-1, keep_n),
            match_result[1].reshape(-1, keep_n),
        )
    simulation_indices, metric_results = match_result
    return simulation_indices, metric_results


def _pattern_match_slice_simulated(
    experimental: Union[da.Array, np.ndarray],
    simulated: Union[da.Array, np.ndarray],
    keep_n: int = 1,
    metric: Union[str, SimilarityMetric] = "zncc",
    n_slices: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """See `pattern_match`.

    Parameters
    ----------
    experimental : da.Array or np.ndarray
        Experimental data
    simulated : da.Array or np.ndarray
        Simulated data
    keep_n : int, optional
        Number of results to keep, by default 1
    metric : str or SimilarityMetric, optional
        Similarity metric, by default "zncc".
    n_slices : int, optional
        Number of simulation slices to process sequentially, by default None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        [description]
    """

    # This is a naive implementation, hopefully not stupid, of slicing the simulated in batches
    # without thinking about aligining with dask chunks or rechunking
    # dask seem to handle the sequential slicing decently

    nav_shape = _get_nav_shape(experimental)
    nav_size = np.prod(nav_shape)
    num_simulated = _get_number_of_simulated(simulated)
    slice_size = num_simulated // n_slices

    n = min(keep_n, slice_size)
    match_result_aggregate = (
        np.zeros((nav_size, n_slices * n), np.int32),
        np.zeros((nav_size, n_slices * n), metric._dtype_out),
    )

    start = 0
    for i in range(n_slices):
        end = start + slice_size if i != n_slices - 1 else num_simulated

        match_result = pattern_match(
            experimental,
            simulated[start:end],
            keep_n=keep_n,
            metric=metric,
            compute=False,
        )
        match_result = list(match_result)

        # adjust simulation indicies matches to correspond with original simulated
        match_result[0] += start

        result_slice = np.s_[:, i * n : (i + 1) * n]
        with ProgressBar():
            print(f"simulation matching {i+1}/{n_slices}:", file=sys.stdout)
            da.store(
                match_result,
                [
                    match_result_aggregate[0][result_slice],
                    match_result_aggregate[1][result_slice],
                ],
                # regions=(slice(......)) # should be possible, but do we gain anything?
            )

        start += slice_size

    match_result = (
        np.zeros((nav_size, n), np.int32),
        np.zeros((nav_size, n), np.float32),
    )
    for i in range(nav_size):
        indices = (metric.sign * -match_result_aggregate[1][i]).argsort(
            kind="mergesort"
        )[:keep_n]
        match_result[0][i] = match_result_aggregate[0][i][indices]
        match_result[1][i] = match_result_aggregate[1][i][indices]

    return match_result
