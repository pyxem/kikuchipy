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
    _get_number_of_templates,
)

# Future Work: mask -> maskedarrays


def template_match(
    patterns: Union[da.Array, np.ndarray],
    templates: Union[da.Array, np.ndarray],
    keep_n: int = 1,
    metric: Union[str, SimilarityMetric] = "zncc",
    compute: bool = True,
    n_slices: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """[summary. Haakon read Returns to understand]

    Parameters
    ----------
    patterns : Union[da.Array, np.ndarray]
        [description]
    templates : Union[da.Array, np.ndarray]
        [description]
    keep_n : int, optional
        [description], by default 1
    metric : Union[str, SimilarityMetric], optional
        [description], by default "zncc"
    compute : bool, optional
        [description, dask compute, computes anyway (for now) if n_slices is given], by default True,
    n_slices : int, optional
        [description], by default None

    Returns
    -------
    match_result : Tuple[np.ndarray, np.ndarray]
        [both arrays have shape (ny*nx,keep_n)
        first array is template indicies and
        second array is metric results
        both are sorted along keep_n axis according to metric used]
    """
    if n_slices is not None:
        # Will (for now) dask compute regardless of compute param
        return _template_match_slice_templates(
            patterns,
            templates,
            keep_n=keep_n,
            metric=metric,
            n_slices=n_slices,
        )

    metric = SIMILARITY_METRICS.get(metric, metric)
    if not isinstance(metric, SimilarityMetric):
        raise ValueError(
            f"{metric} must be either of {list(SIMILARITY_METRICS.keys())} "
            "or an instance of SimilarityMetric. See make_similarity_metric."
        )
    accepeted_scopes = (MetricScope.MANY_TO_MANY, MetricScope.ONE_TO_MANY)
    if not metric.scope in accepeted_scopes:
        raise ValueError(
            f"{metric.scope} must be either of {accepeted_scopes}."
        )

    # check if data is too low scoped
    # could be a function in similarity_metrics for making it cleaner here
    # this check makes _is_compatible uneccesarry
    if (
        not metric._P_T_NDIM_TO_SCOPE.get(
            (patterns.ndim, templates.ndim), False
        )
        in accepeted_scopes
    ):
        raise OSError(
            f"The shape of patterns and templates must correspond with either of {accepeted_scopes}\n"
            f"The shapes; {patterns.shape}, {templates.shape} was given."
        )

    # Expects signal data to be located on the two last axis for all scopes
    sig_data_shape = patterns.shape[-2:]
    t_sig_shape = templates.shape[-2:]
    if sig_data_shape != t_sig_shape:
        raise OSError(
            f"The pattern {sig_data_shape} and template {t_sig_shape} "
            "signal shapes are not identical."
        )

    similarities = metric(patterns, templates)
    if not isinstance(similarities, da.Array):
        similarities = da.from_array(similarities)

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
        # If N is < keep_n => keep_n = N
        keep_n = match_result[0].shape[-1]
        match_result = (
            match_result[0].reshape(-1, keep_n),
            match_result[1].reshape(-1, keep_n),
        )
    return match_result


def _template_match_slice_templates(
    patterns: Union[da.Array, np.ndarray],
    templates: Union[da.Array, np.ndarray],
    keep_n: int = 1,
    metric: Union[str, SimilarityMetric] = "zncc",
    n_slices: int = None,
) -> Tuple[np.ndarray, np.ndarray]:

    # This is a naive implementation, hopefully not stupid, of slicing the templates in batches
    # without thinking about aligining with dask chunks or rechunking
    # dask seem to handle the sequential slicing decently

    metric = SIMILARITY_METRICS.get(metric, metric)

    nav_shape = _get_nav_shape(patterns)
    nav_size = np.prod(nav_shape)
    num_templates = _get_number_of_templates(templates)
    slice_size = num_templates // n_slices

    n = min(keep_n, slice_size)
    match_result_aggregate = (
        np.zeros((nav_size, n_slices * n), np.int32),
        np.zeros((nav_size, n_slices * n), np.float32),
    )

    start = 0
    for i in range(n_slices):
        end = start + slice_size if i != n_slices - 1 else num_templates

        match_result = template_match(
            patterns,
            templates[start:end],
            keep_n=keep_n,
            metric=metric,
            compute=False,
        )
        match_result = list(match_result)

        # adjust template indicies matches to correspond with original templates
        match_result[0] += start

        result_slice = np.s_[:, i * n : (i + 1) * n]
        with ProgressBar():
            print(f"Template matching {i+1}/{n_slices}:", file=sys.stdout)
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
