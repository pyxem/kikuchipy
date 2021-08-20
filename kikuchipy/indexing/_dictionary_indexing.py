# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
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

from typing import Tuple

import dask.array as da
import numba as nb
import numpy as np
from orix.crystal_map import create_coordinate_arrays, CrystalMap
from tqdm import trange

from kikuchipy.indexing.similarity_metrics import SimilarityMetric


def _dictionary_indexing(
    experimental: da.Array,
    experimental_nav_shape: tuple,
    dictionary: da.Array,
    dictionary_size: int,
    step_sizes: tuple,
    dictionary_xmap: CrystalMap,
    metric: SimilarityMetric,
    keep_n: int,
    n_per_iteration: int,
):
    keep_n = min(keep_n, dictionary_size)
    n_iterations = dictionary_size // n_per_iteration

    n_experimental = int(np.prod(experimental_nav_shape))
    aggregate_shape = (n_experimental, n_iterations * keep_n)
    simulated_indices_aggregate = np.zeros(aggregate_shape, dtype=np.int32)
    scores_aggregate = np.zeros(aggregate_shape, dtype=metric.dtype)

    experimental = metric.prepare_all_experimental(experimental, n_per_iteration)

    dictionary = dictionary.reshape((dictionary_size, -1))
    if metric.can_rechunk:
        dictionary = dictionary.rechunk((n_per_iteration, -1))

    lazy_dictionary = isinstance(dictionary, da.Array)

    chunk_start = 0
    for i in trange(n_iterations):
        if i != n_iterations - 1:
            chunk_end = chunk_start + n_per_iteration
        else:  # Last iteration
            chunk_end = dictionary_size

        if lazy_dictionary:
            simulated = dictionary[chunk_start:chunk_end].compute()
        else:
            simulated = dictionary[chunk_start:chunk_end]

        sim_idx, scores = _compare_all_experimental_to_dictionary_chunk(
            experimental,
            simulated,
            keep_n=keep_n,
            metric=metric,
        )
        sim_idx += chunk_start
        da.store(
            sources=[sim_idx, scores],
            targets=[simulated_indices_aggregate, scores_aggregate],
            regions=[np.s_[:, i * keep_n : (i + 1) * keep_n]] * 2,
        )

        chunk_start += n_per_iteration

    simulation_indices, scores = _sort_indices_scores(
        n_experimental=n_experimental,
        keep_n=keep_n,
        dtype=metric.dtype,
        sign=metric.sign,
        indices_aggregate=simulated_indices_aggregate,
        scores_aggregate=scores_aggregate,
    )

    coordinate_arrays, _ = create_coordinate_arrays(
        shape=experimental_nav_shape, step_sizes=step_sizes
    )
    xmap = CrystalMap(
        rotations=dictionary_xmap.rotations[simulation_indices],
        phase_list=dictionary_xmap.phases_in_data,
        prop={"scores": scores, "simulation_indices": simulation_indices},
        **coordinate_arrays,
    )

    return xmap


def _compare_all_experimental_to_dictionary_chunk(
    experimental: da.Array,
    simulated: np.ndarray,
    keep_n: int,
    metric: SimilarityMetric,
) -> Tuple[da.Array, da.Array]:
    sim = metric.prepare_chunk_simulated(simulated)
    similarities = metric.compare(experimental, sim)

    # keep_n_aggregate: If N is < keep_n => keep_n = N
    keep_n = min(keep_n, simulated.shape[0])

    simulated_indices = similarities.argtopk(keep_n, axis=-1)
    simulated_indices = simulated_indices.reshape(-1, keep_n)

    scores = similarities.topk(keep_n, axis=-1)
    scores = scores.reshape(-1, keep_n)

    return simulated_indices, scores


@nb.jit(cache=True, nogil=True, nopython=True)
def _sort_indices_scores(
    n_experimental: int,
    keep_n: int,
    dtype: np.dtype,
    sign: int,
    indices_aggregate: np.ndarray,
    scores_aggregate: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    simulation_indices = np.zeros((n_experimental, keep_n), np.int32)
    scores = np.zeros((n_experimental, keep_n), dtype=dtype)
    for i in nb.prange(n_experimental):
        scores_aggregate_i = sign * -scores_aggregate[i]
        indices = np.argsort(scores_aggregate_i, kind="mergesort")[:keep_n]
        simulation_indices[i] = indices_aggregate[i][indices]
        scores[i] = scores_aggregate[i][indices]
    return simulation_indices, scores
