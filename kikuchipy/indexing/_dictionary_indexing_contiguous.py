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

"""Private tools for approximate dictionary indexing of experimental
 patterns to a dictionary of simulated patterns with known orientations.
"""

from time import sleep, time
from typing import Optional, Tuple, Union

import dask
import dask.array as da
import numpy as np
from orix.crystal_map import create_coordinate_arrays, CrystalMap
from orix.quaternion import Rotation

from kikuchipy.indexing.di_indexers import DIIndexer

from tqdm import tqdm


def _custom_dictionary_indexing(
        experimental: Union[np.ndarray, da.Array],
        experimental_nav_shape: tuple,
        dictionary: Union[np.ndarray, da.Array],
        step_sizes: tuple,
        dictionary_xmap: CrystalMap,
        indexer: DIIndexer,
        keep_n: int,
        n_per_iteration: int,
        navigation_mask: Union[np.ndarray, None],
        signal_mask: Union[np.ndarray, None],
) -> CrystalMap:
    """Dictionary indexing matching experimental patterns to a
    dictionary of simulated patterns of known orientations.

    See :meth:`~kikuchipy.signals.EBSD.custom_di_indexing`.

    Parameters
    ----------
    experimental
    experimental_nav_shape
    dictionary
    step_sizes
    dictionary_xmap
    indexer
    keep_n
    n_per_iteration
    navigation_mask
    signal_mask

    Returns
    -------
    xmap
    """
    # the load and save paths must be different if both are specified (to avoid
    # overwriting the original graph with a new one with different parameters)

    # the entire dictionary must fit in memory
    if isinstance(dictionary, da.Array):
        dictionary = dictionary.compute()
    dictionary_size = dictionary.shape[0]
    dictionary = dictionary.reshape((dictionary_size, -1))

    # handle experimental pattern reshaping
    n_experimental_all = int(np.prod(experimental_nav_shape))
    experimental = experimental.reshape((n_experimental_all, -1))

    # set n_per_iteration to the maximum possible value if it is 0
    if n_per_iteration == 0:
        n_per_iteration = n_experimental_all

    # if n_per_iteration is larger than the number of experimental patterns,
    # set it to the number of experimental patterns
    n_per_iteration = min(n_per_iteration, n_experimental_all)

    if signal_mask is not None:
        dictionary = dictionary[:, ~signal_mask.ravel()]
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            experimental = experimental[:, ~signal_mask.ravel()]

    keep_n = min(keep_n, dictionary_size)

    simulation_indices = np.zeros((n_experimental_all, keep_n), dtype=np.int32)
    scores = np.empty((n_experimental_all, keep_n), dtype=np.float32)

    n_iterations = int(np.ceil(n_experimental_all / n_per_iteration))

    chunk_starts = np.cumsum([0] + [n_per_iteration] * (n_iterations - 1))
    chunk_ends = np.cumsum([n_per_iteration] * n_iterations)
    chunk_ends[-1] = max(chunk_ends[-1], n_experimental_all)

    experimental_is_lazy = isinstance(experimental, da.Array)

    # ingest the dictionary into the indexing function
    time_start = time()
    indexer(dictionary, keep_n)
    build_time = time() - time_start

    time_start = time()
    for start, end in tqdm(zip(chunk_starts, chunk_ends), total=n_iterations):
        experimental_chunk = experimental[start:end]
        if experimental_is_lazy:
            experimental_chunk = experimental_chunk.compute()
        simulation_indices[start:end], scores[start:end] = indexer.query(experimental_chunk)
        del experimental_chunk
    query_time = time() - time_start

    patterns_per_second = n_experimental_all / query_time
    # Without this pause, a part of the red tqdm progressbar background
    # is displayed below this print
    sleep(0.2)
    print(
        f"  Graph build (or load) time: {build_time:.5f} s, "
        f"  Indexing speed: {patterns_per_second:.5f} patterns/s, "
    )

    xmap_kw, _ = create_coordinate_arrays(experimental_nav_shape, step_sizes)
    if navigation_mask is not None:
        nav_mask = ~navigation_mask.ravel()
        xmap_kw["is_in_data"] = nav_mask

        rot = Rotation.identity((n_experimental_all, keep_n))
        rot[nav_mask] = dictionary_xmap.rotations[simulation_indices].data

        scores_all = np.empty((n_experimental_all, keep_n), dtype=scores.dtype)
        scores_all[nav_mask] = scores
        simulation_indices_all = np.empty(
            (n_experimental_all, keep_n), dtype=simulation_indices.dtype
        )
        simulation_indices_all[nav_mask] = simulation_indices
        if keep_n == 1:
            rot = rot.flatten()
            scores_all = scores_all.squeeze()
            simulation_indices_all = simulation_indices_all.squeeze()
        xmap_kw["rotations"] = rot
        xmap_kw["prop"] = {
            "scores": scores_all,
            "simulation_indices": simulation_indices_all,
        }
    else:
        xmap_kw["rotations"] = dictionary_xmap.rotations[simulation_indices.squeeze()]
        xmap_kw["prop"] = {"scores": scores, "simulation_indices": simulation_indices}
    xmap = CrystalMap(phase_list=dictionary_xmap.phases_in_data, **xmap_kw)

    return xmap
