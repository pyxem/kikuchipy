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
from typing import Union

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
        n_dictionary_per_iteration: int,
        keep_dictionary_lazy: bool,
        n_experimental_per_iteration: int,
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
    n_dictionary_per_iteration
    keep_dictionary_lazy
    n_experimental_per_iteration
    navigation_mask
    signal_mask

    Returns
    -------
    xmap
    """
    # handle dictionary reshaping
    dictionary_size = dictionary.shape[0]
    dictionary = dictionary.reshape((dictionary_size, -1))

    # handle experimental pattern reshaping
    n_experimental_all = int(np.prod(experimental_nav_shape))
    experimental = experimental.reshape((n_experimental_all, -1))

    # mask the dictionary and the experimental patterns
    if signal_mask is not None:
        dictionary = dictionary[:, ~signal_mask.ravel()]
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            experimental = experimental[:, ~signal_mask.ravel()]

    # cap out the number of dictionary patterns per iteration
    n_dictionary_per_iteration = min(n_dictionary_per_iteration, dictionary_size)

    # cap out the number of experimental patterns per iteration
    n_experimental_per_iteration = min(n_experimental_per_iteration, n_experimental_all)

    # cap out the number of dictionary patterns to keep
    if keep_n >= dictionary_size:
        raise ValueError(f"keep_n of {keep_n} must be smaller than the dictionary size of {dictionary_size}")

    # prepare the output arrays
    simulation_indices = np.zeros((n_experimental_all, keep_n), dtype=np.int32)
    distances = np.full((n_experimental_all, keep_n), np.inf, dtype=np.float32)

    dictionary_is_lazy = isinstance(dictionary, da.Array)

    # calculate the number of loops for the dictionary and the start and end indices for each loop
    n_dictionary_iterations = int(np.ceil(dictionary_size / n_dictionary_per_iteration))
    dictionary_chunk_starts = np.cumsum([0] + [n_dictionary_per_iteration] * (n_dictionary_iterations - 1))
    dictionary_chunk_ends = np.cumsum([n_dictionary_per_iteration] * n_dictionary_iterations)
    dictionary_chunk_ends[-1] = max(dictionary_chunk_ends[-1], dictionary_size)

    # calculate the number of loops for the experimental patterns and the start and end indices for each loop
    n_experimental_iterations = int(np.ceil(n_experimental_all / n_experimental_per_iteration))
    experimental_chunk_starts = np.cumsum([0] + [n_experimental_per_iteration] * (n_experimental_iterations - 1))
    experimental_chunk_ends = np.cumsum([n_experimental_per_iteration] * n_experimental_iterations)
    experimental_chunk_ends[-1] = max(experimental_chunk_ends[-1], n_experimental_all)

    # start timer
    time_start = time()
    # outer loop over dictionary chunks
    for dictionary_start, dictionary_end in tqdm(zip(dictionary_chunk_starts, dictionary_chunk_ends),
                                                 total=n_dictionary_iterations,
                                                 desc="Dictionary loop",
                                                 position=0):
        dictionary_chunk = dictionary[dictionary_start:dictionary_end]
        if dictionary_is_lazy and not keep_dictionary_lazy:
            dictionary_chunk = dictionary_chunk.compute()

        # calculate the number of (possible) dictionary patterns to keep per experimental pattern chunk
        n_required = min(keep_n, dictionary_end - dictionary_start)

        # set the indexer with the dictionary chunk
        indexer(dictionary_chunk, n_required)

        del dictionary_chunk

        # set the closest dictionary pattern indices and scores for the new dictionary chunk
        simulation_indices_new = np.empty((n_experimental_all, n_required), dtype=np.int32)
        distances_new = np.empty((n_experimental_all, n_required), dtype=np.float32)

        # inner loop over experimental pattern chunks
        for exp_start, exp_end in tqdm(zip(experimental_chunk_starts, experimental_chunk_ends),
                                       total=n_experimental_iterations,
                                       desc="Experiment loop",
                                       position=1,
                                       leave=False):
            experimental_chunk = experimental[exp_start:exp_end]
            # if the experimental chunk is lazy, compute it
            if isinstance(experimental_chunk, da.Array):
                experimental_chunk = experimental_chunk.compute()
            # query the experimental chunk
            simulation_indices_mini, distances_mini = indexer.query(experimental_chunk)
            # simulation_indices_mini, distances_mini = simulation_indices_mini, distances_mini
            # fill the new indices and scores with the mini indices and scores
            simulation_indices_new[exp_start:exp_end] = simulation_indices_mini
            distances_new[exp_start:exp_end] = distances_mini

        # add the dictionary start index to the simulation indices to convert them to the global indices
        simulation_indices_new += dictionary_start
        # concatenate the old and new indices and scores
        all_distances = np.hstack((distances, distances_new))
        all_simulation_indices = np.hstack((simulation_indices, simulation_indices_new))
        best_indices = np.argsort(all_distances, axis=1)[:, :keep_n]
        distances = np.take_along_axis(all_distances, best_indices, axis=1)
        simulation_indices = np.take_along_axis(all_simulation_indices, best_indices, axis=1)

    # stop timer and calculate indexing speed
    query_time = time() - time_start
    patterns_per_second = n_experimental_all / query_time
    # Without this pause, a part of the red tqdm progressbar background is displayed below this print
    sleep(0.2)
    print(
        f"  Indexing speed: {patterns_per_second:.5f} patterns/s, "
    )

    xmap_kw, _ = create_coordinate_arrays(experimental_nav_shape, step_sizes)
    if navigation_mask is not None:
        nav_mask = ~navigation_mask.ravel()
        xmap_kw["is_in_data"] = nav_mask

        rot = Rotation.identity((n_experimental_all, keep_n))
        rot[nav_mask] = dictionary_xmap.rotations[simulation_indices].data

        scores_all = np.empty((n_experimental_all, keep_n), dtype=distances.dtype)
        scores_all[nav_mask] = distances
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
        xmap_kw["prop"] = {"scores": distances, "simulation_indices": simulation_indices}
    xmap = CrystalMap(phase_list=dictionary_xmap.phases_in_data, **xmap_kw)

    return xmap
