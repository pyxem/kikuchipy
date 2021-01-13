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

from math import copysign
from typing import List, Union, Tuple
import warnings

import numpy as np
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion.rotation import Rotation

from kikuchipy.indexing.similarity_metrics import (
    SimilarityMetric,
    _SIMILARITY_METRICS,
)


def merge_crystal_maps(
    crystal_maps: List[CrystalMap],
    mean_n_best: int = 1,
    metric: Union[str, SimilarityMetric] = None,
    simulation_indices_prop: str = "simulation_indices",
    scores_prop: str = "scores",
):
    """Merge a list of at least two single phase
    :class:`~orix.crystal_map.crystal_map.CrystalMap` with a 1D or 2D
    navigation shape into one multi phase map.

    It is required that there are at least as many simulation indices as
    scores per point, and that all maps have the same number of
    rotations, scores and simulation indices per point.

    Parameters
    ----------
    crystal_maps : list of\
            :class:`~orix.crystal_map.crystal_map.CrystalMap`
        A list of crystal maps with simulated indices and scores among
        their properties.
    mean_n_best : int, optional
        Number of best metric results to take the mean of before
        comparing. Default is 1.
    metric : str or SimilarityMetric, optional
        Similarity metric, default is None.
    simulation_indices_prop : str, optional
        Name of simulated indices array in the crystal maps' properties.
        Default is "simulation_indices".
    scores_prop : str, optional
        Name of scores array in the crystal maps' properties. Default
        is "scores".

    Returns
    -------
    merged_xmap : ~orix.crystal_map.crystal_map.CrystalMap
        A crystal map where the rotation of the phase with the best
        matching score(s) is assigned to each point. The best matching
        simulation indices and scores, merge sorted, are added to its
        properties with names equal to whatever passed to `scores_prop`
        and `simulation_indices_prop` with "merged" as a suffix,
        respectively.

    Notes
    -----
    `mean_n_best` can be given with a negative sign if `metric` is not
    given, in order to choose the lowest valued metric results.
    """
    map_shapes = [xmap.shape for xmap in crystal_maps]
    if not np.sum(abs(np.diff(map_shapes, axis=0))) == 0:
        raise ValueError("All crystal maps must have the same navigation shape")

    rot_per_point_per_map = [xmap.rotations_per_point for xmap in crystal_maps]
    if not all(np.diff(rot_per_point_per_map) == 0):
        raise ValueError(
            "All crystal maps must have the same number of rotations, scores "
            "and simulation indices per point."
        )

    if metric is None:
        sign = copysign(1, mean_n_best)
        mean_n_best = abs(mean_n_best)
    else:
        sign = _SIMILARITY_METRICS.get(metric, metric).sign

    # Notation used in the comments below:
    # - M: number of map points
    # - N: number of scores per point
    # - I: number of simulation indices per point
    # - K: number of maps to merge

    # Shape of the combined (unsorted) scores array, and the total
    # number of scores per point. Shape: (M, N, K) or (M, K) if only one
    # score is available (e.g. refined dot products from EMsoft)
    (comb_shape, n_scores_per_point) = _get_combined_scores_shape(
        crystal_maps=crystal_maps, scores_prop=scores_prop
    )

    # Combined (unsorted) scores array of shape (M, N, K) or (M, K)
    combined_scores = np.dstack(
        [xmap.prop[scores_prop] for xmap in crystal_maps]
    )
    combined_scores = combined_scores.reshape(comb_shape)

    # Best score in each map point
    if n_scores_per_point > 1:  # (M, N, K)
        best_scores = np.mean(combined_scores[:, :mean_n_best], axis=1)
    else:  # (M, K)
        best_scores = combined_scores

    # Phase of best score in each map point
    phase_id = np.argmax(sign * best_scores, axis=1)

    # Get the new CrystalMap's rotations, scores and indices, restricted
    # to one phase per point (uncombined)
    new_rotations = Rotation(np.zeros_like(crystal_maps[0].rotations.data))
    new_scores = np.zeros_like(crystal_maps[0].prop[scores_prop])
    new_indices = np.zeros_like(crystal_maps[0].prop[simulation_indices_prop])
    phase_list = PhaseList()
    for i, xmap in enumerate(crystal_maps):
        mask = phase_id == i
        new_rotations[mask] = xmap.rotations[mask]
        new_scores[mask] = xmap.prop[scores_prop][mask]
        new_indices[mask] = xmap.prop[simulation_indices_prop][mask]
        if np.sum(mask) != 0:
            current_id = xmap.phases_in_data.ids[0]
            phase = xmap.phases_in_data[current_id].deepcopy()
            try:
                phase_list.add(phase)
            except ValueError:
                name = phase.name
                warnings.warn(
                    f"There are duplicates of phase {name}, will therefore "
                    f"rename this phase's name to {name + str(i)} in the merged"
                    " PhaseList",
                )
                phase.name = name + str(i)
                phase_list.add(phase)

    # To get the combined, best, sorted scores and simulation indices
    # from all maps (phases), we collapse the second and (potentially)
    # third axis to get (M, N * K) or (M, K)
    mergesort_shape = (comb_shape[0], np.prod(comb_shape[1:]))
    comb_scores_reshaped = combined_scores.reshape(mergesort_shape)
    best_sorted_idx = np.argsort(
        sign * -comb_scores_reshaped, kind="mergesort", axis=1
    )

    # Best, sorted scores in all maps (for all phases) per point
    merged_best_scores = np.take_along_axis(
        comb_scores_reshaped, best_sorted_idx, axis=-1
    )

    # Combined (unsorted) simulation indices array of shape (M, N, K) or
    # (M, K), accounting for the case where there are more simulation
    # indices per point than scores (e.g. refined dot products from
    # EMsoft)
    comb_sim_idx = np.dstack(
        [xmap.prop[simulation_indices_prop] for xmap in crystal_maps]
    )

    # To enable calculation of an orientation similarity map from the
    # combined, sorted simulation indices array, we must make the
    # indices unique across all maps
    for i in range(1, comb_sim_idx.shape[-1]):
        increment = (
            abs(comb_sim_idx[..., i - 1].max() - comb_sim_idx[..., i].min()) + 1
        )
        comb_sim_idx[..., i] += increment

    # Collapse axes as for the combined scores array above
    comb_sim_idx = comb_sim_idx.reshape(mergesort_shape)

    # Best, sorted simulation indices in all maps (for all phases) per
    # point
    merged_simulated_indices = np.take_along_axis(
        comb_sim_idx, best_sorted_idx, axis=-1
    )

    return CrystalMap(
        rotations=new_rotations,
        phase_id=phase_id,
        phase_list=phase_list,
        x=crystal_maps[0].x,
        y=crystal_maps[0].y,
        z=crystal_maps[0].z,
        prop={
            scores_prop: new_scores,
            simulation_indices_prop: new_indices,
            f"merged_{scores_prop}": merged_best_scores,
            f"merged_{simulation_indices_prop}": merged_simulated_indices,
        },
        scan_unit=crystal_maps[0].scan_unit,
    )


def _get_combined_scores_shape(
    crystal_maps: List[CrystalMap], scores_prop: str = "scores"
) -> Tuple[tuple, int]:
    xmap = crystal_maps[0]
    all_scores_shape = (xmap.size,)
    single_scores_shape = xmap.prop[scores_prop].shape
    if len(single_scores_shape) == 1:
        n_scores_per_point = 1
    else:
        n_scores_per_point = single_scores_shape[1]
        all_scores_shape += (single_scores_shape[-1],)
    all_scores_shape += (len(crystal_maps),)
    return all_scores_shape, n_scores_per_point
