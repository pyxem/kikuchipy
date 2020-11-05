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

from math import copysign
from typing import List, Union, Tuple
import warnings

import numpy as np
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion.rotation import Rotation

from kikuchipy.indexing.similarity_metrics import (
    SimilarityMetric,
    SIMILARITY_METRICS,
)


def merge_crystal_maps(
    xmaps: List[CrystalMap],
    mean_n_best: int = 1,
    metric: Union[str, SimilarityMetric] = None,
    simulation_indices_prop: str = "simulation_indices",
    score_prop: str = "scores",
):
    """Merge a list of single phase
    :class:`~orix.crystal_mapCrystalMap`s into one multi phase map. Used
    on results from :class:`~kikuchipy.indexing.DictionaryIndexing`.

    Parameters
    ----------
    xmaps : list of CrystalMap
        A list of crystal maps with simulated indices and scores among
        their properties.
    mean_n_best : int, optional
        Number of best metric results to take the mean of before
        comparing. Default is 1.
    metric : str or SimilarityMetric, optional
        Similarity metric, default is None.
    simulation_indices_prop : str, optional
        Name of simulated indices array in the crystal map's properties.
        Default is "simulation_indices".
    score_prop : str, optional
        Name of scores array in the crystal map's properties. Default
        is "scores".

    Returns
    -------
    merged_xmap : CrystalMap
        A crystal map where the rotation of the phase with the best
        matching score(s) is assigned to each point. The best matching
        simulation indices and scores, merge sorted, are added to its
        properties with names `merged_scores` and
        `merged_simulation_indices`, respectively.

    Notes
    -----
    `mean_n_best` can be given with a negative sign if `metric` is not
    given, in order to choose the lowest valued metric results.
    """
    if len(xmaps) == 1:
        raise ValueError("More than one map must be passed")

    if metric is None:
        sign = copysign(1, mean_n_best)
        mean_n_best = abs(mean_n_best)
    else:
        sign = SIMILARITY_METRICS.get(metric, metric).sign

    # Notation used in the comments below:
    # - M: number of map points
    # - N: number of scores per point
    # - I: number of simulation indices per point
    # - K: number of maps to merge

    # Shape of the combined (unsorted) scores array, and the total
    # number of scores per point. Shape: (M, N, K) or (M, K) if only one
    # score is available (e.g. refined dot products from EMsoft)
    (comb_shape, n_scores_per_point) = _get_combined_scores_shape(
        xmaps=xmaps, score_prop=score_prop
    )

    # Combined (unsorted) scores array of shape (M, N, K) or (M, K)
    combined_scores = np.dstack([xmap.prop[score_prop] for xmap in xmaps])
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
    new_rotations = Rotation(np.zeros_like(xmaps[0].rotations.data))
    new_scores = np.zeros_like(xmaps[0].prop[score_prop])
    new_indices = np.zeros_like(xmaps[0].prop[simulation_indices_prop])
    phase_list = PhaseList()
    for i, xmap in enumerate(xmaps):
        mask = phase_id == i
        new_rotations[mask] = xmap.rotations[mask]
        new_scores[mask] = xmap.prop[score_prop][mask]
        new_indices[mask] = xmap.prop[simulation_indices_prop][mask]
        if np.sum(mask) != 0:
            try:
                phase = xmap.phases_in_data[0]
                phase_list.add(phase)
            except ValueError:
                warnings.warn(f"There are duplicates of phase {phase}")

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
        [
            xmap.prop[simulation_indices_prop][:, :n_scores_per_point]
            for xmap in xmaps
        ]
    )

    # To enable calculation of an orientation similarity map from the
    # combined, sorted simulation indices array, we must make the
    # indices unique across all maps
    max_indices = np.max(comb_sim_idx, axis=(0, 1))
    increment_indices = np.cumsum(max_indices) - max_indices[0]
    comb_sim_idx += increment_indices

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
        x=xmaps[0].x,
        y=xmaps[0].y,
        z=xmaps[0].z,
        prop={
            score_prop: new_scores,
            simulation_indices_prop: new_indices,
            "merged_scores": merged_best_scores,
            "merged_simulated_indices": merged_simulated_indices,
        },
        scan_unit=xmaps[0].scan_unit,
    )


def _get_combined_scores_shape(
    xmaps: List[CrystalMap], score_prop: str = "scores"
) -> Tuple[tuple, int]:
    xmap = xmaps[0]
    all_scores_shape = (xmap.size,)
    single_scores_shape = xmap.prop[score_prop].shape
    if len(single_scores_shape) == 1:
        n_scores_per_point = 1
    else:
        n_scores_per_point = single_scores_shape[1]
        all_scores_shape += (single_scores_shape[-1],)
    n_xmaps = len(xmaps)
    if n_xmaps > 1:
        all_scores_shape += (n_xmaps,)
    return all_scores_shape, n_scores_per_point


# def merge_crystalmaps(
#    xmaps: List[CrystalMap],
#    mean_n_largest: int = 1,
#    metric: Union[str, SimilarityMetric] = None,
# ):
#    """Merge a list of single-phase `CrystalMap`s into one multi-phase map. Used on results from DictionaryIndexing.
#    The given `CrystalMap`s must have "simulated_indices" and "scores" in prop
#    as produced by :class:`~kikuchipy.indexing.StaticDictionaryIndexing`.
#    Both props are merge sorted into the returned map.
#    Parameters
#    ----------
#    xmaps : List[CrystalMap]
#        List of `CrystalMap`s with "simulated_indices" and "scores" in prop
#    mean_n_largest : int, optional
#        Number of top metric results to take the mean of,
#        before comparing and put in the property `merged_top_scores`, by default 1
#    metric : Union[str, SimilarityMetric], optional
#        Similarity metric, by default None
#    Returns
#    -------
#    merged_xmap : CrystalMap
#        A CrystalMap with added prop `merged_top_scores`
#    Notes
#    -----
#    `mean_n_largest` can be given with a negative sign if metric is not given
#    in order to choose the lowest valued metric results.
#    """
#    if metric is None:
#        sign = copysign(1, mean_n_largest)
#        mean_n_largest = abs(mean_n_largest)
#    else:
#        sign = SIMILARITY_METRICS.get(metric, metric).sign
#
#    top_scores_across_xmaps = np.array(
#        [np.mean(xmap.scores[:, :mean_n_largest], axis=1) for xmap in xmaps]
#    )
#
#    phase_id = np.argmax(sign * top_scores_across_xmaps, axis=0)
#    merged_top_scores = np.choose(phase_id, top_scores_across_xmaps)
#
#    scores = np.concatenate([xmap.scores for xmap in xmaps], axis=1)
#
#    score_sorted_indicies = np.argsort(sign * -scores, kind="mergesort", axis=1)
#
#    simulated_indices = np.concatenate(
#        [xmap.simulated_indices for xmap in xmaps],
#        axis=1,
#    )
#    simulated_indices = np.take_along_axis(
#        simulated_indices, score_sorted_indicies, axis=1
#    )
#    scores = np.take_along_axis(scores, score_sorted_indicies, axis=1)
#
#    prop = {
#        "merged_top_scores": merged_top_scores,
#        "simulated_indices": simulated_indices,
#        "scores": scores,
#    }
#
#    # The rotations can be examined more carefully, takes now all keep_n only for the top 1 phase
#    # Maybe not ideal
#    rotations_across_xmaps = np.array([xmap.rotations.data for xmap in xmaps])
#    rotations = np.array(
#        [rotations_across_xmaps[id][i] for i, id in enumerate(phase_id)]
#    )
#    rotations = Rotation(rotations)
#
#    # Warn if phase already in phase_lists
#    phase_list = PhaseList()
#    for xmap in xmaps:
#        try:
#            phase_list.add(xmap.phases_in_data)
#        except ValueError:
#            warnings.warn(
#                f"One or more of the phases in {xmap.phases_in_data} are already"
#                "in the PhaseList.",
#                UserWarning,
#            )
#
#    merged_xmap = CrystalMap(
#        rotations,
#        phase_id=phase_id,
#        x=xmaps[0].x,
#        y=xmaps[0].y,
#        phase_list=phase_list,
#        prop=prop,
#        scan_unit=xmaps[0].scan_unit,
#    )
#    return merged_xmap
