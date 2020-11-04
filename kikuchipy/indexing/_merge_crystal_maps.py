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
from typing import Union, List
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
    mean_n_largest: int = 1,
    metric: Union[str, SimilarityMetric] = None,
    simulated_indices_property: str = "simulated_indices",
    score_property: str = "scores",
):
    """Merge a list of single phase
    :class:`~orix.crystal_mapCrystalMap`s into one multi phase map. Used
    on results from :class:`~kikuchipy.indexing.DictionaryIndexing`.

    The best matching simulation indices are merge sorted and added
    among the properties in the returned map.

    Parameters
    ----------
    xmaps : list of CrystalMap
        A list of crystal maps with simulated indices and scores among
        their properties.
    mean_n_largest : int, optional
        Number of top metric results to take the mean of before
        comparing, and putting in the property `merged_top_scores`.
        Default is 1.
    metric : str or SimilarityMetric, optional
        Similarity metric, default is None.
    simulated_indices_property : str, optional
        Name of simulated indices array in the crystal map's properties.
        Default is "simulated_indices".
    score_property : str, optional
        Name of scores array in the crystal map's properties. Default
        is "scores".

    Returns
    -------
    merged_xmap : CrystalMap
        A crystal map with added property `merged_top_scores`.

    Notes
    -----
    `mean_n_largest` can be given with a negative sign if metric is not
    given in order to choose the lowest valued metric results.
    """
    if metric is None:
        sign = copysign(1, mean_n_largest)
        mean_n_largest = abs(mean_n_largest)
    else:
        sign = SIMILARITY_METRICS.get(metric, metric).sign

    all_scores = np.dstack([xmap.prop[score_property] for xmap in xmaps])

    best_scores = np.mean(all_scores[:, :mean_n_largest], axis=1)
    phase_id = np.argmax(sign * best_scores, axis=1)
    merged_best_scores = np.choose(phase_id, best_scores.T)

    new_rotations = Rotation(np.zeros_like(xmaps[0].rotations.data))
    new_scores = np.zeros_like(xmaps[0].prop[score_property])
    new_indices = np.zeros_like(xmaps[0].prop[simulated_indices_property])
    phase_list = PhaseList()
    for i, xmap in enumerate(xmaps):
        mask = phase_id == i
        new_rotations[mask] = xmap.rotations[mask]
        new_scores[mask] = xmap.prop[score_property][mask]
        new_indices[mask] = xmap.prop[simulated_indices_property][mask]
        if np.sum(mask) != 0:
            try:
                phase = xmap.phases_in_data[0]
                phase_list.add(phase)
            except ValueError:
                warnings.warn(f"There are duplicates of phase {phase}")

    return CrystalMap(
        rotations=new_rotations,
        phase_id=phase_id,
        phase_list=phase_list,
        x=xmaps[0].x,
        y=xmaps[0].y,
        z=xmaps[0].z,
        prop={
            score_property: new_scores,
            simulated_indices_property: new_indices,
            "merged_best_scores": merged_best_scores,
        },
        scan_unit=xmaps[0].scan_unit,
    )


def merge_crystalmaps(
    xmaps: List[CrystalMap],
    mean_n_largest: int = 1,
    metric: Union[str, SimilarityMetric] = None,
    simulated_indices_property: str = "simulated_indices",
    scores_property: str = "scores",
):
    if metric is None:
        sign = copysign(1, mean_n_largest)
        mean_n_largest = abs(mean_n_largest)
    else:
        sign = SIMILARITY_METRICS.get(metric, metric).sign

    top_scores_across_xmaps = np.array(
        [
            np.mean(xmap.prop[scores_property][:, :mean_n_largest], axis=1)
            for xmap in xmaps
        ]
    )

    phase_id = np.argmax(sign * top_scores_across_xmaps, axis=0)
    merged_top_scores = np.choose(phase_id, top_scores_across_xmaps)

    scores = np.concatenate(
        [xmap.prop[scores_property] for xmap in xmaps], axis=1
    )

    score_sorted_indicies = np.argsort(sign * -scores, kind="mergesort", axis=1)

    simulated_indices = np.concatenate(
        [xmap.prop[simulated_indices_property] for xmap in xmaps], axis=1,
    )
    simulated_indices = np.take_along_axis(
        simulated_indices, score_sorted_indicies, axis=1
    )
    scores = np.take_along_axis(scores, score_sorted_indicies, axis=1)

    prop = {
        "merged_top_scores": merged_top_scores,
        "simulated_indices": simulated_indices,
        "scores": scores,
    }

    # The rotations can be examined more carefully, takes now all keep_n only for the top 1 phase
    # Maybe not ideal
    rotations_across_xmaps = np.array([xmap.rotations.data for xmap in xmaps])
    rotations = np.array(
        [rotations_across_xmaps[id][i] for i, id in enumerate(phase_id)]
    )
    rotations = Rotation(rotations)

    # Warn if phase already in phase_lists
    phase_list = PhaseList()
    for xmap in xmaps:
        try:
            phase_list.add(xmap.phases_in_data)
        except ValueError:
            warnings.warn(
                f"One or more of the phases in {xmap.phases_in_data} are already"
                "in the PhaseList.",
                UserWarning,
            )

    merged_xmap = CrystalMap(
        rotations,
        phase_id=phase_id,
        x=xmaps[0].x,
        y=xmaps[0].y,
        phase_list=phase_list,
        prop=prop,
        scan_unit=xmaps[0].scan_unit,
    )
    return merged_xmap
