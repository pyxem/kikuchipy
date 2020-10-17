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

from typing import Union, List
from math import copysign

import numpy as np

from orix.quaternion.rotation import Rotation
from orix.crystal_map.crystal_map import CrystalMap, PhaseList, Phase

from kikuchipy.indexing.similarity_metrics import (
    SimilarityMetric,
    SIMILARITY_METRICS,
)


def _merge(
    xmaps: List[CrystalMap],
    mean_n_largest: int = 1,
    metric: Union[str, SimilarityMetric] = None,
):
    """Merge a list of `CrystalMap`s into a single map. Used on results from DictionaryIndexing.

    The given `CrystalMap`s must have "template_indices" and "metric_results" in prop
    as produced by :class:`~kikuchipy.indexing.StaticDictionaryIndexing`.
    Both props are merge sorted into the returned map.

    Parameters
    ----------
    xmaps : List[CrystalMap]
        List of `CrystalMap`s with "template_indices" and "metric_results" in prop
    mean_n_largest : int, optional
        Compute the "merged_top_metric_result" prop from the mean of n_largest, by default 1
    metric : Union[str, SimilarityMetric], optional
        Similarity metric, by default None

    Returns
    -------
    merged_xmap : CrystalMap
        A CrystalMap with added prop "merged_top_metric_result"

    Notes
    -----
    `mean_n_largest` can be given with a negative sign if metric is not given
    in order to choose the lowest valued metric results.

    Raises
    ------
    ValueError
        If `xmaps` contains CrystalMaps with equal phases.
    """
    if metric is None:
        sign = copysign(1, mean_n_largest)
        mean_n_largest = abs(mean_n_largest)
    else:
        sign = SIMILARITY_METRICS.get(metric, metric).sign

    top_results_across_xmaps = np.array(
        [
            np.mean(xmap.prop["metric_results"][:, 0:mean_n_largest], axis=1)
            for xmap in xmaps
        ]
    )

    phase_id = np.argmax(sign * top_results_across_xmaps, axis=0)
    merged_top_metric_result = np.choose(phase_id, top_results_across_xmaps)

    metric_results = np.concatenate(
        [xmap.prop["metric_results"] for xmap in xmaps], axis=1
    )

    metric_result_sorted_indicies = np.argsort(
        sign * -metric_results, kind="mergesort", axis=1
    )

    template_indices = np.concatenate(
        [xmap.prop["template_indices"] for xmap in xmaps],
        axis=1,
    )
    template_indices = np.take_along_axis(
        template_indices, metric_result_sorted_indicies, axis=1
    )
    metric_results = np.take_along_axis(
        metric_results, metric_result_sorted_indicies, axis=1
    )

    prop = {
        "merged_top_metric_result": merged_top_metric_result,
        "template_indices": template_indices,
        "metric_results": metric_results,
    }

    # The rotations can be examined more carefully, takes now all keep_n only for the top 1 phase
    # Not ideal
    rotations_across_xmaps = np.array([xmap.rotations.data for xmap in xmaps])
    rotations = np.array(
        [rotations_across_xmaps[id][i] for i, id in enumerate(phase_id)]
    )
    rotations = Rotation(rotations)

    # OBS! Can raise ValueError if phase already in phase_lists
    phase_list = PhaseList()
    for xmap in xmaps:
        phase_list.add(xmap.phases_in_data)

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
