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
import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
from h5py import File

from orix.quaternion.rotation import Rotation
from orix.crystal_map.crystal_map import CrystalMap, PhaseList, Phase

from kikuchipy.signals import EBSD, LazyEBSD
from kikuchipy.indexing.template_matching import template_match

from kikuchipy.indexing.osm import orientation_similarity_map
from kikuchipy.indexing.similarity_metrics import (
    SimilarityMetric,
    SIMILARITY_METRICS,
)

from kikuchipy.indexing.merge_crystalmaps import merge_crystalmaps


class StaticDictionaryIndexing:
    """Indexing against pre-computed dictionaries from EMsoft's EMEBSD.f90 program"""

    def __init__(
        self,
        dictionaries: Union[EBSD, LazyEBSD, List[Union[EBSD, LazyEBSD]]],
    ):
        """Initialize with one or more dictionaries before indexing patterns.

        Parameters
        ----------
        dictionaries : Union[EBSD, LazyEBSD, List[Union[EBSD, LazyEBSD]]]
            Dictionaries as EBSD Signals with one-dimensional navigation axis
            and with the `xmap` property set.
        """
        self.dictionaries = (
            dictionaries if isinstance(dictionaries, list) else [dictionaries]
        )

    def index(
        self,
        patterns: Union[EBSD, LazyEBSD],
        metric: Union[str, SimilarityMetric] = "zncc",
        keep_n: int = 1,
        n_slices: int = 1,
        merge_xmaps: bool = True,
        osm: bool = True,
    ) -> List[CrystalMap]:
        """Perform Dictionary Indexing on patterns against preloaded dictionaries.[ref here or elsewhere?]

        Produce a `CrystalMap` for each dictionary with `metric_results` and `template_indices` as properties.

        Parameters
        ----------
        patterns : Union[EBSD, LazyEBSD]
            Patterns
        metric : Union[str, SimilarityMetric], optional
            Similarity metric, by default "zncc".
        keep_n : int, optional
            Number of sorted results to keep, by default 1.
        n_slices : int, optional
            Number of template slices to process sequentially, by default 1.
        merge_xmaps : bool, optional
            Produce a merged crystal map from best results, by default True.
            See also `merge_crystalmaps`.
        osm : bool, optional
            Orientation Similarity Maps as property `osm`, by default True.

        Returns
        -------
        xmaps : List[CrystalMap]
            A crystal map for each dictionary loaded and one merged map if `merge_xmaps = True`.
        """

        # This needs a rework before sent to cluster and possibly more automatic slicing with dask
        num_templates = self.dictionaries[0].data.shape[0]
        if num_templates // n_slices > 13500:
            answer = input(
                f"You should probably increase n_slices depending on your available memory, try above {num_templates // 13500}. Do you want to proceed? [y/n]"
            )
            if answer != "y":
                return

        n_slices = None if n_slices == 1 else n_slices

        metric = SIMILARITY_METRICS.get(metric, metric)

        # Naively let dask compute them seperately, should try in the future combined compute for better performance
        match_results = [
            template_match(
                patterns.data,
                dictionary.data,
                metric=metric,
                keep_n=keep_n,
                n_slices=n_slices,
            )
            for dictionary in self.dictionaries
        ]

        axm = patterns.axes_manager
        scan_unit = axm.navigation_axes[0].units
        x1, x2, y1, y2 = axm.navigation_extent
        scale_x, scale_y = (axm.navigation_axes[i].scale for i in range(2))
        nav_shape = axm.navigation_shape
        x = np.tile(np.arange(x1, x2 + scale_x, scale_x), nav_shape[1])
        y = np.tile(np.arange(y1, y2 + scale_y, scale_y), nav_shape[0])

        #
        # Create crystal map for each match_result, i.e. each phase
        #

        def match_result_2_xmap(i, mr):
            t_indices, coeffs = mr
            xmap = self.dictionaries[i].xmap
            phase_list = xmap.phases_in_data
            rotations = xmap.rotations[t_indices]
            return CrystalMap(
                rotations,
                x=x,
                y=y,
                phase_list=phase_list,
                prop={"metric_results": coeffs, "template_indices": t_indices},
                scan_unit=scan_unit,
            )

        xmaps = [
            match_result_2_xmap(i, mr) for i, mr in enumerate(match_results)
        ]

        #
        # Creating one CrystalMap using best metric result accross all dictionaries
        #
        if merge_xmaps and len(self.dictionaries) > 1:
            # Cummulative summation of the dictionary lengths to create unique template ids across dictionaries
            cum_sum_dict_lengths = np.cumsum(
                [d.data.shape[0] for d in self.dictionaries]
            )

            def adjust_t_ids(i, xmap):
                if i == 0:
                    return xmap
                xmap.prop["template_indices"] += cum_sum_dict_lengths[i - 1]
                return xmap

            xmaps_unique_t_ids = [
                adjust_t_ids(i, xmap) for i, xmap in enumerate(xmaps)
            ]
            xmap_merged = merge_crystalmaps(xmaps_unique_t_ids, metric=metric)
            xmaps.append(xmap_merged)

        # Orientation Similarity Maps
        if osm:
            print("Computing Orientation Similarity Maps.")
            for xmap in xmaps:
                xmap.prop["osm"] = orientation_similarity_map(
                    xmap, n_largest=keep_n
                ).flatten()

        return xmaps
