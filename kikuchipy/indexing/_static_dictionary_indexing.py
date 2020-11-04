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

import numpy as np
from orix.crystal_map import CrystalMap

from kikuchipy.indexing._pattern_matching import _pattern_match
from kikuchipy.indexing.orientation_similarity_map import (
    orientation_similarity_map,
)
from kikuchipy.indexing.similarity_metrics import (
    SimilarityMetric,
    SIMILARITY_METRICS,
)


class StaticDictionaryIndexing:
    """Indexing against pre-computed dictionaries of simulated EBSD
    patterns.
    """

    def __init__(self, dictionaries):
        """Initialize with one or more dictionaries before indexing
        patterns.

        Parameters
        ----------
        dictionaries : EBSD or list of EBSD
            Dictionaries as EBSD Signals with one-dimensional navigation
            axis and with the `xmap` property set.
        """
        if not isinstance(dictionaries, list):
            dictionaries = list(dictionaries)
        self.dictionaries = dictionaries

    def dictionary_indexing(
        self,
        patterns,
        metric: Union[str, SimilarityMetric] = "zncc",
        keep_n: int = 1,
        n_slices: int = 1,
        merge_crystal_maps: bool = True,
        osm: bool = True,
    ) -> List[CrystalMap]:
        """Perform dictionary indexing on patterns against preloaded
        dictionaries, returning a :class:`~orix.crystal_map.CrystalMap`
        for each dictionary with `scores` and `simulated_indices` as
        properties.

        Parameters
        ----------
        patterns : EBSD
            EBSD signal with experimental patterns.
        metric : str or SimilarityMetric, optional
            Similarity metric, by default "zncc".
        keep_n : int, optional
            Number of sorted results to keep, by default 1.
        n_slices : int, optional
            Number of slices of simulations to process sequentially, by
            default 1.
        merge_crystal_maps : bool, optional
            Return a merged crystal map, the best matches determined
            from the similarity scores, in addition to the single phase
            maps. By default True. See also
            :func:`~kikuchipy.indexing.merge_crystal_maps`.
        osm : bool, optional
            Add orientation similarity maps to the returned crystal maps
            as an `osm` property, by default True.

        Returns
        -------
        xmaps : list of CrystalMap
            A crystal map for each dictionary loaded and one merged map
            if `merge_crystal_maps = True`.
        """
        # This needs a rework before sent to cluster and possibly more
        # automatic slicing with dask
        num_simulations = self.dictionaries[0].data.shape[0]
        if num_simulations // n_slices > 13500:
            answer = input(
                "You should probably increase n_slices depending on your "
                f"available memory, try above {num_simulations // 13500}. Do "
                "you want to proceed? [y/n]"
            )
            if answer != "y":
                return

        metric = SIMILARITY_METRICS.get(metric, metric)

        # Naively let dask compute them seperately, should try in the
        # future combined compute for better performance
        match_results = [
            _pattern_match(
                patterns.data,
                dictionary.data,
                metric=metric,
                keep_n=keep_n,
                n_slices=n_slices,
            )
            for dictionary in self.dictionaries
        ]

        # Create spatial arrays
        nav_axes = patterns.axes_manager
        step_size = nav_axes[0].units

        #        axm = patterns.axes_manager
        #        scan_unit = axm.navigation_axes[0].units
        #        col1, col2, row1, row2 = axm.navigation_extent
        #        scale_x, scale_y = (axm.navigation_axes[i].scale for i in range(2))
        #        nav_shape = axm.navigation_shape
        #        x = np.tile(np.arange(col1, col2 + scale_x, scale_x), nav_shape[1])
        #        y = np.tile(np.arange(row1, row2 + scale_y, scale_y), nav_shape[0])

        #
        # Create crystal map for each match_result, i.e. each phase
        #

        def match_result_2_xmap(i, mr):
            simulated_indices, scores = mr
            xmap = self.dictionaries[i].xmap
            phase_list = xmap.phases_in_data
            rotations = xmap.rotations[simulated_indices]
            return CrystalMap(
                rotations,
                x=x,
                y=y,
                phase_list=phase_list,
                prop={"scores": scores, "simulated_indices": simulated_indices},
                scan_unit=scan_unit,
            )

        xmaps = [
            match_result_2_xmap(i, mr) for i, mr in enumerate(match_results)
        ]

        #
        # Creating one CrystalMap using best metric result accross all dictionaries
        #
        if merge_crystal_maps and len(self.dictionaries) > 1:
            # Cummulative summation of the dictionary lengths to create unique simulation ids across dictionaries
            cum_sum_dict_lengths = np.cumsum(
                [d.data.shape[0] for d in self.dictionaries]
            )

            def adjust_sim_ids(i, xmap):
                if i == 0:
                    return xmap
                xmap.simulated_indices += cum_sum_dict_lengths[i - 1]
                return xmap

            xmaps_unique_sim_ids = [
                adjust_sim_ids(i, xmap) for i, xmap in enumerate(xmaps)
            ]
            xmap_merged = merge_crystalmaps(xmaps_unique_sim_ids, metric=metric)
            xmaps.append(xmap_merged)

        # Orientation Similarity Maps
        if osm:
            print("Computing Orientation Similarity Maps.")
            for xmap in xmaps:
                xmap.prop["osm"] = orientation_similarity_map(
                    xmap, n_largest=keep_n
                ).flatten()

        return xmaps


# def _get_spatial_arrays()
