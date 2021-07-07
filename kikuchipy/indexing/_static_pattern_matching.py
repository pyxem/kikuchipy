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

from typing import List, Union

from orix.crystal_map import CrystalMap, create_coordinate_arrays

from kikuchipy.indexing import merge_crystal_maps, orientation_similarity_map
from kikuchipy.indexing._pattern_matching import _pattern_match
from kikuchipy.indexing.similarity_metrics import SimilarityMetric, _SIMILARITY_METRICS


class StaticPatternMatching:
    """Pattern matching of experimental patterns to simulated patterns,
    of known crystal orientations in pre-computed dictionaries
    :cite:`chen2015dictionary,jackson2019dictionary`, for phase and
    orientation determination.
    """

    def __init__(self, dictionaries):
        """Set up pattern matching with one or more dictionaries of
        pre-computed simulated patterns of known crystal orientations.

        Parameters
        ----------
        dictionaries : EBSD or list of EBSD
            Dictionaries as EBSD signals with a 1D navigation axis and
            the `xmap` property with known crystal orientations set.
        """
        if not isinstance(dictionaries, list):
            dictionaries = [dictionaries]
        self.dictionaries = dictionaries

    def __call__(
        self,
        signal,
        metric: Union[str, SimilarityMetric] = "ncc",
        keep_n: int = 50,
        n_slices: int = 1,
        return_merged_crystal_map: bool = False,
        get_orientation_similarity_map: bool = False,
    ) -> Union[CrystalMap, List[CrystalMap], None]:
        """Match each experimental pattern to all simulated patterns, of
        known crystal orientations in pre-computed dictionaries
        :cite:`chen2015dictionary,jackson2019dictionary`, to determine
        their phase and orientation.

        A suitable similarity metric, the normalized cross-correlation
        (:func:`~kikuchipy.indexing.similarity_metrics.ncc`), is used by
        default, but a valid user-defined similarity metric may be used
        instead.

        :class:`~orix.crystal_map.CrystalMap`'s for each dictionary with
        "scores" and "simulation_indices" as properties are returned.

        Parameters
        ----------
        signal : EBSD
            EBSD signal with experimental patterns.
        metric : str or SimilarityMetric, optional
            Similarity metric, by default "ncc" (normalized
            cross-correlation).
        keep_n : int, optional
            Number of best matches to keep, by default 50 or the number
            of simulated patterns if fewer than 50 are available.
        n_slices : int, optional
            Number of simulation slices to process sequentially, by
            default 1 (no slicing).
        return_merged_crystal_map : bool, optional
            Whether to return a merged crystal map, the best matches
            determined from the similarity scores, in addition to the
            single phase maps. By default False.
        get_orientation_similarity_map : bool, optional
            Add orientation similarity maps to the returned crystal
            maps' properties named "osm". By default False.

        Returns
        -------
        xmaps : orix.crystal_map.CrystalMap or list of \
                orix.crystal_map.CrystalMap
            A crystal map for each dictionary loaded and one merged map
            if `return_merged_crystal_map = True`.

        Notes
        -----
        Merging of crystal maps and calculations of orientation
        similarity maps can be done afterwards with
        :func:`~kikuchipy.indexing.merge_crystal_maps` and
        :func:`~kikuchipy.indexing.orientation_similarity_map`,
        respectively.

        See Also
        --------
        ~kikuchipy.indexing.similarity_metrics.make_similarity_metric
        ~kikuchipy.indexing.similarity_metrics.ndp
        """
        # This needs a rework before sent to cluster and possibly more
        # automatic slicing with dask
        n_simulations = max([d.axes_manager.navigation_size for d in self.dictionaries])
        good_number = 13500
        if (n_simulations // n_slices) > good_number:
            answer = input(
                "You should probably increase n_slices depending on your available "
                f"memory, try above {n_simulations // good_number}. Do you want to "
                "proceed? [y/n]"
            )
            if answer != "y":
                return

        # Get metric from optimized metrics if it is available, or
        # return the metric if it is not
        metric = _SIMILARITY_METRICS.get(metric, metric)

        am = signal.axes_manager

        step_sizes = tuple([i.scale for i in am.navigation_axes])
        coordinate_arrays, _ = create_coordinate_arrays(
            shape=am.navigation_shape[::-1], step_sizes=step_sizes
        )

        n_nav_dims = am.navigation_dimension
        if n_nav_dims == 0:
            xmap_kwargs = dict()
        elif n_nav_dims == 1:
            scan_unit = am.navigation_axes[0].units
            xmap_kwargs = dict(x=coordinate_arrays["x"], scan_unit=scan_unit)
        else:  # 2d
            scan_unit = am.navigation_axes[0].units
            xmap_kwargs = dict(
                x=coordinate_arrays["x"], y=coordinate_arrays["y"], scan_unit=scan_unit
            )

        keep_n = min([keep_n] + [d.xmap.size for d in self.dictionaries])

        # Naively let dask compute them separately, should try in the
        # future combined compute for better performance
        xmaps = []
        patterns = signal.data
        for dictionary in self.dictionaries:
            simulation_indices, scores = _pattern_match(
                patterns,
                dictionary.data,
                metric=metric,
                keep_n=keep_n,
                n_slices=n_slices,
                phase_name=dictionary.xmap.phases_in_data.names[0],
            )
            new_xmap = CrystalMap(
                rotations=dictionary.xmap.rotations[simulation_indices],
                phase_list=dictionary.xmap.phases_in_data,
                prop={"scores": scores, "simulation_indices": simulation_indices},
                **xmap_kwargs,
            )
            xmaps.append(new_xmap)

        # Create a merged CrystalMap using best metric result across all
        # dictionaries
        if return_merged_crystal_map and len(self.dictionaries) > 1:
            xmap_merged = merge_crystal_maps(xmaps, metric=metric)
            xmaps.append(xmap_merged)

        # Compute orientation similarity map
        if get_orientation_similarity_map:
            for xmap in xmaps:
                osm = orientation_similarity_map(xmap, n_best=keep_n)
                xmap.prop["osm"] = osm.flatten()

        if len(xmaps) == 1:
            xmaps = xmaps[0]

        return xmaps
