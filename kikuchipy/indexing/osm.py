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

from typing import Tuple, List
from math import copysign

import numpy as np
from scipy.ndimage import generic_filter

from orix.crystal_map import CrystalMap

from kikuchipy.indexing.similarity_metrics import (
    SimilarityMetric,
    SIMILARITY_METRICS,
)


def orientation_similarity_map(
    xmap: CrystalMap,
    n_largest: int = None,
    normalize_by_n: bool = True,
    from_n_largest: int = None,
    footprint: np.ndarray = None,
    center_index: int = 2,
) -> np.ndarray:
    """Create Orientation Similarity Map(OSM), following [Marquardt2017]_.

    The given CrystalMap `xmap` must contain `"template_indices"` in `xmap.prop`
    as produced by :class:`~kikuchipy.indexing.StaticDictionaryIndexing`.

    Parameters
    ----------
    xmap : CrystalMap
        CrystalMap with "template_indices" in prop
    n_largest : int, optional
        Number of sorted results to be used, by default use all.
    normalize_by_n : bool, optional
        Whether to return Orientation Similarity in range [0, 1], by default True
    from_n_largest : int, optional
        Create OSM for each n in range [`from_n_largest`, `n_largest`], by default None.
    footprint : [np.ndarray], optional
        Boolean 2D array specifying which neighbours to be used in computation of OS
        in each navigation point, by default nearest neighbours.
    center_index : int, optional
        Flat index of central navigation point in the truthy values of footprint, by default 2.

    Returns
    -------
    osm : np.ndarray
        Orientation Similarity Map(s)

    Notes
    -----
    If a range of OSM are to be created n_largest is at osm[:,:,0]
    and from_n_largest at osm[:,:,-1].

    Raises
    ------
    NotImplementedError
        [might be interesting to try different footprints]

    References
    ----------
    .. [Marquardt2017] K. Marquardt, M. De Graef,\
        S. Singh, H. Marquardt, A. Rosenthal, S. Koizuimi\
        "Quantitative electron backscatter diffraction (EBSD) data analyses \
        using the dictionary indexing (DI) approach"\
        , American Mineralogist: Journal of Earth and Planetary Materials 102(9), 2017.
    """

    if footprint is None:
        footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    template_indices = xmap.prop["template_indices"]

    nav_size, keep_n = template_indices.shape

    if n_largest is None:
        n_largest = keep_n
    elif n_largest > keep_n:
        raise ValueError(
            f"n_largest {n_largest} can not be larger than keep_n {keep_n}"  # keep_n * len(dictionaries) for merged maps
        )

    data_shape = xmap.shape
    flat_index_map = np.arange(nav_size).reshape(data_shape)

    if from_n_largest is None:
        from_n_largest = n_largest

    osm = np.zeros(data_shape + (n_largest - from_n_largest + 1,))

    # Cardinality of the intersection between a and b
    f = lambda a, b: len(np.intersect1d(a, b))

    def os_per_pixel(v, match_indicies, n):
        # v is indices picked out with footprint from flat_index_map
        v = v.astype(np.int)

        # Filter only true neighbours, -1 out of image and not include itself
        neighbours = v[np.where((v != -1) & (v != center_index))]

        number_of_equal_matches_to_its_neighbours = [
            f(match_indicies[center_index], mi)
            for mi in match_indicies[neighbours]
        ]
        os = np.mean(number_of_equal_matches_to_its_neighbours)
        if normalize_by_n:
            os /= n
        return os

    for i, n in enumerate(range(n_largest, from_n_largest - 1, -1)):
        match_indicies = template_indices[:, :n]
        osm[:, :, i] = generic_filter(
            flat_index_map,
            lambda v: os_per_pixel(v, match_indicies, n),
            footprint=footprint,
            mode="constant",
            cval=-1,
            output=np.float32,
        )

    return osm.squeeze()
