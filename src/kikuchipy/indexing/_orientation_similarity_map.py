# Copyright 2019-2024 The kikuchipy developers
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

"""Compute an orientation similarity map, where the ranked list of the
array indices of the best matching simulated patterns in one map point
is compared to the corresponding lists in the nearest neighbour points.
"""

# TODO: Consider moving to orix.

import numpy as np
from orix.crystal_map import CrystalMap
from scipy.ndimage import generic_filter


def orientation_similarity_map(
    xmap: CrystalMap,
    n_best: int | None = None,
    simulation_indices_prop: str = "simulation_indices",
    normalize: bool = False,
    from_n_best: int | None = None,
    footprint: np.ndarray | None = None,
    center_index: int = 2,
) -> np.ndarray:
    r"""Compute an orientation similarity map (OSM) where the ranked
    list of the dictionary indices of the best matching simulated
    patterns in one point is compared to the corresponding lists in the
    nearest neighbour points :cite:`marquardt2017quantitative`.

    Parameters
    ----------
    xmap
        A crystal map with a ranked list of the array indices of the
        best matching simulated patterns among its properties.
    n_best
        Number of ranked indices to compare. If not given (default), all
        indices are compared.
    simulation_indices_prop
        Name of simulated indices array in the crystal maps' properties.
        Default is ``"simulation_indices"``.
    normalize
        Whether to normalize the number of equal indices to the range
        [0, 1], by default ``False``.
    from_n_best
        Return an OSM for each n in the range ``[from_n_best, n_best]``.
        If not given (default), the OSM for ``n_best`` indices is
        returned.
    footprint
        Boolean 2D array specifying which neighbouring points to compare
        lists with, by default the four nearest neighbours.
    center_index
        Flat index of central navigation point in the truthy values of
        footprint, by default ``2``.

    Returns
    -------
    osm
        Orientation similarity map(s). If ``from_n_best`` is given, the
        returned array has three dimensions, where ``n_best`` is at
        ``osm[:, :, 0]`` and ``from_n_best`` at ``osm[:, :, -1]``.

    Notes
    -----
    If the set :math:`S_{r,c}` is the ranked list of best matching
    indices for a given point :math:`(r,c)`, then the orientation
    similarity index :math:`\eta_{r,c}` is the average value of the
    cardinalities (\#) of the intersections with the neighbouring sets

    .. math::

        \eta_{r,c} = \frac{1}{4}
            \left(
                \#(S_{r,c} \cap S_{r-1,c}) +
                \#(S_{r,c} \cap S_{r+1,c}) +
                \#(S_{r,c} \cap S_{r,c-1}) +
                \#(S_{r,c} \cap S_{r,c+1})
            \right).

    .. versionchanged:: 0.5
       Default value of ``normalize`` changed to ``False``.
    """
    simulation_indices = xmap.prop[simulation_indices_prop]
    nav_size, keep_n = simulation_indices.shape

    if n_best is None:
        n_best = keep_n
    elif n_best > keep_n:
        raise ValueError(f"n_best {n_best} cannot be greater than keep_n {keep_n}")

    data_shape = xmap.shape
    flat_index_map = np.arange(nav_size).reshape(data_shape)

    if from_n_best is None:
        from_n_best = n_best

    osm = np.zeros(data_shape + (n_best - from_n_best + 1,), dtype=np.float32)

    if footprint is None:
        footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    for i, n in enumerate(range(n_best, from_n_best - 1, -1)):
        match_indices = simulation_indices[:, :n]
        osm[:, :, i] = generic_filter(
            flat_index_map,
            lambda v: _orientation_similarity_per_pixel(
                v, center_index, match_indices, n, normalize
            ),
            footprint=footprint,
            mode="constant",
            cval=-1,
            output=np.float32,
        )

    return osm.squeeze()


def _orientation_similarity_per_pixel(
    v: np.ndarray, center_index: int, match_indices: np.ndarray, n: int, normalize: bool
) -> np.ndarray:
    # v are indices picked out with the footprint from flat_index_map
    v = v.astype(int)
    center_value = v[center_index]
    # Filter only true neighbours, -1 out of image and not include
    # itself
    neighbours = v[np.where((v != -1) & (v != center_value))]

    # Cardinality of the intersection between a and b
    number_of_equal_matches_to_its_neighbours = [
        len(np.intersect1d(match_indices[center_value], mi))
        for mi in match_indices[neighbours]
    ]

    os_i = np.nanmean(number_of_equal_matches_to_its_neighbours)

    if normalize:
        os_i /= n

    return os_i
