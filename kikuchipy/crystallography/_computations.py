# Copyright 2019-2022 The kikuchipy developers
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

"""Crystallographic computations."""

from collections import defaultdict
from typing import List, Optional, Tuple, Union

from diffsims.crystallography import ReciprocalLatticeVector
import matplotlib.colors as mcolors
import numpy as np
from orix.crystal_map import Phase

from kikuchipy.draw.colors import TSL_COLORS


def _get_uvw_from_hkl(hkl: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """Return a unique set of zone axes from the cross product of Miller
    indices, without 000.
    """
    hkl = np.atleast_2d(hkl)
    uvw = np.cross(hkl[:, np.newaxis, :], hkl).reshape((len(hkl) ** 2, 3))
    not000 = np.count_nonzero(uvw, axis=1) != 0
    uvw = uvw[not000]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvw = uvw / np.gcd.reduce(uvw.astype(int), axis=1)[:, np.newaxis]
    uvw = np.unique(uvw, axis=0).astype(int)
    return uvw


def _get_hkl_family(hkl: np.ndarray, gspacing: np.ndarray) -> Tuple[dict, np.ndarray]:
    _, idx, inv = np.unique(gspacing, return_index=True, return_inverse=True)
    families = defaultdict(list)
    x = np.arange(inv.size)
    for i, j in enumerate(idx):
        mask = inv == i
        families[tuple(hkl[j])].extend(x[mask].tolist())
    return families, idx


def _get_colors_for_allowed_vectors(
    phase: Phase,
    highest_hkl: Union[List[int], np.ndarray, None] = None,
    color_cycle: Optional[List[str]] = None,
):
    """Return an array of Miller indices of allowed reciprocal lattice
    vectors for a point group and a corresponding color.

    The idea with this function is to always get the same color for the
    same band in the same point group.

    Parameters
    ----------
    phase
        A phase with a space and point group describing the allowed
        symmetry operations.
    highest_hkl
        Highest Miller indices to consider. If not given, [999] is used.
    color_cycle
        A list of color names recognized by Matplotlib. If not given, a
        color palette based on EDAX TSL's coloring of bands is cycled
        through.

    Returns
    -------
    hkl_color
        Array with Miller indices and corresponding colors of shape
        (n hkl, 2, 3), with hkl and color in index 0 and 1 along axis=1,
        respectively.
    """
    if highest_hkl is None:
        highest_hkl = [9, 9, 9]
    rlv = ReciprocalLatticeVector.from_highest_indices(phase=phase, hkl=highest_hkl)

    # Keep only allowed vectors
    try:
        rlv2 = rlv[rlv.allowed]
    except NotImplementedError:
        rlv2 = rlv

    # Reduce indices, e.g. [222] to [111], to assign these same color
    rlv3 = rlv2.unit.round()

    # Sort based on gspacing, to get red for family with highest
    # gspacing, green for family with second highest gspacing, etc.
    g_order = np.argsort(rlv3.gspacing)
    rlv2 = rlv2[g_order]

    # Assign each vector to a family
    hkl2 = np.round(rlv2.hkl).astype(int)
    gspacing = np.round(rlv2.gspacing, decimals=11)
    families, idx = _get_hkl_family(hkl=hkl2, gspacing=gspacing)

    # Get colors from a color cycle, repeating the cycle if there are
    # more families than cycle colors
    if color_cycle is None:
        color_cycle = TSL_COLORS
    n_families = len(families)
    n_times = int(np.ceil(n_families / len(color_cycle)))
    colors = (color_cycle * n_times)[:n_families]
    colors = [mcolors.to_rgb(i) for i in colors]

    return rlv2[idx].hkl, rlv2[idx].gspacing, colors
