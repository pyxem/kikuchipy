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

"""Crystallographic computations."""

from collections import defaultdict
from typing import List, Optional, Tuple, Union

from diffsims.crystallography import ReciprocalLatticePoint
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
        uvw = uvw / np.gcd.reduce(uvw, axis=1)[:, np.newaxis]
    uvw = np.unique(uvw, axis=0).astype(int)
    return uvw


def _get_hkl_family(
    hkl: Union[np.ndarray, list], reduce: bool = False
) -> Tuple[dict, dict]:
    # TODO: Almost identical to
    #  diffsims.crystallography.ReciprocalLatticePoint.unique, improve
    #  this instead!
    hkl = np.atleast_2d(hkl)
    # Remove [0, 0, 0] points
    hkl = hkl[~np.all(np.isclose(hkl, 0), axis=1)]
    families = defaultdict(list)
    families_idx = defaultdict(list)
    for i, this_hkl in enumerate(hkl.tolist()):
        for that_hkl in families.keys():
            if _is_equivalent(this_hkl, that_hkl, reduce=reduce):
                families[tuple(that_hkl)].append(this_hkl)
                families_idx[tuple(that_hkl)].append(i)
                break
        else:
            families[tuple(this_hkl)].append(this_hkl)
            families_idx[tuple(this_hkl)].append(i)
    n_families = len(families)
    unique_hkl = np.zeros((n_families, 3), dtype=int)
    for i, all_hkl_in_family in enumerate(families.values()):
        unique_hkl[i] = np.sort(all_hkl_in_family)[-1]
    return families, families_idx


def _is_equivalent(
    this_hkl: list, that_hkl: list, reduce: bool = False
) -> bool:
    """Determine whether two Miller index 3-tuples are equivalent.
    Symmetry is not considered.
    """
    if reduce:
        this_hkl, _ = _reduce_hkl(this_hkl)
        that_hkl, _ = _reduce_hkl(that_hkl)
    return np.allclose(
        np.sort(np.abs(this_hkl).astype(int)),
        np.sort(np.abs(that_hkl).astype(int)),
    )


def _reduce_hkl(hkl: Union[list, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce Miller indices 3-tuples by a greatest common divisor."""
    hkl = np.atleast_2d(hkl)
    divisor = np.gcd.reduce(hkl, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hkl = hkl / divisor[:, np.newaxis]
    return hkl.astype(int), divisor


def _get_colors_for_allowed_bands(
    phase: Phase,
    highest_hkl: Union[List[int], np.ndarray, None] = None,
    color_cycle: Optional[List[str]] = None,
):
    """Return an array of Miller indices of allowed Kikuchi bands for a
    point group and a corresponding color.

    The idea with this function is to always get the same color for the
    same band in the same point group.

    Parameters
    ----------
    phase
        A phase container with a space and point group describing the
        allowed symmetry operations.
    highest_hkl
        Highest Miller indices to consider. If None (default), [9, 9, 9]
        is used.
    color_cycle
        A list of color names recognized by Matplotlib. If None
        (default), a color palette based on EDAX TSL's coloring of bands
        is cycled through.

    Returns
    -------
    hkl_color
        Array with Miller indices and corresponding colors of shape
        (nhkl, 2, 3), with hkl and color in index 0 and 1 along axis=1,
        respectively.
    """
    if highest_hkl is None:
        highest_hkl = [9, 9, 9]
    rlp = ReciprocalLatticePoint.from_highest_hkl(
        phase=phase, highest_hkl=highest_hkl,
    )

    rlp2 = rlp[rlp.allowed]
    # TODO: Replace this ordering with future ordering method in
    #  diffsims
    g_order = np.argsort(rlp2.gspacing)
    new_hkl = np.atleast_2d(rlp2._hkldata)[g_order]
    rlp3 = ReciprocalLatticePoint(phase=rlp.phase, hkl=new_hkl)
    hkl = np.atleast_2d(rlp3._hkldata)
    families, families_idx = _get_hkl_family(hkl=hkl, reduce=True)

    if color_cycle is None:
        color_cycle = TSL_COLORS
    n_families = len(families)
    n_times = int(np.ceil(n_families / len(color_cycle)))
    colors = (color_cycle * n_times)[:n_families]
    colors = [mcolors.to_rgb(i) for i in colors]

    hkl_colors = np.zeros(shape=(rlp3.size, 2, 3))
    for hkl_idx, color in zip(families_idx.values(), colors):
        hkl_colors[hkl_idx, 0] = hkl[hkl_idx]
        hkl_colors[hkl_idx, 1] = color

    return hkl_colors
