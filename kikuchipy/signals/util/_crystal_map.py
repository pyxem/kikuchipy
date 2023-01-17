# Copyright 2019-2023 The kikuchipy developers
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

"""Utilities for working with a :class:`~orix.crystal_map.CrystalMap`
and an :class:`~kikuchipy.signals.EBSD` signal.
"""

from typing import Optional, Tuple
import warnings

import numpy as np
from orix.crystal_map import CrystalMap, Phase


def _xmap_is_compatible_with_signal(
    xmap: CrystalMap, navigation_axes: tuple, raise_if_not: bool = False
) -> bool:
    """Check whether a signal's navigation axes are compatible with a
    crystal map.
    """
    nav_shape = tuple([a.size for a in navigation_axes])
    nav_scale = list([a.scale for a in navigation_axes])

    try:
        xmap_scale = list([xmap._step_sizes[a.name] for a in navigation_axes])
    except KeyError:
        warnings.warn(
            "Signal navigation axes must be named 'x' and/or 'y' in order to compare "
            "the signal navigation scales to the crystal map step sizes 'dx' and 'dy' "
            "(see `EBSD.axes_manager`)"
        )
        xmap_scale = list(xmap._step_sizes.values())[-len(navigation_axes) :]

    compatible = xmap.shape == nav_shape
    if compatible and not np.allclose(xmap_scale, nav_scale, atol=1e-6):
        warnings.warn(
            f"Crystal map step size(s) {xmap_scale} and signal's step size(s) "
            f"{nav_scale} must be the same (see `EBSD.axes_manager`)"
        )

    if not compatible and raise_if_not:
        raise ValueError(
            f"Crystal map shape {xmap.shape} and signal's navigation shape {nav_shape} "
            "must be the same (see `EBSD.axes_manager`)"
        )
    else:
        return compatible


# TODO: Move to orix' Phase.__eq__
def _equal_phase(phase1: Phase, phase2: Phase) -> bool:
    try:
        equal_sg = phase1.space_group.number == phase2.space_group.number
    except AttributeError:
        equal_sg = True
    equal_pg = phase1.point_group == phase2.point_group
    equal_structure = len(phase1.structure) == len(phase2.structure)
    if equal_structure:
        for atom1, atom2 in zip(phase1.structure, phase2.structure):
            equal_structure *= atom1.element == atom2.element
            equal_structure *= np.allclose(atom1.xyz, atom2.xyz)
            equal_structure *= np.isclose(atom1.occupancy, atom2.occupancy)
            if not equal_structure:
                break
    return bool(equal_sg * equal_pg * equal_structure)


def _get_points_in_data_in_xmap(
    xmap: CrystalMap,
    navigation_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    points_to_refine = xmap.is_in_data

    if navigation_mask is not None:
        nav_mask_shape = navigation_mask.shape
        if navigation_mask.shape != xmap.shape:
            raise ValueError(
                f"Navigation mask shape {nav_mask_shape} and crystal map shape "
                f"{xmap.shape} must be the same"
            )
        points_to_refine_in_mask = ~navigation_mask.ravel()
        points_to_refine_in_data = points_to_refine_in_mask[points_to_refine]
        points_to_refine = np.logical_and(points_to_refine, points_to_refine_in_mask)
        phase_id = np.unique(xmap.phase_id[points_to_refine_in_data])
    else:
        phase_id = np.unique(xmap.phase_id)

    if phase_id.size != 1:
        raise ValueError(
            "Points to refine in crystal map must have only one phase, but had the "
            f"phase IDs {list(phase_id)}"
        )
    unique_phase_id = phase_id[0]

    return points_to_refine, unique_phase_id
