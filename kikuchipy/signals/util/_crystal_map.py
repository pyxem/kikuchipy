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

from typing import Optional, Tuple, Union
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
def _equal_phase(phase1: Phase, phase2: Phase) -> Tuple[bool, Union[str, None]]:
    if phase1.name != phase2.name:
        return False, "names"

    space_groups = []
    point_groups = []
    for phase in [phase1, phase2]:
        if hasattr(phase.space_group, "number"):
            space_groups.append(phase.space_group.number)
        else:
            space_groups.append(np.nan)
        if phase.point_group is not None:
            point_groups.append(phase.point_group.data)
        else:
            point_groups.append(np.nan)

    # Check space groups
    if not np.allclose(*space_groups, equal_nan=True):
        return False, "space groups"

    # Check point groups
    if np.size(point_groups[0]) != np.size(point_groups[1]) or not np.allclose(
        *point_groups, equal_nan=True
    ):
        return False, "point groups"

    # Compare number of atoms, lattice parameters and atom element,
    # coordinate and occupancy
    structure1 = phase1.structure
    structure2 = phase2.structure
    if len(structure1) != len(structure2):
        return False, "number of atoms"
    if not np.allclose(structure1.lattice.abcABG(), structure2.lattice.abcABG()):
        return False, "lattice parameters"

    for atom1, atom2 in zip(structure1, structure2):
        if (
            atom1.element != atom2.element
            or not np.allclose(atom1.xyz, atom2.xyz)
            or not np.isclose(atom1.occupancy, atom2.occupancy)
        ):
            return False, "atoms"

    return True, None


def _get_indexed_points_in_data_in_xmap(
    xmap: CrystalMap,
    navigation_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, Union[Tuple[int], Tuple[int, int], None]]:
    in_data = xmap.is_in_data.copy()

    if navigation_mask is not None:
        # Keep only points in the mask that are already in the data
        nav_mask_shape = navigation_mask.shape
        if navigation_mask.shape != xmap.shape:
            raise ValueError(
                f"Navigation mask shape {nav_mask_shape} and crystal map shape "
                f"{xmap.shape} must be the same"
            )

        in_mask = ~navigation_mask.ravel()
        in_mask_and_data = in_mask[in_data]
        in_data = np.logical_and(in_data, in_mask)
        phase_id = np.unique(xmap.phase_id[in_mask_and_data])
    else:
        phase_id = np.unique(xmap.phase_id[xmap.is_indexed])

    if not (phase_id.size == 1 or (phase_id.size == 2 and -1 in phase_id)):
        raise ValueError(
            "Points in data in crystal map must have only one phase, but had the phase "
            f"IDs {list(phase_id)}"
        )
    unique_phase_id = phase_id[phase_id != -1][0]

    # xmap.is_indexed might have fewer elements than the in_data array
    is_indexed = np.zeros_like(in_data)
    is_indexed[xmap.is_in_data] = xmap.is_indexed
    in_data_indexed = np.logical_and(in_data, is_indexed)

    # Check if the (possibly combined) mask is continuous
    if xmap.ndim == 1:
        points_in_data_idx = np.where(in_data)[0]
        mask_size = points_in_data_idx[-1] - points_in_data_idx[0] + 1
        mask_is_continuous = mask_size == in_data.sum()
        mask_shape = (mask_size,)
    else:
        in_data2d = in_data.reshape(xmap.shape)
        r_points, c_points = np.where(in_data2d)
        r_size = r_points.max() - r_points.min() + 1
        c_size = c_points.max() - c_points.min() + 1
        mask_is_continuous = (r_size * c_size) == in_data.sum()
        mask_shape = (r_size, c_size)
    if not np.allclose(in_data_indexed, in_data) or not mask_is_continuous:
        mask_shape = None

    return in_data_indexed, in_data, unique_phase_id, mask_shape
