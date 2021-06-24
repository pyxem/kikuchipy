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

"""Utilities for combining a crystallographic map,
:class:`orix.crystal_map.CrystalMap` with an EBSD signal,
:class:`kikuchipy.signals.EBSD`.
"""

import warnings

from hyperspy.axes import AxesManager
from orix.crystal_map import CrystalMap


def crystal_map_is_compatible_with_signal(
    xmap: CrystalMap, axes_manager: AxesManager, raise_if_false: bool = False
) -> bool:
    nav_axes = axes_manager.navigation_axes[::-1]
    nav_shape = tuple([a.size for a in nav_axes])
    nav_scale = tuple([a.scale for a in nav_axes])
    compatible = None
    try:
        xmap_scale = tuple([xmap._step_sizes[a.name] for a in nav_axes])
    except KeyError:
        warnings.warn(
            "The signal navigation axes must be named 'x' and/or 'y' in order to "
            "compare the signal navigation scale to the CrystalMap step sizes 'dx' and "
            "'dy' (see `EBSD.axes_manager`)"
        )
        xmap_scale = (None,) * len(nav_axes)
        compatible = False
    compatible = xmap.shape == nav_shape and xmap_scale == nav_scale
    if not compatible and raise_if_false:
        raise AttributeError(
            f"The crystal map shape {xmap.shape} and step sizes {xmap_scale} aren't "
            f"compatible with the signal navigation shape {nav_shape} and step sizes "
            f"{nav_scale} (see `EBSD.axes_manager`)"
        )
    else:
        return compatible
