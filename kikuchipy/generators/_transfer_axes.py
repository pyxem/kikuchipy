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


def _transfer_navigation_axes_to_signal_axes(new_axes, old_axes):
    """Transfer navigation axis calibrations from an old signal to the
    signal axes of a new signal produced from it by a generator. Used
    from methods that generate a signal with a single value at each
    navigation position.

    Adapted from the pyxem package.

    Parameters
    ----------
    new_axes
        The new signal axes with undefined navigation axes.
    old_axes
        The old signal axes with calibrated navigation axes.

    Returns
    -------
    new_axes
        The new signal with calibrated signal axes.
    """
    for i in range(
        min(new_axes.signal_dimension, old_axes.navigation_dimension)
    ):
        ax_new = new_axes.signal_axes[i]
        ax_old = old_axes.navigation_axes[i]
        ax_new.name = ax_old.name
        ax_new.scale = ax_old.scale
        ax_new.units = ax_old.units
    return new_axes
