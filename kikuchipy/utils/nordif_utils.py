# -*- coding: utf-8 -*-
# Copyright 2019 The KikuchiPy developers
#
# This file is part of KikuchiPy.
#
# KikuchiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KikuchiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KikuchiPy. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import dask.array as da


def fix_pattern_order(data, shift=-1, overwrite=False, corrupt_idx=(0, 0),
                      overwrite_idx=(-1, -1)):
    """Shift the patterns a number of steps equal to `shift` using
    `numpy.roll` or the dask equivalent. If a pattern specified by
    `corrupt_idx` is corrupted this pattern can be overwritten by
    another pattern specified by `overwrite_idx` before shifting, if
    the data is not lazy.

    Parameters
    ----------
    data : array_like
        Two-dimensional array containing signal data.
    shift : int, optional
        Number of steps to shift patterns.
    overwrite : bool, optional
        Whether to overwrite a pattern or not.
    corrupt_idx : tuple, optional
        Index of corrupted pattern.
    overwrite_idx : tuple, optional
        Index of pattern to overwrite corrupted pattern with.

    Returns
    -------
    data : array_like
        Two-dimensional array containing shifted data.
    """
    # Overwrite patterns before shifting
    if overwrite:
        # Check if lazy
        if isinstance(data, da.Array):
            raise ValueError("Cannot overwrite data in dask array.")
        data[corrupt_idx] = data[overwrite_idx]

    # Shift patterns
    sx, sy = data.shape[2:]
    shift_by = shift * sx * sy
    if isinstance(data, da.Array):
        data = da.roll(data, shift=shift_by)
    else:
        data = np.roll(data, shift=shift_by)

    return data
