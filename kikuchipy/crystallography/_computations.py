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

from typing import Union

import numpy as np


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
