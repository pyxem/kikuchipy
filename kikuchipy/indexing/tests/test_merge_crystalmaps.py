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

import pytest
import warnings
import numpy as np

from orix.crystal_map import CrystalMap
from orix.quaternion import Rotation

from kikuchipy.indexing.merge_crystalmaps import merge_crystalmaps


def test_merge_crystalmaps():
    xmap1 = CrystalMap(
        Rotation(np.zeros((9, 4))),
        x=np.arange(9),
        prop={
            "scores": np.ones((9, 1)),
            "simulated_indices": np.arange(9).reshape((9, 1)),
        },
    )
    xmap2 = CrystalMap(
        Rotation(np.zeros((9, 4))),
        x=np.arange(9),
        prop={
            "scores": np.zeros((9, 1)),
            "simulated_indices": np.arange(9, 18).reshape((9, 1)),
        },
    )
    xmap1.phases._dict[0].name = "1"
    xmap2.phases._dict[0].name = "2"
    xmap_merged = merge_crystalmaps([xmap2, xmap1])
    assert np.allclose(xmap_merged.simulated_indices[:, 0], np.arange(9))


def test_warning_merge_maps_with_same_phase():
    xmap1 = CrystalMap(
        Rotation(np.zeros((9, 4))),
        x=np.arange(9),
        prop={
            "scores": np.ones((9, 1)),
            "simulated_indices": np.arange(9).reshape((9, 1)),
        },
    )
    xmap2 = CrystalMap(
        Rotation(np.zeros((9, 4))),
        x=np.arange(9),
        prop={
            "scores": np.zeros((9, 1)),
            "simulated_indices": np.arange(9, 18).reshape((9, 1)),
        },
    )
    with pytest.warns(UserWarning):
        merge_crystalmaps([xmap2, xmap1])
