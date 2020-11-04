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

import numpy as np
from orix.crystal_map import CrystalMap
from orix.quaternion import Rotation
import pytest

from kikuchipy.data import nickel_ebsd_small
from kikuchipy.indexing._static_dictionary_indexing import (
    StaticDictionaryIndexing,
    _get_spatial_arrays,
)
from kikuchipy.signals import EBSD


class TestStaticDictionaryIndexing:
    def test_init_static_dictionary_indexing(self):
        s = nickel_ebsd_small()
        _ = StaticDictionaryIndexing(s)

    def test_static_dictionary_indexing(self):
        s = nickel_ebsd_small()
        s_dict1 = EBSD(s.data.reshape(-1, 60, 60))
        s_dict2 = EBSD(s.data.reshape(-1, 60, 60))
        s_dict1._xmap = CrystalMap(Rotation(np.zeros((9, 4))), x=np.arange(9))
        s_dict2._xmap = CrystalMap(Rotation(np.zeros((9, 4))), x=np.arange(9))
        s_dict1.xmap.phases._dict[0].name = "1"
        s_dict2.xmap.phases._dict[0].name = "2"
        sd = StaticDictionaryIndexing([s_dict1, s_dict2])
        res = sd(s)
        cm1, _, _ = res

        assert np.allclose(cm1.scores, 1)
        # np.isin(["scores","simulated_indices","osm"],list(cm.prop.keys()))
        assert np.all(
            [
                "scores" in cm.prop
                and "simulated_indices" in cm.prop
                and "osm" in cm.prop
                for cm in res
            ]
        )

    @pytest.mark.parametrize(
        "nav_slice, desired_arrays",
        [
            # 0d
            ((0, 0), ()),
            ((slice(0, 0), slice(0, 0)), (np.array([]),) * 2),
            # 1d
            ((0, slice(None)), np.tile(np.arange(0, 4.5, 1.5), 3)),
            # 2d
            (
                (slice(None), slice(0, 2)),
                (
                    np.tile(np.arange(0, 4.5, 1.5), 3),
                    np.tile(np.arange(0, 3, 1.5), 2),
                ),
            ),
        ],
    )
    def test_get_spatial_arrays(self, nav_slice, desired_arrays):
        """Ensure spatial arrays for 0d, 1d and 2d EBSD signals are
        returned correctly.
        """
        s = nickel_ebsd_small()
        spatial_arrays = _get_spatial_arrays(s.inav[nav_slice].axes_manager)

        if len(spatial_arrays) == 0:
            assert spatial_arrays == desired_arrays
        else:
            assert [
                np.allclose(spatial_arrays[i], desired_arrays[i])
                for i in range(len(spatial_arrays))
            ]
