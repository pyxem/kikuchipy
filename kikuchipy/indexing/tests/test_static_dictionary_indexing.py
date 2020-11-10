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
        sdi = StaticDictionaryIndexing(s)

        assert isinstance(sdi.dictionaries, list)
        assert sdi.dictionaries[0] == s
        assert isinstance(sdi.dictionaries[0], EBSD)
        assert np.may_share_memory(sdi.dictionaries[0].data, s.data)

    def test_static_dictionary_indexing_osm(self):
        s = nickel_ebsd_small()

        s_dict1 = EBSD(s.data.reshape(-1, 60, 60))
        s_dict2 = EBSD(s.data.reshape(-1, 60, 60))
        s_dict1._xmap = CrystalMap(Rotation(np.zeros((9, 4))))
        s_dict2._xmap = CrystalMap(Rotation(np.zeros((9, 4))))
        s_dict1.xmap.phases[0].name = "a"
        s_dict2.xmap.phases[0].name = "b"

        sd = StaticDictionaryIndexing([s_dict1, s_dict2])
        res = sd(s, return_merged_crystal_map=False)
        xmap1, _ = res

        assert np.allclose(xmap1.scores, 1)
        # np.isin(["scores","simulated_indices","osm"],list(cm.prop.keys()))
        assert np.all(["osm" in xmap.prop for xmap in res])

    @pytest.mark.parametrize(
        "nav_slice, step_sizes, desired_arrays",
        [
            # 0d
            ((0, 0), (1, 1), ()),
            ((slice(0, 0), slice(0, 0)), (1, 1), (np.array([]),) * 2),
            # 1d
            ((0, slice(None)), (1, 1.5), np.tile(np.arange(0, 4.5, 1.5), 3)),
            # 2d
            (
                (slice(None), slice(0, 2)),
                (2, 1.5),
                (
                    np.tile(np.arange(0, 6, 2), 2),
                    np.tile(np.arange(0, 3, 1.5), 3),
                ),
            ),
            (
                (slice(None), slice(0, 2)),
                (0.5, 1),
                (
                    np.tile(np.arange(0, 1.5, 0.5), 2),
                    np.tile(np.arange(0, 2, 1), 3),
                ),
            ),
        ],
    )
    def test_get_spatial_arrays(self, nav_slice, step_sizes, desired_arrays):
        """Ensure spatial arrays for 0d, 1d and 2d EBSD signals are
        returned correctly.
        """
        s = nickel_ebsd_small()
        s.axes_manager["x"].scale = step_sizes[0]
        s.axes_manager["y"].scale = step_sizes[1]
        axes_manager = s.inav[nav_slice].axes_manager
        spatial_arrays = _get_spatial_arrays(
            shape=axes_manager.navigation_shape,
            extent=axes_manager.navigation_extent,
            step_sizes=[i.scale for i in axes_manager.navigation_axes],
        )

        if len(spatial_arrays) == 0:
            assert spatial_arrays == desired_arrays
        else:
            assert [
                np.allclose(spatial_arrays[i], desired_arrays[i])
                for i in range(len(spatial_arrays))
            ]
