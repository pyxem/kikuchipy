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

import io

import numpy as np
from orix.crystal_map import CrystalMap
from orix.quaternion import Rotation
import pytest

from kikuchipy.data import nickel_ebsd_small
from kikuchipy.indexing._static_pattern_matching import (
    StaticPatternMatching,
    _get_spatial_arrays,
)
from kikuchipy.io.tests.test_util import replace_stdin
from kikuchipy.signals import EBSD


class TestStaticPatternMatching:
    def test_init_static_pattern_matching(self):
        s = nickel_ebsd_small()
        sdi = StaticPatternMatching(s)

        assert isinstance(sdi.dictionaries, list)
        assert sdi.dictionaries[0] == s
        assert isinstance(sdi.dictionaries[0], EBSD)
        assert np.may_share_memory(sdi.dictionaries[0].data, s.data)

    def test_get_orientation_similarity_map(self):
        s = nickel_ebsd_small()

        s_dict1 = EBSD(s.data.reshape(-1, 60, 60))
        s_dict2 = EBSD(s.data.reshape(-1, 60, 60))
        n_patterns = s_dict1.axes_manager.navigation_size
        s_dict1._xmap = CrystalMap(Rotation(np.zeros((n_patterns, 4))))
        s_dict2._xmap = CrystalMap(Rotation(np.zeros((n_patterns, 4))))
        s_dict1.xmap.phases[0].name = "a"
        s_dict2.xmap.phases[0].name = "b"

        sd = StaticPatternMatching([s_dict1, s_dict2])
        res = sd(s, keep_n=1, get_orientation_similarity_map=True)
        xmap1, _ = res

        assert np.allclose(xmap1.scores, 1)
        assert np.all(["osm" in xmap.prop for xmap in res])

    @pytest.mark.parametrize(
        "n_rot_in, n_rot_out, keep_n", [(60, 50, 10), (40, 40, 5)]
    )
    def test_keep_n(self, n_rot_in, n_rot_out, keep_n):
        s = nickel_ebsd_small()
        s_dict = EBSD(np.random.random((n_rot_in, 60, 60)).astype(np.float32))
        s_dict._xmap = CrystalMap(Rotation(np.zeros((n_rot_in, 4))))
        sd = StaticPatternMatching(s_dict)
        xmap = sd(s)

        assert xmap.rotations_per_point == n_rot_out

        xmap2 = sd(s, keep_n=keep_n)

        assert xmap2.rotations_per_point == keep_n

    @pytest.mark.parametrize(
        "return_merged_xmap, desired_n_xmaps_out", [(True, 3), (False, 2)]
    )
    def test_return_merged_crystal_map(
        self, return_merged_xmap, desired_n_xmaps_out
    ):
        s = nickel_ebsd_small()
        s_dict1 = EBSD(s.data.reshape(-1, 60, 60))
        s_dict2 = s_dict1.deepcopy()
        n_patterns = s_dict1.axes_manager.navigation_size
        s_dict1._xmap = CrystalMap(Rotation(np.zeros((n_patterns, 4))))
        s_dict2._xmap = s_dict1.xmap.deepcopy()
        s_dict1.xmap.phases[0].name = "a"
        s_dict2.xmap.phases[0].name = "b"

        sd = StaticPatternMatching([s_dict1, s_dict2])
        res1 = sd(s, return_merged_crystal_map=return_merged_xmap)

        assert len(res1) == desired_n_xmaps_out

        sd.dictionaries.pop(-1)
        res2 = sd(s, return_merged_crystal_map=True)

        assert isinstance(res2, CrystalMap)

        res3 = sd(s)

        assert isinstance(res3, CrystalMap)

    def test_n_slices_input(self, dummy_signal):
        sig_shape = dummy_signal.axes_manager.signal_shape
        n_px = np.prod(sig_shape)
        n_sim = 13500 + 1
        rand_data = (
            np.random.randint(0, 255, n_sim * n_px)
            .reshape((n_sim,) + sig_shape)
            .astype(np.uint8)
        )
        s_dict1 = EBSD(rand_data)
        s_dict1._xmap = CrystalMap(Rotation(np.zeros((n_sim, 4))))
        sd = StaticPatternMatching(s_dict1)

        with replace_stdin(io.StringIO("y")):
            res = sd(dummy_signal, n_slices=1)
            assert isinstance(res, CrystalMap)

        with replace_stdin(io.StringIO("n")):
            res = sd(dummy_signal, n_slices=1)
            assert res is None

    @pytest.mark.parametrize(
        "slices, desired_xmap_shape", [((0, 0), ()), ((0, slice(0, 2)), (2,))],
    )
    def test_signal_varying_dimensions(
        self, dummy_signal, slices, desired_xmap_shape
    ):
        s = dummy_signal.inav[slices]
        sig_shape = dummy_signal.axes_manager.signal_shape
        s_dict1 = EBSD(dummy_signal.data.reshape((-1,) + sig_shape))
        n_sim = s_dict1.axes_manager.navigation_size
        s_dict1._xmap = CrystalMap(Rotation(np.zeros((n_sim, 4))))
        sd = StaticPatternMatching(s_dict1)
        res = sd(s)

        assert res.shape == desired_xmap_shape

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
