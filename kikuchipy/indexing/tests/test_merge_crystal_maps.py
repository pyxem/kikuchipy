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

from orix.crystal_map import CrystalMap, PhaseList, Phase
from orix.quaternion import Rotation
import numpy as np
import pytest

from kikuchipy.indexing._merge_crystal_maps import merge_crystal_maps


class TestMergeCrystalMaps:
    def test_merge_crystal_maps_1d(self):
        nav_size = 9
        nav_shape = (nav_size,)
        x = np.arange(nav_size)
        scores = np.ones(nav_shape + (1,))

        xmap1 = CrystalMap(
            Rotation(np.zeros(nav_shape + (4,))),
            x=x,
            phase_list=PhaseList(Phase("a")),
            prop={
                "scores": scores,
                "simulated_indices": np.arange(nav_size).reshape(nav_shape),
            },
        )
        xmap2 = CrystalMap(
            Rotation(np.zeros(nav_shape + (4,))),
            x=x,
            phase_list=PhaseList(Phase("b")),
            prop={
                "scores": scores,
                "simulated_indices": np.arange(nav_size, 18).reshape(nav_shape),
            },
        )
        xmap_merged = merge_crystal_maps([xmap2, xmap1])

        assert np.allclose(
            xmap_merged.simulated_indices[:, 0], np.arange(nav_size)
        )

    def test_warning_merge_maps_with_same_phase(self):
        xmap1 = CrystalMap(
            Rotation(np.zeros((9, 4))),
            x=np.arange(9),
            phase_list=PhaseList(Phase("a")),
            prop={
                "scores": np.ones((9, 1)),
                "simulated_indices": np.arange(9).reshape((9, 1)),
            },
        )
        xmap2 = CrystalMap(
            Rotation(np.zeros((9, 4))),
            x=np.arange(9),
            phase_list=PhaseList(Phase("a")),
            prop={
                "scores": np.zeros((9, 1)),
                "simulated_indices": np.arange(9, 18).reshape((9, 1)),
            },
        )
        with pytest.warns(UserWarning):
            merge_crystal_maps([xmap2, xmap1])
