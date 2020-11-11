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

from kikuchipy.indexing.orientation_similarity_map import (
    orientation_similarity_map,
)


class TestOrientationSimilarityMap:
    def test_orientation_similarity_map(self):
        xmap = CrystalMap(
            rotations=Rotation(np.zeros((100, 4))),
            prop={"simulation_indices": np.tile(np.arange(5), (100, 1))},
            x=np.tile(np.arange(10), 10),
            y=np.tile(np.arange(10), 10),
        )
        assert np.allclose(orientation_similarity_map(xmap), np.ones((10, 10)))

    def test_n_best_too_great(self):
        xmap = CrystalMap(
            rotations=Rotation(np.zeros((100, 4))),
            prop={"simulation_indices": np.ones((100, 5))},
            x=np.tile(np.arange(10), 10),
            y=np.tile(np.arange(10), 10),
        )
        with pytest.raises(ValueError, match="n_best 6 cannot be greater than"):
            orientation_similarity_map(xmap, n_best=6)

    def test_from_n_best(self):
        sim_idx_prop = "simulated_indices"
        xmap = CrystalMap(
            rotations=Rotation(np.zeros((100, 4))),
            prop={sim_idx_prop: np.ones((100, 5))},
            x=np.tile(np.arange(10), 10),
            y=np.tile(np.arange(10), 10),
        )
        osm = orientation_similarity_map(
            xmap, simulation_indices_prop=sim_idx_prop, from_n_best=2
        )
        assert osm.shape == (10, 10, 4)
