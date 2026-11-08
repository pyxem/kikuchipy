#
# Copyright 2019-2026 the kikuchipy developers
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
#


import numpy as np
import orix.crystal_map as ocm
import orix.quaternion as oqu

from kikuchipy.signals.util._crystal_map import (
    _create_coordinate_arrays,
    _get_indexed_points_in_data_in_xmap,
)


class TestCreateCoordinateArrays:
    def test_empty_shape(self):
        """An empty shape, the navigation shape of a single pattern,
        gives coordinates of a single point.
        """
        for step_sizes in [None, ()]:
            coords, map_size = _create_coordinate_arrays((), step_sizes)
            assert map_size == 1
            assert list(coords) == ["x"]
            assert coords["x"].size == 1

    def test_equal_to_orix(self):
        for shape, step_sizes in [((3,), (1.5,)), ((2, 3), (1, 2))]:
            coords1, map_size1 = _create_coordinate_arrays(shape, step_sizes)
            coords2, map_size2 = ocm.create_coordinate_arrays(shape, step_sizes)
            assert map_size1 == map_size2
            for key in coords2:
                assert np.allclose(coords1[key], coords2[key])

    def test_crystal_map_of_single_point(self):
        coords, _ = _create_coordinate_arrays(())
        xmap = ocm.CrystalMap(oqu.Rotation.identity(), **coords)
        assert xmap.size == 1
        assert xmap.shape == ()


class TestGetIndexedPointsInDataInXmap:
    def test_single_point(self):
        """A crystal map of a single point, of shape (), is supported."""
        xmap = ocm.CrystalMap(
            oqu.Rotation.identity(), phase_list=ocm.PhaseList(ocm.Phase("a", 225))
        )
        assert xmap.shape == ()
        in_data_indexed, in_data, phase_id, mask_shape = (
            _get_indexed_points_in_data_in_xmap(xmap)
        )
        assert np.array_equal(in_data_indexed, [True])
        assert np.array_equal(in_data, [True])
        assert phase_id == 0
        assert mask_shape == (1,)
