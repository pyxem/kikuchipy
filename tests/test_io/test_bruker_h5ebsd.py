# Copyright 2019-2024 The kikuchipy developers
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

from h5py import File
import pytest

import kikuchipy as kp

from ..conftest import assert_dictionary


class TestBrukerH5EBSD:
    def test_load(self, bruker_path, ni_small_axes_manager):
        bruker_file = bruker_path / "patterns.h5"
        # Cover grid type check
        with File(bruker_file, mode="r+") as f:
            grid = f["Scan 0/EBSD/Header/Grid Type"]
            grid[()] = "hexagonal".encode()
        with pytest.raises(IOError, match="Only square grids are"):
            kp.load(bruker_file)
        with File(bruker_file, mode="r+") as f:
            grid = f["Scan 0/EBSD/Header/Grid Type"]
            grid[()] = "isometric".encode()

        s = kp.load(bruker_file)
        assert s.data.shape == (3, 3, 60, 60)
        assert_dictionary(s.axes_manager.as_dictionary(), ni_small_axes_manager)

    def test_load_roi(self, bruker_path):
        s = kp.load(bruker_path / "patterns_roi.h5")
        assert s.data.shape == (3, 2, 60, 60)

        with pytest.raises(ValueError, match="Only a rectangular region of"):
            kp.load(bruker_path / "patterns_roi_nonrectangular.h5")
