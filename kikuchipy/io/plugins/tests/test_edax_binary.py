# Copyright 2019-2022 The kikuchipy developers
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

import os

import dask.array as da
import numpy as np
import pytest

import kikuchipy as kp


DIR_PATH = os.path.dirname(__file__)
EDAX_PATH = os.path.join(DIR_PATH, "../../../data/edax_binary")
FILE_UP1 = os.path.join(EDAX_PATH, "edax_binary.up1")
FILE_UP2 = os.path.join(EDAX_PATH, "edax_binary.up2")


class TestEDAXBinaryReader:
    def test_load_up1(self):
        """Load UP1 file into memory."""
        s = kp.data.nickel_ebsd_small()
        s1 = kp.load(FILE_UP1)
        assert isinstance(s1, kp.signals.EBSD)
        assert np.allclose(s1.data, s.data.reshape((-1, 60, 60)))

    def test_load_up2(self):
        """Load UP2 file into memory."""
        s = kp.data.nickel_ebsd_small()
        data_up2 = s.data.reshape((-1, 60, 60)).astype("uint16")
        data_up2 = np.append(data_up2, np.zeros((1, 60, 60), "uint16"), axis=0)

        with pytest.warns(UserWarning, match="Returned signal has one navigation "):
            s2 = kp.load(FILE_UP2)
            assert isinstance(s2, kp.signals.EBSD)
            assert np.allclose(s2.data, data_up2)

    def test_load_lazy(self):
        """Load lazily."""
        s = kp.load(FILE_UP1, lazy=True)
        s2 = kp.data.nickel_ebsd_small()

        assert isinstance(s, kp.signals.LazyEBSD)
        assert isinstance(s.data, da.Array)
        s.compute()
        assert np.allclose(s.data, s2.data.reshape((-1, 60, 60)))

    @pytest.mark.parametrize(
        "edax_binary_file",
        [(1, (2, 3), (10, 10), "uint8", 2, False)],
        indirect=["edax_binary_file"],
    )
    def test_version_raises(self, edax_binary_file):
        """Ensure files written with version 2 raises an error."""
        with pytest.raises(ValueError, match="Only files with version 1 or >= 3"):
            _ = kp.load(edax_binary_file.name)

    def test_nav_shape(self):
        """Ensure navigation shape can be set, and that an error is
        raised if the shape does not correspond to the number of
        patterns.
        """
        s = kp.load(FILE_UP1, nav_shape=(3, 3))
        s2 = kp.data.nickel_ebsd_small()
        assert np.allclose(s.data, s2.data)

        with pytest.raises(ValueError, match=r"Given `nav_shape` \(3, 4\) does not "):
            _ = kp.load(FILE_UP1, nav_shape=(3, 4))

    @pytest.mark.parametrize(
        "edax_binary_file",
        [(2, (2, 3), (10, 10), "uint16", 4, False)],
        indirect=["edax_binary_file"],
    )
    def test_nav_shape_up2_not_hex(self, edax_binary_file):
        """Test navigation shape when not hex."""
        s = kp.load(edax_binary_file.name)
        assert s.axes_manager.navigation_shape[::-1] == (2, 3)
