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
import shutil

from h5py import File
import pytest

from kikuchipy import load
from kikuchipy.conftest import assert_dictionary


DIR_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIR_PATH, "../../../data")
EDAX_FILE = os.path.join(DATA_PATH, "edax_h5ebsd/patterns.h5")


class TestEDAXH5EBSD:
    def test_load(self, tmp_path, ni_small_axes_manager):
        s = load(EDAX_FILE)
        assert s.data.shape == (3, 3, 60, 60)
        assert_dictionary(s.axes_manager.as_dictionary(), ni_small_axes_manager)
        mag = s.metadata.Acquisition_instrument.SEM.magnification

        # No 'SEM-PRIAS' group available
        tmp_file = tmp_path / os.path.basename(EDAX_FILE)
        shutil.copy(EDAX_FILE, tmp_file)
        with File(tmp_file, mode="r+") as f:
            f["Scan 1"].move("SEM-PRIAS Images", "SSEM-PRIAS Images")
        s2 = load(tmp_file)
        assert s2.metadata.Acquisition_instrument.SEM.magnification != mag

        # Not a square grid
        with File(tmp_file, mode="r+") as f:
            grid = f["Scan 1/EBSD/Header/Grid Type"]
            grid[()] = "HexGrid".encode()
        with pytest.raises(IOError, match="Only square grids are"):
            _ = load(tmp_file)

    def test_save_error(self):
        s = load(EDAX_FILE)
        with pytest.raises(OSError, match="(.*) is not a supported kikuchipy"):
            s.save(EDAX_FILE, add_scan=True)
