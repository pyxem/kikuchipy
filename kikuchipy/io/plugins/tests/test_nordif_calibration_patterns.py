# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
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

import pytest

from kikuchipy.io.plugins.nordif_calibration_patterns import _get_coordinates
from kikuchipy import load


DIR_PATH = os.path.dirname(__file__)
NORDIF_DIR = os.path.join(DIR_PATH, "../../../data/nordif")


class TestNORDIFCalibrationPatterns:
    def test_read(self):
        s = load(os.path.join(NORDIF_DIR, "Setting.txt"))
        assert s.data.shape == (2, 60, 60)

    @pytest.mark.parametrize("setting_file", ["Setting_bad1.txt", "Setting_bad2.txt"])
    def test_get_coordinates_raises(self, setting_file):
        with pytest.raises(ValueError, match="No calibration patterns found"):
            _ = _get_coordinates(os.path.join(NORDIF_DIR, setting_file))

    def test_background_file_not_found(self):
        fname_orig = "Background calibration pattern.bmp"
        file_orig = os.path.join(NORDIF_DIR, fname_orig)
        file_new = os.path.join(NORDIF_DIR, file_orig + ".bak")
        os.rename(file_orig, file_new)
        with pytest.warns(UserWarning, match="Could not read static"):
            _ = load(os.path.join(NORDIF_DIR, "Setting.txt"))
        os.rename(file_new, file_orig)
