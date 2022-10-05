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

from matplotlib.pyplot import imread
import numpy as np
import pytest

import kikuchipy as kp
from kikuchipy.io.plugins.nordif_calibration_patterns import _get_coordinates


DIR_PATH = os.path.dirname(__file__)
NORDIF_DIR = os.path.join(DIR_PATH, "../../../data/nordif")
BG_FILE = os.path.join(NORDIF_DIR, "Background acquisition pattern.bmp")


class TestNORDIFCalibrationPatterns:
    def test_read(self):
        s = kp.load(os.path.join(NORDIF_DIR, "Setting.txt"))
        assert s.data.shape == (2, 60, 60)
        assert np.allclose(s.static_background, imread(BG_FILE))
        assert isinstance(s.detector, kp.detectors.EBSDDetector)
        assert s.detector.shape == s.data.shape[1:]
        assert s.detector.sample_tilt == 70

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
            _ = kp.load(os.path.join(NORDIF_DIR, "Setting.txt"))
        os.rename(file_new, file_orig)
