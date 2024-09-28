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

from matplotlib.pyplot import imread
import numpy as np
import pytest

import kikuchipy as kp


class TestNORDIFCalibrationPatterns:
    def test_read(self, nordif_path, caplog):
        caplog.set_level("DEBUG", logger="kikuchipy")

        s = kp.load(nordif_path / "Setting.txt")
        assert s.data.shape == (2, 60, 60)
        assert np.allclose(
            s.static_background,
            imread(nordif_path / "Background acquisition pattern.bmp"),
        )
        assert isinstance(s.detector, kp.detectors.EBSDDetector)
        assert s.detector.shape == s.data.shape[1:]
        assert s.detector.sample_tilt == 70

        # Calibration specific original metadata
        omd = s.original_metadata.as_dictionary()
        # Calibration pattern indices
        assert np.allclose(
            omd["calibration_patterns"]["indices"], [[447, 425], [532, 294]]
        )
        assert np.allclose(
            omd["calibration_patterns"]["indices_scaled"], [[122, 116], [145, 80]]
        )
        # Area
        assert omd["area"]["shape"] == (1000, 1000)
        assert omd["area"]["shape_scaled"] == (273, 273)
        # ROI within area
        assert omd["roi"]["origin"] == (223, 152)
        assert omd["roi"]["shape"] == (11, 11)
        assert omd["roi"]["origin_scaled"] == (61, 41)
        assert omd["roi"]["shape_scaled"] == (3, 3)
        # Area image
        assert "area_image" not in omd
        record = caplog.records[0]
        assert (record.levelname, record.message) == ("DEBUG", "No area image found")

    @pytest.mark.parametrize("setting_file", ["Setting_bad1.txt", "Setting_bad2.txt"])
    def test_no_patterns_raises(self, nordif_path, setting_file):
        with pytest.raises(ValueError, match="No calibration patterns found"):
            _ = kp.load(nordif_path / setting_file)

    def test_background_file_not_found(
        self, nordif_path, nordif_renamed_calibration_pattern
    ):
        # The unused fixture renames a permanent file, yields, and then
        # reverts the renaming
        with pytest.warns(UserWarning, match="Could not read static"):
            _ = kp.load(nordif_path / "Setting.txt")

    def test_logs_incorrect_shapes(self, caplog, nordif_path):
        caplog.set_level("DEBUG", logger="kikuchipy")

        s = kp.load(nordif_path / "Setting_bad3.txt")
        omd = s.original_metadata.as_dictionary()

        assert omd["area"]["shape"] is None
        assert "area_image" not in omd

        assert len(caplog.records) == 4
        for record, message in zip(
            caplog.records,
            [
                "Could not read area (electron image) shape",
                "Could not read area ROI 'Top'",
                "Number of samples (3, 3) differs from the one calculated from area/ROI shapes (4, 4)",
                "No area image found",
            ],
        ):
            assert record.levelname == "DEBUG"
            assert record.message == message
