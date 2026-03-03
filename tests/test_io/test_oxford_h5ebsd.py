# Copyright 2019-2026 The kikuchipy developers
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
import pytest

import kikuchipy as kp
from kikuchipy.io.plugins.oxford_h5ebsd._api import get_binning


class TestOxfordH5EBSD:
    @pytest.mark.parametrize(
        "oxford_h5ebsd_file", ["7.0", "6.0"], indirect=["oxford_h5ebsd_file"]
    )
    def test_load_oxford_h5ebsd(
        self, oxford_h5ebsd_file, ni_small_axes_manager, assert_dictionary_func
    ):
        s = kp.load(oxford_h5ebsd_file)
        assert s.data.shape == (3, 3, 60, 60)
        assert_dictionary_func(s.axes_manager.as_dictionary(), ni_small_axes_manager)
        assert s.metadata.Acquisition_instrument.SEM.beam_energy == 20

        s2 = kp.data.nickel_ebsd_small()
        s2.remove_static_background()
        assert np.allclose(s.data, s2.data)
        assert np.allclose(s.static_background, s2.static_background)

        # Detector
        det = s.detector
        assert det.pc.shape == (3, 3, 3)
        assert det.binning == 17.0
        assert np.isclose(det.sample_tilt, 69.9, atol=0.1)
        assert np.isclose(det.tilt, 1.5)

    def test_load_unprocessed_patterns(self, oxford_h5ebsd_file):
        s1 = kp.load(oxford_h5ebsd_file)
        s2 = kp.load(oxford_h5ebsd_file, processed=False)
        assert np.allclose(s1.data + 1, s2.data)

    @pytest.mark.parametrize(
        ["version", "header_group", "binning"],
        [
            ("5.0", {"Camera Binning Mode": "8x8 (168x128 px)"}, 8),
            ("6.0", {"Camera Binning Mode": "Resolution (1244x1024 px)"}, 1),
            ("6.0", {"Camera Binning Mode": "Speed 3 (156x88 px)"}, 8),
            ("7.0", {"Camera Mode": "Sensitivity (622x512 px)"}, 2),
        ],
    )
    def test_get_binning(self, version, header_group, binning):
        assert get_binning(header_group, version) == binning
