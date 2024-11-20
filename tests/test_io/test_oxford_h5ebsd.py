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

import numpy as np

import kikuchipy as kp


class TestOxfordH5EBSD:
    def test_load(
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
        assert np.isclose(det.sample_tilt, 69.9, atol=0.1)
        assert det.binning == 8
        assert np.isclose(det.tilt, 1.5)

    def test_load_unprocessed_patterns(self, oxford_h5ebsd_file):
        s1 = kp.load(oxford_h5ebsd_file)
        s2 = kp.load(oxford_h5ebsd_file, processed=False)
        assert np.allclose(s1.data + 1, s2.data)
