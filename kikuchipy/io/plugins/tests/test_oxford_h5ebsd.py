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

import numpy as np

import kikuchipy as kp
from kikuchipy.conftest import assert_dictionary


DIR_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIR_PATH, "../../../data")
OXFORD_FILE = os.path.join(DATA_PATH, "oxford_h5ebsd/patterns.h5oina")


class TestOxfordH5EBSD:
    def test_load(self, tmp_path, ni_small_axes_manager, nickel_ebsd_small_di_xmap):
        s = kp.load(OXFORD_FILE)
        assert s.data.shape == (3, 3, 60, 60)
        assert_dictionary(s.axes_manager.as_dictionary(), ni_small_axes_manager)
        assert s.metadata.Acquisition_instrument.SEM.beam_energy == 20
        assert s.detector.pc.shape == (3, 3, 3)
        assert np.isclose(s.detector.sample_tilt, 69.9, atol=0.1)

        s2 = kp.data.nickel_ebsd_small()
        s2.remove_static_background()
        assert np.allclose(s.data, s2.data)
        assert np.allclose(s.static_background, s2.static_background)
