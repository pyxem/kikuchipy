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

import numpy as np
import pytest

import kikuchipy as kp


DIR_PATH = os.path.dirname(__file__)
OXFORD_PATH = os.path.join(DIR_PATH, "../../../data/oxford_binary")
OXFORD_FILE = os.path.join(OXFORD_PATH, "patterns.ebsp")


class TestOxfordBinary:
    def test_load(self):
        s = kp.load(OXFORD_FILE, navigation_shape=(3, 3))
        s2 = kp.data.nickel_ebsd_small()

        assert isinstance(s, kp.signals.EBSD)
        assert np.allclose(s.data, s2.data)

    def test_load_lazy(self):
        s = kp.load(OXFORD_FILE, navigation_shape=(3, 3), lazy=True)
        s2 = kp.data.nickel_ebsd_small()

        assert isinstance(s, kp.signals.LazyEBSD)
        s.compute()
        assert np.allclose(s.data, s2.data)

    def test_load_no_navigation_shape(self):
        s = kp.load(OXFORD_FILE)
        s2 = kp.load(OXFORD_FILE, navigation_shape=(3, 3))

        sig_shape = s2.axes_manager.signal_shape[::-1]
        assert np.allclose(s.data, s2.data.reshape((9,) + sig_shape))

    def test_raises(self):
        with pytest.raises(ValueError, match="An assumed number of patterns"):
            _ = kp.load(OXFORD_FILE, navigation_shape=(40, 30))

        with pytest.raises(ValueError, match="The assumed number of patterns"):
            _ = kp.load(OXFORD_FILE, navigation_shape=(3, 4))
