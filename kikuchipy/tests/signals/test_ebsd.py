# -*- coding: utf-8 -*-
# Copyright 2019 The KikuchiPy developers
#
# This file is part of KikuchiPy.
#
# KikuchiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KikuchiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KikuchiPy. If not, see <http://www.gnu.org/licenses/>.

from kikuchipy.signals import EBSD
import numpy as np
import pytest


class TestEBSD:

    def test_init(self):
        array0 = np.zeros(shape=(10, 10, 10, 10))
        s0 = EBSD(array0)
        assert array0.shape == s0.axes_manager.shape

        with pytest.raises(ValueError):
            EBSD(np.zeros(10))

        array1 = np.zeros(shape=(10, 10))
        s1 = EBSD(array1)
        assert array1.shape == s1.axes_manager.shape
