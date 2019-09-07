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

import os
import pytest
import numpy as np
import kikuchipy as kp
import matplotlib.pyplot as plt

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def signal():
    """Signal with shape <3, 3|60, 60> initialised from a NumPy array of
    shape (3, 3, 60, 60) to be used in tests.
    """

    return kp.signals.EBSD(
        np.load(os.path.join(DIR_PATH, 'test_data/numpy/patterns.npy')))


@pytest.fixture
def background_pattern():
    """Static background pattern of shape (60, 60)."""

    return plt.imread(
        os.path.join(
            DIR_PATH, 'test_data/nordif/Background acquisition pattern.bmp'))
