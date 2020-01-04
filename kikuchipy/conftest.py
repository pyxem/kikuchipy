# -*- coding: utf-8 -*-
# Copyright 2019-2020 The KikuchiPy developers
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

import gc
import os
import tempfile

import numpy as np
import pytest

import kikuchipy as kp


@pytest.fixture
def dummy_signal():
    """Dummy signal of shape <3, 3|3, 3>. If this is changed, all tests
    using this signal will fail since they compare the output from
    methods using this signal (as input) to hard-coded outputs.
    """

    # fmt: off
    dummy_array = np.array(
        [5, 6, 5, 7, 6, 5, 6, 1, 0, 9, 7, 8, 7, 0, 8, 8, 7, 6, 0, 3, 3, 5, 2, 9,
         3, 3, 9, 8, 1, 7, 6, 4, 8, 8, 2, 2, 4, 0, 9, 0, 1, 0, 2, 2, 5, 8, 6, 0,
         4, 7, 7, 7, 6, 0, 4, 1, 6, 3, 4, 0, 1, 1, 0, 5, 9, 8, 4, 6, 0, 2, 9, 2,
         9, 4, 3, 6, 5, 6, 2, 5, 9],
        dtype=np.uint8
    ).reshape((3, 3, 3, 3))
    # fmt: on

    return kp.signals.EBSD(dummy_array)


@pytest.fixture
def dummy_background():
    """Dummy static background pattern for the dummy signal. If this is
    changed, all tests using this background will fail since they
    compare the output from methods using this background (as input) to
    hard-coded outputs.
    """

    return np.array([5, 4, 5, 4, 3, 4, 4, 4, 3], dtype=np.uint8).reshape((3, 3))


@pytest.fixture()
def save_path_h5ebsd():
    """Temporary file in a temporary directory for use when tests need
    to write, and sometimes read again, a signal to, and from, a file.
    """

    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, "patterns_temp.h5")
        yield file_path
        gc.collect()
