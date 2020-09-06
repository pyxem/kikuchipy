# -*- coding: utf-8 -*-
# Copyright 2019-2020 The kikuchipy developers
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

import pytest

from kikuchipy.detectors import EBSDDetector


class TestEBSDDetector:
    def test_init(self):
        """Initialization is OK."""
        nrows = 1
        ncols = 2
        px_size = 3
        binning = 4
        tilt = 5
        det = EBSDDetector(
            shape=(nrows, ncols), px_size=px_size, binning=binning, tilt=tilt,
        )

        assert det.shape == (nrows, ncols)
        assert det.aspect_ratio == 0.5


#    def test_nav_shape(self):
