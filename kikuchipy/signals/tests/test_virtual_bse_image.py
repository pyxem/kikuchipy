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

import pytest

from kikuchipy.generators import VirtualBSEGenerator


class TestVirtualBSEImage:
    def test_rescale_rgb_raises(self, dummy_signal):
        vbse_gen = VirtualBSEGenerator(dummy_signal)
        vbse_gen.grid_shape = (3, 3)
        rgb = vbse_gen.get_rgb_image(r=(0, 0), g=(0, 1), b=(1, 0))
        with pytest.raises(NotImplementedError):
            rgb.rescale_intensity()

    def test_normalize_rgb_raises(self, dummy_signal):
        vbse_gen = VirtualBSEGenerator(dummy_signal)
        vbse_gen.grid_shape = (3, 3)
        rgb = vbse_gen.get_rgb_image(r=(0, 0), g=(0, 1), b=(1, 0))
        with pytest.raises(NotImplementedError):
            rgb.normalize_intensity()
