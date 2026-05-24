#
# Copyright 2019-2026 the kikuchipy developers
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
#

import matplotlib.pyplot as plt
import numpy as np

import kikuchipy as kp
from kikuchipy._utils.rosettasciio_utils import RGB_DTYPES


def test_get_rgb_navigator():
    s = kp.data.nickel_ebsd_small(lazy=True).inav[:2, :3]
    nav_shape = s._navigation_shape_rc
    image = np.random.random(np.prod(nav_shape) * 3).reshape(nav_shape + (3,))

    s_rgb8 = kp.draw.get_rgb_navigator(image, dtype=np.uint8)
    rgb8_data = s_rgb8.data
    assert np.issubdtype(rgb8_data.dtype, RGB_DTYPES["rgb8"])
    assert rgb8_data.shape == nav_shape
    s.plot(navigator=s_rgb8)

    # Both rgb8 and rgb16 works
    s_rgb16 = kp.draw.get_rgb_navigator(image, dtype=np.dtype("uint16"))
    rgb16_data = s_rgb16.data
    assert rgb16_data.shape == nav_shape
    assert np.issubdtype(rgb16_data.dtype, RGB_DTYPES["rgb16"])

    plt.close("all")
