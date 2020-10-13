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

"""Color palettes for Kikuchi bands."""

import numpy as np


# TSL Kikuchi band colors (custom names)
RED = (1, 0, 0)
GREEN = (0, 1, 0)
BLUE = (0, 0, 1)
YELLOW = (1.0, 1, 0)
CYAN = (0, 1, 1)
PINK = (1, 0, 1)
BROWN = (0.5, 0, 0)
EARTH = (0.5, 0.5, 0)
PURPLE = (0.5, 0, 0.5)
DARK_GREEN = (0, 0.5, 0)
DARKER_GREEN = (0, 0.5, 0.5)
DARK_BLUE = (0, 0, 0.5)
# TODO: More colors!
KIKUCHI_BAND_COLORS_TSL = {
    "m-3m": np.array(
        [
            ([1, -1, -1], RED),
            ([0, -2, 0], YELLOW),
            ([0, -2, 2], GREEN),
            ([1, -3, -1], BLUE),
            ([2, -2, -2], RED),
            ([0, -4, 0], YELLOW),
            ([1, -3, -3], PINK),
            ([0, -4, 2], CYAN),
            ([2, -4, -2], BROWN),
            ([1, -5, -1], DARK_GREEN),
            ([3, -3, -3], RED),
            ([0, -4, 4], GREEN),
            ([1, -5, -3], PURPLE),
            ([2, -4, -4], DARK_BLUE),
            ([0, -6, 0], YELLOW),
            ([3, -5, -3], GREEN),
            ([2, -6, -2], BLUE),
            ([4, -4, -4], RED),
            ([1, -5, -5], RED),
            ([1, -7, -1], YELLOW),
            ([0, -6, 4], GREEN),
            ([2, -6, -4], BLUE),
            ([1, -7, -3], PINK),
            ([3, -5, -5], CYAN),
            ([0, -8, 0], YELLOW),
            ([3, -7, -3], BROWN),
            ([0, -8, 2], DARK_GREEN),
            ([4, -6, -4], PURPLE),
            ([0, -6, 6], GREEN),
            ([2, -8, -2], DARK_BLUE),
            ([5, -5, -5], RED),
            ([1, -7, -5], EARTH),
            ([2, -6, -6], PINK),
            ([0, -8, 4], CYAN),
            ([1, -9, -1], DARKER_GREEN),
            ([3, -7, -5], RED),
            ([2, -8, -4], YELLOW),
            ([4, -6, -6], GREEN),
            ([1, -9, -3], BLUE),
            ([4, -8, -4], BROWN),
            ([3, -9, -3], BLUE),
            ([1, -7, -7], PINK),
            ([5, -7, -5], CYAN),
            ([0, -8, 6], BROWN),
            ([2, -8, -6], DARK_GREEN),
            ([3, -7, -7], PURPLE),
            ([1, -9, -5], DARK_BLUE),
            ([6, -6, -6], RED),
            ([3, -9, -5], EARTH),
            ([4, -8, -6], DARKER_GREEN),
            ([5, -7, -7], RED),
            ([0, -8, 8], GREEN),
            ([5, -9, -5], YELLOW),
            ([1, -9, -7], GREEN),
            ([2, -8, -8], BLUE),
            ([6, -8, -6], PINK),
            ([3, -9, -7], CYAN),
            ([4, -8, -8], DARK_BLUE),
            ([7, -7, -7], RED),
            ([5, -9, -7], BROWN),
            ([1, -9, -9], DARK_GREEN),
            ([6, -8, -8], PURPLE),
            ([3, -9, -9], PINK),
            ([7, -9, -7], DARK_BLUE),
            ([5, -9, -9], EARTH),
            ([8, -8, -8], RED),
            ([7, -9, -9], DARKER_GREEN),
            ([9, -9, -9], RED),
        ],
    )
}
