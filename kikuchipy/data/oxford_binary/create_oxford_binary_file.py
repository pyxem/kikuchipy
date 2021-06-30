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

import kikuchipy as kp


s = kp.data.nickel_ebsd_small()

sc, sr = s.axes_manager.signal_shape
n_pixels = sr * sc
nc, nr = s.axes_manager.navigation_shape
n_patterns = nr * nc

dir_data = os.path.abspath(os.path.dirname(__file__))
fname = os.path.join(dir_data, "patterns.ebsp")
file = open(fname, mode="w")

# Write file header: 8 bytes with ?
file_header = np.ones(1, dtype=np.int64)
file_header.tofile(file)

# Pattern header: 16 bytes with pattern height, pattern width and ?
pattern_header_size = 16
pattern_header = np.zeros(8, dtype=np.uint16)
pattern_header[2] = sr
pattern_header[4] = sc

# Pattern footer: 18 bytes with ?
pattern_footer_size = 18
pattern_footer = np.zeros(9, dtype=np.uint16)

# Write pattern positions: 8 bytes per position
pattern_positions = np.arange(n_patterns, dtype=np.int64)
pattern_positions *= pattern_header_size + n_pixels + pattern_footer_size
pattern_positions += file_header.nbytes + n_patterns * 8
# Shift positions one step to the right
pattern_positions = np.roll(pattern_positions, shift=1)
pattern_positions.tofile(file)
# And thus the patterns must be shifted one step to the left
new_order = np.roll(np.arange(n_patterns), shift=-1)

# Write patterns with header and footer
pattern_data = s.data
for i in new_order:
    r, c = np.unravel_index(i, (nr, nc))
    pattern_header.tofile(file)
    pattern_data[r, c].tofile(file)
    pattern_footer.tofile(file)

file.close()
