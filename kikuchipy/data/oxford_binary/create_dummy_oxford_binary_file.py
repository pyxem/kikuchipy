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

"""Script to create a dummy Oxford Instruments' binary .ebsp file
for testing and use in the user guide.
"""

import os

import numpy as np

import kikuchipy as kp


s = kp.data.nickel_ebsd_small()

sc, sr = s.axes_manager.signal_shape
n_pixels = sr * sc
nc, nr = s.axes_manager.navigation_shape
n_patterns = nr * nc
step_size = s.axes_manager["x"].scale

dir_data = os.path.abspath(os.path.dirname(__file__))
fname = os.path.join(dir_data, "patterns.ebsp")
file = open(fname, mode="w")

pattern_header_size = 16
pattern_footer_size = 18

# Write file header: 8 bytes with the file version
version = np.array(-2, dtype=np.int64)
version.tofile(file)
# Write pattern starts: 8 bytes per position
pattern_starts = np.arange(n_patterns, dtype=np.int64)
pattern_starts *= pattern_header_size + n_pixels + pattern_footer_size
pattern_starts += version.nbytes + n_patterns * 8
# Shift positions one step to the right
pattern_starts = np.roll(pattern_starts, shift=1)
pattern_starts.tofile(file)
# And thus the patterns must be shifted one step to the left
new_order = np.roll(np.arange(n_patterns), shift=-1)

# Pattern header: 16 bytes with whether the pattern is compressed,
# pattern height, pattern width and the number of pattern bytes
pattern_header = np.array([0, sr, sc, sr * sc], dtype=np.int32)

# Write patterns with header and footer
pattern_data = s.data
for i in new_order:
    r, c = np.unravel_index(i, (nr, nc))
    pattern_header.tofile(file)
    pattern_data[r, c].tofile(file)

    # Pattern footer
    np.array(1, dtype=bool).tofile(file)  # has_beam_x
    np.array(c * step_size, dtype=np.float64).tofile(file)  # beam_x
    np.array(1, dtype=bool).tofile(file)  # has_beam_y
    np.array(r * step_size, dtype=np.float64).tofile(file)  # beam_y

file.close()
