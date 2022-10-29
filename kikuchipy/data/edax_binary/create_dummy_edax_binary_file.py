# Copyright 2019-2022 The kikuchipy developers
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

"""Script to create dummy EDAX binary UP1/UP2 files for testing and use
in the documentation.
"""

import os
from pathlib import Path

import numpy as np

import kikuchipy as kp


s = kp.data.nickel_ebsd_small()

sx, sy = s.axes_manager.signal_shape
n_pixels = sy * sx
nx, ny = s.axes_manager.navigation_shape
n_patterns = ny * nx
step_size = s.axes_manager["x"].scale

dir_data = Path(os.path.dirname(__file__))

# UP1, version 1
# --------------
with open(dir_data / "edax_binary.up1", mode="w") as file1:
    # File header: 16 bytes
    # 4 bytes with the file version
    np.array([1], "uint32").tofile(file1)
    # 12 bytes with the pattern width, height and file offset position
    np.array([sx, sy, 16], "uint32").tofile(file1)

    # Patterns
    s.data.ravel().tofile(file1)

# UP2, version 3
# --------------
# With hexagonal grid, assuming this grid (patterns as "x"):
# ------- #
#  x x x  #
# x x x x #
#  x x x  #
# ------- #
with open(dir_data / "edax_binary.up2", mode="w") as file2:
    # File header: 42 bytes
    # 4 bytes with the file version
    np.array([3], "uint32").tofile(file2)
    # 12 bytes with the pattern width, height and file offset position
    np.array([sx, sy, 42], "uint32").tofile(file2)
    # 1 byte with any "extra patterns" (?)
    np.array([1], "uint8").tofile(file2)
    # 8 bytes with the map width and height (same as square)
    np.array([nx, ny], "uint32").tofile(file2)
    # 1 byte to say whether the grid is hexagonal
    np.array([1], "uint8").tofile(file2)
    # 16 bytes with the horizontal and vertical step sizes
    np.array([np.pi, np.pi / 2], "float64").tofile(file2)

    # Patterns
    data = s.data.ravel().astype("uint16")
    np.append(data, np.zeros(sx * sy, "uint16")).tofile(file2)
