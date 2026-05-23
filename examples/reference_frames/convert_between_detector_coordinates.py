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

"""
===========================================
Convert between detector coordinate formats
===========================================

This example shows how to convert between pixel and gnomonic detector coordinates.
"""

# %%
# Imports.
import numpy as np

import kikuchipy as kp

# %%
# Convert a set of detector pixel coordinates (row, column) to gnomonic coordinates on
# an EBSD detector for all sample positions (patterns).
pc = np.random.random(5 * 10 * 3).reshape(5, 10, 3)
det = kp.detectors.EBSDDetector((60, 60), pc=pc)

print(det)
print(det.navigation_shape)

# 20 coordinates
px_coords1 = np.random.randint(0, det.shape[0], (20, 2))
gn_coords1 = det.to_gnomonic_coords(px_coords1)
print(gn_coords1.shape)

# %%
# This also works for a number of sets of coordinates, each set different for each
# sample position.
# Here, we have 7 sets of 20 coordinates.
px_coords2 = np.random.randint(0, det.shape[0], det.navigation_shape + (7, 20, 2))
gn_coords2 = det.to_gnomonic_coords(px_coords2)
print(gn_coords2.shape)

# %%
# Convert 7 sets of 20 coordinates, the same for all sample positions.
# This time, from gnomonic coordinates to pixels.
px_coords3 = det.to_pixel_coords(gn_coords2[:, :, 0])
print(np.allclose(px_coords3, px_coords2[:, :, 0]))
