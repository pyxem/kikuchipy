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
Plot nice master pattern image
==============================

This example shows you how to plot a nice and clean image of an EBSD master pattern.
More details are given in the
:doc:`visualization tutorial </tutorials/visualizing_patterns>`.
"""

# %%
# Imports.
import matplotlib.pyplot as plt
import numpy as np

import kikuchipy as kp

# %%
# Load both hemispheres of master pattern in stereographic projection.
mp = kp.data.nickel_ebsd_master_pattern_small(hemisphere="both")
print(mp)

# %%
# Extract the underlying data of both hemipsheres and mask out the
# surrounding black pixels.
data = mp.data.astype("float32")
mask = data[0] == 0
data[:, mask] = np.nan

# %%
# Plot both hemispheres with labels.
fig, (ax0, ax1) = plt.subplots(ncols=2, layout="tight")
ax0.imshow(data[0], cmap="gray")
ax1.imshow(data[1], cmap="gray")
ax0.axis("off")
ax1.axis("off")
ax0.set_title("Upper")
ax1.set_title("Lower")
