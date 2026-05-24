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
Plot distribution of projection centers
=======================================

This example shows how to plot a distribution of projection/pattern centers (PCs) with
the :class:`~kikuchipy.detectors.EBSDDetector`.
See the detector class documentation for further details on the definition of the PC and
gnomonic coordinates.
"""

# %%
# Imports.
import matplotlib.pyplot as plt

import kikuchipy as kp

# %%
# Create a detector with smoothly varying PC values, extrapolated from a single PC
# (assumed to be in the upper left corner of a map)
det0 = kp.detectors.EBSDDetector(
    shape=(480, 640), pc=(0.4, 0.3, 0.5), px_size=70, sample_tilt=70
)
print(det0)

det = det0.extrapolate_pc(
    pc_indices=[0, 0], navigation_shape=(5, 10), step_sizes=(20, 20)
)
print(det)

# %%
# Plot PC values in maps.

det.plot_pc()

# %%
# Plot in scatter plots in vertical orientation.

det.plot_pc("scatter", annotate=True)

# %%
# Plot in a 3D scatter plot, returning the figure for saving etc.
fig = det.plot_pc("3d", return_figure=True)

plt.show()
