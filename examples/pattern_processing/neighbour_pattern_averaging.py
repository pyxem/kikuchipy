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
===========================
Neighbour pattern averaging
===========================

This example shows how to average each pattern in a scan with its nearest neighbours
using :meth:`~kikuchipy.signals.EBSD.average_neighbour_patterns`.

More details are given in the
:doc:`pattern processing tutorial </tutorials/pattern_processing>` and the
:doc:`feature maps tutorial </tutorials/feature_maps>`.
"""

# %%
# Imports.
import hyperspy.api as hs
import matplotlib.pyplot as plt

import kikuchipy as kp

# Silence progressbars
hs.preferences.General.show_progressbar = False

# %%
# Load Ni patterns and subtract static and dynamic background.
s = kp.data.nickel_ebsd_large()
print(s)

s.remove_static_background()
s.remove_dynamic_background()

# %%
# Get image quality before averaging
iq0 = s.get_image_quality()

# %%
# Keep one pattern for comparison.
x, y = (50, 8)
pattern0 = s.inav[x, y].deepcopy()

# %%
# Average in a (3, 3) window with a Gaussian kernel with a standard deviation of 1.
s.average_neighbour_patterns(window="gaussian", std=1)

iq1 = s.get_image_quality()
pattern1 = s.inav[x, y]

# %%
# Plot pattern and histograms of image qualities before and after averaging.
fig, axes = plt.subplots(2, 2, height_ratios=[3, 1.5], layout="tight")
for ax, pattern, title in zip(
    axes[0],
    [pattern0, pattern1],
    ["Static + Dynamic", "Static + Dynamic + Averaging"],
):
    ax.imshow(pattern, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
for ax, iq in zip(axes[1], [iq0, iq1]):
    ax.hist(iq.ravel(), bins=100)
