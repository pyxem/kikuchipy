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
=============================
Dynamic background correction
=============================

This example shows how to remove the dynamic background of an EBSD pattern using
:meth:`~kikuchipy.signals.EBSD.remove_dynamic_background`.

More details are given in the
:doc:`pattern processing tutorial </tutorials/pattern_processing>`.
"""

# %%
# Imports.
import matplotlib.pyplot as plt

import kikuchipy as kp

# %%
# Load low resolution Ni patterns and check that the *static* background
# pattern is stored with the signal
s = kp.data.nickel_ebsd_small()
print(s.static_background)

s.remove_static_background()
s2 = s.remove_dynamic_background(inplace=False)

# %%
# Plot pattern before and after correction and the intensity histograms
patterns = [s.inav[0, 0].data, s2.inav[0, 0].data]
fig, axes = plt.subplots(2, 2, height_ratios=[3, 1.5], layout="tight")
for ax, pattern, title in zip(axes[0], patterns, ["Static", "Dynamic"]):
    ax.imshow(pattern, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
for ax, pattern in zip(axes[1], patterns):
    ax.hist(pattern.ravel(), bins=100)
