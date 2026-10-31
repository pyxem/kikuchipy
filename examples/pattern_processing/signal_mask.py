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
==================================
Signal mask from pattern intensity
==================================

This example shows how to get a signal mask of pixels without Kikuchi diffraction, like
the corners outside a circular phosphor screen, using
:func:`~kikuchipy.pattern.get_signal_mask`.

The mask is found by intensity thresholding, by default with the minimum method in
:func:`skimage.filters.threshold_minimum`. Pixels to exclude are True, following the
convention of signal masks in kikuchipy, e.g. in
:meth:`~kikuchipy.signals.EBSD.refine_orientation` and
:func:`~kikuchipy.pattern.optimize_pattern_processing`.
"""

# %%
# Imports.
import matplotlib.pyplot as plt

import kikuchipy as kp

# %%
# Load a single crystal silicon pattern acquired on a detector with a circular phosphor
# screen.
s = kp.data.si_ebsd_moving_screen(allow_download=True)

# %%
# Get the signal mask from the static background. An average pattern, like the static
# background or the mean pattern of a dataset, is less influenced by noise and Kikuchi
# bands than a single pattern.
signal_mask = kp.pattern.get_signal_mask(s.static_background)
print(f"Masked out {signal_mask.mean():.1%} of the pixels")

# %%
# Plot the pattern, the mask, and the pattern with the masked out pixels set to zero.
pattern = s.data
fig, axes = plt.subplots(ncols=3, figsize=(12, 4), layout="constrained")
for ax, image, title in zip(
    axes,
    [pattern, signal_mask, pattern * ~signal_mask],
    ["Pattern", "Signal mask", "Pattern * ~signal mask"],
):
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.show()
