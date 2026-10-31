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
====================
Band-pass FFT filter
====================

This example shows how to create a band-pass filter transfer function with
:func:`~kikuchipy.filters.bandpass_fft_filter` and apply it to EBSD patterns in the
frequency domain with :meth:`~kikuchipy.signals.EBSD.fft_filter`.

The band-pass filter is the product of a high-pass and a low-pass filter. The high-pass
filter removes slow intensity variations across the detector, while the low-pass filter
removes noise.

The filter parameters have the same names as the parameters returned from
:func:`~kikuchipy.pattern.optimize_bandpass_filter`, so that optimized parameters can be
passed directly. See
:ref:`sphx_glr_examples_pattern_processing_pattern_processing_optimization.py` for how
to optimize them.
"""

# %%
# Imports.
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np

import kikuchipy as kp

# %%
# Load patterns and remove the static and dynamic background.
s = kp.data.nickel_ebsd_small()
s.remove_static_background()
s.remove_dynamic_background()

# %%
# Create the band-pass filter. The cutoffs and the widths of the transitions are given
# in pixels in the frequency domain, from the center of the shifted spectrum.
w = kp.filters.bandpass_fft_filter(
    s.detector.shape,
    highpass_cutoff=3,  # 0 to pass all high frequencies
    lowpass_cutoff=23,  # > pattern width to pass all low frequencies
    highpass_cutoff_width=2,
    lowpass_cutoff_width=10,
)

# %%
# Plot the transfer function and its profile through the center. Frequencies close to
# the center (slow variations) and far from the center (noise) are suppressed.
center = s.detector.shape[0] // 2
distance = np.arange(s.detector.shape[1]) - s.detector.shape[1] // 2

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(9, 4), layout="constrained")
im = ax0.imshow(w, cmap="viridis")
ax0.set_title("Transfer function")
ax0.axis("off")
fig.colorbar(im, ax=ax0)
ax1.plot(distance, w[center])
_ = ax1.set(xlabel="Distance from center (px)", ylabel="Transfer function value")

# %%
# Apply the filter to all patterns. The filter must be applied in the frequency domain
# with the zero-frequency component shifted to the center of the spectrum.
s2 = s.fft_filter(w, function_domain="frequency", shift=True, inplace=False)

# %%
# Plot the pattern before and after bandpass filtering.
hs.plot.plot_images(
    [s.inav[0, 0], s2.inav[0, 0]],
    axes_decor="off",
    label=["Before", "After"],
    vmin=0,
    vmax=255,
    tight_layout=True,
)
