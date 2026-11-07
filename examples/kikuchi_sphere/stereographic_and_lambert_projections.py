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
=====================================
Stereographic and Lambert projections
=====================================

This example shows how to project a :class:`~kikuchipy.signals.EBSDMasterPattern` from
the stereographic projection to the square Lambert projection with
:meth:`~kikuchipy.signals.EBSDMasterPattern.as_lambert`, and back with
:meth:`~kikuchipy.signals.EBSDMasterPattern.as_stereo`.

Pattern matching requires a master pattern in the square Lambert projection, which
preserves areas.
The stereographic projection preserves angles instead, which makes it well suited for
inspecting a master pattern.

See :cite:`rosca2014anew` and :cite:`callahan2013dynamical` for details of the
Lambert projection and its use.
"""

# %%
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np

import kikuchipy as kp

hs.preferences.General.show_progressbar = False

# %%
# Load the packaged low-resolution nickel master pattern in the stereographic
# projection.
# It only contains the upper hemisphere.
mp_stereo = kp.data.nickel_ebsd_master_pattern_small(projection="stereographic")
print(mp_stereo)

# %%
# Get a mask of pixels inside the equator of the stereographic projection.
size = mp_stereo.axes_manager.signal_shape[0]
x = np.linspace(-1, 1, size)
xx, yy = np.meshgrid(x, x)
is_inside_equator = xx**2 + yy**2 <= 1

# %%
# Define a helper function to prepare master patterns for plotting.
# Intensities are normalized to a mean of zero and a standard deviation of one, and
# plotted within three standard deviations of the mean.
# Pixels outside the equator in the stereographic projection are set to NaN, so that
# they are ignored in the normalization and are not plotted.
vmin, vmax = -3, 3


def normalize(
    mp: kp.signals.EBSDMasterPattern, exclude_outside_equator: bool = True
) -> kp.signals.EBSDMasterPattern:
    mp_norm = mp.deepcopy()
    mp_norm.change_dtype("float64")
    if mp.projection == "stereographic" and exclude_outside_equator:
        mp_norm.data[..., ~is_inside_equator] = np.nan
    mp_norm.normalize_intensity()
    return mp_norm


mp_stereo_norm = normalize(mp_stereo)

# %%
# Stereographic to Lambert
# ========================
#
# Project the master pattern to the square Lambert projection.
# The returned master pattern has the same shape, but a 32-bit floating point data
# type.
mp_lam = mp_stereo.as_lambert()
print(mp_lam)

# %%
# Plot both projections.
# The Lambert projection preserves areas, not angles, so bands are distorted towards
# the edges of the square.
# The equator of the stereographic projection maps to the edges of the square.
mp_lam_norm = normalize(mp_lam)

_ = hs.plot.plot_images(
    [mp_stereo_norm, mp_lam_norm],
    label=["Stereographic", "Lambert"],
    axes_decor="off",
    colorbar=None,
    vmin=vmin,
    vmax=vmax,
    tight_layout=True,
)

# %%
# Lambert to stereographic
# ========================
#
# Project the Lambert master pattern back to the stereographic projection.
# By default, pixels outside the equator are set to zero.
mp_stereo2 = mp_lam.as_stereo()
mp_stereo2_norm = normalize(mp_stereo2)

# %%
# Compare the projected master pattern to the original one.
# The difference is small, and is plotted in a narrower range than the master patterns.
# It is mostly in fine details, which are smoothed by the bilinear interpolation in
# both projections.
mp_diff = mp_stereo_norm - mp_stereo2_norm

_ = hs.plot.plot_images(
    [mp_stereo_norm, mp_stereo2_norm, mp_diff],
    cmap=["gray", "gray", "seismic"],
    label=["Stereographic", "Stereo -> Lambert -> stereo", "Difference (±0.5)"],
    axes_decor="off",
    colorbar=None,
    vmin=[vmin, vmin, -0.5],
    vmax=[vmax, vmax, 0.5],
    tight_layout=True,
)

# %%
# Including intensities outside the equator
# =========================================
#
# Pixels outside the equator of the stereographic projection map to the opposite
# hemisphere.
# We can include their intensities from the opposite hemisphere of the Lambert master
# pattern.
#
# Our master pattern only contains the upper hemisphere.
# Since the point group *m-3m* of nickel has inversion symmetry, the lower hemisphere is
# obtained from the upper one.
# For point groups without inversion symmetry, both hemispheres must be present in the
# master pattern.
mp_stereo3 = mp_lam.as_stereo(exclude_outside_equator=False)
mp_stereo3_norm = normalize(mp_stereo3, exclude_outside_equator=False)

# %%
# Plot the master pattern with the equator outlined.
# Bands continue across the equator into the opposite hemisphere.
fig, axes = plt.subplots(ncols=2, figsize=(9, 4.8), layout="constrained")
for ax, mp_norm, title in zip(
    axes,
    [mp_stereo2_norm, mp_stereo3_norm],
    ["Excluding outside equator", "Including outside equator"],
):
    ax.imshow(mp_norm.data, cmap="gray", vmin=vmin, vmax=vmax)
    ax.contour(is_inside_equator, levels=[0.5], colors="tab:orange", linewidths=2)
    ax.set_title(title)
    ax.axis("off")
