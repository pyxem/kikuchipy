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
================================================
Project patterns from the detector to the sphere
================================================

This example shows how to project EBSD patterns from the detector onto the sphere, in
the stereographic projection, using :meth:`~kikuchipy.detectors.EBSDDetector.to_stereo`.

This requires the detector-sample geometry and the crystal orientation of each
pattern to be known.
The result is an :class:`~kikuchipy.signals.EBSDMasterPattern` in the stereographic
projection with both hemispheres.

We project patterns in four ways:

1. One pattern
2. One pattern, symmetrized
3. Multiple patterns
4. Multiple patterns, symmetrized
"""

# %%
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np

import kikuchipy as kp

hs.preferences.General.show_progressbar = False

# %%
# Load the packaged low-resolution nickel EBSD dataset.
# Its crystal map contains orientations refined against a dynamically simulated master
# pattern, and its detector contains one projection center (PC) per pattern.
s = kp.data.nickel_ebsd_small()
s.remove_static_background()
s.remove_dynamic_background()

xmap = s.xmap
det = s.detector
print(xmap)
print(det)

# %%
# Load the simulated master pattern in the stereographic projection.
# It only contains the upper hemisphere, but the lower hemisphere is identical due to
# the m-3m point group symmetry of nickel.
mp_sim = kp.data.nickel_ebsd_master_pattern_small(projection="stereographic")
size = mp_sim.axes_manager.signal_shape[0]

# %%
# Prepare the simulated master pattern for plotting.
# Pixels outside the equator are set to NaN, like in the projected master patterns.
x = np.linspace(-1, 1, size)
xx, yy = np.meshgrid(x, x)
is_inside_equator = xx**2 + yy**2 <= 1

mp_sim_plot = mp_sim.deepcopy()
mp_sim_plot.change_dtype("float64")
mp_sim_plot.data[~is_inside_equator] = np.nan

# %%
# Define helper functions to plot master patterns.
# Intensities are normalized to a mean of zero and a standard deviation of one, and
# plotted within three standard deviations of the mean.
# Pixels without intensities (NaN) are ignored in the normalization and are not
# plotted.
# This way, simulated and projected master patterns look similar, and bands can be
# followed from the simulated into the projected region.
vmin, vmax = -3, 3

mp_sim_norm = mp_sim_plot.normalize_intensity(inplace=False)


def plot_on_simulated_sphere(mp: kp.signals.EBSDMasterPattern, title: str) -> None:
    mp_norm = mp.normalize_intensity(inplace=False)

    fig, axes = plt.subplots(ncols=2, figsize=(9, 4.8), layout="constrained")
    for ax, data, hemisphere in zip(axes, mp_norm.data, ["Upper", "Lower"]):
        ax.imshow(mp_sim_norm.data, cmap="gray", vmin=vmin, vmax=vmax)

        # Pixels without intensities are NaN and thus transparent
        ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax)

        # Outline the projected region
        is_valid = np.isfinite(data)
        ax.contour(is_valid, levels=[0.5], colors="tab:orange", linewidths=2)

        ax.set_title(hemisphere)
        ax.axis("off")
    fig.suptitle(title)


# %%
# 1. One pattern
# ==============
#
# Get the first pattern.
# Slicing the signal also slices its crystal map and detector.
s0: kp.signals.EBSD = s.inav[0, 0]
print(s0.detector.pc)

# %%
# Project the pattern with its orientation, and plot it on top of the simulated master
# pattern.
# The pattern covers only a small part of the sphere.
# Bands in the projected pattern align with bands in the simulated master pattern.
mp1 = s0.detector.to_stereo(s0.data, s0.xmap.orientations, size=size)
print(mp1)

plot_on_simulated_sphere(mp1, "One pattern (orientation outside fundamental zone)")

# %%
# The pattern is placed on the side of the sphere because the grain orientation is not
# inside the orientation fundamental zone.
# Let's re-do the projection using the symmetrically equivalent orientation inside the
# fundamental zone.
mp1 = s0.detector.to_stereo(s0.data, s0.xmap.orientations.reduce(), size=size)
print(mp1)

plot_on_simulated_sphere(mp1, "One pattern")

# %%
# The returned master pattern has both hemispheres, and its phase has the point group of
# the orientation.
print(mp1.hemisphere)
print(mp1.phase)

# %%
# 2. One pattern, symmetrized
# ===========================
#
# Project the same pattern for all symmetrically equivalent orientations.
# This requires the orientation to have the crystal symmetry, here point group *m-3m*
# for nickel (space group 225 *Fm-3m*).
# A single pattern then covers almost the whole sphere.
# Pixels covered by more than one equivalent orientation get the mean intensity.
mp2 = s0.detector.to_stereo(s0.data, s0.xmap.orientations, size=size, symmetrize=True)

# %%
# Compare the upper hemisphere to the simulated master pattern.
mp2_norm = mp2.normalize_intensity(inplace=False)

_ = hs.plot.plot_images(
    [mp2_norm.inav[0], mp_sim_norm],
    label=["One pattern, symmetrized", "Simulated"],
    axes_decor="off",
    colorbar=None,
    vmin=vmin,
    vmax=vmax,
    tight_layout=True,
)

# %%
# Points in the stereographic projection outside the equator are set to NaN by default,
# because they belong to the opposite hemisphere.
# We can however include them, if we really want to.
mp22 = s0.detector.to_stereo(
    s0.data,
    s0.xmap.orientations,
    size=size,
    exclude_outside_equator=False,
    symmetrize=True,
)
print(mp22)

plot_on_simulated_sphere(mp22, "One pattern, symmetrized, including outside equator")

# %%
# Note the missing intensities at the <111> zone axes; they are missing because the
# single experimental pattern lacks the a [111] zone axis.

# %%
# 3. Multiple patterns
# ====================
#
# Project all nine patterns.
# Patterns may have any navigation shape, but the number of orientations must match the
# number of patterns.
# The detector must have either one PC or one PC per pattern.
# Intensities are averaged in pixels covered by more than one pattern.
#
# The dataset contains only two grains, so the patterns cover only two regions of the
# sphere.
mp3 = det.to_stereo(s.data, xmap.orientations.reduce(), size=size)

plot_on_simulated_sphere(mp3, "Multiple patterns")

# %%
# 4. Multiple patterns, symmetrized
# =================================
#
# Project all patterns for all symmetrically equivalent orientations.
mp4 = det.to_stereo(s.data, xmap.orientations, size=size, symmetrize=True)

# %%
# Compare the upper hemisphere to the one obtained from a single pattern.
# With more patterns, noise is averaged out and bands become clearer.
mp4_norm = mp4.normalize_intensity(inplace=False)

_ = hs.plot.plot_images(
    [mp2_norm.inav[0], mp4_norm.inav[0]],
    label=["One pattern, symmetrized", "Multiple patterns, symmetrized"],
    axes_decor="off",
    colorbar=None,
    vmin=vmin,
    vmax=vmax,
    tight_layout=True,
)

# %%
# Compare how much of the sphere each master pattern covers, i.e. the fraction
# of pixels inside the equator with intensities.
for mp, label in zip([mp1, mp2, mp3, mp4], ["1", "2", "3", "4"]):
    coverage = np.isfinite(mp.data[:, is_inside_equator]).mean()
    print(f"{label}. Sphere coverage: {coverage:.1%}")

# %%
# Comparison to simulation
# ========================
#
# Compare the two symmetrized master patterns to the simulated master pattern by the
# normalized cross-correlation (NCC), calculated over both hemispheres with the
# :class:`~kikuchipy.indexing.NormalizedCrossCorrelationMetric`.
# Only pixels with intensities in all three master patterns are compared, by excluding
# the other pixels with a signal mask.
# A higher NCC means that the projected master pattern is more similar to the simulated
# one.
#
# The lower hemisphere of the simulated master pattern is identical to the upper one.
sim = np.stack([mp_sim_plot.data] * 2)
projected = np.stack([mp2.data, mp4.data])

is_valid = np.isfinite(sim) & np.isfinite(mp2.data) & np.isfinite(mp4.data)

metric = kp.indexing.NormalizedCrossCorrelationMetric(
    n_experimental_patterns=2,
    n_dictionary_patterns=1,
    signal_mask=~is_valid,
    dtype="float64",
)
ncc = metric(projected, sim).compute()

for score, label in zip(ncc[:, 0], ["2. One pattern", "4. Multiple patterns"]):
    print(f"{label}, symmetrized: NCC = {score:.3f}")

# %%
# The master pattern from all nine patterns is more similar to the simulated master
# pattern than the one from a single pattern, since noise is averaged out.

# sphinx_gallery_thumbnail_number = 3
