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
==============================
Detector-sample geometry views
==============================

This example shows how to plot multiple views of the detector-sample geometry in a
single figure using the :class:`~kikuchipy.detectors.EBSDDetector`.
Both a side view and top view are available.

These can be useful when demonstrating variations in geometry parameters in a single
figure.

For a more in-depth tutorial on the various reference frames in kikuchipy, see the
:doc:`reference frames tutorial </tutorials/reference_frames>`.

.. note::

    The side view shows changes to the following geometry parameters:

    - :attr:`~kikuchipy.detectors.EBSDDetector.tilt`
    - :attr:`~kikuchipy.detectors.EBSDDetector.sample_tilt`
    - :attr:`~kikuchipy.detectors.EBSDDetector.pcy`
    - :attr:`~kikuchipy.detectors.EBSDDetector.pcz`

    The top view shows changes to the following geometry parameters:

    - :attr:`~kikuchipy.detectors.EBSDDetector.sample_tilt`
    - :attr:`~kikuchipy.detectors.EBSDDetector.tilt`
    - :attr:`~kikuchipy.detectors.EBSDDetector.azimuthal`
    - :attr:`~kikuchipy.detectors.EBSDDetector.pcx`
    - :attr:`~kikuchipy.detectors.EBSDDetector.pcy`
    - :attr:`~kikuchipy.detectors.EBSDDetector.pcz`

    None of these views show changes in the detector
    :attr:`~kikuchipy.detectors.EBSDDetector.twist` angle.

.. note::

    Use :class:`~kikuchipy.draw.EBSDDetectorPlotter` for a plot where all
    detector-sample geometry parameters can be changed interactively.
"""

# %%
# Imports and some reusable functions.

import matplotlib.pyplot as plt

import kikuchipy as kp


def tilt_string(detector: kp.detectors.EBSDDetector, fmt: str = "short") -> str:
    if fmt == "long":
        return (
            rf"Sample tilt $\sigma = {detector.sample_tilt:.1f}$"
            "\N{DEGREE SIGN}\n"
            rf"Detector tilt $\theta = {detector.tilt:.1f}$"
            "\N{DEGREE SIGN}"
        )
    else:
        return (
            r"($\sigma$, $\theta$) = "
            f"({detector.sample_tilt:.1f}"
            "\N{DEGREE SIGN}, "
            f"{detector.tilt:.1f}"
            "\N{DEGREE SIGN})"
        )


def pc_string(detector: kp.detectors.EBSDDetector) -> str:
    pcx, pcy, pcz = detector.pc_average.round(2)
    return f"(PCx, PCy, PCz) = ({pcx}, {pcy}, {pcz})"


# %%
# Get the default detector with a rectangular shape.
shape = (75, 100)
det1 = kp.detectors.EBSDDetector(shape)
print(det1)

# %%
# And view the detector-sample geometry.
fig = det1.plot_side_view(legend=True, return_figure=True)
_ = fig.suptitle(f"Default detector-sample geometry\n{tilt_string(det1, fmt='long')}")

# %%
# Changing sample tilt
# ====================

det2 = kp.detectors.EBSDDetector(shape)

fig = plt.figure(figsize=(9, 6), layout="constrained")
for i, stilt in enumerate([0, 45, 90]):
    det2.sample_tilt = stilt

    ax1 = fig.add_subplot(2, 3, i + 1)
    ax1.set_title("Side view\n" + tilt_string(det2))
    det2.plot_side_view(ax=ax1)

    ax2 = fig.add_subplot(2, 3, i + 4)
    ax2.set_title("Top view")
    det2.plot_top_view(ax=ax2)

# %%
# Changing detector tilt
# ======================

det3 = kp.detectors.EBSDDetector(shape)

fig = plt.figure(figsize=(9, 6), layout="constrained")
for i, tilt in enumerate([0, 45, 90]):
    det3.tilt = tilt

    ax1 = fig.add_subplot(2, 3, i + 1)
    ax1.set_title("Side view\n" + tilt_string(det3))
    det3.plot_side_view(ax=ax1)

    ax2 = fig.add_subplot(2, 3, i + 4)
    ax2.set_title("Top view")
    det3.plot_top_view(ax=ax2)

# %%
# Changing sample and detector tilts
# ==================================
#
# Let's start with the sample tilt at :math:`\sigma = 90^{\circ}` and then decrease it
# while increasing the detector tilt :math:`\theta` in steps of :math:`10^{\circ}`.

det4 = kp.detectors.EBSDDetector(shape)

sample_tilts = [90, 45, 0]
detector_tilts = sample_tilts[::-1]

fig = plt.figure(figsize=(9, 6), layout="constrained")
for i, (stilt, tilt) in enumerate(zip(sample_tilts, detector_tilts)):
    det4.sample_tilt = stilt
    det4.tilt = tilt

    ax1 = fig.add_subplot(2, 3, i + 1)
    ax1.set_title("Side view\n" + tilt_string(det4))
    det4.plot_side_view(ax=ax1)

    ax2 = fig.add_subplot(2, 3, i + 4)
    ax2.set_title("Top view")
    det4.plot_top_view(ax=ax2)

# %%
# Changing projection center
# ==========================
#
# See the documentation for the :class:`~kikuchipy.detectors.EBSDDetector` for a
# definition of the projection center (PC).

# %%
# Changing PCx
# ------------
#
# Vary PCx from :math:`0 \rightarrow 1` (left :math:`\rightarrow` right).

det5 = kp.detectors.EBSDDetector(shape)

fig = plt.figure(figsize=(9, 3), layout="constrained")
for i, pcx in enumerate([0, 0.5, 1]):
    det5.pcx = pcx

    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title("Top view\n" + pc_string(det5))
    det5.plot_top_view(ax=ax)

# %%
# Changing PCy
# ------------
#
# Vary PCy from :math:`0 \rightarrow 1` (top :math:`\rightarrow` bottom).

det6 = kp.detectors.EBSDDetector(shape)

fig = plt.figure(figsize=(9, 3), layout="constrained")
for i, pcy in enumerate([0, 0.5, 1]):
    det6.pcy = pcy

    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title("Side view\n" + pc_string(det6))
    det6.plot_side_view(ax=ax)

# %%
# Changing PCz
# ------------
#
# Vary PCz from :math:`0.2 \rightarrow 1` (increasing distance from the detector)

det7 = kp.detectors.EBSDDetector(shape)

fig = plt.figure(figsize=(9, 6), layout="constrained")
for i, pcz in enumerate([0.2, 0.6, 1]):
    det7.pcz = pcz

    ax1 = fig.add_subplot(2, 3, i + 1)
    ax1.set_title("Side view\n" + pc_string(det7))
    det7.plot_side_view(ax=ax1)

    ax2 = fig.add_subplot(2, 3, i + 4)
    ax2.set_title("Top view")
    det7.plot_top_view(ax=ax2)
