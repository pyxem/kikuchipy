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
Detector-sample geometry side-view
==================================

This example shows how to view the detector-sample geometry using the
:class:`~kikuchipy.detectors.EBSDDetector`.
This can be useful when understanding the geometry and the definition of the detector
parameters.

.. note::

    This example only shows the side-view.
    It ignores the detector :attr:`~kikuchipy.detectors.EBSDDetector.azimuthal` and
    :attr:`~kikuchipy.detectors.EBSDDetector.twist`.
"""

# %%
# Imports and some reusable functions.

import matplotlib.pyplot as plt
import numpy as np

import kikuchipy as kp


def tilt_string(detector: kp.detectors.EBSDDetector) -> str:
    return (
        rf"sample tilt $\sigma = {detector.sample_tilt:.1f}$"
        "\N{DEGREE SIGN}\n"
        rf"detector tilt $\theta = {detector.tilt:.1f}$"
        "\N{DEGREE SIGN}"
    )


def pc_string(detector: kp.detectors.EBSDDetector) -> str:
    pcx, pcy, pcz = detector.pc_average.round(2)
    return f"(PCx, PCy, PCz) = ({pcx}, {pcy}, {pcz})"


# %%
# Get the default detector.
det1 = kp.detectors.EBSDDetector()
print(det1)

# %%
# And view the detector-sample geometry.
fig = det1.plot_side_view(annotate=True, return_figure=True)
_ = fig.suptitle(f"Default detector-sample geometry\n{tilt_string(det1)}")

# %%
# Changing sample tilt
# ====================

det2 = kp.detectors.EBSDDetector()

sample_tilts = np.linspace(0, 90, num=9)
nrows = 3
ncols = int(np.ceil(sample_tilts.size / nrows))

fig = plt.figure(figsize=(3 * ncols, 3 * nrows), layout="constrained")
for i, (j, k) in enumerate(np.ndindex(nrows, ncols)):
    if i > sample_tilts.size - 1:
        break

    det2.sample_tilt = sample_tilts[i]

    ax = fig.add_subplot(nrows, ncols, i + 1)
    ax.set_title(tilt_string(det2))
    det2.plot_side_view(ax=ax)

# %%
# Changing detector tilt
# ======================

det3 = kp.detectors.EBSDDetector()

detector_tilts = np.linspace(0, 90, num=9)
nrows = 3
ncols = int(np.ceil(detector_tilts.size / nrows))

fig = plt.figure(figsize=(3 * ncols, 3 * nrows), layout="constrained")
for i, (j, k) in enumerate(np.ndindex(nrows, ncols)):
    if i > detector_tilts.size - 1:
        break

    det3.tilt = detector_tilts[i]

    ax = fig.add_subplot(nrows, ncols, i + 1)
    ax.set_title(tilt_string(det3))
    det3.plot_side_view(ax=ax)

# %%
# Changing sample and detector tilts
# ==================================
#
# Let's start with the sample tilt at :math:`\sigma = 90^{\circ}` and then decrease it
# while increasing the detector tilt :math:`\theta` in steps of :math:`10^{\circ}`.

det4 = kp.detectors.EBSDDetector()

sample_tilts = np.linspace(90, 0, num=9)
detector_tilts = np.linspace(0, 90, num=9)
nrows = 3
ncols = int(np.ceil(detector_tilts.size / nrows))

fig = plt.figure(figsize=(3 * ncols, 3 * nrows), layout="constrained")
for i, (j, k) in enumerate(np.ndindex(nrows, ncols)):
    if i > detector_tilts.size - 1:
        break

    det4.sample_tilt = sample_tilts[i]
    det4.tilt = detector_tilts[i]

    ax = fig.add_subplot(nrows, ncols, i + 1)
    ax.set_title(tilt_string(det4))
    det4.plot_side_view(ax=ax)

# %%
# Changing projection center
# ==========================
#
# See the documentation for the :class:`~kikuchipy.detectors.EBSDDetector` for a
# definition of the projection center (PC).
#
# Since we're looking at the geometry in side-view along the microscope X, which is
# parallel to the detector X (and since we keep the detector azimuthal angle
# :math:`0^{\circ}`), a change in the PCx won't show.

# %%
# Changing PCy
# ------------
#
# Vary PCy from :math:`0 \rightarrow 1` (top :math:`\rightarrow` bottom).

det5 = kp.detectors.EBSDDetector()

pcy = np.linspace(0, 1, num=9)
nrows = 3
ncols = int(np.ceil(pcy.size / nrows))

fig = plt.figure(figsize=(3 * ncols, 3 * nrows), layout="constrained")
for i, (j, k) in enumerate(np.ndindex(nrows, ncols)):
    if i > pcy.size - 1:
        break

    det5.pcy = pcy[i]

    ax = fig.add_subplot(nrows, ncols, i + 1)
    ax.set_title(pc_string(det5))
    det5.plot_side_view(ax=ax)

# %%
# Changing PCz
# ------------
#
# Vary PCz from :math:`0.2 \rightarrow 1` (increasing distance from the detector)

det6 = kp.detectors.EBSDDetector(px_size=560)

pcz = np.linspace(0.2, 1, num=9)
nrows = 3
ncols = int(np.ceil(pcz.size / nrows))

fig = plt.figure(figsize=(3 * ncols, 3 * nrows), layout="tight")
for i, (j, k) in enumerate(np.ndindex(nrows, ncols)):
    if i > pcz.size - 1:
        break

    det6.pcz = pcz[i]

    ax = fig.add_subplot(nrows, ncols, i + 1)
    ax.set_title(pc_string(det6))
    det6.plot_side_view(ax=ax)

# %%
# Interactive changes
# ===================
#
# We can also change the sample and detector tilts and the PC values interactively.
#
# .. note::
#
#   This requires that :mod:`ipywidgets` is installed and that the code is running in
#   an interactive environment with an interactive Matplotlib backend.
#
# The default behavior is not to affect the detector.
# Here, we want the changes to affect the detector inplace.

det6.plot_side_view_interactive(inplace=True)

plt.show()
