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
Plot projection center on the detector
======================================

This example shows how to plot the projection/pattern center (PC) on an
:class:`~kikuchipy.detectors.EBSDDetector`.
See the detector class documentation for further details on the definition of the PC and
gnomonic coordinates.
"""

# %%
# Imports.
import matplotlib.pyplot as plt

import kikuchipy as kp

# %%
# Create an EBSD detector with a binning of 8 and a single projection/pattern (PC)
# center given in EDAX' definition,
# :math:`(x^{*}, y^{*}, z^{*}) = (0.421, 0.779, 0.505)`.
det = kp.detectors.EBSDDetector(
    shape=(60, 60),
    pc=[0.421, 0.779, 0.505],
    convention="edax",
    px_size=70,  # Microns
    tilt=5,  # Degrees
    sample_tilt=70,  # Degrees
    binning=8,
)
print(det)

# %%
# Load a small test dataset with (60, 60) patterns to plot the PC over.
s = kp.data.nickel_ebsd_small()
print(s)

# %%
# Plot the PC on top of the pattern.
# Instead of pixels, we show the detector extent in gnomonic coordinates along the x
# and y axes.
# We also draw gnomonic circles at an interval of :math:`10^{\circ}`.
det.plot(
    pattern=s.inav[0, 0].data,
    coordinates="gnomonic",
    draw_gnomonic_circles=True,
)

plt.show()
