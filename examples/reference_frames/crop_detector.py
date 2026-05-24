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
Crop an EBSD detector
=====================

This example shows how to crop an :class:`~kikuchipy.detectors.EBSDDetector`.
"""

# %%
# Imports.
import matplotlib.pyplot as plt

import kikuchipy as kp

# %%
# Create an EBSD detector.
det = kp.detectors.EBSDDetector((150, 200), pc=[0.6, 0.5, 0.5])
print(det)

det.plot(coordinates="gnomonic", draw_gnomonic_circles=True)

# %%
# Crop away the upper 30 pixels and notice how the PC is adjusted accordingly.
det2 = det.crop(extent=(30, 150, 0, 200))
print(det2)

det2.plot(coordinates="gnomonic", draw_gnomonic_circles=True)

# %%
# Plot a cropped detector with the PC on a cropped pattern.
s = kp.data.nickel_ebsd_small()
print(s)

s.remove_static_background(show_progressbar=False)

det3 = s.detector
det3.plot(pattern=s.inav[0, 0].data)

det4 = det3.crop((0, 50, 0, 50))
det4.plot(pattern=s.inav[0, 0].data[:50, :50])

plt.show()
