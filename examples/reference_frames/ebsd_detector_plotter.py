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
=================================
Interactive EBSD detector plotter
=================================

This example shows how to set up an interactive plot showing the side and top view of
the microscope as well as the detector plane and the effects of changing detector-sample
geometry parameters as well as the crystal orientation.

We use the :class:`~kikuchipy.draw.EBSDDetectorPlotter`.

.. note::

    The plotter requires :doc:`ipywidgets <ipywidgets:index>` to be installed.

    Interactivity is only achived when run locally, not on this web page. An interactive
    Matplotlib backend must be set.
"""

# %%
# Imports.
from diffsims.crystallography import ReciprocalLatticeVector
from IPython.display import display
import matplotlib
import matplotlib.pyplot as plt

import kikuchipy as kp

matplotlib.use("ipympl")
plt.ion()

# %%
# Load a small (not large) nickel EBSD map.
s = kp.data.nickel_ebsd_large()
print(s)

# %%
# The EBSD dataset has both a crystal map and detector attached.
xmap = s.xmap
print(xmap)

det = s.detector
print(det)

# %%
# Load the builtin nickel master pattern (of low resolution).
mp = kp.data.nickel_ebsd_master_pattern_small(projection="lambert", energy=20)
print(mp)

# %%
# Set up a reflector list and pick a rotation from the crystal map.
phase = xmap.phases[0]
print(phase)

ref0 = ReciprocalLatticeVector(
    phase=phase, hkl=[[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]]
)
ref0.sanitise_phase()
ref = ref0.symmetrise().unique()

rot = s.xmap.rotations[0]

# %%
# Create a plotter and add the geometrical and dynamical simulations.
pl = kp.draw.EBSDDetectorPlotter(detector=det, rotation=rot, inplace=True)
pl.set_geometrical_simulation(reflectors=ref)
pl.set_master_pattern(mp)
print(pl)

# %%
# Show the plotter.
fig, controls = pl.show(figsize=(9, 3))

display(controls)

plt.show()
