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
Rotate vectors from sample to detector reference frame
======================================================

This example shows how to transform column vectors given in the sample reference frame
to the detector reference frame using
:class:`~kikuchipy.detectors.EBSDDetector.sample_to_detector`.
"""

# %%
# Imports.
import numpy as np
import orix.vector as ove

import kikuchipy as kp

# %%
# Create a detector.
det = kp.detectors.EBSDDetector()
print(det)

# %%
# Get rotation and orientation matrix.
R_s2d = det.sample_to_detector
om_s2d = R_s2d.to_matrix().squeeze()

print(R_s2d)
print(om_s2d.round(4))

# %%
# We see that the detector :math:`X_d` is parallel to the sample :math:`Y_s`.
# If we change the detector :attr:`~kikuchipy.detectors.EBSDDetector.azimuthal` angle,
# this would not be the case.
# We can check the effect of various other angles easily.
print(
    kp.detectors.EBSDDetector(azimuthal=5)
    .sample_to_detector.to_matrix()
    .squeeze()
    .round(4)
)

# %%
# Create some vectors given in the sample reference frame.
v_s = ove.Vector3d.random(4)
print(v_s)

# %%
# Rotate to the detector using two paths:
# * Use the returned :class:`~orix.quaternion.Rotation` directly
# * Via NumPy's matrix multiplication sign '@' (making sure to transpose the underlying
#   vectors data to make the column vectors)
# * Via NumPy's :func:`~numpy.dot` function (making sure to transpose the underlying
#   vectors data to make the column vectors)
v_d = R_s2d * v_s
v_d_np1 = (om_s2d @ v_s.data.T).T
v_d_np2 = np.dot(om_s2d, v_s.data.T).T

print(v_d.data.round(4))
print(v_d_np1.round(4))
print(v_d_np2.round(4))

# Check equality
print(np.allclose(v_d.data, v_d_np1, atol=1e-4))
print(np.allclose(v_d.data, v_d_np2, atol=1e-4))

# %%
# Rotate from the detector to the sample.
R_d2s = ~R_s2d
om_d2s = om_s2d.T

v_s2 = R_d2s * v_d
v_s2_np1 = (om_d2s @ v_d.data.T).T
v_s2_np2 = np.dot(om_d2s, v_d.data.T).T

print(v_s2.data.round(4))
print(v_s2_np1.round(4))
print(v_s2_np2.round(4))

# Check equality
print(np.allclose(v_s2.data, v_s2_np1, atol=1e-4))
print(np.allclose(v_s2.data, v_s2_np2, atol=1e-4))
