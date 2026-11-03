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

"""Orientation of the EBSD detector from the sample tilt and the
detector tilt, azimuthal, and twist angles.

Matrices returned here are passive: their rows are the basis vectors
of one reference frame expressed in another. Three frames are used:

- The microscope frame, with Z up.
- The sample frame, which is tilted by the sample tilt about the
  microscope X.
- The gnomonic frame (X_g, Y_g, Z_g) = (X_d, -Y_d, -Z_d), where X_d is
  right, Y_d down, and Z_d toward the sample in the detector frame
  (CS_d) shown in the reference frames tutorial.

The detector tilt, azimuthal, and twist are intrinsic, right-handed
rotations about X_d, Y_d, and Z_d, applied in that order. In the
gnomonic frame, the azimuthal and twist angles are therefore negated.
"""

import numpy as np
import orix.quaternion as oqu

# Passive microscope-to-gnomonic matrix of the default detector (tilt,
# azimuthal, and twist all zero), with Bunge Euler angles (0, 90, 0)
_DEFAULT_DETECTOR_MATRIX = np.array(
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ],
    dtype=np.float64,
)


def _rotation_matrix_x(angle: float) -> np.ndarray:
    """Return the active, right-handed rotation matrix about X by
    *angle* in radians.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_matrix_y(angle: float) -> np.ndarray:
    """Return the active, right-handed rotation matrix about Y by
    *angle* in radians.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotation_matrix_z(angle: float) -> np.ndarray:
    """Return the active, right-handed rotation matrix about Z by
    *angle* in radians.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _microscope_to_gnomonic_matrix(
    theta: float, omega: float, gamma: float
) -> np.ndarray:
    """Return the passive microscope-to-gnomonic matrix.

    Rows of the returned matrix are the gnomonic basis vectors
    (X_g, Y_g, Z_g) expressed in microscope coordinates, so that
    ``G @ v_microscope = v_gnomonic`` for column vectors.

    Parameters
    ----------
    theta
        Detector tilt in radians.
    omega
        Detector azimuthal angle in radians.
    gamma
        Detector twist in radians.
    """
    # Active rotation of the detector in the gnomonic frame of the
    # default detector. The rotations are intrinsic, so the matrices are
    # multiplied in the order the rotations are applied.
    rot = (
        _rotation_matrix_x(theta)
        @ _rotation_matrix_y(-omega)
        @ _rotation_matrix_z(-gamma)
    )
    return rot.T @ _DEFAULT_DETECTOR_MATRIX


def _sample_to_microscope_matrix(sigma: float) -> np.ndarray:
    """Return the passive sample-to-microscope matrix.

    Rows of the returned matrix are the microscope basis vectors
    expressed in sample coordinates, so that
    ``S @ v_sample = v_microscope`` for column vectors.

    Parameters
    ----------
    sigma
        Sample tilt in radians.
    """
    rot = _rotation_matrix_z(np.pi / 2) @ _rotation_matrix_x(-sigma)
    return rot.T


def _sample_to_detector_matrix(
    sigma: float, theta: float, omega: float, gamma: float
) -> np.ndarray:
    """Return the passive sample-to-gnomonic matrix.

    Rows of the returned matrix are the gnomonic basis vectors
    (X_g, Y_g, Z_g) expressed in sample coordinates, so that
    ``M @ v_sample = v_gnomonic`` for column vectors.

    Parameters
    ----------
    sigma
        Sample tilt in radians.
    theta
        Detector tilt in radians.
    omega
        Detector azimuthal angle in radians.
    gamma
        Detector twist in radians.
    """
    g = _microscope_to_gnomonic_matrix(theta, omega, gamma)
    s = _sample_to_microscope_matrix(sigma)
    return g @ s


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    """Wrap angle(s) in degrees to the range (-180, 180]."""
    return -((180 - np.asarray(angle, dtype=np.float64)) % 360 - 180)


def _detector_euler_from_angles(
    tilt: float, azimuthal: float, twist: float
) -> np.ndarray:
    """Return the Bunge Euler angles of the detector orientation
    relative to the microscope from the detector tilt, azimuthal, and
    twist angles, all in degrees.

    The first and third Euler angles are in the range (-180, 180].
    """
    theta, omega, gamma = np.deg2rad([tilt, azimuthal, twist])
    g = _microscope_to_gnomonic_matrix(theta, omega, gamma)
    euler = np.rad2deg(oqu.Rotation.from_matrix(g).to_euler().squeeze())
    euler[[0, 2]] = _wrap_angle(euler[[0, 2]])
    return euler


def _detector_angles_from_euler(euler: np.ndarray) -> tuple[float, float, float]:
    """Return the detector tilt, azimuthal, and twist angles from the
    Bunge Euler angles of the detector orientation relative to the
    microscope, all in degrees.

    This is the inverse of :func:`_detector_euler_from_angles`. It is
    singular only at an azimuthal angle of +/- 90 degrees.
    """
    rot = oqu.Rotation.from_euler(np.deg2rad(euler))
    g = rot.to_matrix().squeeze()
    # Rx(tilt) @ Ry(-azimuthal) @ Rz(-twist), see
    # _microscope_to_gnomonic()
    r = _DEFAULT_DETECTOR_MATRIX @ g.T
    azimuthal = -np.arcsin(np.clip(r[0, 2], -1, 1))
    tilt = np.arctan2(-r[1, 2], r[2, 2])
    twist = np.arctan2(r[0, 1], r[0, 0])
    tilt, azimuthal, twist = np.rad2deg([tilt, azimuthal, twist])
    return float(tilt), float(azimuthal), float(twist)
