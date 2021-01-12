# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
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

"""Rotations to align the EBSD detector with the tilted sample. Notation
from :cite:`britton2016tutorial`.
"""

from typing import Optional
from diffpy.structure import Lattice
import numpy as np
from orix.quaternion.rotation import Rotation
from orix.vector import neo_euler, Vector3d

from kikuchipy.crystallography import get_direct_structure_matrix


def detector2sample(
    sample_tilt: float, detector_tilt: float, convention: Optional[str] = None,
) -> Rotation:
    """Rotation U_S to align detector frame D with sample frame S.

    Parameters
    ----------
    sample_tilt
        Sample tilt in degrees.
    detector_tilt
        Detector tilt in degrees.
    convention
        Which sample reference frame to use, either the one used by EDAX
        TSL (default), "tsl", or the one used by Bruker, "bruker".

    Returns
    -------
    Rotation
    """
    # Rotation about sample (microscope) X axis
    tilt = -np.deg2rad((sample_tilt - 90) - detector_tilt)
    ax_angle = neo_euler.AxAngle.from_axes_angles(Vector3d.xvector(), tilt)
    r = Rotation.from_neo_euler(ax_angle)

    if convention != "bruker":
        # Followed by a 90 degree rotation about the sample Z axis,
        # if the TSL sample reference frame is used
        ax_angle_bruker2tsl = neo_euler.AxAngle.from_axes_angles(
            Vector3d.zvector(), np.pi / 2
        )
        r = Rotation.from_neo_euler(ax_angle_bruker2tsl) * r

    return r.to_matrix()[0]


def detector2direct_lattice(
    sample_tilt: float,
    detector_tilt: float,
    lattice: Lattice,
    rotation: Rotation,
) -> np.ndarray:
    """Rotation U_K from detector frame D to direct crystal lattice
    frame K.

    Parameters
    ----------
    sample_tilt
        Sample tilt in degrees.
    detector_tilt
        Detector tilt in degrees.
    lattice
        Crystal lattice.
    rotation
        Unit cell rotation from the sample frame S.

    Returns
    -------
    np.ndarray
    """
    # Rotation U_S to align the detector frame D with the sample frame S
    _detector2sample = detector2sample(sample_tilt, detector_tilt)

    # Rotation U_O from S to the Cartesian crystal frame C
    sample2cartesian = rotation.to_matrix()

    # Rotation U_A from C to the direct crystal lattice frame K
    structure_matrix = get_direct_structure_matrix(lattice)
    cartesian2direct = structure_matrix.T

    return cartesian2direct.dot(sample2cartesian).dot(_detector2sample)


def detector2reciprocal_lattice(
    sample_tilt: float,
    detector_tilt: float,
    lattice: Lattice,
    rotation: Rotation,
) -> np.ndarray:
    """Rotation U_Kstar from detector to reciprocal crystal lattice
    frame Kstar.

    Parameters
    ----------
    sample_tilt
        Sample tilt in degrees.
    detector_tilt
        Detector tilt in degrees.
    lattice
        Crystal lattice.
    rotation
        Unit cell rotation from the sample frame S.

    Returns
    -------
    np.ndarray
    """
    # Rotation U_S to align the detector frame D with the sample frame S
    _detector2sample = detector2sample(sample_tilt, detector_tilt)

    # Rotation U_O from S to the Cartesian crystal frame C
    sample2cartesian = rotation.to_matrix()

    # Rotation U_A from C to the reciprocal crystal lattice frame Kstar
    structure_matrix = get_direct_structure_matrix(lattice)
    cartesian2reciprocal = np.linalg.inv(structure_matrix).T

    return cartesian2reciprocal.dot(sample2cartesian).dot(_detector2sample)
