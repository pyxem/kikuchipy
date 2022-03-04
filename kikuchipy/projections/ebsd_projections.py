# Copyright 2019-2022 The kikuchipy developers
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

from diffpy.structure import Lattice
import numpy as np
from orix.quaternion import Rotation

from kikuchipy.crystallography.matrices import get_direct_structure_matrix2


def detector2sample(sample_tilt: float, detector_tilt: float) -> np.ndarray:
    r"""Rotation :math:`U_S` to align Bruker's detector frame :math:`D`
    with EDAX TSL's sample frame :math:`S` (RD-TD-ND).

    Parameters
    ----------
    sample_tilt
        Sample tilt in degrees.
    detector_tilt
        Detector tilt in degrees.

    Returns
    -------
    numpy.ndarray
    """
    # Rotation from Bruker's detector frame D to Bruker's sample frame
    tilt = np.deg2rad(sample_tilt - 90 - detector_tilt)
    rot = Rotation.from_axes_angles((-1, 0, 0), tilt)

    # Rotation from Bruker's sample frame to EDAX TSL's sample frame
    rot_bruker2tsl = Rotation.from_axes_angles((0, 0, 1), np.pi / 2)
    rot = rot_bruker2tsl * rot

    return rot.to_matrix()[0]


def detector2direct_lattice(
    sample_tilt: float, detector_tilt: float, lattice: Lattice, rotation: Rotation
) -> np.ndarray:
    r"""Rotation :math:`U_K` from Bruker's detector frame :math:`D` to
    direct crystal lattice frame :math:`K`.

    Parameters
    ----------
    sample_tilt
        Sample tilt in degrees.
    detector_tilt
        Detector tilt in degrees.
    lattice
        Crystal lattice.
    rotation
        Rotation from the sample frame :math:`S` to the cartesian
        crystal lattice frame :math:`C`.

    Returns
    -------
    numpy.ndarray
    """
    # Rotation U_S to align Bruker's detector frame D with EDAX TSL's
    # sample reference frame S (RD-TD-ND)
    _detector2sample = detector2sample(sample_tilt, detector_tilt)

    # Rotation U_O from S to the Cartesian crystal lattice frame C
    sample2cartesian = rotation.to_matrix()

    # Rotation U_A from C to the direct crystal lattice frame K
    structure_matrix = get_direct_structure_matrix2(lattice)
    cartesian2direct = structure_matrix

    return cartesian2direct.dot(sample2cartesian) @ _detector2sample


def detector2reciprocal_lattice(
    sample_tilt: float, detector_tilt: float, lattice: Lattice, rotation: Rotation
) -> np.ndarray:
    r"""Rotation :math:`U_{K^*}` from Bruker's detector frame :math:`D`
    to reciprocal crystal lattice frame :math:`K^*`.

    Parameters
    ----------
    sample_tilt
        Sample tilt in degrees.
    detector_tilt
        Detector tilt in degrees.
    lattice
        Crystal lattice.
    rotation
        Rotation from the sample frame :math:`S` to the cartesian
        crystal lattice frame :math:`C`.

    Returns
    -------
    numpy.ndarray
    """
    # Rotation U_S to align Bruker's detector frame D with EDAX TSL's
    # sample reference frame S (RD-TD-ND)
    _detector2sample = detector2sample(sample_tilt, detector_tilt)

    # Rotation U_O from S to the Cartesian crystal lattice frame C
    sample2cartesian = rotation.to_matrix()

    # Rotation U_A from C to the reciprocal crystal lattice frame Kstar
    structure_matrix = get_direct_structure_matrix2(lattice)
    cartesian2reciprocal = np.linalg.inv(structure_matrix)

    return cartesian2reciprocal.dot(sample2cartesian) @ _detector2sample
