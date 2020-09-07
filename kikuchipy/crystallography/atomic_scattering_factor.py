# -*- coding: utf-8 -*-
# Copyright 2017-2020 The diffsims developers
#
# This file is part of diffsims.
#
# diffsims is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# diffsims is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with diffsims.  If not, see <http://www.gnu.org/licenses/>.

# TODO: This file will be moved to diffsims

import numpy as np

from kikuchipy.crystallography.atomic_scattering_parameters import (
    get_atomic_scattering_parameters,
    get_element_id_from_string,
)


def get_kinematical_atomic_scattering_factor(atom, scattering_parameter):
    """Return the kinematical (X-ray) atomic scattering factor f for a
    certain atom and scattering parameter.

    Assumes structure's Debye-Waller factors are expressed in Ångströms.

    Parameters
    ----------
    atom : diffpy.structure.Atom
        Atom with element type, Debye-Waller factor and occupancy number.
    scattering_parameter : float
        The scattering parameter s for these Miller indices describing
        the crystal plane in which the atom lies.

    Returns
    -------
    f : float
        Scattering factor for this atom on this plane.
    """
    # Get the atomic scattering parameters
    element_id = get_element_id_from_string(atom.element)
    a, b = get_atomic_scattering_parameters(element_id)

    # Get the scattering parameter squared
    s2 = scattering_parameter ** 2

    # Get the atomic scattering factor
    f = element_id - (41.78214 * s2 * np.sum(a * np.exp(-b * s2)))

    # Correct for occupancy and the Debye-Waller factor
    dw_factor = np.exp(-atom.Bisoequiv * s2)
    f *= atom.occupancy * dw_factor

    return f


def get_doyleturner_atomic_scattering_factor(
    atom, scattering_parameter, unit_cell_volume
):
    """Return the atomic scattering factor f for a certain atom and
    scattering parameter using Doyle-Turner atomic scattering parameters
    [Doyle1968]_.

    Assumes structure's Debye-Waller factors are expressed in Ångströms.

    Parameters
    ----------
    atom : diffpy.structure.Atom
        Atom with element type, Debye-Waller factor and occupancy number.
    scattering_parameter : float
        The scattering parameter s for these Miller indices describing
        the crystal plane in which the atom lies.
    unit_cell_volume : float
        Volume of the unit cell.

    Returns
    -------
    f : float
        Scattering factor for this atom on this plane.
    """
    # Get the atomic scattering parameters
    element_id = get_element_id_from_string(atom.element)
    a, b = get_atomic_scattering_parameters(element_id)

    # Get the scattering parameter squared
    s2 = scattering_parameter ** 2

    # Get the atomic scattering factor
    f = (47.87801 / unit_cell_volume) * np.sum(a * np.exp(-b * s2))

    # Correct for occupancy and the Debye-Waller factor
    dw_factor = np.exp(-atom.Bisoequiv * s2)
    f *= atom.occupancy * dw_factor

    return f
