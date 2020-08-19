# -*- coding: utf-8 -*-
# Copyright 2019-2020 The kikuchipy developers
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

from diffpy.structure import Atom
from diffpy.structure.symmetryutilities import (
    expandPosition,
    SymmetryConstraints,
)
from diffpy.structure.spacegroupmod import SpaceGroup
import numpy as np
from orix.crystal_map import Phase

from kikuchipy.diffraction.atomic_scattering_parameters import (
    get_atomic_scattering_parameters,
    get_element_id_from_string,
)


def get_atomic_scattering_factor(
    atom: Atom, scattering_parameter: float,
) -> float:
    """Return the atomic scattering factor f for a certain atom and
    scattering parameter.
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


def find_asymmetric_positions(
    positions: list, spacegroup: SpaceGroup
) -> np.ndarray:
    asymmetric_positions = SymmetryConstraints(spacegroup, positions).corepos
    return [
        np.array([np.allclose(xyz, asym_xyz) for xyz in positions])
        for asym_xyz in asymmetric_positions
    ][0]


def get_xray_structure_factor(
    phase: Phase, hkl: np.ndarray, scattering_parameter: float,
) -> complex:
    """Assumes structure's lattice parameters and Debye-Waller factors
    are expressed in Angstroms.
    """
    # Initialize real and imaginary parts of the structure factor
    structure_factor = 0 + 0j

    structure = phase.structure
    space_group = phase.space_group

    # Loop over asymmetric unit
    asymmetric_positions = find_asymmetric_positions(structure.xyz, space_group)
    for is_asymmetric, atom in zip(asymmetric_positions, structure):
        if not is_asymmetric:
            continue

        # Get atomic scattering factor for this atom
        f = get_atomic_scattering_factor(atom, scattering_parameter)

        # Loop over all atoms in the orbit
        equiv_pos = expandPosition(spacegroup=space_group, xyz=atom.xyz)[0]
        for xyz in equiv_pos:
            arg = 2 * np.pi * np.sum(hkl * xyz)
            structure_factor += f * (np.cos(arg) - (np.sin(arg) * 1j))

    return structure_factor.real
