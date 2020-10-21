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

"""Crystallographic matrices not computed in diffpy.structure."""

from diffpy.structure import Lattice
import numpy as np

# TODO: Implement these in diffpy.structure (see
#  https://github.com/diffpy/diffpy.structure/issues/46)


def get_direct_structure_matrix(lattice: Lattice) -> np.ndarray:
    """Direct structure matrix as defined in EMsoft.

    Parameters
    ----------
    lattice
        Crystal structure lattice.

    Returns
    -------
    """
    a, b, c = lattice.abcABG()[:3]
    ca, cb, cg = lattice.ca, lattice.cb, lattice.cg
    sa, sb, sg = lattice.sa, lattice.sb, lattice.sg
    return np.array(
        [
            [a, b * cg, c * cb],
            [0, b * sg, -c * (cb * cg - ca) / sg],
            [0, 0, lattice.volume / (a * b * sg)],
        ]
    )


def get_reciprocal_structure_matrix(lattice: Lattice) -> np.ndarray:
    """Reciprocal structure matrix as defined in EMsoft.

    Parameters
    ----------
    lattice
        Crystal structure lattice.

    Returns
    -------
    """
    return np.linalg.inv(get_direct_structure_matrix(lattice)).T


def get_reciprocal_metric_tensor(lattice: Lattice) -> np.ndarray:
    """Reciprocal metric tensor as defined in EMsoft.

    Parameters
    ----------
    lattice
        Crystal structure lattice.

    Returns
    -------
    """
    a, b, c = lattice.abcABG()[:3]
    ca, cb, cg = lattice.ca, lattice.cb, lattice.cg
    sa, sb, sg = lattice.sa, lattice.sb, lattice.sg
    terms_12_21 = a * b * (c ** 2) * (ca * cb - cg)
    terms_13_31 = a * (b ** 2) * c * (cg * ca - cb)
    terms_23_32 = (a ** 2) * b * c * (cb * cg - ca)
    return np.array(
        [
            [(b * c * sa) ** 2, terms_12_21, terms_13_31],
            [terms_12_21, (a * c * sb) ** 2, terms_23_32],
            [terms_13_31, terms_23_32, (a * b * sg) ** 2],
        ]
    ) / np.linalg.det(lattice.metrics)
