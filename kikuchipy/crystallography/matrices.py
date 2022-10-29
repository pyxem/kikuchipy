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

"""Crystallographic matrices not computed in diffpy.structure."""

from diffpy.structure import Lattice
import numpy as np

from kikuchipy._util import deprecated


@deprecated(
    since="0.6",
    alternative="diffpy.structure.Lattice.base",
    alternative_is_function=False,
    removal="0.7",
)
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
