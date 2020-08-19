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

from __future__ import annotations  # Allows -> ReciprocalLatticePoint (v4.0)
from collections import defaultdict
from itertools import product
from typing import Tuple, Union

from diffpy.structure import Lattice
import numpy as np
from orix.crystal_map import Phase
from orix.symmetry import Symmetry
from orix.vector import Vector3d

from kikuchipy.diffraction.structure_factor import get_xray_structure_factor


class ReciprocalLatticePoint:
    """Reciprocal lattice points (reflectors) g with Miller indices,
    length of the reciprocal lattice vectors and other relevant
    diffraction parameters.
    """

    def __init__(
        self, phase: Phase, hkl: Union[np.ndarray, list, tuple, Vector3d],
    ):
        """A container for Miller indices, structure factors and related
        parameters for reciprocal lattice points (reflectors) g.

        Parameters
        ----------
        phase
            A phase container with a crystal structure and a point group
            describing allowed symmetry operations.
        hkl
            Miller indices.
        """
        self._hkl = Vector3d(hkl)
        self.phase = phase
        self._structure_factor = [None] * self.size

    def __repr__(self):
        return (
            f"{self.__class__.__name__} {self.hkl.shape}\n"
            f"Phase: {self.phase.name} ({self.phase.point_group.name})\n"
            f"{np.array_str(self.hkl.data, precision=4, suppress_small=True)}"
        )

    def __getitem__(self, key) -> ReciprocalLatticePoint:
        new_rlp = self.__class__(self.phase, self.hkl[key])
        new_rlp._structure_factor = self.structure_factor[key]
        return new_rlp

    @property
    def hkl(self) -> Vector3d:
        return Vector3d(self._hkl.data.astype(int))

    @property
    def _hkldata(self) -> np.ndarray:
        return np.squeeze(self.hkl.data)

    @property
    def size(self) -> int:
        return self.hkl.size

    @property
    def shape(self) -> tuple:
        return self._hkldata.shape

    @property
    def multiplicity(self) -> Union[int, np.ndarray]:
        return self.symmetrise(antipodal=True, return_multiplicity=True)[1]

    @property
    def gspacing(self) -> np.ndarray:
        return self.phase.structure.lattice.rnorm(self._hkldata)

    @property
    def dspacing(self) -> np.ndarray:
        return 1 / self.gspacing

    @property
    def scattering_parameter(self) -> np.ndarray:
        """The scattering parameter s."""
        return 0.5 * self.gspacing

    @property
    def structure_factor(self):
        """The structure factor F."""
        return self._structure_factor

    @classmethod
    def from_min_dspacing(
        cls, phase: Phase, min_dspacing: Union[int, float] = 0.5,
    ) -> ReciprocalLatticePoint:
        """Creates a ReciprocalLatticePoint object populated by unique
        Miller indices with a direct space interplanar spacing greater
        than a lower threshold.

        Parameters
        ----------
        phase
            A phase container with a crystal structure and a point group
            describing the allowed symmetry operations.
        min_dspacing
            Smallest interplanar spacing to consider. Default is 0.5 Å.
        """
        highest_hkl = get_highest_hkl(
            lattice=phase.structure.lattice, min_dspacing=min_dspacing
        )
        hkl = get_hkl(highest_hkl=highest_hkl)
        return cls(phase=phase, hkl=hkl).unique()

    @classmethod
    def from_highest_hkl(
        cls, phase: Phase, highest_hkl: Union[np.ndarray, list, tuple],
    ) -> ReciprocalLatticePoint:
        hkl = get_hkl(highest_hkl=highest_hkl)
        return cls(phase=phase, hkl=hkl).unique()

    @classmethod
    def from_nfamilies(
        cls, phase: Phase, nfamilies: int = 5,
    ) -> ReciprocalLatticePoint:
        raise NotImplementedError

    def calculate_structure_factor(self, method: str = None):
        """Populate `self.structure_factor` with the structure factor F
        for each point.

        Parameters
        ----------
        method
            Either "xray" for the X-ray structure factor or
            "doyleturner" for the structure factor using Doyle-Turner
            atomic scattering factors.
        """
        structure_factors = np.zeros(self.size)
        hkls = self._hkldata
        scattering_parameters = self.scattering_parameter
        for i, (hkl, s) in enumerate(zip(hkls, scattering_parameters)):
            structure_factors[i] = get_xray_structure_factor(
                phase=self.phase, hkl=hkl, scattering_parameter=s
            )
        self._structure_factor = structure_factors

    def unique(self, use_symmetry: bool = True) -> ReciprocalLatticePoint:
        if use_symmetry:
            all_hkl = self._hkldata
            all_hkl = all_hkl[~np.all(np.isclose(all_hkl, 0), axis=1)]
            families = defaultdict(list)
            for this_hkl in all_hkl.tolist():
                for that_hkl in families.keys():
                    if is_equivalent(this_hkl, that_hkl):
                        families[tuple(that_hkl)].append(this_hkl)
                        break
                else:
                    families[tuple(this_hkl)].append(this_hkl)

            n_families = len(families)
            unique_hkl = np.zeros((n_families, 3))
            for i, all_hkl_in_family in enumerate(families.values()):
                unique_hkl[i] = sorted(all_hkl_in_family)[-1]
        else:
            unique_hkl = self.hkl.unique()
        return self.__class__(phase=self.phase, hkl=unique_hkl)

    def symmetrise(
        self,
        antipodal: bool = True,
        unique: bool = True,
        return_multiplicity: bool = False,
    ) -> Union[
        ReciprocalLatticePoint,
        Tuple[ReciprocalLatticePoint, Union[int, np.ndarray]],
    ]:
        """Return a new object with symmetrically equivalent Miller
        indices.

        Parameters
        ----------
        antipodal
            Whether to include antipodal symmetry operations. Default is
            True.
        unique
            Whether to return only distinct indices. Default is True.
            If true, zero entries which are assumed to be degenerate are
            removed.
        return_multiplicity
            Whether to return the multiplicity of the indices. This
            option is only available if `unique` is True. Default is
            False.

        Returns
        -------
        ReciprocalLatticePoint
            A new object with symmetrically equivalent Miller indices.
        multiplicity : np.ndarray
            Multiplicity of each original Miller indices. Only returned
            if `return_multiplicity` is True.

        Notes
        -----
        Should be the same as EMsoft's CalcFamily in their symmetry.f90
        module.
        """
        # Get symmetry operations
        pg = self.phase.point_group
        operations = pg[~pg.improper] if not antipodal else pg

        out = get_equivalent_hkl(
            hkl=self.hkl,
            operations=operations,
            unique=unique,
            return_multiplicity=return_multiplicity,
        )

        # Format output and return
        if unique and return_multiplicity:
            multiplicity = out[1]
            if multiplicity.size == 1:
                multiplicity = multiplicity[0]
            return self.__class__(phase=self.phase, hkl=out[0]), multiplicity
        else:
            return self.__class__(phase=self.phase, hkl=out)


def get_highest_hkl(lattice: Lattice, min_dspacing: float = 0.5) -> np.ndarray:
    """Return the highest Miller indices hkl of the reciprocal
    lattice point with a direct space interplanar spacing greater
    than but closest to a lower threshold.

    Parameters
    ----------
    lattice
        Crystal structure lattice.
    min_dspacing
        Smallest interplanar spacing to consider. Default is 0.5 Å.

    Returns
    -------
    highest_hkl
        Highest Miller indices.
    """
    highest_hkl = np.ones(3, dtype=int)
    for i in range(3):
        hkl = np.zeros(3)
        d = min_dspacing + 1
        while d > min_dspacing:
            hkl[i] += 1
            d = 1 / lattice.rnorm(hkl)
        highest_hkl[i] = hkl[i]
    return highest_hkl


def get_hkl(highest_hkl: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """Get a list of reciprocal lattice points from some highest Miller
    indices.

    Parameters
    ----------
    highest_hkl
        Highest Miller indices to consider.

    Returns
    -------
    hkl
        An array of reciprocal lattice points.
    """
    index_ranges = [np.arange(-i, i + 1) for i in highest_hkl]
    return np.asarray(list(product(*index_ranges)))


def get_equivalent_hkl(
    hkl: Vector3d,
    operations: Symmetry,
    unique: bool = False,
    return_multiplicity: bool = False,
) -> Union[Vector3d, Tuple[Vector3d, np.ndarray]]:
    new_hkl = operations.outer(hkl)
    new_hkl = new_hkl.flatten().reshape(*new_hkl.shape[::-1])

    multiplicity = None
    if unique:
        n_families = new_hkl.shape[0]
        multiplicity = np.zeros(n_families, dtype=int)
        temp_hkl = new_hkl[0].unique().data
        multiplicity[0] = temp_hkl.shape[0]
        if n_families > 1:
            for i, hkl in enumerate(new_hkl[1:]):
                temp_hkl2 = hkl.unique()
                multiplicity[i + 1] = temp_hkl2.size
                temp_hkl = np.append(temp_hkl, temp_hkl2.data, axis=0)
        new_hkl = Vector3d(temp_hkl[: multiplicity.sum()])

    # Remove 1-dimensions
    new_hkl = new_hkl.squeeze()

    if unique and return_multiplicity:
        return new_hkl, multiplicity
    else:
        return new_hkl


def is_equivalent(this_hkl: list, that_hkl: list) -> bool:
    return sorted(np.abs(this_hkl)) == sorted(np.abs(that_hkl))
