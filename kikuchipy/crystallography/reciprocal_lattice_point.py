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

from collections import defaultdict
from itertools import product

import numpy as np
from orix.vector import Vector3d

from kikuchipy.crystallography.structure_factor import (
    get_kinematical_structure_factor,
    get_doyleturner_structure_factor,
    get_refraction_corrected_wavelength,
)


_FLOAT_EPS = np.finfo(np.float).eps  # Used to round values below 1e-16 to zero


class ReciprocalLatticePoint:
    """Reciprocal lattice point (or crystal plane, reflector, g, etc.)
    with Miller indices, length of the reciprocal lattice vectors and
    other relevant diffraction parameters.
    """

    def __init__(self, phase, hkl):
        """A container for Miller indices, structure factors and related
        parameters for crystal planes (reciprocal lattice points,
        reflectors, g, etc.).

        Parameters
        ----------
        phase : orix.crystal_map.phase_list.Phase
            A phase container with a crystal structure and a space and
            point group describing the allowed symmetry operations.
        hkl : orix.vector.Vector3d, np.ndarray, list, or tuple
            Miller indices.
        """
        self._hkl = Vector3d(hkl)
        self.phase = phase
        self._structure_factor = [None] * self.size
        self._theta = [None] * self.size

    def __repr__(self):
        return (
            f"{self.__class__.__name__} {self.hkl.shape}\n"
            f"Phase: {self.phase.name} ({self.phase.point_group.name})\n"
            f"{np.array_str(self.hkl.data, precision=4, suppress_small=True)}"
        )

    def __getitem__(self, key):
        new_cp = self.__class__(self.phase, self.hkl[key])
        if self.structure_factor[0] is None:
            new_cp._structure_factor = [None] * new_cp.size
        else:
            new_cp._structure_factor = self.structure_factor[key]
        return new_cp

    @property
    def hkl(self):
        """Return :class:`~orix.vector.Vector3d` of Miller indices."""
        return Vector3d(self._hkl.data.astype(int))

    @property
    def _hkldata(self):
        """Return :class:`np.ndarray` without 1-dimensions."""
        return np.squeeze(self.hkl.data)

    @property
    def h(self):
        """Return :class:`np.ndarray` of Miller index h."""
        return self._hkldata[..., 0]

    @property
    def k(self):
        """Return :class:`np.ndarray` of Miller index k."""
        return self._hkldata[..., 1]

    @property
    def l(self):
        """Return :class:`np.ndarray` of Miller index l."""
        return self._hkldata[..., 2]

    @property
    def size(self):
        """Return `int`."""
        return self.hkl.size

    @property
    def shape(self):
        """Return `tuple`."""
        return self._hkldata.shape

    @property
    def multiplicity(self):
        """Return either `int` or :class:`np.ndarray` of `int`."""
        return self.symmetrise(antipodal=True, return_multiplicity=True)[1]

    @property
    def gspacing(self):
        """Return :class:`np.ndarray` of reciprocal lattice point
        spacings.
        """
        return self.phase.structure.lattice.rnorm(self._hkldata)

    @property
    def dspacing(self):
        """Return :class:`np.ndarray` of direct lattice interplanar
        spacings.
        """
        return 1 / self.gspacing

    @property
    def scattering_parameter(self):
        """Return :class:`np.ndarray` of scattering parameters s."""
        return 0.5 * self.gspacing

    @property
    def structure_factor(self):
        """Return :class:`np.ndarray` of structure factors F or None."""
        return self._structure_factor

    @property
    def allowed(self):
        """Return whether planes diffract according to diffraction
        selection rules assuming kinematical scattering theory.
        """
        # Translational symmetry
        centering = self.phase.space_group.short_name[0]

        if centering == "P":  # Primitive
            if self.phase.space_group.crystal_system == "HEXAGONAL":
                # TODO: See rules in e.g.
                #  https://mcl1.ncifcrf.gov/dauter_pubs/284.pdf, Table 4
                #  http://xrayweb.chem.ou.edu/notes/symmetry.html, Systematic Absences
                raise NotImplementedError
            else:  # Any hkl
                return np.ones(self.size, dtype=bool)
        elif centering == "F":  # Face-centred, hkl all odd/even
            selection = np.sum(np.mod(self._hkldata, 2), axis=1)
            return np.array([i not in [1, 2] for i in selection], dtype=bool)
        elif centering == "I":  # Body-centred, h + k + l = 2n (even)
            return np.mod(np.sum(self._hkldata, axis=1), 2) == 0
        elif centering == "A":  # Centred on A faces only
            return np.mod(self.k + self.l, 2) == 0
        elif centering == "B":  # Centred on B faces only
            return np.mod(self.h + self.l, 2) == 0
        elif centering == "C":  # Centred on C faces only
            return np.mod(self.h + self.k, 2) == 0
        elif centering == "R":  # Rhombohedral
            return np.mod(-self.h + self.k + self.l, 3) == 0

    @property
    def theta(self):
        """Return :class:`np.ndarray` of twice the Bragg angle."""
        return self._theta

    @classmethod
    def from_min_dspacing(cls, phase, min_dspacing=0.5):
        """Create a CrystalPlane object populated by unique Miller indices
        with a direct space interplanar spacing greater than a lower
        threshold.

        Parameters
        ----------
        phase : orix.crystal_map.phase_list.Phase
            A phase container with a crystal structure and a space and
            point group describing the allowed symmetry operations.
        min_dspacing : float, optional
            Smallest interplanar spacing to consider. Default is 0.5 Å.
        """
        highest_hkl = get_highest_hkl(
            lattice=phase.structure.lattice, min_dspacing=min_dspacing
        )
        hkl = get_hkl(highest_hkl=highest_hkl)
        return cls(phase=phase, hkl=hkl).unique()

    @classmethod
    def from_highest_hkl(cls, phase, highest_hkl):
        """Create a CrystalPlane object populated by unique Miller indices
        below, but including, a set of higher indices.

        Parameters
        ----------
        phase : orix.crystal_map.phase_list.Phase
            A phase container with a crystal structure and a space and
            point group describing the allowed symmetry operations.
        highest_hkl : np.ndarray, list, or tuple of int
            Highest Miller indices to consider (including).
        """
        hkl = get_hkl(highest_hkl=highest_hkl)
        return cls(phase=phase, hkl=hkl).unique()

    def unique(self, use_symmetry=True):
        """Return planes with unique Miller indices.

        Parameters
        ----------
        use_symmetry : bool, optional
            Whether to use symmetry to remove the planes with indices
            symmetrically equivalent to another set of indices.

        Returns
        -------
        ReciprocalLatticePoint
        """
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
        # TODO: Enable inheriting classes pass on their properties in this new object
        return self.__class__(phase=self.phase, hkl=unique_hkl)

    def symmetrise(
        self, antipodal=True, unique=True, return_multiplicity=False,
    ):
        """Return planes with symmetrically equivalent Miller indices.

        Parameters
        ----------
        antipodal : bool, optional
            Whether to include antipodal symmetry operations. Default is
            True.
        unique : bool, optional
            Whether to return only distinct indices. Default is True.
            If true, zero entries which are assumed to be degenerate are
            removed.
        return_multiplicity : bool, optional
            Whether to return the multiplicity of indices. This option is
            only available if `unique` is True. Default is False.

        Returns
        -------
        ReciprocalLatticePoint
            Planes with Miller indices symmetrically equivalent to the
            original planes.
        multiplicity : np.ndarray
            Multiplicity of the original Miller indices. Only returned if
            `return_multiplicity` is True.

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

        # TODO: Enable inheriting classes pass on their properties in this new object
        # Format output and return
        if unique and return_multiplicity:
            multiplicity = out[1]
            if multiplicity.size == 1:
                multiplicity = multiplicity[0]
            return self.__class__(phase=self.phase, hkl=out[0]), multiplicity
        else:
            return self.__class__(phase=self.phase, hkl=out)

    def calculate_structure_factor(self, method=None, voltage=None):
        """Populate `self.structure_factor` with the structure factor F
        for each plane.

        Parameters
        ----------
        method : str, optional
            Either "kinematical" for kinematical X-ray structure factors
            or "doyleturner" for structure factors using Doyle-Turner
            atomic scattering factors. If None (default), kinematical
            structure factors are calculated.
        voltage : float, optional
            Beam energy in V used when `method=doyleturner`.
        """
        if method is None:
            method = "kinematical"
        methods = ["kinematical", "doyleturner"]
        if method not in methods:
            raise ValueError(f"method={method} must be among {methods}")
        elif method == "doyleturner" and voltage is None:
            raise ValueError(
                "'voltage' parameter must set when method='doyleturner'"
            )

        structure_factors = np.zeros(self.size)
        hkls = self._hkldata
        scattering_parameters = self.scattering_parameter
        phase = self.phase
        # TODO: Find a better way to call different methods in the loop
        for i, (hkl, s) in enumerate(zip(hkls, scattering_parameters)):
            if method == "kinematical":
                structure_factors[i] = get_kinematical_structure_factor(
                    phase=phase, hkl=hkl, scattering_parameter=s,
                )
            else:
                structure_factors[i] = get_doyleturner_structure_factor(
                    phase=phase,
                    hkl=hkl,
                    scattering_parameter=s,
                    voltage=voltage,
                )
        self._structure_factor = np.where(
            structure_factors < _FLOAT_EPS, 0, structure_factors
        )

    def calculate_theta(self, voltage):
        """Populate `self.theta` with the Bragg angle :math:`theta_B` for
        each plane.

        Parameters
        ----------
        voltage : float
            Beam energy in V.
        """
        wavelength = get_refraction_corrected_wavelength(self.phase, voltage)
        self._theta = np.arcsin(0.5 * wavelength * self.gspacing)


def get_highest_hkl(lattice, min_dspacing=0.5):
    """Return the highest Miller indices hkl of the plane with a direct
    space interplanar spacing greater than but closest to a lower
    threshold.

    Parameters
    ----------
    lattice : diffpy.structure.Lattice
        Crystal lattice.
    min_dspacing : float, optional
        Smallest interplanar spacing to consider. Default is 0.5 Å.

    Returns
    -------
    highest_hkl : np.ndarray
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


def get_hkl(highest_hkl):
    """Return a list of planes from a set of highest Miller indices.

    Parameters
    ----------
    highest_hkl : orix.vector.Vector3d, np.ndarray, list, or tuple of int
        Highest Miller indices to consider.

    Returns
    -------
    hkl : np.ndarray
        An array of Miller indices.
    """
    index_ranges = [np.arange(-i, i + 1) for i in highest_hkl]
    return np.asarray(list(product(*index_ranges)))


def get_equivalent_hkl(
    hkl, operations, unique=False, return_multiplicity=False
):
    """Return symmetrically equivalent Miller indices.

    Parameters
    ----------
    hkl : orix.vector.Vector3d, np.ndarray, list or tuple of int
        Miller indices.
    operations : orix.quaternion.symmetry.Symmetry
        Point group describing allowed symmetry operations.
    unique : bool, optional
        Whether to return only unique Miller indices. Default is False.
    return_multiplicity : bool, optional
        Whether to return the multiplicity of the input indices. Default
        is False.

    Returns
    -------
    new_hkl : orix.vector.Vector3d
        The symmetrically equivalent Miller indices.
    multiplicity : np.ndarray
        Number of symmetrically equivalent indices. Only returned if
        `return_multiplicity` is True.
    """
    new_hkl = operations.outer(Vector3d(hkl))
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
