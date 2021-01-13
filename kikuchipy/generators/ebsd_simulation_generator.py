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

from copy import deepcopy
from typing import Optional, Tuple

from diffsims.crystallography import ReciprocalLatticePoint
import numpy as np
from orix.crystal_map import Phase
from orix.quaternion.rotation import Rotation

from kikuchipy.crystallography._computations import _get_uvw_from_hkl
from kikuchipy.detectors import EBSDDetector
from kikuchipy.projections.ebsd_projections import (
    detector2reciprocal_lattice,
    detector2direct_lattice,
)
from kikuchipy.simulations import GeometricalEBSDSimulation
from kikuchipy.simulations.features import KikuchiBand, ZoneAxis


class EBSDSimulationGenerator:
    """A generator storing necessary parameters to simulate geometrical
    EBSD patterns.
    """

    def __init__(
        self, detector: EBSDDetector, phase: Phase, rotations: Rotation,
    ):
        """A generator storing necessary parameters to simulate
        geometrical EBSD patterns.

        Parameters
        ----------
        detector
            Detector describing the detector-sample geometry.
        phase
            A phase container with a crystal structure and a space and
            point group describing the allowed symmetry operations.
        rotations
            Unit cell rotations to simulate patterns for. The
            navigation shape of the resulting simulation is determined
            from the rotations' shape, with a maximum dimension of 2.

        Examples
        --------
        >>> from orix.crystal_map import Phase
        >>> from orix.quaternion import Rotation
        >>> from kikuchipy.detectors import EBSDDetector
        >>> from kikuchipy.generators import EBSDSimulationGenerator
        >>> det = EBSDDetector(
        ...     shape=(60, 60), sample_tilt=70, pc=[0.5,] * 3
        ... )
        >>> p = Phase(name="ni", space_group=225)
        >>> p.structure.lattice.setLatPar(3.52, 3.52, 3.52, 90, 90, 90)
        >>> simgen = EBSDSimulationGenerator(
        ...     detector=det,
        ...     phase=p,
        ...     rotations=Rotation.from_euler([90, 45, 90])
        ... )
        >>> simgen
        EBSDSimulationGenerator (1,)
        EBSDDetector (60, 60), px_size 1 um, binning 1, tilt 0, pc
        (0.5, 0.5, 0.5)
        <name: . space group: None. point group: None. proper point
        group: None. color: tab:blue>
        Rotation (1,)
        """
        self.detector = detector.deepcopy()
        self.phase = phase.deepcopy()
        self.rotations = deepcopy(rotations)

    @property
    def rotations(self) -> Rotation:
        """Unit cell rotations to simulate patterns for."""
        return self._rotations

    @rotations.setter
    def rotations(self, value: Rotation):
        """Set unit cell rotations, also reshaping detector PC array."""
        ndim = len(value.shape)
        if ndim > 2:
            raise ValueError(f"A maximum dimension of 2 is allowed, {ndim} > 2")
        else:
            self._rotations = value
            self._align_pc_with_rotations_shape()

    @property
    def navigation_shape(self) -> tuple:
        """Navigation shape of the rotations and detector projection
        center array (maximum of 2).
        """
        return self._rotations.shape

    @navigation_shape.setter
    def navigation_shape(self, value: tuple):
        """Set the navigation shape of the rotations and detector
        projection center array (maximum of 2).
        """
        ndim = len(value)
        if ndim > 2:
            raise ValueError(f"A maximum dimension of 2 is allowed, {ndim} > 2")
        else:
            self.detector.navigation_shape = value
            self.rotations = self.rotations.reshape(*value)

    @property
    def navigation_dimension(self) -> int:
        """Number of navigation dimensions (a maximum of 2)."""
        return len(self.navigation_shape)

    def __repr__(self):
        rotation_repr = repr(self.rotations).split("\n")[0]
        return (
            f"{self.__class__.__name__} {self.navigation_shape}\n"
            f"{self.detector}\n"
            f"{self.phase}\n"
            f"{rotation_repr}\n"
        )

    def __getitem__(self, key):
        new_detector = self.detector.deepcopy()
        new_detector.pc = new_detector.pc[key]
        new_rotations = self.rotations[key]
        return self.__class__(
            detector=new_detector, phase=self.phase, rotations=new_rotations
        )

    def geometrical_simulation(
        self, reciprocal_lattice_point: Optional[ReciprocalLatticePoint] = None
    ) -> GeometricalEBSDSimulation:
        """Project a set of Kikuchi bands and zone axes onto the
        detector, one set for each rotation of the unit cell.

        The zone axes are calculated from the Kikuchi bands.

        Parameters
        ----------
        reciprocal_lattice_point
            Crystal planes to project onto the detector. If None, and
            the generator has a phase with a unit cell with a point
            group, a set of planes with minimum distance of 1 Å and
            their symmetrically equivalent planes are used.

        Returns
        -------
        GeometricalEBSDSimulation

        Examples
        --------
        >>> from diffsims.crystallography import ReciprocalLatticePoint
        >>> simgen
        EBSDSimulationGenerator (1,)
        EBSDDetector (60, 60), px_size 1 um, binning 1, tilt 0, pc
        (0.5, 0.5, 0.5)
        <name: . space group: None. point group: None. proper point
        group: None. color: tab:blue>
        Rotation (1,)
        >>> sim1 = simgen.geometrical_simulation()
        >>> sim1.bands.size
        94
        >>> rlp = ReciprocalLatticePoint(
        ...     phase=simgen.phase, hkl=[[1, 1, 1], [2, 0, 0]]
        ... )
        >>> sim2 = simgen.geometrical_simulation()
        >>> sim2.bands.size
        13
        """
        rlp = reciprocal_lattice_point
        if rlp is None and (
            hasattr(self.phase.point_group, "name")
            and hasattr(self.phase.structure.lattice, "abcABG")
        ):
            rlp = ReciprocalLatticePoint.from_min_dspacing(
                self.phase, min_dspacing=1
            )
            rlp = rlp[rlp.allowed].symmetrise()
        elif rlp is None:
            raise ValueError("A ReciprocalLatticePoint object must be passed")
        self._rlp_phase_is_compatible(rlp)

        # Unit cell parameters (called more than once)
        phase = rlp.phase
        hkl = rlp.hkl.data

        # Get number of navigation dimensions and navigation axes
        # indices
        n_nav_dims = self.navigation_dimension
        navigation_axes = (1, 2)[:n_nav_dims]

        # Get Kikuchi band coordinates for all bands in all patterns
        # U_Kstar, transformation from detector frame D to reciprocal
        # crystal lattice frame Kstar.
        # Output shape is (3, n, 3) or (3, ny, nx, 3)
        det2recip = detector2reciprocal_lattice(
            sample_tilt=self.detector.sample_tilt,
            detector_tilt=self.detector.tilt,
            lattice=phase.structure.lattice,
            rotation=self.rotations,
        )
        # Output shape is (nhkl, n, 3) or (nhkl, ny, nx, 3)
        hkl_detector = np.tensordot(hkl, det2recip, axes=(1, 0))
        if n_nav_dims == 0:
            hkl_detector = hkl_detector.squeeze()
        # Get bands that are in some pattern
        hkl_is_upper, hkl_in_a_pattern = _get_coordinates_in_upper_hemisphere(
            z_coordinates=hkl_detector[..., 2], navigation_axes=navigation_axes
        )
        hkl = hkl[hkl_in_a_pattern, ...]
        hkl_in_pattern = hkl_is_upper[hkl_in_a_pattern, ...].T
        # Get coordinates in the proper shape
        hkl_detector = np.moveaxis(
            hkl_detector[hkl_in_a_pattern], source=0, destination=n_nav_dims
        )
        # And store it all
        bands = KikuchiBand(
            phase=phase,
            hkl=hkl,
            hkl_detector=hkl_detector,
            in_pattern=hkl_in_pattern,
            gnomonic_radius=self.detector.r_max,
        )

        # Get zone axes coordinates from Kikuchi bands
        uvw = _get_uvw_from_hkl(hkl=hkl)
        det2direct = detector2direct_lattice(
            sample_tilt=self.detector.sample_tilt,
            detector_tilt=self.detector.tilt,
            lattice=self.phase.structure.lattice,
            rotation=self.rotations,
        )
        uvw_detector = np.tensordot(uvw, det2direct, axes=(1, 0))
        if n_nav_dims == 0:
            uvw_detector = uvw_detector.squeeze()
        uvw_is_upper, uvw_in_a_pattern = _get_coordinates_in_upper_hemisphere(
            z_coordinates=uvw_detector[..., 2], navigation_axes=navigation_axes
        )
        uvw = uvw[uvw_in_a_pattern, ...]
        uvw_in_pattern = uvw_is_upper[uvw_in_a_pattern, ...].T
        uvw_detector = np.moveaxis(
            uvw_detector[uvw_in_a_pattern], source=0, destination=n_nav_dims
        )
        zone_axes = ZoneAxis(
            phase=phase,
            uvw=uvw,
            uvw_detector=uvw_detector,
            in_pattern=uvw_in_pattern,
            gnomonic_radius=self.detector.r_max,
        )

        return GeometricalEBSDSimulation(
            detector=self.detector,
            rotations=self.rotations,
            bands=bands,
            zone_axes=zone_axes,
        )

    def _rlp_phase_is_compatible(self, rlp: ReciprocalLatticePoint):
        if (
            not np.allclose(
                rlp.phase.structure.lattice.abcABG(),
                self.phase.structure.lattice.abcABG(),
                atol=1e-4,
            )
            or rlp.phase.point_group.name != self.phase.point_group.name
        ):
            raise ValueError(
                f"The lattice parameters and/or point group of {rlp.phase} "
                f"are not the same as for {self.phase}"
            )

    def _align_pc_with_rotations_shape(self):
        """Ensure that the PC and rotation arrays have matching
        navigation shapes, e.g. (2, 5, 3) and (2, 5, 4), respectively.
        """
        nav_shape = self.navigation_shape  # From rotations
        detector_nav_shape = self.detector.navigation_shape
        if detector_nav_shape == (1,):
            self.detector.pc = np.ones(nav_shape + (3,)) * self.detector.pc[0]
        elif detector_nav_shape != nav_shape:
            raise ValueError(
                f"The detector navigation shape {detector_nav_shape} must be "
                f"(1,) or equal to the rotations's shape {self.rotations.shape}"
            )
        self.detector.navigation_shape = self.navigation_shape


def _get_coordinates_in_upper_hemisphere(
    z_coordinates: np.ndarray, navigation_axes: tuple
) -> Tuple[np.ndarray, np.ndarray]:
    """Return two boolean arrays with True if a coordinate is in the
    upper hemisphere and if it is in the upper hemisphere in some
    pattern, respectively.
    """
    upper_hemisphere = z_coordinates > 0
    if len(navigation_axes) == 0:
        upper_hemisphere = upper_hemisphere.squeeze()
    in_a_pattern = np.sum(upper_hemisphere, axis=navigation_axes) != 0
    return upper_hemisphere, in_a_pattern
