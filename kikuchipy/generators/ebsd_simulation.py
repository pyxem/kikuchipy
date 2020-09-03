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

from typing import Optional

from diffsims.crystallography import CrystalPlane
import numpy as np
from orix.crystal_map import Phase
from orix.quaternion import Rotation

from kikuchipy.detectors import EBSDDetector
from kikuchipy.projections.ebsd import (
    detector2reciprocal_lattice,
    detector2direct_lattice,
)
from kikuchipy.simulations import GeometricalEBSDSimulation
from kikuchipy.simulations.features import KikuchiBand, ZoneAxis


class EBSDSimulationGenerator:
    def __init__(
        self,
        detector: Optional[EBSDDetector] = None,
        phase: Optional[Phase] = None,
        orientations: Optional[Rotation] = None,
    ):
        if detector is None:
            detector = EBSDDetector()
        if phase is None:
            phase = Phase()
        self.detector = detector
        self.phase = phase
        self.orientations = orientations

    def __repr__(self):
        return (
            f"{self.__class__.__name__}\n"
            f"{self.detector}\n  {self.phase}\n"
            f"{self.orientations}\n"
        )

    def geometrical_simulation(
        self, reciprocal_lattice_point: Optional[CrystalPlane] = None,
    ) -> GeometricalEBSDSimulation:
        """Project a set of center positions of Kikuchi bands on the
        detector, one set for each orientation of the unit cell.

        Parameters
        ----------
        reciprocal_lattice_point :
            Crystal planes to project onto the detector. If None, and
            the generator has a phase with a unit cell with a point
            group, a set of planes with minimum distance of 1 Å is used.

        Returns
        -------
        GeometricalEBSDSimulation
        """
        if self.orientations is None:
            raise ValueError("Unit cell orientations must be set")
        rlp = reciprocal_lattice_point
        if rlp is None and (
            hasattr(self.phase.point_group, "name")
            and hasattr(self.phase.structure.lattice, "abcABG")
        ):
            rlp = CrystalPlane.from_min_dspacing(self.phase, min_dspacing=1)
            rlp.calculate_structure_factor(voltage=15e3)
            rlp = rlp[rlp.allowed].symmetrise()
        elif rlp is None:
            raise ValueError("A ReciprocalLatticePoint object must be passed")
        self._rlp_phase_is_compatible(rlp)

        # Unit cell parameters (called more than once)
        phase = rlp.phase
        hkl = rlp._hkldata
        hkl_transposed = hkl.T

        # Get Kikuchi band coordinates for all bands in all patterns
        # U_Kstar, transformation from detector frame D to reciprocal crystal
        # lattice frame Kstar
        # TODO: Possible bottleneck due to large dot products! Room for
        #  lots of improvements with dask.
        det2recip = detector2reciprocal_lattice(
            sample_tilt=self.detector.sample_tilt,
            detector_tilt=self.detector.tilt,
            lattice=phase.structure.lattice,
            orientation=self.orientations,
        )  # (3, n, 3)
        band_coordinates = det2recip.T.dot(hkl_transposed).T  # (n hkl, n, 3)

        # Determine whether a band is visible in a pattern
        upper_hemisphere = band_coordinates[..., 2] > 0
        is_in_some_pattern = np.sum(upper_hemisphere, axis=1) != 0

        # Get bands that were in some pattern and their coordinates in the
        # proper shape
        hkl = hkl[is_in_some_pattern]
        hkl_in_pattern = upper_hemisphere[is_in_some_pattern].T
        band_coordinates = np.rollaxis(
            band_coordinates[is_in_some_pattern], axis=1
        )

        # And store it all
        bands = KikuchiBand(
            phase=phase,
            hkl=hkl,
            coordinates=band_coordinates,
            in_pattern=hkl_in_pattern,
            gnomonic_radius=self.detector.r_max,
        )

        # Get zone axes coordinates
        # U_K, transformation from detector frame D to direct crystal lattice
        # frame K
        #        det2direct = detector2direct_lattice(
        #            sample_tilt=self.detector.sample_tilt,
        #            detector_tilt=self.detector.tilt,
        #            lattice=phase.structure.lattice,
        #            orientation=self.orientations,
        #        )
        #        hkl_transposed_upper = hkl_transposed[..., upper_hemisphere]
        #        axis_coordinates = det2direct.T.dot(hkl_transposed_upper).T
        #        zone_axes = ZoneAxis(
        #            phase=phase, hkl=upper_hkl, coordinates=axis_coordinates
        #        )

        return GeometricalEBSDSimulation(
            detector=self.detector,
            reciprocal_lattice_point=rlp,
            orientations=self.orientations,
            bands=bands,
            #            zone_axes=zone_axes,
        )

    def _rlp_phase_is_compatible(self, rlp: CrystalPlane):
        if (
            rlp.phase.structure.lattice.abcABG
            != self.phase.structure.lattice.abcABG
            or rlp.phase.point_group.name != self.phase.point_group.name
        ):
            raise ValueError(
                f"The unit cell with the reciprocal lattice points {rlp.phase} "
                f"is not the same as {self.phase}"
            )
