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

from diffsims.crystallography import CrystalPlane
import numpy as np

from kikuchipy.projections import (
    detector2reciprocal_lattice,
    detector2direct_lattice,
    get_polar,
)


class GeometricEBSD:
    def __init__(
        self, detector, reciprocal_lattice_point, orientation,
    ):
        self.detector = detector
        self.orientation = orientation

        sample_tilt = detector.sample_tilt
        detector_tilt = detector.tilt

        phase = reciprocal_lattice_point.phase
        hkl = reciprocal_lattice_point._hkldata
        hkl_transposed = hkl.T
        lattice = phase.structure.lattice

        # Get Kikuchi bands
        # U_Kstar, transformation from detector frame D to reciprocal crystal
        # lattice frame Kstar
        det2recip = detector2reciprocal_lattice(
            sample_tilt=sample_tilt,
            detector_tilt=detector_tilt,
            lattice=lattice,
            orientation=orientation,
        )
        band_coordinates = det2recip.T.dot(hkl_transposed).T
        upper_hemisphere = band_coordinates[..., 2] > 0  # To include zero
        upper_hkl = hkl[upper_hemisphere]
        self.bands = KikuchiBand(
            phase=phase,
            hkl=upper_hkl,
            coordinates=band_coordinates[upper_hemisphere],
        )
        structure_factor = reciprocal_lattice_point.structure_factor
        self.bands._structure_factor = structure_factor

        # Get zone axes
        # U_K, transformation from detector frame D to direct crystal lattice
        # frame K
        det2direct = detector2direct_lattice(
            sample_tilt=sample_tilt,
            detector_tilt=detector_tilt,
            lattice=lattice,
            orientation=orientation,
        )
        hkl_transposed_upper = hkl_transposed[..., upper_hemisphere]
        axis_coordinates = det2direct.T.dot(hkl_transposed_upper).T
        self.zone_axes = ZoneAxis(
            phase=phase, hkl=upper_hkl, coordinates=axis_coordinates,
        )
        self.zone_axes._structure_factor = structure_factor


class KikuchiBand(CrystalPlane):
    hesse_radius = 2

    def __init__(self, phase, hkl, coordinates=None):
        super().__init__(phase=phase, hkl=hkl)
        self._coordinates = coordinates

    def __getitem__(self, key):
        new_band = super().__getitem__(key)
        new_band._coordinates = self.coordinates[key]
        return new_band

    @classmethod
    def from_nfamilies(cls, phase, nfamilies=5):
        pass

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def x_d(self):
        return self.coordinates[..., 0]

    @property
    def y_d(self):
        return self.coordinates[..., 1]

    @property
    def z_d(self):
        return self.coordinates[..., 2]

    @property
    def x_g(self):
        return self.x_d / self.z_d

    @property
    def y_g(self):
        return self.y_d / self.z_d

    @property
    def polar_coordinates(self):
        return get_polar(self.coordinates)

    @property
    def hesse_distance(self):
        """Distance from origin is right-angle component of distance of
        pole.
        """
        theta, _, _ = self.polar_coordinates
        return np.tan(0.5 * np.pi - theta)

    @property
    def within_hesse_radius(self):
        is_full_upper = self.z_d > -1e-5
        in_circle = np.abs(self.hesse_distance) < self.hesse_radius
        return np.logical_and(in_circle, is_full_upper)


class ZoneAxis(KikuchiBand):
    pass
