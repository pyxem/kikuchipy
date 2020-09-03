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

from typing import Optional, Union

from diffsims.crystallography import CrystalPlane
import numpy as np
from orix.crystal_map import Phase
from orix.vector import Vector3d

from kikuchipy.projections.spherical import get_phi, get_theta, get_r


class KikuchiBand(CrystalPlane):
    gnomonic_radius = 10

    def __init__(
        self,
        phase: Phase,
        hkl: Union[Vector3d, np.ndarray, list, tuple],
        coordinates: Optional[np.ndarray] = None,
    ):
        """Center positions of Kikuchi bands on the detector.

        Parameters
        ----------
        phase
            A phase container with a crystal structure and a space and
            point group describing the allowed symmetry operations.
        hkl
            Miller indices.
        coordinates
            Kikuchi band coordinates on the detector.
        """
        super().__init__(phase=phase, hkl=hkl)
        self._coordinates = coordinates

    def __getitem__(self, key, **kwargs):
        return super().__getitem__(key, coordinates=self.coordinates[key])

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @property
    def x_detector(self) -> np.ndarray:
        return self.coordinates[..., 0]

    @property
    def y_detector(self) -> np.ndarray:
        return self.coordinates[..., 1]

    @property
    def z_detector(self) -> np.ndarray:
        return self.coordinates[..., 2]

    @property
    def x_gnomonic(self) -> np.ndarray:
        return self.x_detector / self.z_detector

    @property
    def y_gnomonic(self) -> np.ndarray:
        return self.y_detector / self.z_detector

    @property
    def phi_polar(self) -> np.ndarray:
        return get_phi(self.coordinates)

    @property
    def theta_polar(self) -> np.ndarray:
        return get_theta(self.coordinates)

    @property
    def hesse_distance(self) -> np.ndarray:
        """Distance from the PC (origin), i.e. the right-angle component
        of the distance to pole.
        """
        return np.tan(0.5 * np.pi - self.theta_polar)

    @property
    def within_gnomonic_radius(self) -> np.ndarray:
        is_full_upper = self.z_detector > -1e-5
        in_circle = np.abs(self.hesse_distance) < self.gnomonic_radius
        return np.logical_and(in_circle, is_full_upper)

    @property
    def hesse_alpha(self) -> np.ndarray:
        """Only angles for the planes within the Gnomonic radius are
        returned.
        """
        within = self.within_gnomonic_radius
        return np.arccos(self.hesse_distance[within] / self.gnomonic_radius)

    @property
    def plane_trace_coordinates(self) -> np.ndarray:
        """Plane trace coordinates P1, P2 in the plane of the detector.

        Only coordinates for the planes within the Gnomonic radius are
        returned.
        """
        within = self.within_gnomonic_radius

        phi = self.phi_polar[within]
        hesse_alpha = self.hesse_alpha

        size = hesse_alpha.size
        plane_trace = np.zeros((size, 4), dtype=np.float32)
        alpha1 = phi - np.pi + hesse_alpha
        alpha2 = phi - np.pi - hesse_alpha

        plane_trace[:, 0] = np.cos(alpha1)
        plane_trace[:, 1] = np.cos(alpha2)
        plane_trace[:, 2] = np.sin(alpha1)
        plane_trace[:, 3] = np.sin(alpha2)

        return self.gnomonic_radius * plane_trace

    @property
    def hesse_line_x(self) -> np.ndarray:
        return -self.hesse_distance * np.cos(self.phi_polar)

    @property
    def hesse_line_y(self) -> np.ndarray:
        return -self.hesse_distance * np.sin(self.phi_polar)


class ZoneAxis(CrystalPlane):
    gnomonic_radius = 10

    def __init__(
        self,
        phase: Phase,
        hkl: Union[Vector3d, np.ndarray, list, tuple],
        coordinates: Optional[np.ndarray] = None,
    ):
        """Positions of zone axes on the detector.

        Parameters
        ----------
        phase
            A phase container with a crystal structure and a space and
            point group describing the allowed symmetry operations.
        hkl
            Miller indices.
        coordinates
            Zone axes coordinates on the detector.
        """
        super().__init__(phase=phase, hkl=hkl)
        self._coordinates = coordinates

    def __getitem__(self, key, **kwargs):
        return super().__getitem__(key, coordinates=self.coordinates[key])

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @property
    def x_detector(self) -> np.ndarray:
        return self.coordinates[..., 0]

    @property
    def y_detector(self) -> np.ndarray:
        return self.coordinates[..., 1]

    @property
    def z_detector(self) -> np.ndarray:
        return self.coordinates[..., 2]

    @property
    def x_gnomonic(self) -> np.ndarray:
        """Only coordinates for the axes within the Gnomonic radius are
        returned.
        """
        within = self.within_gnomonic_radius
        return self.x_detector[within] / self.z_detector[within]

    @property
    def y_gnomonic(self) -> np.ndarray:
        """Only coordinates for the axes within the Gnomonic radius are
        returned.
        """
        within = self.within_gnomonic_radius
        return self.y_detector[within] / self.z_detector[within]

    @property
    def r_gnomonic(self) -> np.ndarray:
        return get_r(self.coordinates) / self.z_detector

    @property
    def theta_polar(self) -> np.ndarray:
        return get_theta(self.coordinates)

    @property
    def within_gnomonic_radius(self) -> np.ndarray:
        return self.r_gnomonic < self.gnomonic_radius
