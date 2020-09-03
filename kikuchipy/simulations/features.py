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
    def __init__(
        self,
        phase: Phase,
        hkl: Union[Vector3d, np.ndarray, list, tuple],
        coordinates: np.ndarray,
        in_pattern: np.ndarray,
        gnomonic_radius: Union[float, np.ndarray] = 10,
    ):
        """Center positions of Kikuchi bands on the detector for n
        simulated patterns.

        Parameters
        ----------
        phase
            A phase container with a crystal structure and a space and
            point group describing the allowed symmetry operations.
        hkl
            All Miller indices present in any of the n patterns.
        coordinates
            Detector coordinates per pattern for each hkl, in the shape
            (n, n_hkl, 3).
        in_pattern
            Boolean array of shape (n, n_hkl) indicating whether an hkl
            is visible in a pattern.
        gnomonic_radius
            Only plane trace coordinates of bands with Hesse normal
            form distances below this radius is returned when called
            for.
        """
        super().__init__(phase=phase, hkl=hkl)
        if coordinates.ndim == 2:
            self._coordinates = coordinates[np.newaxis, ...]
        else:  # ndim == 3
            self._coordinates = coordinates
        self._in_pattern = np.atleast_2d(in_pattern)
        self.gnomonic_radius = gnomonic_radius

    def __getitem__(self, key, **kwargs):
        # TODO: Index by patterns or bands, not only bands!
        return super().__getitem__(
            key,
            coordinates=self.coordinates[key],
            in_pattern=self.in_pattern[key],
        )

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @property
    def gnomonic_radius(self) -> np.ndarray:
        """Only plane trace coordinates of bands with Hesse normal form
        distances below this radius is returned when called for.
        """
        return self._gnomonic_radius

    @gnomonic_radius.setter
    def gnomonic_radius(self, value: Union[float, np.ndarray]):
        """Only plane trace coordinates of bands with Hesse normal form
        distances below this radius is returned when called for.
        """
        self._gnomonic_radius = np.asarray(value)

    @property
    def n_patterns(self) -> int:
        return self.coordinates.shape[0]

    @property
    def in_pattern(self) -> np.ndarray:
        return self._in_pattern

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
        """Return whether a plane trace is within the `gnomonic_radius`
        as a boolean array.
        """
        is_full_upper = self.z_detector > -1e-5
        gnomonic_radius = self.gnomonic_radius[:, np.newaxis]
        in_circle = np.abs(self.hesse_distance) < gnomonic_radius
        return np.logical_and(in_circle, is_full_upper)

    @property
    def hesse_alpha(self) -> np.ndarray:
        """Hesse angle alpha. Only angles for the planes within the
        `gnomonic_radius` are returned.
        """
        hesse_distance = self.hesse_distance
        hesse_distance[~self.within_gnomonic_radius] = np.nan
        return np.arccos(hesse_distance / self.gnomonic_radius[:, np.newaxis])

    @property
    def plane_trace_coordinates(self) -> np.ndarray:
        """Plane trace coordinates P1, P2 in the plane of the detector.

        Only coordinates for the planes within the `gnomonic_radius` are
        returned.
        """
        # Get alpha1 and alpha2 angles
        phi = self.phi_polar
        hesse_alpha = self.hesse_alpha
        plane_trace = np.zeros((self.n_patterns, self.size, 4))
        alpha1 = phi - np.pi + hesse_alpha
        alpha2 = phi - np.pi - hesse_alpha

        # Calculate start and end points for the plane traces
        plane_trace[..., 0] = np.cos(alpha1)
        plane_trace[..., 1] = np.cos(alpha2)
        plane_trace[..., 2] = np.sin(alpha1)
        plane_trace[..., 3] = np.sin(alpha2)

        # And remember to multiply by the gnomonic radius
        # TODO: Surely, this can be done better!
        return (self.gnomonic_radius * plane_trace.T).T

    @property
    def hesse_line_x(self) -> np.ndarray:
        return -self.hesse_distance * np.cos(self.phi_polar)

    @property
    def hesse_line_y(self) -> np.ndarray:
        return -self.hesse_distance * np.sin(self.phi_polar)


class ZoneAxis(CrystalPlane):
    def __init__(
        self,
        phase: Phase,
        hkl: Union[Vector3d, np.ndarray, list, tuple],
        coordinates: np.ndarray,
        in_pattern: np.ndarray,
        gnomonic_radius: Union[float, np.ndarray] = 10,
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
        in_pattern
            Boolean array of shape (n, n_hkl) indicating whether an hkl
            is visible in a pattern.
        gnomonic_radius
            Only plane trace coordinates of bands with Hesse normal
            form distances below this radius is returned when called
            for.
        """
        super().__init__(phase=phase, hkl=hkl)
        if coordinates.ndim == 2:
            self._coordinates = coordinates[np.newaxis, ...]
        else:  # ndim == 3
            self._coordinates = coordinates
        self._in_pattern = np.atleast_2d(in_pattern)
        self.gnomonic_radius = gnomonic_radius

    @property
    def gnomonic_radius(self) -> np.ndarray:
        """Only plane trace coordinates of bands with Hesse normal form
        distances below this radius is returned when called for.
        """
        return self._gnomonic_radius

    @gnomonic_radius.setter
    def gnomonic_radius(self, value: Union[float, np.ndarray]):
        """Only plane trace coordinates of bands with Hesse normal form
        distances below this radius is returned when called for.
        """
        r = np.asarray(value)
        if r.ndim == 1:
            self._gnomonic_radius = r[:, np.newaxis]
        else:
            self._gnomonic_radius = r

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
