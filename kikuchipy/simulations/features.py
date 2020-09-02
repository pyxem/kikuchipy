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

from kikuchipy.projections import get_polar


class KikuchiBand(CrystalPlane):
    hesse_radius = 10

    def __init__(self, phase, hkl, coordinates=None):
        """Center position of a Kikuchi band on a detector.

        Parameters
        ----------
        phase
        hkl
        coordinates
        """
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
        """Distance from origin, i.e. the right-angle component of
        distance of the pole.
        """
        theta = self.polar_coordinates[..., 0]
        return np.tan(0.5 * np.pi - theta)

    @property
    def hesse_alpha(self):
        return np.arccos(self.hesse_distance / self.hesse_radius)

    @property
    def alpha_g(self):
        """Angle from PC (0, 0) to point on circle where the line cuts
        the circle.
        """
        phi = self.polar_coordinates[..., 1]
        return np.row_stack(
            (phi - np.pi + self.hesse_alpha, phi - np.pi - self.hesse_alpha)
        )

    @property
    def within_hesse_radius(self):
        is_full_upper = self.z_d > -1e-5
        in_circle = np.abs(self.hesse_distance) < self.hesse_radius
        return np.logical_and(in_circle, is_full_upper)

    @property
    def plane_trace_x_g(self):
        a1, a2 = self.alpha_g
        return self.hesse_radius * np.row_stack((np.cos(a1), np.cos(a2)))

    @property
    def plane_trace_y_g(self):
        a1, a2 = self.alpha_g
        return self.hesse_radius * np.row_stack((np.sin(a1), np.sin(a2)))

    @property
    def plane_trace_g(self):
        a1, a2 = self.alpha_g
        return self.hesse_radius * np.row_stack(
            (np.cos(a1), np.sin(a1), np.cos(a2), np.sin(a2))
        )

    @property
    def hesse_line_x(self):
        phi = self.polar_coordinates[..., 1]
        return -self.hesse_distance * np.cos(phi)

    @property
    def hesse_line_y(self):
        phi = self.polar_coordinates[..., 1]
        return -self.hesse_distance * np.sin(phi)


class ZoneAxis(KikuchiBand):
    def __init__(self, phase, hkl, coordinates=None):
        """Position of a zone axis on a detector.

        Parameters
        ----------
        phase
        hkl
        coordinates
        """
        super().__init__(phase=phase, hkl=hkl, coordinates=coordinates)
