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

import numpy as np
from orix.vector import Miller, Vector3d


class KikuchiPatternFeature:
    def __init__(
        self,
        vector: Miller,
        vector_detector: Vector3d,
        in_pattern: np.ndarray,
        max_r_gnomonic: float = 10,
    ):
        self.vector = vector
        self.vector_detector = vector_detector
        self.in_pattern = np.atleast_2d(in_pattern)
        self.max_r_gnomonic = max_r_gnomonic
        self.ndim = vector.ndim

    @property
    def x_gnomonic(self) -> np.ndarray:
        return np.atleast_2d(self.vector_detector.x / self.vector_detector.z)

    @property
    def y_gnomonic(self) -> np.ndarray:
        return np.atleast_2d(self.vector_detector.y / self.vector_detector.z)

    @property
    def within_r_gnomonic(self) -> np.ndarray:
        return self._within_r_gnomonic

    def _set_within_r_gnomonic(self, coordinates):
        is_full_upper = np.atleast_2d(self.vector_detector.z) > -1e-5
        in_circle = coordinates < self.max_r_gnomonic
        self._within_r_gnomonic = np.logical_and(in_circle, is_full_upper)


class KikuchiPatternLine(KikuchiPatternFeature):
    def __init__(
        self,
        hkl: Miller,
        hkl_detector: Vector3d,
        in_pattern: np.ndarray,
        max_r_gnomonic: float = 10,
    ):
        super().__init__(hkl, hkl_detector, in_pattern, max_r_gnomonic)
        self._set_hesse_distance()
        self._set_within_r_gnomonic(np.abs(self.hesse_distance))
        self._set_hesse_alpha()
        self._set_plane_trace_coordinates()

    @property
    def hesse_distance(self) -> np.ndarray:
        return self._hesse_distance

    @property
    def hesse_alpha(self) -> np.ndarray:
        return self._hesse_alpha

    @property
    def plane_trace_coordinates(self) -> np.ndarray:
        """x0, y0, x1, y1"""
        return self._plane_trace_coordinates

    def _set_hesse_distance(self):
        hesse_distance = np.tan(0.5 * np.pi - self.vector_detector.polar)
        self._hesse_distance = np.atleast_2d(hesse_distance)

    def _set_hesse_alpha(self):
        hesse_distance = self.hesse_distance
        hesse_distance[~self.within_r_gnomonic] = np.nan
        self._hesse_alpha = np.arccos(hesse_distance / self.max_r_gnomonic)

    def _set_plane_trace_coordinates(self):
        # Get alpha1 and alpha2 angles (NaN for bands outside gnomonic radius)
        azimuth = np.atleast_2d(self.vector_detector.azimuth)
        hesse_alpha = self.hesse_alpha
        a1 = azimuth - np.pi + hesse_alpha
        a2 = azimuth - np.pi - hesse_alpha

        # Calculate start and end points for the plane traces
        plane_trace = np.stack((np.cos(a1), np.sin(a1), np.cos(a2), np.sin(a2)))
        plane_trace = np.moveaxis(plane_trace, 0, -1)
        plane_trace *= self.max_r_gnomonic

        self._plane_trace_coordinates = plane_trace


class KikuchiPatternZoneAxis(KikuchiPatternFeature):
    def __init__(
        self,
        uvw: Miller,
        uvw_detector: Vector3d,
        in_pattern: np.ndarray,
        max_r_gnomonic: float = 10,
    ):
        super().__init__(uvw, uvw_detector, in_pattern, max_r_gnomonic)
        self._set_r_gnomonic()
        self._set_within_r_gnomonic(self.r_gnomonic)
        self._set_xy_within_r_gnomonic()

    @property
    def r_gnomonic(self) -> np.ndarray:
        return self._r_gnomonic

    def _set_r_gnomonic(self):
        self._r_gnomonic = np.sqrt(self.x_gnomonic**2 + self.y_gnomonic**2)

    def _set_xy_within_r_gnomonic(self):
        xy = np.stack((self.x_gnomonic, self.y_gnomonic))
        xy = np.moveaxis(xy, 0, -1)
        xy[~self.within_r_gnomonic] = np.nan
        self._xy_within_r_gnomonic = xy
