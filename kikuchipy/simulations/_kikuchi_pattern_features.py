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
from orix.vector import Vector3d


class KikuchiPatternFeature:
    def __init__(
        self,
        vector: Vector3d,
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

    #    def __getitem__(self, key):
    #        """Get a deepcopy subset of the simulation instance.
    #
    #        Properties have different shapes, so care must be taken when
    #        slicing. As an example, consider a 2 x 3 map with 4 vectors.
    #        Three data shapes are considered:
    #        * navigation shape (2, 3) (x_gnomonic)
    #        * vector shape (4,) (vector)
    #        * full shape (2, 3, 4) (vector_detector, in_pattern)
    #        """
    #        n_keys = len(key) if hasattr(key, "__iter__") else 1
    #        key = np.atleast_2d(key)
    #        # These are overwritten as the input key length is investigated
    #        nav_slice, sim_slice = key, key  # full_slice = key
    #        ndim = self.ndim
    #        if n_keys == 1:
    #            if ndim != 0:
    #                sim_slice = slice(None)
    #        elif n_keys == 2:
    #            if ndim == 0:
    #                raise IndexError("Not enough axes to slice")
    #            elif ndim == 1:
    #                sim_slice = key[1]
    #            else:  # nav_slice = key
    #                sim_slice = slice(None)
    #        elif n_keys == 3:  # Maximum number of slices
    #            if ndim < 2:
    #                raise IndexError("Not enough axes to slice")
    #            else:
    #                sim_slice = key[2]
    #        return self.__class__(
    #            self.vector[sim_slice],
    #            self.vector_detector[sim_slice],
    #            in_pattern=self.in_pattern[key],
    #            max_r_gnomonic=self.max_r_gnomonic,
    #        )

    def _set_within_r_gnomonic(self, coordinates):
        is_full_upper = np.atleast_2d(self.vector_detector.z) > -1e-5
        in_circle = coordinates < self.max_r_gnomonic
        self._within_r_gnomonic = np.logical_and(in_circle, is_full_upper)


class KikuchiPatternLine(KikuchiPatternFeature):
    def __init__(
        self,
        hkl: Vector3d,
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
        uvw: Vector3d,
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


def _get_dimension_str(nav_shape: tuple, sig_shape: tuple):
    """Adapted from HyperSpy's AxesManager._get_dimension_str."""
    dim_str = "("
    if len(nav_shape) > 0:
        for axis in nav_shape:
            dim_str += f"{axis}, "
    dim_str = dim_str.rstrip(", ")
    dim_str += "|"
    if len(sig_shape) > 0:
        for axis in sig_shape:
            dim_str += f"{axis}, "
    dim_str = dim_str.rstrip(", ")
    dim_str += ")"
    return dim_str
