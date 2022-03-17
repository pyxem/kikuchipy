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
        r_gnomonic: float = 10,
    ):
        self.vector = vector
        self.vector_detector = vector_detector
        self.in_pattern = in_pattern
        self.r_gnomonic = r_gnomonic

    @property
    def x_gnomonic(self) -> np.ndarray:
        return self.vector_detector.x / self.vector_detector.z

    @property
    def y_gnomonic(self) -> np.ndarray:
        return self.vector_detector.y / self.vector_detector.z

    @property
    def hesse_distance(self) -> np.ndarray:
        return np.tan(0.5 * np.pi - self.vector_detector.polar)

    @property
    def within_r_gnomonic(self) -> np.ndarray:
        is_full_upper = self.vector_detector.z > -1e-5
        in_circle = np.abs(self.hesse_distance) < self.r_gnomonic
        return np.logical_and(in_circle, is_full_upper)

    def __getitem__(self, key):
        """Get a deepcopy subset of the simulation instance.

        Properties have different shapes, so care must be taken when
        slicing. As an example, consider a 2 x 3 map with 4 lines. Three
        data shapes are considered:
        * navigation shape (2, 3) (gnomonic_radius)
        * line shape (4,) (hkl, structure_factor, theta)
        * full shape (2, 3, 4) (hkl_detector, in_pattern)
        """
        # These are overwritten as the input key length is investigated
        nav_slice, sim_slice = key, key  # full_slice = key
        nav_ndim = 1  # self.navigation_dimension
        n_keys = len(key) if hasattr(key, "__iter__") else 1
        if n_keys == 0:  # The case with key = ()/slice(None). Return everything
            sim_slice = slice(None)
        elif n_keys == 1:
            if nav_ndim != 0:
                sim_slice = slice(None)
        elif n_keys == 2:
            if nav_ndim == 0:
                raise IndexError("Not enough axes to slice")
            elif nav_ndim == 1:
                nav_slice = key[0]
                sim_slice = key[1]
            else:  # nav_slice = key
                sim_slice = slice(None)
        elif n_keys == 3:  # Maximum number of slices
            if nav_ndim < 2:
                raise IndexError("Not enough axes to slice")
            else:
                nav_slice = key[:2]
                sim_slice = key[2]
        new_features = self.__class__(
            vector=self.vector[sim_slice],
            vector_detector=self.vector_detector[sim_slice],
            in_pattern=self.in_pattern[key],
            r_gnomonic=self.r_gnomonic,
        )
        #        new_features._structure_factor = self.structure_factor[band_slice]
        #        new_features._theta = self.theta[band_slice]
        return new_features


class KikuchiPatternLine(KikuchiPatternFeature):
    def __init__(
        self,
        hkl: Vector3d,
        hkl_detector: Vector3d,
        in_pattern: np.ndarray,
        r_gnomonic: float = 10,
    ):
        super().__init__(hkl, hkl_detector, in_pattern, r_gnomonic)

    @property
    def hesse_alpha(self) -> np.ndarray:
        hesse_distance = self.hesse_distance
        hesse_distance[~self.within_r_gnomonic] = np.nan
        return np.arccos(hesse_distance / self.r_gnomonic)

    @property
    def plane_trace_coordinates(self) -> np.ndarray:
        # Get alpha1 and alpha2 angles (NaN for bands outside gnomonic radius)
        azimuth = self.vector_detector.azimuth
        hesse_alpha = self.hesse_alpha
        plane_trace = np.zeros((self.vector.size, 4))
        alpha1 = azimuth - np.pi + hesse_alpha
        alpha2 = azimuth - np.pi - hesse_alpha

        # Calculate start and end points for the plane traces
        plane_trace[..., 0] = np.cos(alpha1)
        plane_trace[..., 1] = np.cos(alpha2)
        plane_trace[..., 2] = np.sin(alpha1)
        plane_trace[..., 3] = np.sin(alpha2)

        # And remember to multiply by the gnomonic radius
        return self.r_gnomonic * plane_trace

    @property
    def hesse_line_x(self) -> np.ndarray:
        return -self.hesse_distance * np.cos(self.vector_detector.azimuth)

    @property
    def hesse_line_y(self) -> np.ndarray:
        return -self.hesse_distance * np.sin(self.vector_detector.azimuth)


class KikuchiPatternZoneAxis(KikuchiPatternFeature):
    def __init__(
        self,
        uvw: Vector3d,
        uvw_detector: Vector3d,
        in_pattern: np.ndarray,
        r_gnomonic: float = 10,
    ):
        super().__init__(uvw, uvw_detector, in_pattern, r_gnomonic)

    @property
    def _xy_within_gnomonic_radius(self) -> np.ndarray:
        xy = np.ones((self.vector.size, 2)) * np.nan
        within = self.within_r_gnomonic
        xy[within, 0] = self.x_gnomonic[within]
        xy[within, 1] = self.y_gnomonic[within]
        return xy


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
