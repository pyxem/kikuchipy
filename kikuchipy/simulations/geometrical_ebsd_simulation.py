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
from orix import quaternion

from kikuchipy.detectors import EBSDDetector
from kikuchipy.draw.markers import get_line_segment_list, get_point_list
from kikuchipy.simulations.features import KikuchiBand, ZoneAxis


class GeometricalEBSDSimulation:
    def __init__(
        self,
        detector: EBSDDetector,
        reciprocal_lattice_point: CrystalPlane,
        orientations: quaternion.Rotation,
        bands: KikuchiBand,
        zone_axes: Optional[ZoneAxis] = None,
    ):
        """Create a geometrical EBSD simulation storing a set of center
        positions of Kikuchi bands on the detector, one set for each
        orientation of the unit cell.

        Parameters
        ----------
        detector
            An EBSD detector with a shape, pixel size, binning, and
            projection center(s) (PC(s)).
        reciprocal_lattice_point
            Crystal planes projected onto the detector.
        orientations
            Orientations of the unit cell.
        bands
            Kikuchi band(s) projected onto the detector.
        zone_axes
            Zone axis/axes projected onto the detector. Default is None.

        Returns
        -------
        GeometricalEBSDSimulation
        """
        self.detector = detector
        self.orientations = orientations
        self.reciprocal_lattice_point = reciprocal_lattice_point
        self.bands = bands
        self.zone_axes = zone_axes

    @property
    def plane_trace_detector_coordinates(self) -> np.ndarray:
        """Start and end point coordinates of plane traces in
        uncalibrated detector coordinates.

        Returns
        -------
        np.ndarray
            Column sorted, on the form
            [[x00, y00, x01, y01], [x10, y10, x11, y11], ...].
        """
        pcx, pcy, pcz = self.detector.pc
        x_g = self.bands.plane_trace_x_g
        x_g = (x_g + (pcx / pcz)) / self.detector.x_scale
        y_g = -self.bands.plane_trace_y_g
        y_g = (y_g + (pcy / pcz)) / self.detector.y_scale
        return np.column_stack((x_g[0], y_g[0], x_g[1], y_g[1]))

    @property
    def zone_axes_detector_coordinates(self) -> np.ndarray:
        """Coordinates of zone axes in uncalibrated detector
        coordinates.

        Returns
        -------
        np.ndarray
            Column sorted, on the form [[x0, y0], [x1, y1], ...].
        """
        pcx, pcy, pcz = self.detector.pc
        x_g = self.zone_axes.x_g
        x_g = (x_g + (pcx / pcz)) / self.detector.x_scale
        y_g = -self.zone_axes.y_g
        y_g = (y_g + (pcy / pcz)) / self.detector.y_scale
        return np.column_stack((x_g, y_g))

    @property
    def zone_axes_label_detector_coordinates(self) -> np.ndarray:
        """Coordinates of zone axes labels in uncalibrated detector
        coordinates.

        Returns
        -------
        np.ndarray
            Column sorted, on the form [[x0, y0], [x1, y1], ...].
        """
        zone_axes_coords = self.zone_axes_detector_coordinates
        zone_axes_coords[1] -= 0.02 * self.detector.nrows
        return zone_axes_coords

    def bands_as_line_segments(self, **kwargs):
        """Coordinates of zone axes labels in uncalibrated detector
        coordinates.

        Returns
        -------
        np.ndarray
            Column sorted, on the form [[x0, y0], [x1, y1], ...].
        """
        return get_line_segment_list(
            lines=self.plane_trace_detector_coordinates,
            linewidth=kwargs.pop("linewidth", 2),
            color=kwargs.pop("color", "lime"),
            **kwargs,
        )

    def zone_axes_as_points(self, **kwargs):
        return get_point_list(
            points=self.zone_axes_detector_coordinates,
            size=kwargs.pop("size", 40),
            marker=kwargs.pop("marker", "o"),
            facecolor=kwargs.pop("facecolor", "w"),
            edgecolor=kwargs.pop("edgecolor", "k"),
            **kwargs,
        )
