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

from re import sub
from typing import Optional

from diffsims.crystallography import CrystalPlane
import numpy as np
from orix import quaternion

from kikuchipy.detectors import EBSDDetector
from kikuchipy.draw.markers import (
    get_line_segment_list,
    get_point_list,
    get_text_list,
)
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
    def bands_detector_coordinates(self) -> np.ndarray:
        """Start and end point coordinates of bands in uncalibrated
        detector coordinates.

        Returns
        -------
        band_coords
            On the form [[x00, y00, x01, y01], [x10, y10, x11, y11],
            ...].
        """
        band_coords = np.zeros((self.bands.size, 4), dtype=np.float32)
        pcx, pcy, pcz = self.detector.pc
        band_coords[:, ::2] = (
            self.bands.plane_trace_x_g + pcx / pcz
        ) / self.detector.x_scale
        band_coords[:, 1::2] = (
            -self.bands.plane_trace_y_g + pcy / pcz
        ) / self.detector.y_scale
        return band_coords

    @property
    def zone_axes_detector_coordinates(self) -> np.ndarray:
        """Coordinates of zone axes in uncalibrated detector
        coordinates.

        Returns
        -------
        zone_axes_coords
            Column sorted, on the form [[x0, y0], [x1, y1], ...].
        """
        zone_axes_coords = np.zeros((self.zone_axes.size, 2), dtype=np.float32)
        pcx, pcy, pcz = self.detector.pc
        zone_axes_coords[:, 0] = (
            self.zone_axes.x_gnomonic + pcx / pcz
        ) / self.detector.x_scale
        zone_axes_coords[:, 1] = (
            -self.zone_axes.y_gnomonic + pcy / pcz
        ) / self.detector.y_scale
        return zone_axes_coords

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
        zone_axes_coords[1] -= 0.05 * self.detector.nrows
        return zone_axes_coords

    def bands_as_markers(self, **kwargs) -> list:
        return get_line_segment_list(
            lines=self.bands_detector_coordinates,
            linewidth=kwargs.pop("linewidth", 2),
            color=kwargs.pop("color", "lime"),
            **kwargs,
        )

    def zone_axes_as_markers(self, **kwargs) -> list:
        return get_point_list(
            points=self.zone_axes_detector_coordinates,
            size=kwargs.pop("size", 40),
            marker=kwargs.pop("marker", "o"),
            facecolor=kwargs.pop("facecolor", "w"),
            edgecolor=kwargs.pop("edgecolor", "k"),
            zorder=kwargs.pop("zorder", 500),
            **kwargs,
        )

    def zone_axes_labels_as_markers(self, **kwargs) -> list:
        return get_text_list(
            texts=sub("[][ ]", "", str(self.zone_axes._hkldata)).split("\n"),
            coordinates=self.zone_axes_label_detector_coordinates,
            color=kwargs.pop("color", "k"),
            zorder=kwargs.pop("zorder", 600),
            ha=kwargs.pop("ha", "center"),
            bbox=kwargs.pop(
                "bbox",
                dict(
                    facecolor="w",
                    edgecolor="k",
                    boxstyle="round, rounding_size=0.2",
                    pad=0.1,
                ),
            ),
        )

    def pc_as_markers(self, **kwargs) -> list:
        pcxy = self.detector.pc[:2]
        pcxy[0, ...] *= self.detector.ncols - 1
        pcxy[1, ...] *= self.detector.nrows - 1
        return get_point_list(
            points=pcxy,
            size=kwargs.pop("size", 150),
            marker=kwargs.pop("marker", "*"),
            facecolor=kwargs.pop("facolor", "C1"),
            edgecolor=kwargs.pop("edgecolor", "k"),
            zorder=kwargs.pop("zorder", 6),
        )
