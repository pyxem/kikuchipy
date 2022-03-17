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

import matplotlib.collections as mcollections
import numpy as np


class KikuchiPatternSimulation:
    def __init__(
        self,
        detector,
        rotation,
        reflectors,
        lines,
        zone_axes,
    ):
        self.detector = detector
        self.rotation = rotation
        self.reflectors = reflectors
        self.lines = lines
        self.zone_axes = zone_axes


class GeometricalKikuchiPatternSimulation(KikuchiPatternSimulation):
    def __init__(self, detector, rotation, reflectors, lines, zone_axes):
        super().__init__(detector, rotation, reflectors, lines, zone_axes)

    @property
    def lines_detector_coordinates(self) -> np.ndarray:
        """Start and end point coordinates of bands in uncalibrated
        detector coordinates (a scale of 1 and offset of 0).

        Returns
        -------
        band_coords_detector : numpy.ndarray
            Band coordinates (x0, y0, x1, y1) on the detector.
        """
        # Get PC coordinates and add two axes to get the shape
        # (navigation shape, 1, 1)
        det = self.detector
        pcx = det.pcx[..., None, None]
        pcy = det.pcy[..., None, None]
        pcz = det.pcz[..., None, None]

        # Convert coordinates
        coords_d = self.lines.plane_trace_coordinates.copy()
        coords_d[..., [0, 2]] = coords_d[..., [0, 2]] + (pcx / pcz) * det.aspect_ratio
        coords_d[..., [0, 2]] = coords_d[..., [0, 2]] / det.x_scale[..., None, None]
        coords_d[..., [1, 3]] = -coords_d[..., [1, 3]] + (pcy / pcz)
        coords_d[..., [1, 3]] = coords_d[..., [1, 3]] / det.y_scale[..., None, None]

        return coords_d

    def as_collections(
        self, coordinates: str = "detector"
    ) -> mcollections.LineCollection:
        # Lines
        if coordinates == "detector":
            line_coords = self.lines_detector_coordinates
        else:
            line_coords = self.lines.plane_trace_coordinates
        is_nan = np.isnan(line_coords).any(axis=-1)
        line_coords = line_coords[~is_nan]
        line_coords = line_coords.reshape((line_coords.shape[0], 2, 2))
        collection = mcollections.LineCollection(
            segments=list(line_coords), linewidth=1, color="r", alpha=1, zorder=1
        )

        # Zone axes
        #        xy = self.zone_axes._xy_within_gnomonic_radius
        #        collection = mcollections.PathCollection()

        return collection
