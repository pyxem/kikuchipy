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
import matplotlib.path as mpath
import numpy as np


class KikuchiPatternSimulation:

    exclude_outside_detector = True

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

    @property
    def zone_axes_within_gnomonic_bounds(self) -> np.ndarray:
        """Return a boolean array with True for the zone axes within
        the detector's gnomonic bounds.

        Returns
        -------
        within_gnomonic_bounds : numpy.ndarray
            Boolean array with True for zone axes within the detector's
            gnomonic bounds.
        """
        det = self.detector

        # Get gnomonic bounds
        x_range = det.x_range
        y_range = det.y_range

        # Extend gnomonic bounds by one detector pixel to include zone
        # axes on the detector border
        x_scale = det.x_scale
        y_scale = det.y_scale
        x_range[..., 0] -= x_scale
        x_range[..., 1] += x_scale
        y_range[..., 0] -= y_scale
        y_range[..., 1] += y_scale

        # Get gnomonic coordinates
        xg = self.zone_axes.x_gnomonic
        yg = self.zone_axes.y_gnomonic

        # Add an extra dimension to account for n number of zone axes in
        # the last dimension for the gnomonic coordinate arrays
        x_range = np.expand_dims(x_range, axis=-2)
        y_range = np.expand_dims(y_range, axis=-2)

        # Get boolean array
        within_x = np.logical_and(xg >= x_range[..., 0], xg <= x_range[..., 1])
        within_y = np.logical_and(yg >= y_range[..., 0], yg <= y_range[..., 1])
        within_gnomonic_bounds = within_x * within_y

        return within_gnomonic_bounds.reshape(self.zone_axes.vector.size)

    @property
    def zone_axes_detector_coordinates(self) -> np.ndarray:
        """Coordinates of zone axes in uncalibrated detector
        coordinates (a scale of 1 and offset of 0).

        If `GeometricalEBSDSimulation.exclude_outside_detector` is True,
        the coordinates of the zone axes outside the detector are set to
        `np.nan`.

        Returns
        -------
        za_coords : numpy.ndarray
            Zone axis coordinates (x, y) on the detector.
        """
        xyg = self.zone_axes._xy_within_gnomonic_radius
        xg = xyg[..., 0]
        yg = xyg[..., 1]
        coords_d = np.empty_like(xyg)

        # Get projection center coordinates, and add one axis to get the
        # shape (navigation shape, 1)
        det = self.detector
        pcx = det.pcx[..., np.newaxis]
        pcy = det.pcy[..., np.newaxis]
        pcz = det.pcz[..., np.newaxis]

        coords_d[..., 0] = (xg + (pcx / pcz) * det.aspect_ratio) / det.x_scale[
            ..., np.newaxis
        ]
        coords_d[..., 1] = (-yg + (pcy / pcz)) / det.y_scale[..., np.newaxis]

        if self.exclude_outside_detector:
            coords_d[~self.zone_axes_within_gnomonic_bounds] = np.nan

        return coords_d

    def as_collections(self, coordinates: str = "detector") -> list:
        # Lines
        if coordinates == "detector":
            line_coords = self.lines_detector_coordinates
            za_coords = self.zone_axes_detector_coordinates
            za_scale = 2
        else:
            line_coords = self.lines.plane_trace_coordinates
            za_coords = self.zone_axes._xy_within_gnomonic_radius
            za_scale = self.detector.x_scale * 3

        line_coords = line_coords[~np.isnan(line_coords).any(axis=-1)]
        line_coords = line_coords.reshape((line_coords.shape[0], 2, 2))
        line_collection = mcollections.LineCollection(
            segments=list(line_coords),
            linewidth=1,
            color="C0",
            alpha=1,
            zorder=1,
            label="kikuchi_lines",
        )

        # Zone axes
        za_coords = za_coords[~np.isnan(za_coords).any(axis=-1)]
        paths = []
        for x, y in za_coords:
            paths.append(mpath.Path.circle((x, y), za_scale))
        za_collection = mcollections.PathCollection(
            paths, color="C1", zorder=1, label="zone_axes"
        )

        return [line_collection, za_collection]
