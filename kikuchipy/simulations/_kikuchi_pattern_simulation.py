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
    def __init__(self, detector, rotation):
        self.detector = detector
        self.rotation = rotation


class GeometricalKikuchiPatternSimulation(KikuchiPatternSimulation):
    def __init__(self, detector, rotation, lines):
        super().__init__(detector, rotation)
        self.lines = lines

    @property
    def lines_detector_coordinates(self) -> np.ndarray:
        """Start and end point coordinates of bands in uncalibrated
        detector coordinates (a scale of 1 and offset of 0).

        Returns
        -------
        band_coords_detector : numpy.ndarray
            Band coordinates (x0, y0, x1, y1) on the detector.
        """
        # Get start and end points for the plane traces in gnomonic coordinates
        # and set up output array in uncalibrated detector coordinates
        band_coords_gnomonic = self.lines.plane_trace_coordinates
        band_coords_detector = np.zeros_like(band_coords_gnomonic)

        # Get projection center coordinates, and add two axis to get the shape
        # (navigation shape, 1, 1)
        pcx = self.detector.pcx[..., np.newaxis, np.newaxis]
        pcy = self.detector.pcy[..., np.newaxis, np.newaxis]
        pcz = self.detector.pcz[..., np.newaxis, np.newaxis]

        # X and Y coordinates are now in place (0, 2) and (1, 3) respectively
        band_coords_detector[..., ::2] = (
            band_coords_gnomonic[..., :2] + (pcx / pcz) * self.detector.aspect_ratio
        ) / self.detector.x_scale[..., np.newaxis, np.newaxis]
        band_coords_detector[..., 1::2] = (
            -band_coords_gnomonic[..., 2:] + (pcy / pcz)
        ) / self.detector.y_scale[..., np.newaxis, np.newaxis]

        return band_coords_detector

    def as_collections(
        self, coordinates: str = "detector"
    ) -> mcollections.LineCollection:
        if coordinates == "detector":
            line_coords = self.lines_detector_coordinates
        else:
            line_coords = self.lines.plane_trace_coordinates
            line_coords[:, [0, 1, 2, 3]] = line_coords[:, [0, 2, 1, 3]]

        is_nan = np.isnan(line_coords).any(axis=-1)
        line_coords = line_coords[~is_nan]
        line_coords = line_coords.reshape((line_coords.shape[0], 2, 2))

        collection = mcollections.LineCollection(
            segments=list(line_coords), linewidth=1, color="r", alpha=1, zorder=1
        )

        return collection
