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

import re
from typing import Optional, Union

import matplotlib.collections as mcollections
import matplotlib.path as mpath
import matplotlib.text as mtext
import numpy as np


class KikuchiPatternSimulation:
    def __init__(
        self,
        detector,
        rotations,
        reflectors,
        lines,
        zone_axes,
    ):
        self._detector = detector
        self._rotations = rotations
        self._reflectors = reflectors
        self._lines = lines
        self._zone_axes = zone_axes


class GeometricalKikuchiPatternSimulation(KikuchiPatternSimulation):
    def __init__(self, detector, rotations, reflectors, lines, zone_axes):
        super().__init__(detector, rotations, reflectors, lines, zone_axes)
        self._set_lines_detector_coordinates()
        self._set_zone_axes_detector_coordinates()
        self.ndim = rotations.ndim

    @property
    def detector(self):
        return self._detector

    @property
    def rotations(self):
        return self._rotations

    @property
    def reflectors(self):
        return self._reflectors

    @property
    def lines(self):
        return self._lines

    @property
    def zone_axes(self):
        return self._zone_axes

    def plot(
        self,
        index: Union[int, tuple, None] = None,
        coordinates: str = "detector",
        pattern: Optional[np.ndarray] = None,
        lines: bool = True,
        zone_axes: bool = True,
        zone_axes_labels: bool = True,
        pc: bool = True,
        pattern_kwargs: dict = None,
        lines_kwargs: dict = None,
        zone_axes_kwargs: dict = None,
        zone_axes_labels_kwargs: dict = None,
        pc_kwargs: dict = None,
        return_figure: bool = False,
    ):
        fig, ax = self.detector.plot(
            coordinates=coordinates,
            pattern=pattern,
            show_pc=pc,
            pc_kwargs=pc_kwargs,
            pattern_kwargs=pattern_kwargs,
            return_fig_ax=True,
        )
        collections = self.as_collections(
            index,
            coordinates,
            lines,
            zone_axes,
            zone_axes_labels,
            lines_kwargs,
            zone_axes_kwargs,
            zone_axes_labels_kwargs,
        )
        for c in collections:
            try:
                ax.add_collection(c)
            except AttributeError:
                for text in c:
                    ax.add_artist(text)
        if return_figure:
            return fig

    def as_collections(
        self,
        index: Union[int, tuple, None] = None,
        coordinates: str = "detector",
        lines: bool = True,
        zone_axes: bool = True,
        zone_axes_labels: bool = True,
        lines_kwargs: dict = None,
        zone_axes_kwargs: dict = None,
        zone_axes_labels_kwargs: dict = None,
    ) -> list:
        if index is None:
            index = (0, 0)[: self.ndim]
        collections = []
        if lines:
            if lines_kwargs is None:
                lines_kwargs = {}
            collections.append(
                self._lines_as_collection(index, coordinates, **lines_kwargs)
            )
        if zone_axes:
            if zone_axes_kwargs is None:
                zone_axes_kwargs = {}
            collections.append(
                self._zone_axes_as_collection(index, coordinates, **zone_axes_kwargs)
            )
        if zone_axes_labels:
            if zone_axes_labels_kwargs is None:
                zone_axes_labels_kwargs = {}
            collections.append(
                self._zone_axes_labels_as_list(
                    index, coordinates, **zone_axes_labels_kwargs
                )
            )
        return collections

    def lines_coordinates(
        self,
        index: Union[int, tuple] = None,
        coordinates: str = "detector",
        exclude_nan: bool = True,
    ) -> np.ndarray:
        if index is None:
            index = (0, 0)[: self.ndim]
        if coordinates == "detector":
            coords = self._lines_detector_coordinates[index]
        else:  # gnomonic
            coords = self.lines.plane_trace_coordinates[index]
        if exclude_nan:
            coords = coords[~np.isnan(coords).any(axis=-1)]
        return coords

    def zone_axes_coordinates(
        self,
        index,
        coordinates: str = "detector",
        exclude_nan: bool = True,
    ) -> np.ndarray:
        if index is None:
            index = (0, 0)[: self.ndim]
        if coordinates == "detector":
            coords = self._zone_axes_detector_coordinates[index]
        else:  # gnomonic
            coords = self.zone_axes._xy_within_r_gnomonic[index]
        if exclude_nan:
            coords = coords[~np.isnan(coords).any(axis=-1)]
        return coords

    def _set_lines_detector_coordinates(self) -> np.ndarray:
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

        self._lines_detector_coordinates = coords_d

    def _set_zone_axes_detector_coordinates(self) -> np.ndarray:
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
        xyg = self.zone_axes._xy_within_r_gnomonic
        xg = xyg[..., 0]
        yg = xyg[..., 1]
        coords_d = np.empty_like(xyg)

        # Get projection center coordinates, and add one axis to get the
        # shape (navigation shape, 1)
        det = self.detector
        pcx = det.pcx[..., None]
        pcy = det.pcy[..., None]
        pcz = det.pcz[..., None]

        coords_d[..., 0] = (xg + (pcx / pcz) * det.aspect_ratio) / det.x_scale[
            ..., None
        ]
        coords_d[..., 1] = (-yg + (pcy / pcz)) / det.y_scale[..., None]

        self._zone_axes_detector_coordinates = coords_d

    def _lines_as_collection(
        self, index: Union[int, tuple], coordinates: str, **kwargs
    ) -> mcollections.LineCollection:
        coords = self.lines_coordinates(index, coordinates)
        coords = coords.reshape((coords.shape[0], 2, 2))
        line_defaults = dict(
            color="r", linewidth=1, alpha=1, zorder=1, label="kikuchi_lines"
        )
        for k, v in line_defaults.items():
            kwargs.setdefault(k, v)
        return mcollections.LineCollection(segments=list(coords), **kwargs)

    def _zone_axes_as_collection(
        self, index: Union[int, tuple], coordinates: str, **kwargs
    ) -> mcollections.PathCollection:
        coords = self.zone_axes_coordinates(index, coordinates)
        if coordinates == "detector":
            scatter_size = 0.01 * self.detector.nrows
        else:  # gnomonic
            scatter_size = 0.01 * np.diff(self.detector.x_range)[0]
        circles = []
        for x, y in coords:
            circles.append(mpath.Path.circle((x, y), scatter_size))
        path_defaults = dict(color="w", ec="k", zorder=1, label="zone_axes")
        for k, v in path_defaults.items():
            kwargs.setdefault(k, v)
        return mcollections.PathCollection(circles, **kwargs)

    def _zone_axes_labels_as_list(
        self, index: Union[int, tuple], coordinates: str, **kwargs
    ) -> list:
        za = self.zone_axes
        za_labels = za.vector.coordinates.round(0).astype(int)
        za_labels = za_labels[za.within_r_gnomonic[index]]
        za_labels_str = np.array2string(za_labels, threshold=za_labels.size)
        za_labels_list = re.sub(" ", "", za_labels_str[1:-1]).split("\n")
        xy = self.zone_axes_coordinates(index, coordinates)
        if coordinates == "detector":
            xy[..., 1] -= 0.03 * self.detector.nrows
        else:  # gnomonic
            xy[..., 1] += 0.03 * np.diff(self.detector.y_range)[0]
        text_defaults = dict(ha="center", bbox=dict(boxstyle="square", fc="w", pad=0.1))
        for k, v in text_defaults.items():
            kwargs.setdefault(k, v)
        texts = []
        for (x, y), label in zip(xy, za_labels_list):
            if np.all(~np.isnan([x, y])):
                text_i = mtext.Text(x, y, label, **kwargs)
                texts.append(text_i)
        return texts
