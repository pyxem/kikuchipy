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

import numpy as np
from orix.quaternion.rotation import Rotation

from kikuchipy.detectors import EBSDDetector
from kikuchipy.draw.markers import (
    get_line_segment_list,
    get_point_list,
    get_text_list,
)
from kikuchipy.simulations.features import KikuchiBand, ZoneAxis


class GeometricalEBSDSimulation:
    """Geometrical EBSD simulation with Kikuchi bands and zone axes.
    """

    def __init__(
        self,
        detector: EBSDDetector,
        rotations: Rotation,
        bands: Optional[KikuchiBand] = None,
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
        rotations
            Orientations of the unit cell.
        bands
            Kikuchi bands projected onto the detector. Default is None.
        zone_axes
            Zone axes projected onto the detector. Default is None.

        Returns
        -------
        GeometricalEBSDSimulation
        """
        self.detector = detector
        self.rotations = rotations
        self.bands = bands
        self.zone_axes = zone_axes

    @property
    def bands_detector_coordinates(self) -> np.ndarray:
        """Start and end point coordinates of bands in uncalibrated
        detector coordinates (a scale of 1 and offset of 0).

        Returns
        -------
        band_coords
            On the form [[x00, y00, x01, y01], [x10, y10, x11, y11],
            ...].
        """
        # Get start and end points for the plane traces in gnomonic coordinates
        # and set up output array in uncalibrated detector coordinates
        band_coords_gnomonic = self.bands.plane_trace_coordinates
        band_coords_detector = np.zeros_like(band_coords_gnomonic)

        # Get projection center coordinates, and add two axis to get the shape
        # (navigation shape, 1, 1)
        pcx = self.detector.pcx[..., np.newaxis, np.newaxis]
        pcy = self.detector.pcy[..., np.newaxis, np.newaxis]
        pcz = self.detector.pcz[..., np.newaxis, np.newaxis]

        # X and Y coordinates are now in place (0, 2) and (1, 3) respectively
        band_coords_detector[..., ::2] = (
            band_coords_gnomonic[..., :2] + (pcx / pcz)
        ) / self.detector.x_scale[..., np.newaxis, np.newaxis]
        band_coords_detector[..., 1::2] = (
            -band_coords_gnomonic[..., 2:] + (pcy / pcz)
        ) / self.detector.y_scale[..., np.newaxis, np.newaxis]

        return band_coords_detector

    @property
    def zone_axes_detector_coordinates(self) -> np.ndarray:
        """Coordinates of zone axes in uncalibrated detector
        coordinates.

        Returns
        -------
        zone_axes_coords
            Column sorted, on the form [[x0, y0], [x1, y1], ...].
        """
        x_gnomonic = self.zone_axes.x_gnomonic
        y_gnomonic = self.zone_axes.y_gnomonic
        size = x_gnomonic.size
        zone_axes_coords = np.zeros((size, 2), dtype=np.float32)
        pcx, pcy, pcz = self.detector.pc
        zone_axes_coords[:, 0] = (
            x_gnomonic + (pcx / pcz)
        ) / self.detector.x_scale
        zone_axes_coords[:, 1] = (
            -y_gnomonic + (pcy / pcz)
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
        zone_axes_coords[:, 1] -= 0.02 * self.detector.nrows
        return zone_axes_coords

    def bands_as_markers(self, **kwargs) -> list:
        """Return a list of Kikuchi band line segment markers.

        Parameters
        ----------
        kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.draw.markers.get_line_segment_list`.

        Returns
        -------
        list
        """
        if self.bands.navigation_shape == (1,):
            lines = np.squeeze(self.bands_detector_coordinates)
        else:
            lines = self.bands_detector_coordinates
        return get_line_segment_list(
            lines=lines,
            linewidth=kwargs.pop("linewidth", 2),
            color=kwargs.pop("color", "lime"),
            alpha=kwargs.pop("alpha", 0.7),
            zorder=kwargs.pop("zorder", 1),
            **kwargs,
        )

    def zone_axes_as_markers(self, **kwargs) -> list:
        """Return a list of zone axes point markers.

        Parameters
        ----------
        kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.draw.markers.get_point_list`.

        Returns
        -------
        list
        """
        return get_point_list(
            points=self.zone_axes_detector_coordinates,
            size=kwargs.pop("size", 40),
            marker=kwargs.pop("marker", "o"),
            facecolor=kwargs.pop("facecolor", "w"),
            edgecolor=kwargs.pop("edgecolor", "k"),
            zorder=kwargs.pop("zorder", 5),
            alpha=kwargs.pop("alpha", 0.7),
            **kwargs,
        )

    def zone_axes_labels_as_markers(self, **kwargs) -> list:
        """Return a list of zone axes label text markers.

        Parameters
        ----------
        kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.draw.markers.get_text_list`.

        Returns
        -------
        list
        """
        zone_axes = self.zone_axes[self.zone_axes.within_gnomonic_radius]
        return get_text_list(
            texts=sub("[][ ]", "", str(zone_axes._hkldata)).split("\n"),
            coordinates=self.zone_axes_label_detector_coordinates,
            color=kwargs.pop("color", "k"),
            zorder=kwargs.pop("zorder", 5),
            ha=kwargs.pop("ha", "center"),
            bbox=kwargs.pop(
                "bbox",
                dict(
                    facecolor="w",
                    edgecolor="k",
                    boxstyle="round, rounding_size=0.2",
                    pad=0.1,
                    alpha=0.7,
                ),
            ),
        )

    def pc_as_markers(self, **kwargs) -> list:
        """Return a list of projection center point markers.

        Parameters
        ----------
        kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.draw.markers.get_point_list`.

        Returns
        -------
        list
        """
        pcxy = self.detector.pc[..., :2]
        nrows, ncols = self.detector.shape
        x_scale = ncols - 1 if ncols > 1 else 1
        y_scale = nrows - 1 if nrows > 1 else 1
        pcxy[..., 0] *= x_scale
        pcxy[..., 1] *= y_scale
        return get_point_list(
            points=pcxy,
            size=kwargs.pop("size", 150),
            marker=kwargs.pop("marker", "*"),
            facecolor=kwargs.pop("facolor", "C1"),
            edgecolor=kwargs.pop("edgecolor", "k"),
            zorder=kwargs.pop("zorder", 6),
        )

    def as_markers(
        self,
        bands: bool = True,
        zone_axes: bool = True,
        zone_axes_labels: bool = True,
        pc: bool = True,
        bands_kwargs: Optional[dict] = None,
        zone_axes_kwargs: Optional[dict] = None,
        zone_axes_labels_kwargs: Optional[dict] = None,
        pc_kwargs: Optional[dict] = None,
    ) -> list:
        """Return a list of all or some of the simulation markers.

        Parameters
        ----------
        bands
            Whether to return band markers. Default is True.
        zone_axes
            Whether to return zone axes markers. Default is True.
        zone_axes_labels
            Whether to return zone axes label markers. Default is True.
        pc
            Whether to return projection center markers. Default is
            True.
        bands_kwargs
            Keyword arguments passed to
            :func:`kikuchipy.draw.markers.get_line_segment_list`.
        zone_axes_kwargs
            Keyword arguments passed to
            :func:`kikuchipy.draw.markers.get_point_list`.
        zone_axes_labels_kwargs
            Keyword arguments passed to
            :func:`kikuchipy.draw.markers.get_text_list`.
        pc_kwargs
            Keyword arguments passed to
            :func:`kikuchipy.draw.markers.get_point_list`.

        Returns
        -------
        markers
            A list with all markers.
        """
        markers = []
        if bands:
            if bands_kwargs is None:
                bands_kwargs = {}
            markers += self.bands_as_markers(**bands_kwargs)
        if zone_axes:
            if zone_axes_kwargs is None:
                zone_axes_kwargs = {}
            markers += self.zone_axes_as_markers(**zone_axes_kwargs)
        if zone_axes_labels:
            if zone_axes_labels_kwargs is None:
                zone_axes_labels_kwargs = {}
            markers += self.zone_axes_labels_as_markers(
                **zone_axes_labels_kwargs
            )
        if pc:
            if pc_kwargs is None:
                pc_kwargs = {}
            markers += self.pc_as_markers(**pc_kwargs)
        return markers

    def __repr__(self):
        rotation_repr = repr(self.rotations).split("\n")[0]
        band_repr = repr(self.bands).split("\n")[0]
        return (
            f"{self.__class__.__name__} {self.bands.navigation_shape}\n"
            f"{self.detector}\n"
            f"{self.bands.phase}\n"
            f"{band_repr}\n"
            f"{rotation_repr}\n"
        )
