# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
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

import logging
from re import sub
from typing import List, Optional

import numpy as np
from orix.quaternion.rotation import Rotation

from kikuchipy.crystallography._computations import (
    _get_hkl_family,
    _get_colors_for_allowed_bands,
    _is_equivalent,
)
from kikuchipy.detectors import EBSDDetector
from kikuchipy.draw.markers import (
    get_line_segment_list,
    get_point_list,
    get_text_list,
)
from kikuchipy.simulations.features import KikuchiBand, ZoneAxis


class DisableMatplotlibWarningFilter(logging.Filter):
    def filter(self):
        message_to_disable = "posx and posy should be finite values"
        return not self.msg == message_to_disable


logging.getLogger("matplotlib.text").addFilter(DisableMatplotlibWarningFilter)


class GeometricalEBSDSimulation:
    """Geometrical EBSD simulation with Kikuchi bands and zone axes."""

    exclude_outside_detector = True

    def __init__(
        self,
        detector: EBSDDetector,
        rotations: Rotation,
        bands: KikuchiBand,
        zone_axes: ZoneAxis,
    ):
        """Create a geometrical EBSD simulation storing a set of center
        positions of Kikuchi bands and zone axes on the detector, one
        set for each orientation of the unit cell.

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
        band_coords_detector : numpy.ndarray
            Band coordinates (y0, x0, y1, x1) on the detector.
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
        za_coords = np.zeros_like(xyg)

        # Get projection center coordinates, and add one axis to get the
        # shape (navigation shape, 1)
        pcx = self.detector.pcx[..., np.newaxis]
        pcy = self.detector.pcy[..., np.newaxis]
        pcz = self.detector.pcz[..., np.newaxis]

        za_coords[..., 0] = (xg + (pcx / pcz)) / self.detector.x_scale[
            ..., np.newaxis
        ]
        za_coords[..., 1] = (-yg + (pcy / pcz)) / self.detector.y_scale[
            ..., np.newaxis
        ]

        if self.exclude_outside_detector:
            on_detector = self.zone_axes_within_gnomonic_bounds
            za_coords[~on_detector] = np.nan

        return za_coords

    @property
    def zone_axes_label_detector_coordinates(self) -> np.ndarray:
        """Coordinates of zone axes labels in uncalibrated detector
        coordinates (a scale of 1 and offset of 0).

        Returns
        -------
        za_coords : numpy.ndarray
            Zone axes labels (x, y) placed just above the zone axes on
            the detector.
        """
        za_coords = self.zone_axes_detector_coordinates
        za_coords[..., 1] -= 0.02 * self.detector.nrows
        return za_coords

    def bands_as_markers(
        self, family_colors: Optional[List[str]] = None, **kwargs
    ) -> list:
        """Return a list of Kikuchi band line segment markers.

        Parameters
        ----------
        family_colors
            A list of at least as many colors as unique HKL families,
            either as RGB iterables or colors recognizable by
            Matplotlib, used to color each unique family of bands. If
            None (default), this is determined from a list similar to
            the one used in EDAX TSL's software.
        kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.draw.markers.get_line_segment_list`.

        Returns
        -------
        list
            List with line segment markers.
        """
        if self.bands.navigation_shape == (1,):
            lines = np.squeeze(self.bands_detector_coordinates)
        else:
            lines = self.bands_detector_coordinates

        # Get dictionaries of families and in which a band belongs
        families, families_idx = _get_hkl_family(self.bands.hkl.data)

        # Get family colors
        # TODO: Perhaps move this outside this function (might be useful
        #  elsewhere)
        if family_colors is None:
            family_colors = []
            colors = _get_colors_for_allowed_bands(
                phase=self.bands.phase,
                highest_hkl=np.max(np.abs(self.bands.hkl.data), axis=0),
            )
            for hkl in families.keys():
                for table_hkl, color in colors:
                    if _is_equivalent(hkl, table_hkl):
                        family_colors.append(color)
                        break
        #                else:  # Hopefully we never arrive here
        #                    family_colors.append([1, 0, 0])

        # Append list of markers per family (colors changing with
        # family)
        marker_list = []
        for i, idx in enumerate(families_idx.values()):
            marker_list += get_line_segment_list(
                lines=lines[..., idx, :],
                linewidth=kwargs.pop("linewidth", 1),
                color=family_colors[i],
                alpha=kwargs.pop("alpha", 1),
                zorder=kwargs.pop("zorder", 1),
                **kwargs,
            )
        return marker_list

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
            List with point markers.
        """
        # TODO: Give them some descriptive colors (facecolor)!
        # TODO: Marker style based on symmetry (2, 3, 4 and 6-fold):
        #  https://matplotlib.org/3.3.2/api/markers_api.html#module-matplotlib.markers
        return get_point_list(
            points=self.zone_axes_detector_coordinates,
            size=kwargs.pop("size", 40),
            marker=kwargs.pop("marker", "o"),
            facecolor=kwargs.pop("facecolor", "w"),
            edgecolor=kwargs.pop("edgecolor", "k"),
            zorder=kwargs.pop("zorder", 5),
            alpha=kwargs.pop("alpha", 1),
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
            List of text markers.
        """
        return get_text_list(
            texts=sub("[][ ]", "", str(self.zone_axes.hkl.data)).split("\n"),
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
                    alpha=1,
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
            List of point markers.
        """
        # Set up (x, y) detector coordinate array of final shape
        # nav_shape + (n_patterns, 2)
        nav_shape = self.bands.navigation_shape
        n = int(np.prod(nav_shape))  # Number of patterns
        pcxy = np.ones((n, n, 2)) * np.nan
        i = np.arange(n)
        pcxy[i, i, :2] = self.detector.pc[..., :2].reshape((n, 2))
        pcxy = pcxy.reshape(nav_shape + (n, 2))

        nrows, ncols = self.detector.shape
        x_scale = ncols - 1 if ncols > 1 else 1
        y_scale = nrows - 1 if nrows > 1 else 1
        pcxy[..., 0] *= x_scale
        pcxy[..., 1] *= y_scale
        return get_point_list(
            points=pcxy,
            size=kwargs.pop("size", 150),
            marker=kwargs.pop("marker", "*"),
            facecolor=kwargs.pop("facecolor", "C1"),
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
        markers : hyperspy.drawing.marker.MarkerBase
            List with all markers.
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
        # Get gnomonic bounds
        x_range = self.detector.x_range
        y_range = self.detector.y_range

        # Extend gnomonic bounds by one detector pixel to include zone
        # axes on the detector border
        x_scale = self.detector.x_scale
        y_scale = self.detector.y_scale
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

        return within_gnomonic_bounds.reshape(self.zone_axes._data_shape)

    def __repr__(self):
        rotation_repr = repr(self.rotations).split("\n")[0]
        band_repr = repr(self.bands).split("\n")[0]
        return (
            f"{self.__class__.__name__} {self.bands.navigation_shape}\n"
            f"{self.detector}\n"
            f"{self.bands.phase}\n"
            f"{band_repr}\n"
            f"{rotation_repr}"
        )
