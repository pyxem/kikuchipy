# Copyright 2019-2024 The kikuchipy developers
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

from copy import deepcopy
import re
from typing import Literal

from diffsims.crystallography import ReciprocalLatticeVector
import hyperspy.api as hs
import matplotlib.collections as mcollections
import matplotlib.figure as mfigure
import matplotlib.path as mpath
import matplotlib.text as mtext
import numpy as np
from orix.quaternion import Rotation

from kikuchipy.detectors.ebsd_detector import EBSDDetector
from kikuchipy.simulations._kikuchi_pattern_features import (
    KikuchiPatternLine,
    KikuchiPatternZoneAxis,
)

LINE_COLOR = "r"
ZONE_AXES_COLOR = "w"
ZONE_AXES_LABEL_COLOR = "k"


class GeometricalKikuchiPatternSimulation:
    """Collection of coordinates of Kikuchi lines and zone axes on an
    EBSD detector for simple plotting or creation of HyperSpy markers to
    plot onto :class:`~kikuchipy.signals.EBSD` signals.

    Instances of this class are returned from
    :meth:`kikuchipy.simulations.KikuchiPatternSimulator.on_detector`,
    and *not* ment to be created directly.

    Parameters
    ----------
    detector
        EBSD detector.
    rotations
        Crystal orientations for which coordinates of Kikuchi lines and
        zone axes have been generated.
    reflectors
        Reciprocal lattice vectors used in the simulation.
    lines
        Collection of coordinates of Kikuchi lines on the detector.
    zone_axes
        Collection of coordinates of zone axes on the detector.
    """

    def __init__(
        self,
        detector: EBSDDetector,
        rotations: Rotation,
        reflectors: ReciprocalLatticeVector,
        lines: KikuchiPatternLine,
        zone_axes: KikuchiPatternZoneAxis,
    ) -> None:
        self._detector = detector.deepcopy()
        self._rotations = deepcopy(rotations)
        self._reflectors = reflectors.deepcopy()
        self._lines = lines
        self._zone_axes = zone_axes
        self._set_lines_detector_coordinates()
        self._set_zone_axes_detector_coordinates()
        self.ndim = rotations.ndim

    # -------------------------- Properties -------------------------- #

    @property
    def detector(self) -> EBSDDetector:
        """Return the EBSD detector onto which simulations were
        generated.
        """
        return self._detector

    @property
    def rotations(self) -> Rotation:
        """Return the crystal orientations for which simulations were
        generated.
        """
        return self._rotations

    @property
    def reflectors(self) -> ReciprocalLatticeVector:
        """Return the reciprocal lattice vectors used in the
        simulations.
        """
        return self._reflectors

    @property
    def navigation_shape(self) -> tuple:
        """Return the navigation shape of the simulations, equal to the
        shape of :attr:`rotations`.
        """
        return self._rotations.shape

    # ------------------------ Dunder methods ------------------------ #

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.navigation_shape}:\n" + repr(
            self.reflectors
        )

    # ------------------------ Public methods ------------------------ #

    def as_collections(
        self,
        index: int | tuple[int, ...] | None = None,
        coordinates: Literal["detector", "gnomonic"] = "detector",
        lines: bool = True,
        zone_axes: bool = False,
        zone_axes_labels: bool = False,
        lines_kwargs: dict | None = None,
        zone_axes_kwargs: dict | None = None,
        zone_axes_labels_kwargs: dict | None = None,
    ) -> list:
        """Return a list of Matplotlib collections for a single
        simulation.

        Parameters
        ----------
        index
            Index of the simulation to get collections from. This is the
            first simulation if not given.
        coordinates
            Coordinate space for the plot axes, either "detector"
            (default) or "gnomonic".
        lines
            Whether to get the collection of Kikuchi lines. Default is
            True. Returned as
            :class:`matplotlib.collections.LineCollection`.
        zone_axes
            Whether to get the collection of zone axes. Default is
            False. Returned as
            :class:`matplotlib.collections.PathCollection`.
        zone_axes_labels
            Whether to get the collection of zone axes labels. Default
            is False. Return as a list of :class:`matplotlib.text.Text`.
        lines_kwargs
            Keyword arguments passed to
            :class:`matplotlib.collections.LineCollection` to format
            Kikuchi lines if *lines* True.
        zone_axes_kwargs
            Keyword arguments passed to
            :class:`matplotlib.collections.PathCollection` to format
            zone axes if *zone_axes* True.
        zone_axes_labels_kwargs
            Keyword arguments passed to :class:`matplotlib.text.Text` to
            format zone axes labels if *zone_axes_labels* True.

        Returns
        -------
        collection_list
            List of Matplotlib collections.

        See Also
        --------
        as_markers, plot
        """
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

    def as_markers(
        self,
        lines: bool = True,
        zone_axes: bool = False,
        zone_axes_labels: bool = False,
        pc: bool = False,
        lines_kwargs: dict | None = None,
        zone_axes_kwargs: dict | None = None,
        zone_axes_labels_kwargs: dict | None = None,
        pc_kwargs: dict | None = None,
    ) -> list:
        """Return a list of simulation markers.

        Parameters
        ----------
        lines
            Whether to return Kikuchi line markers. Default is True.
        zone_axes
            Whether to return zone axes markers. Default is False.
        zone_axes_labels
            Whether to return zone axes label markers. Default is False.
        pc
            Whether to return projection center (PC) markers. Default is
            False.
        lines_kwargs
            Keyword arguments passed to
            :func:`~matplotlib.pyplot.axvline` to format the lines.
        zone_axes_kwargs
            Keyword arguments passed to
            :func:`~matplotlib.pyplot.scatter` to format the zone axes
            markers.
        zone_axes_labels_kwargs
            Keyword arguments passed to :func:`~matplotlib.text.Text` to
            format the zone axes labels.
        pc_kwargs
            Keyword arguments passed to
            :func:`~matplotlib.pyplot.scatter` to format the PC markers.

        Returns
        -------
        markers
            List with all markers.

        See Also
        --------
        as_collections, plot
        """
        markers = []
        if lines:
            if lines_kwargs is None:
                lines_kwargs = {}
            markers.append(self._lines_as_markers(**lines_kwargs))
        if zone_axes:
            if zone_axes_kwargs is None:
                zone_axes_kwargs = {}
            markers.append(self._zone_axes_as_markers(**zone_axes_kwargs))
        if zone_axes_labels:
            if zone_axes_labels_kwargs is None:
                zone_axes_labels_kwargs = {}
            markers.append(self._zone_axes_labels_as_markers(**zone_axes_labels_kwargs))
        if pc:
            if pc_kwargs is None:
                pc_kwargs = {}
            markers.append(self._pc_as_markers(**pc_kwargs))
        return markers

    def lines_coordinates(
        self,
        index: int | tuple | None = None,
        coordinates: Literal["detector", "gnomonic"] = "detector",
        exclude_nan: bool = True,
    ) -> np.ndarray:
        """Return Kikuchi line coordinates for a single simulation.

        Parameters
        ----------
        index
            Index of the simulation to get line coordinates for. This is
            the first simulation if not given.
        coordinates
            Coordinate space, either "detector" (default) or "gnomonic".
        exclude_nan
            Whether to exclude coordinates of Kikuchi lines not present
            in the pattern. Default is True. If False, all simulations
            (by varying *index*) return an array of the same shape.

        Returns
        -------
        coords
            Kikuchi line coordinates.

        See Also
        --------
        zone_axes_coordinates
        """
        if index is None:
            index = (0, 0)[: self.ndim]
        if coordinates == "detector":
            coords = self._lines_detector_coordinates[index]
        else:  # gnomonic
            coords = self._lines.plane_trace_coordinates[index]
        if exclude_nan:
            coords = coords[~np.isnan(coords).any(axis=-1)]
        return coords.copy()

    def plot(
        self,
        index: int | tuple | None = None,
        coordinates: Literal["detector", "gnomonic"] = "detector",
        pattern: np.ndarray | None = None,
        lines: bool = True,
        zone_axes: bool = True,
        zone_axes_labels: bool = True,
        pc: bool = True,
        pattern_kwargs: dict | None = None,
        lines_kwargs: dict | None = None,
        zone_axes_kwargs: dict | None = None,
        zone_axes_labels_kwargs: dict | None = None,
        pc_kwargs: dict | None = None,
        return_figure: bool = False,
    ) -> mfigure.Figure | None:
        """Plot a single simulation on the detector.

        Parameters
        ----------
        index
            Index of the simulation to plot. This is the first
            simulation if not given. Must be a 2-tuple if
            :attr:`navigation_shape` is 2D.
        coordinates
            Coordinate space of the plot axes, either "detector"
            (default) or "gnomonic".
        pattern
            Pattern to plot the simulation onto. The simulation is
            plotted on a gray background if not given.
        lines
            Whether to show Kikuchi lines. Default is True.
        zone_axes
            Whether to show zone axes. Default is True.
        zone_axes_labels
            Whether to show zone axes labels. Default is True.
        pc
            Whether to show the projection/pattern centre (PC). Default
            is True.
        pattern_kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.imshow` if a *pattern* is given.
        lines_kwargs
            Keyword arguments passed to
            :class:`matplotlib.collections.LineCollection` if *lines* is
            True.
        zone_axes_kwargs
            Keyword arguments passed to
            :class:`matplotlib.collections.PathCollection` if
            *zone_axes* is True.
        zone_axes_labels_kwargs
            Keyword arguments passed to :class:`matplotlib.text.Text` if
            *zone_axes_labels* is True.
        pc_kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.scatter` if *pc* is True.
        return_figure
            Whether to return the figure. Default is False.

        Returns
        -------
        fig
            Returned if *return_figure* is True.

        See Also
        --------
        as_collections, as_markers
        """
        fig = self.detector.plot(
            coordinates=coordinates,
            pattern=pattern,
            show_pc=pc,
            pc_kwargs=pc_kwargs,
            pattern_kwargs=pattern_kwargs,
            return_figure=True,
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
        ax = fig.axes[0]
        for c in collections:
            if isinstance(c, list) and isinstance(c[0], mtext.Text):
                for text_i in c:
                    ax.add_artist(text_i)
            else:
                ax.add_collection(c)
        if return_figure:
            return fig

    def zone_axes_coordinates(
        self,
        index: int | tuple | None = None,
        coordinates: Literal["detector", "gnomonic"] = "detector",
        exclude_nan: bool = True,
    ) -> np.ndarray:
        """Return zone axes coordinates for a single simulation.

        Parameters
        ----------
        index
            Index of the simulation to get zone axis coordinates for.
            This is the first simulation if not given.
        coordinates
            Coordinate space, either "detector" (default) or "gnomonic".
        exclude_nan
            Whether to exclude coordinates of zone axes not present in
            the pattern. Default is True. If False, all simulations (by
            varying *index*) return an array of the same shape.

        Returns
        -------
        coords
            Zone axes coordinates.

        See Also
        --------
        lines_coordinates
        """
        if index is None:
            index = (0, 0)[: self.ndim]
        if coordinates == "detector":
            coords = self._zone_axes_detector_coordinates[index]
        else:  # gnomonic
            coords = self._zone_axes._xy_within_r_gnomonic[index]
        if exclude_nan:
            coords = coords[~np.isnan(coords).any(axis=-1)]
        return coords.copy()

    # ------------------------ Private methods ----------------------- #

    def _set_lines_detector_coordinates(self) -> None:
        """Set the start and end point coordinates of bands in
        uncalibrated detector coordinates (a scale of 1 and offset of
        0).
        """
        # Get PC coordinates and add two axes to get the shape
        # (navigation shape, 1, 1)
        det = self.detector
        pcx = det.pcx[..., None, None]
        pcy = det.pcy[..., None, None]
        pcz = det.pcz[..., None, None]

        # Convert coordinates
        coords_d = self._lines.plane_trace_coordinates.copy()
        coords_d[..., [0, 2]] = coords_d[..., [0, 2]] + (pcx / pcz) * det.aspect_ratio
        coords_d[..., [0, 2]] = coords_d[..., [0, 2]] / det.x_scale[..., None, None]
        coords_d[..., [1, 3]] = -coords_d[..., [1, 3]] + (pcy / pcz)
        coords_d[..., [1, 3]] = coords_d[..., [1, 3]] / det.y_scale[..., None, None]

        self._lines_detector_coordinates = coords_d

    def _set_zone_axes_detector_coordinates(self) -> None:
        """Set the coordinates of zone axes in uncalibrated detector
        coordinates (a scale of 1 and offset of 0) inside the gnomonic
        bounds of the detector.
        """
        xyg = self._zone_axes._xy_within_r_gnomonic
        xg = xyg[..., 0]
        yg = xyg[..., 1]
        coords_d = np.zeros_like(xyg)

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

        # Set those outside gnomonic bounds in each pattern to None
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
        # Add an extra dimension to account for n number of zone axes in
        # the last dimension for the gnomonic coordinate arrays
        x_range = np.expand_dims(x_range, axis=-2)
        y_range = np.expand_dims(y_range, axis=-2)
        # Get boolean array
        within_x = np.logical_and(xg >= x_range[..., 0], xg <= x_range[..., 1])
        within_y = np.logical_and(yg >= y_range[..., 0], yg <= y_range[..., 1])
        within_gnomonic_bounds = np.logical_and(within_x, within_y)

        coords_d[~within_gnomonic_bounds] = np.nan

        self._zone_axes_detector_coordinates = coords_d

    def _lines_as_collection(
        self,
        index: int | tuple[int, ...] | None,
        coordinates: Literal["detector", "gnomonic"],
        **kwargs,
    ) -> mcollections.LineCollection:
        coords = self.lines_coordinates(index, coordinates)
        coords = coords.reshape((coords.shape[0], 2, 2))
        kw = {
            "color": LINE_COLOR,
            "linewidth": 1,
            "alpha": 1,
            "zorder": 1,
            "label": "kikuchi_lines",
        }
        kw.update(kwargs)
        return mcollections.LineCollection(segments=coords, **kw)

    def _zone_axes_as_collection(
        self,
        index: int | tuple[int, ...] | None,
        coordinate_fmt: Literal["detector", "gnomonic"],
        **kwargs,
    ) -> mcollections.PathCollection:
        coords = self.zone_axes_coordinates(index, coordinate_fmt)
        offset = 0.01
        if coordinate_fmt == "detector":
            scatter_size = offset * self.detector.nrows
        else:  # gnomonic
            scatter_size = offset * np.diff(self.detector.x_range)[0]
        circles = []
        for x, y in coords:
            circle = mpath.Path.circle((x, y), scatter_size)
            circles.append(circle)
        kw = {"ec": "k", "fc": ZONE_AXES_COLOR, "zorder": 1, "label": "zone_axes"}
        kw.update(kwargs)
        return mcollections.PathCollection(circles, **kw)

    def _zone_axes_labels_as_list(
        self,
        index: int | tuple[int, ...] | None,
        coordinates: Literal["detector", "gnomonic"],
        **kwargs,
    ) -> list[mtext.Text]:
        labels = self._zone_axes_labels_as_array().tolist()
        coords = self.zone_axes_coordinates(index, coordinates, exclude_nan=False)
        y_offset = 0.03
        if coordinates == "detector":
            coords[..., 1] -= y_offset * self.detector.nrows
        else:  # gnomonic
            coords[..., 1] += y_offset * np.diff(self.detector.y_range)[0]
        kw = {
            "color": ZONE_AXES_LABEL_COLOR,
            "horizontalalignment": "center",
            "bbox": {"boxstyle": "square", "fc": "w", "pad": 0.1},
        }
        kw.update(kwargs)
        texts = []
        for (x, y), label in zip(coords, labels):
            if ~np.isnan(x):
                text_i = mtext.Text(x, y, label, **kw)
                texts.append(text_i)
        return texts

    def _lines_as_markers(self, **kwargs) -> hs.plot.markers.Lines:
        coords = self.lines_coordinates(index=(), exclude_nan=False)
        nav_shape = self.navigation_shape
        coords = coords.reshape(*nav_shape, -1, 2, 2)
        if nav_shape == (1,):
            segments = coords.reshape(-1, 2, 2)
        else:
            segments = np.empty(nav_shape[::-1], dtype=object)
            keep = ~np.isnan(coords[..., 0, 0])
            for idx in np.ndindex(segments.shape):
                idx_rc = idx[::-1]
                segments[idx] = coords[idx_rc][keep[idx_rc]]
        kw = {"colors": LINE_COLOR, "zorder": 1}
        kw.update(kwargs)
        markers = hs.plot.markers.Lines(segments, **kw)
        return markers

    def _pc_as_markers(self, **kwargs) -> hs.plot.markers.Markers:
        kw = {"sizes": 300, "fc": "gold", "ec": "k", "zorder": 4}
        kw.update(kwargs)
        marker = hs.plot.markers.Markers(
            collection=mcollections.StarPolygonCollection,
            offsets=self._pc_xy_offsets(),
            numsides=5,
            **kw,
        )
        return marker

    def _pc_xy_offsets(self) -> np.ndarray:
        if self.detector.navigation_shape == self.navigation_shape != (1,):
            return self._pc_xy_offsets_multiple()
        else:
            return self._pc_xy_offsets_single()

    def _pc_xy_offsets_single(self) -> np.ndarray:
        pc = self.detector.pc_average[:2].copy()
        for i, shape in enumerate(self.detector.shape[::-1]):
            if shape > 1:
                pc[..., i] *= shape - 1
        return pc

    def _pc_xy_offsets_multiple(self) -> np.ndarray:
        pc = self.detector.pc[..., :2].copy()
        for i, shape in enumerate(self.detector.shape[::-1]):
            if shape > 1:
                pc[..., i] *= shape - 1
        pc_object_arr = np.empty(self.navigation_shape[::-1], dtype=object)
        for idx in np.ndindex(pc_object_arr.shape):
            pc_object_arr[idx] = pc[idx[::-1]]
        return pc_object_arr

    def _zone_axes_as_markers(self, **kwargs) -> hs.plot.markers.Lines:
        coords = self.zone_axes_coordinates(index=(), exclude_nan=False)
        nav_shape = self.navigation_shape
        if nav_shape == (1,):
            offsets = coords.reshape(-1, 2)
        else:
            offsets = np.empty(nav_shape[::-1], dtype=object)
            keep = ~np.isnan(coords[..., 0])
            for idx in np.ndindex(offsets.shape):
                idx_rc = idx[::-1]
                offsets[idx] = coords[idx_rc][keep[idx_rc]]
        kw = {"fc": ZONE_AXES_COLOR, "ec": "none", "zorder": 2}
        kw.update(kwargs)
        markers = hs.plot.markers.Points(offsets, **kw)
        return markers

    def _zone_axes_labels_as_array(self) -> np.ndarray:
        uvw = self._zone_axes.vector.coordinates.round().astype(int)
        uvw_str = np.array2string(uvw, threshold=uvw.size)
        texts = re.sub("[][ ]", "", uvw_str).split("\n")
        texts = np.asanyarray(texts)
        return texts

    def _zone_axes_labels_as_markers(self, **kwargs) -> hs.plot.markers.Texts:
        coords = self.zone_axes_coordinates(index=(), exclude_nan=False)
        labels = self._zone_axes_labels_as_array()
        nav_shape = self.navigation_shape
        if nav_shape == (1,):
            offsets = coords.reshape(-1, 2)
            texts = labels
        else:
            offsets = np.empty(nav_shape[::-1], dtype=object)
            texts = np.empty_like(offsets)
            keep = ~np.isnan(coords[..., 0])
            for idx in np.ndindex(offsets.shape):
                idx_rc = idx[::-1]
                keep_i = keep[idx_rc]
                offsets[idx] = coords[idx_rc][keep_i]
                texts[idx] = labels[keep_i]
        kw = {
            "color": ZONE_AXES_LABEL_COLOR,
            "zorder": 3,
            "horizontalalignment": "center",
            "verticalalignment": "bottom",
            # TODO: Uncomment once supported by HyperSpy again
            # "bbox": {"fc": "w", "ec": "k", "boxstyle": "square", "pad": 0.2},
        }
        kw.update(kwargs)
        marker = hs.plot.markers.Texts(offsets, texts=texts, **kw)
        return marker
