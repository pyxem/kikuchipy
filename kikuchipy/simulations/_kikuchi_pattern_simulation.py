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

from copy import deepcopy
import re
from typing import List, Optional, Union

from diffsims.crystallography import ReciprocalLatticeVector
from hyperspy.drawing.marker import MarkerBase
from hyperspy.utils.markers import line_segment, point, text
import matplotlib.collections as mcollections
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
from orix.quaternion import Rotation

from kikuchipy.detectors import EBSDDetector
from kikuchipy.simulations._kikuchi_pattern_features import (
    KikuchiPatternLine,
    KikuchiPatternZoneAxis,
)


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
    ):
        self._detector = detector.deepcopy()
        self._rotations = deepcopy(rotations)
        self._reflectors = reflectors.deepcopy()
        self._lines = lines
        self._zone_axes = zone_axes
        self._set_lines_detector_coordinates()
        self._set_zone_axes_detector_coordinates()
        self.ndim = rotations.ndim

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

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__} {self.navigation_shape}:\n" + repr(
            self.reflectors
        )

    def as_collections(
        self,
        index: Union[int, tuple, None] = None,
        coordinates: str = "detector",
        lines: bool = True,
        zone_axes: bool = False,
        zone_axes_labels: bool = False,
        lines_kwargs: dict = None,
        zone_axes_kwargs: dict = None,
        zone_axes_labels_kwargs: dict = None,
    ) -> list:
        """Get a single simulation as a list of Matplotlib objects.

        Parameters
        ----------
        index
            Index of the simulation to get collections from. If not
            given, this is the first simulation.
        coordinates
            The coordinates of the plot axes, either ``"detector"``
            (default) or ``"gnomonic"``.
        lines
            Whether to get the collection of Kikuchi lines. Default is
            ``True``. These are returned as
            :class:`matplotlib.collections.LineCollection`.
        zone_axes
            Whether to get the collection of zone axes. Default is
            ``False``. These are returned as
            :class:`matplotlib.collections.PathCollection`.
        zone_axes_labels
            Whether to get the collection of zone axes labels. Default
            is ``False``. These are returned as a class:`list` of
            :class:`matplotlib.text.Text`.
        lines_kwargs
            Keyword arguments passed to
            :class:`matplotlib.collections.LineCollection` to format
            Kikuchi lines if ``lines=True``.
        zone_axes_kwargs
            Keyword arguments passed to
            :class:`matplotlib.collections.PathCollection` to format
            zone axes if ``zone_axes=True``.
        zone_axes_labels_kwargs
            Keyword arguments passed to :class:`matplotlib.text.Text` to
            format zone axes labels if ``zone_axes_labels=True``.

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
        lines_kwargs: Optional[dict] = None,
        zone_axes_kwargs: Optional[dict] = None,
        zone_axes_labels_kwargs: Optional[dict] = None,
        pc_kwargs: Optional[dict] = None,
    ) -> List[MarkerBase]:
        """Return a list of simulation markers.

        Parameters
        ----------
        lines
            Whether to return Kikuchi line markers. Default is ``True``.
        zone_axes
            Whether to return zone axes markers. Default is ``False``.
        zone_axes_labels
            Whether to return zone axes label markers. Default is
            ``False``.
        pc
            Whether to return projection center (PC) markers. Default is
            ``False``.
        lines_kwargs
            Keyword arguments passed to
            :func:`~matplotlib.pyplot.axvline` to format the lines.
        zone_axes_kwargs
            Keyword arguments passed to
            :func:`~matplotlib.pyplot.scatter` to format the markers.
        zone_axes_labels_kwargs
            Keyword arguments passed to :func:`~matplotlib.text.Text` to
            format the labels.
        pc_kwargs
            Keyword arguments passed to
            :func:`~matplotlib.pyplot.scatter` to format the markers.

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
            markers += self._lines_as_markers(**lines_kwargs)
        if zone_axes:
            if zone_axes_kwargs is None:
                zone_axes_kwargs = {}
            markers += self._zone_axes_as_markers(**zone_axes_kwargs)
        if zone_axes_labels:
            if zone_axes_labels_kwargs is None:
                zone_axes_labels_kwargs = {}
            markers += self._zone_axes_labels_as_markers(**zone_axes_labels_kwargs)
        if pc:
            if pc_kwargs is None:
                pc_kwargs = {}
            markers += self._pc_as_markers(**pc_kwargs)
        return markers

    def lines_coordinates(
        self,
        index: Union[int, tuple, None] = None,
        coordinates: str = "detector",
        exclude_nan: bool = True,
    ) -> np.ndarray:
        """Get Kikuchi line coordinates of a single simulation.

        Parameters
        ----------
        index
            Index of the simulation to get line coordinates for. If not
            given, this is the first simulation.
        coordinates
            The type of coordinates, either ``"detector"`` (default) or
            ``"gnomonic"``.
        exclude_nan
            Whether to exclude coordinates of Kikuchi lines not present
            in the pattern. Default is ``True``. By passing ``False``,
            all simulations (by varying ``index``) returns an array of
            the same shape.

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
        index: Union[int, tuple, None] = None,
        coordinates: str = "detector",
        pattern: Optional[np.ndarray] = None,
        lines: bool = True,
        zone_axes: bool = True,
        zone_axes_labels: bool = True,
        pc: bool = True,
        pattern_kwargs: Optional[dict] = None,
        lines_kwargs: Optional[dict] = None,
        zone_axes_kwargs: Optional[dict] = None,
        zone_axes_labels_kwargs: Optional[dict] = None,
        pc_kwargs: Optional[dict] = None,
        return_figure: bool = False,
    ) -> plt.Figure:
        """Plot a single simulation on the detector.

        Parameters
        ----------
        index
            Index of the simulation to plot. If not given, this is the
            first simulation. If :attr:`navigation_shape` is 2D, and
            ``index`` is passed, it must be a 2-tuple.
        coordinates
            The coordinates of the plot axes, either ``"detector"``
            (default) or ``"gnomonic"``.
        pattern
            A pattern to plot the simulation onto. If not given, the
            simulation is plotted on a gray background.
        lines
            Whether to show Kikuchi lines. Default is ``True``.
        zone_axes
            Whether to show zone axes. Default is ``True``.
        zone_axes_labels
            Whether to show zone axes labels. Default is ``True``.
        pc
            Whether to show the projection/pattern centre (PC). Default
            is ``True``.
        pattern_kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.imshow` if ``pattern`` is given.
        lines_kwargs
            Keyword arguments passed to
            :class:`matplotlib.collections.LineCollection` to format
            Kikuchi lines if ``lines=True``.
        zone_axes_kwargs
            Keyword arguments passed to
            :class:`matplotlib.collections.PathCollection` to format
            zone axes if ``zone_axes=True``.
        zone_axes_labels_kwargs
            Keyword arguments passed to :class:`matplotlib.text.Text` to
            format zone axes labels if ``zone_axes_labels=True``.
        pc_kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.scatter` to format the PC if
            ``pc=True``.
        return_figure
            Whether to return the figure. Default is ``False``.

        Returns
        -------
        fig
            Returned if ``return_figure=True``.

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
        index: Union[int, tuple, None] = None,
        coordinates: str = "detector",
        exclude_nan: bool = True,
    ) -> np.ndarray:
        """Get zone axis coordinates of a single simulation.

        Parameters
        ----------
        index
            Index of the simulation to get zone axis coordinates for. If
            not given, this is the first simulation.
        coordinates
            The type of coordinates, either ``"detector"`` (default) or
            ``"gnomonic"``.
        exclude_nan
            Whether to exclude coordinates of zone axes not present in
            the pattern. Default is ``True``. By passing ``False``, all
            simulations (by varying ``index``) returns an array of
            the same shape.

        Returns
        -------
        coords
            Zone axis coordinates.

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

    def _lines_as_collection(
        self, index: Union[int, tuple], coordinates: str, **kwargs
    ) -> mcollections.LineCollection:
        """Get Kikuchi lines as a Matplotlib collection.

        Parameters
        ----------
        index
            Index of the simulation to get collections from. If not
            given, this is the first simulation.
        coordinates
            The coordinates of the lines, either ``"detector"``
            (default) or ``"gnomonic"``.
        **kwargs
            Keyword arguments passed to
            :class:`~matplotlib.collections.LineCollection` to format
            Kikuchi lines.

        Returns
        -------
        collection
            Collection of lines.
        """
        coords = self.lines_coordinates(index, coordinates)
        coords = coords.reshape((coords.shape[0], 2, 2))
        line_defaults = dict(
            color="r", linewidth=1, alpha=1, zorder=1, label="kikuchi_lines"
        )
        for k, v in line_defaults.items():
            kwargs.setdefault(k, v)
        return mcollections.LineCollection(segments=list(coords), **kwargs)

    def _lines_as_markers(self, **kwargs) -> List[line_segment]:
        """Get Kikuchi lines as a list of HyperSpy markers.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to
            :func:`~matplotlib.pyplot.axvline` to format the lines.

        Returns
        -------
        lines_list
            List with line segment markers.
        """
        coords = self.lines_coordinates(index=(), exclude_nan=False)
        lines_list = []
        segment_defaults = dict(color="r", zorder=1)
        for k, v in segment_defaults.items():
            kwargs.setdefault(k, v)

        for i in range(self._lines.vector.size):
            line = coords[..., i, :]
            if not np.all(np.isnan(line)):
                # TODO: Inefficient, squeeze before the loop if possible
                x1 = line[..., 0].squeeze()
                y1 = line[..., 1].squeeze()
                x2 = line[..., 2].squeeze()
                y2 = line[..., 3].squeeze()
                marker = line_segment(x1=x1, y1=y1, x2=x2, y2=y2, **kwargs)
                lines_list.append(marker)

        return lines_list

    def _pc_as_markers(self, **kwargs) -> list:
        """Return a list of projection center (PC) point markers.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to
            :func:`~matplotlib.pyplot.scatter` to format the markers.

        Returns
        -------
        pc_marker
            List with a single PC marker.
        """
        det = self.detector
        if det.navigation_shape == self.navigation_shape:
            pcx = det.pc[..., 0]
            pcy = det.pc[..., 1]
        else:
            pcx1, pcy1 = det.pc_average[:2]
            pcx = np.full(self.navigation_shape, pcx1)
            pcy = np.full(self.navigation_shape, pcy1)

        if pcx.shape[0] == 1:
            pcx = pcx.squeeze()
            pcy = pcy.squeeze()

        nrows, ncols = det.shape
        if nrows > 1:
            pcy *= nrows - 1
        if ncols > 1:
            pcx *= ncols - 1

        for k, v in dict(size=300, marker="*", fc="gold", ec="k", zorder=4).items():
            kwargs.setdefault(k, v)

        pc_marker = point(x=pcx, y=pcy, **kwargs)

        return [pc_marker]

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

    def _zone_axes_as_collection(
        self, index: Union[int, tuple], coordinates: str, **kwargs
    ) -> mcollections.PathCollection:
        """Get zone axes as a Matplotlib collection.

        Parameters
        ----------
        index
            Index of the simulation to get collections from. If not
            given, this is the first simulation.
        coordinates
            The coordinates of the plot axes, either ``"detector"``
            (default) or ``"gnomonic"``.
        **kwargs
            Keyword arguments passed to
            :class:`~matplotlib.collections.PathCollection` to format
            zone axes.

        Returns
        -------
        collection
            Collection of zone axes.
        """
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
        """Get zone axes labels as a list of texts.

        Parameters
        ----------
        index
            Index of the simulation to get labels from. If not
            given, this is the first simulation.
        coordinates
            The coordinates of the zone axes labels, either
            ``"detector"`` (default) or ``"gnomonic"``.
        **kwargs
            Keyword arguments passed to :class:`~matplotlib.text.Text`
            to format zone axes labels.

        Returns
        -------
        texts
            List of zone axes labels.
        """
        za = self._zone_axes
        za_labels = za.vector.coordinates.round(0).astype(np.int64)
        za_labels_str = np.array2string(za_labels, threshold=za_labels.size)
        za_labels_list = re.sub("[][ ]", "", za_labels_str[1:-1]).split("\n")
        xy = self.zone_axes_coordinates(index, coordinates, exclude_nan=False)
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

    def _zone_axes_as_markers(self, **kwargs) -> list:
        """Return a list of zone axes point markers.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to
            :func:`~matplotlib.pyplot.scatter` to format the markers.

        Returns
        -------
        zone_axes_list
            List with zone axes markers.
        """
        coords = self.zone_axes_coordinates(index=(), exclude_nan=False)
        zone_axes_list = []

        for k, v in dict(ec="none", zorder=2).items():
            kwargs.setdefault(k, v)

        for i in range(self._zone_axes.vector.size):
            # TODO: Inefficient, squeeze before the loop if possible
            zone_axis = coords[..., i, :].squeeze()
            if not np.all(np.isnan(zone_axis)):
                marker = point(x=zone_axis[..., 0], y=zone_axis[..., 1], **kwargs)
                zone_axes_list.append(marker)

        return zone_axes_list

    def _zone_axes_labels_as_markers(self, **kwargs) -> list:
        """Return a list of zone axes label text markers.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to :func:`~matplotlib.text.Text` to
            format the labels.

        Returns
        -------
        zone_axes_label_list
            List of text markers.
        """
        coords = self.zone_axes_coordinates(index=(), exclude_nan=False)

        zone_axes = self._zone_axes.vector.coordinates.round(0).astype(np.int64)
        array_str = np.array2string(zone_axes, threshold=zone_axes.size)
        texts = re.sub("[][ ]", "", array_str).split("\n")

        for k, v in dict(
            color="k",
            zorder=3,
            ha="center",
            va="bottom",
            bbox=dict(fc="w", ec="k", boxstyle="square", pad=0.2),
        ).items():
            kwargs.setdefault(k, v)

        zone_axes_label_list = []
        is_finite = np.isfinite(coords)[..., 0]
        coords[~is_finite] = -1

        for i in range(zone_axes.shape[0]):
            if not np.allclose(coords[..., i, :], -1):  # All NaNs
                x = coords[..., i, 0]
                y = coords[..., i, 1]
                x[~is_finite[..., i]] = np.nan
                y[~is_finite[..., i]] = np.nan
                # TODO: Inefficient, squeeze before the loop if possible
                x = x.squeeze()
                y = y.squeeze()
                text_marker = text(x=x, y=y, text=texts[i], **kwargs)
                zone_axes_label_list.append(text_marker)

        return zone_axes_label_list
