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

from collections import defaultdict
from re import sub
from typing import List, Optional, Tuple, Union
import warnings

from diffsims.crystallography import ReciprocalLatticePoint
import matplotlib
import numpy as np
from orix.crystal_map import Phase
from orix.quaternion.rotation import Rotation

from kikuchipy.detectors import EBSDDetector
from kikuchipy.draw.markers import (
    get_line_segment_list,
    get_point_list,
    get_text_list,
)
from kikuchipy.draw.colors import TABLEAU_COLORS, TSL_COLORS
from kikuchipy.simulations.features import KikuchiBand, ZoneAxis


class GeometricalEBSDSimulation:
    """Geometrical EBSD simulation with Kikuchi bands and zone axes."""

    exclude_outside_detector = True

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
        za_coords
            Column sorted, on the form [[x0, y0], [x1, y1], ...].
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
        coordinates.

        Returns
        -------
        np.ndarray
            Column sorted, on the form [[x0, y0], [x1, y1], ...].
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
            A list of colors, either as RGB iterables or colors
            recognizable by Matplotlib, used to color each unique family
            of bands. If None (default), this is determined from a list
            similar to the one used in EDAX TSL's software.
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

        # Get dictionaries of families and in which a band belongs
        families, families_idx = _get_hkl_family(self.bands.hkl.data)

        # Get family colors
        # TODO: Perhaps move this outside this function (might be useful
        #  elsewhere)
        if family_colors is None:
            family_colors = []
            colors = _get_colors_for_allowed_bands(
                phase=self.bands.phase,
                highest_hkl=np.max(np.abs(self.bands._hkldata), axis=0),
                color_cycle=TSL_COLORS,
            )
            for hkl in families.keys():
                for table_hkl, color in colors:
                    if _is_equivalent(hkl, table_hkl):
                        family_colors.append(color)
                        break
                else:  # Hopefully we never arrive here
                    family_colors.append([1, 0, 0])

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
        """
        # TODO: Remove warning after HyperSpy merges this PR
        #  https://github.com/hyperspy/hyperspy/pull/2558 and publishes
        #  a minor release with that update
        if matplotlib._log.level < 40:
            warnings.warn(
                message=(
                    "Matplotlib will print log warnings when EBSD.plot() is "
                    "called due to zone axes NaN values, unless it's log level "
                    "is set to 'error' via `matplotlib.set_loglevel('error')`. "
                    "This will (hopefully) be fixed when HyperSpy releases a "
                    "minor version with this update: "
                    "https://github.com/hyperspy/hyperspy/pull/2558"
                ),
                category=UserWarning,
            )
        return get_text_list(
            texts=sub("[][ ]", "", str(self.zone_axes._hkldata)).split("\n"),
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

    @property
    def zone_axes_within_gnomonic_bounds(self) -> np.ndarray:
        """Return a boolean array with True for the zone axes within
        the detector's gnomonic bounds.

        Returns
        -------
        within_gnomonic_bounds
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

        return within_gnomonic_bounds

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


def _get_hkl_family(hkl: np.ndarray, reduce: bool = False) -> Tuple[dict, dict]:
    # TODO: Almost identical to
    #  diffsims.crystallography.ReciprocalLatticePoint.unique, improve
    #  this instead!
    # Remove [0, 0, 0] points
    hkl = hkl[~np.all(np.isclose(hkl, 0), axis=1)]
    families = defaultdict(list)
    families_idx = defaultdict(list)
    for i, this_hkl in enumerate(hkl.tolist()):
        for that_hkl in families.keys():
            if _is_equivalent(this_hkl, that_hkl, reduce=reduce):
                families[tuple(that_hkl)].append(this_hkl)
                families_idx[tuple(that_hkl)].append(i)
                break
        else:
            families[tuple(this_hkl)].append(this_hkl)
            families_idx[tuple(this_hkl)].append(i)
    n_families = len(families)
    unique_hkl = np.zeros((n_families, 3), dtype=int)
    for i, all_hkl_in_family in enumerate(families.values()):
        unique_hkl[i] = sorted(all_hkl_in_family)[-1]
    return families, families_idx


def _is_equivalent(
    this_hkl: list, that_hkl: list, reduce: bool = False
) -> bool:
    """Determine whether two Miller index 3-tuples are equivalent.
    Symmetry is not considered.
    """
    if reduce:
        this_hkl, _ = _reduce_hkl(this_hkl)
        that_hkl, _ = _reduce_hkl(that_hkl)
    return np.allclose(
        sorted(np.abs(this_hkl).astype(int)),
        sorted(np.abs(that_hkl).astype(int)),
    )


def _reduce_hkl(hkl: Union[list, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce Miller indices 3-tuples by a greatest common divisor."""
    hkl = np.atleast_2d(hkl)
    divisor = np.gcd.reduce(hkl, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hkl = hkl / divisor[:, np.newaxis]
    return hkl.astype(int), divisor


def _get_colors_for_allowed_bands(
    phase: Phase,
    highest_hkl: Union[List[int], np.ndarray, None] = None,
    color_cycle: Optional[List[str]] = None,
):
    """Return an array of Miller indices of allowed Kikuchi bands for a
    point group and a corresponding color.

    The idea with this function is to always get the same color for the
    same band in the same point group.

    Parameters
    ----------
    phase
        A phase container with a crystal structure and a space and point
        group describing the allowed symmetry operations.
    highest_hkl
        Highest Miller indices to consider. If None (default),
        [9, 9, 9] is used.
    color_cycle
        A list of color names recognized by Matplotlib. If None
        (default), the Matplotlib Tableau colors are cycled through.

    Returns
    -------
    hkl_color
        Array with Miller indices and corresponding color.
    """
    if highest_hkl is None:
        highest_hkl = [9, 9, 9]
    rlp = ReciprocalLatticePoint.from_highest_hkl(
        phase=phase, highest_hkl=highest_hkl,
    )

    rlp2 = rlp[rlp.allowed]
    # TODO: Replace this ordering with future ordering method in
    #  diffsims
    g_order = np.argsort(rlp2.gspacing)
    new_hkl = np.atleast_2d(rlp2._hkldata)[g_order]
    rlp3 = ReciprocalLatticePoint(phase=rlp.phase, hkl=new_hkl)
    hkl = np.atleast_2d(rlp3._hkldata)
    families, families_idx = _get_hkl_family(hkl=hkl, reduce=True)

    if color_cycle is None:
        color_cycle = TABLEAU_COLORS
    n_color_cycle = len(color_cycle)
    n_families = len(families)
    colors = np.tile(
        color_cycle, (int(np.ceil(n_families / n_color_cycle)), 1)
    )[:n_families]
    colors = [matplotlib.colors.to_rgb(i) for i in colors]

    hkl_colors = np.zeros(shape=(rlp3.size, 2, 3))
    for hkl_idx, color in zip(families_idx.values(), colors):
        hkl_colors[hkl_idx, 0] = hkl[hkl_idx]
        hkl_colors[hkl_idx, 1] = color

    return hkl_colors
