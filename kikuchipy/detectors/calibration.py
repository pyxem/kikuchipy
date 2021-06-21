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

"""Calibration of the EBSD projection/pattern center."""

from itertools import combinations
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


class PCCalibrationMovingScreen:
    """A class to perform and inspect the calibration of the EBSD
    projection center (PC) using the "moving screen" technique from
    :cite:`hjelen1991electron`.

    The technique requires two patterns acquired with a stationary beam
    but with different detector distances (DDs) where the difference is
    known. First, the goal is to find the pattern region which does not
    shift between the two camera positions, (PCx, PCy). This point can
    be estimated by selecting the same pattern features in both
    patterns. Second, the DD (PCz) can be estimated in the same unit as
    the known camera distance difference. If also the detector pixel
    size is known, PCz can be given in the fraction of the detector
    screen height.
    """

    def __init__(
        self,
        pattern_in: np.ndarray,
        pattern_out: np.ndarray,
        points_in: Union[np.ndarray, List[Tuple[float]]],
        points_out: Union[np.ndarray, List[Tuple[float]]],
        delta_z: float = 1,
        px_size: Optional[float] = None,
        binning: int = 1,
        convention: str = "tsl",
    ):
        r"""Return a class instance storing the PC estimates, the
        average PC, and other parameters relevant for the estimation.

        Parameters
        ----------
        pattern_in
            Pattern acquired with the shortest detector distance (DD) in
            the "in" position.
        pattern_out
            Pattern acquired with the longer DD in the "out" position,
            with the camera a known distance `delta_z` from the "in"
            position.
        points_in
            Set of :math:`n` coordinates [(x1, y1), (x2, y2), ...] of
            pattern features in `pattern_in`.
        points_out
            Set of :math:`n` coordinates [(x1, y1), (x2, y2), ...] of
            pattern features, the same as in `points_in`, in
            `pattern_out`. They must be in the same order as in
            `points_in`.
        delta_z
            Known distance between the "in" and "out" camera positions
            in which the `pattern_in` and `pattern_out` were acquired,
            respectively. Default is 1. The output PCz value will be in
            the same unit as this value, unless `px_size` is provided.
        px_size
            Known size of the detector pixels, in the same unit as
            `delta_z`. If this is None (default), the PCz will not be
            scaled to fractions of detector height.
        binning
            Detector pixel binning. Default is 1, meaning no binning.
            This is used together with `px_size` to scale PCz.
        convention
            Whether to present PCy as the value from bottom to top
            (TSL), or top to bottom (Bruker). Default is "tsl".
        """
        self.patterns = np.stack([pattern_in, pattern_out])
        self.points = np.stack([points_in, points_out])

        self.delta_z = delta_z
        self.px_size = px_size
        self.binning = binning
        self.convention = convention

        self.make_lines()

    @property
    def shape(self) -> Tuple[int, int]:
        """Detector shape, (nrows, ncols)."""
        return self.patterns[0].shape

    @property
    def nrows(self) -> int:
        """Number of detector rows."""
        return self.shape[0]

    @property
    def ncols(self) -> int:
        """Number of detector columns."""
        return self.shape[1]

    @property
    def n_points(self) -> int:
        """Number of points of pattern features in each pattern."""
        return len(self.points[0])

    @property
    def lines(self) -> np.ndarray:
        """Start and end points of all possible lines between all points
        per pattern, of shape (2, n_lines, 4), where the last axis is
        (x1, y1, x2, y2).
        """
        return self._lines

    @property
    def n_lines(self) -> int:
        """Number of lines in each pattern."""
        return len(self.lines[0])

    @property
    def lines_start(self) -> np.ndarray:
        """Starting points of lines within the patterns, of shape
        (2, n_lines, 2).
        """
        return np.stack([self.lines[0, :, :2], self.lines[1, :, :2]])

    @property
    def lines_end(self) -> np.ndarray:
        """End points of lines within both patterns, of shape
        (2, n_lines, 2).
        """
        return np.stack([self.lines[0, :, 2:], self.lines[1, :, 2:]])

    @property
    def line_lengths(self) -> np.ndarray:
        """Length of lines within the patterns in pixels."""
        length_in = _line_lengths(self.lines_start[0], self.lines_end[0])
        length_out = _line_lengths(self.lines_start[1], self.lines_end[1])
        return np.stack([length_in, length_out])

    @property
    def lines_out_in(self) -> np.ndarray:
        """Start (out) and end (in) points of the lines between
        corresponding points in the patterns, of shape (n_points, 4).
        """
        return np.hstack([self.points[1], self.points[0]])

    @property
    def lines_out_in_start(self) -> np.ndarray:
        """Starting points of the lines between corresponding points in
        the patterns, of shape (n_points, 2).
        """
        return self.lines_out_in[:, :2]

    @property
    def lines_out_in_end(self) -> np.ndarray:
        """End points of the lines between corresponding points in the
        patterns, of shape (n_points, 2).
        """
        return self.lines_out_in[:, 2:]

    @property
    def _pxy_all(self) -> np.array:
        l_iter = combinations(range(self.n_points), 2)
        l = self.lines_out_in
        return np.array([_get_intersection_from_lines(l[i], l[j]) for i, j in l_iter])

    @property
    def pxy_within_detector(self) -> np.ndarray:
        """A boolean array stating whether each intersection of lines
        between corresponding points in the patterns are inside the
        detector (True), or outside (False).
        """
        px_all = self._pxy_all[:, 0]
        py_all = self._pxy_all[:, 1]
        return np.logical_and(
            np.logical_and(px_all > 0, px_all < self.ncols),
            np.logical_and(py_all > 0, py_all < self.nrows),
        )

    @property
    def pxy_all(self) -> np.ndarray:
        """Intersections of the lines between the corresponding points
        in the patterns, i.e. estimates of (PCx, PCy), of shape
        (n_points, 2).
        """
        return self._pxy_all[self.pxy_within_detector]

    @property
    def pxy(self) -> float:
        """Average of intersections of the lines between corresponding
        points in the patterns.
        """
        return np.nanmean(self.pxy_all, axis=0)

    @property
    def pcx_all(self) -> np.ndarray:
        """All estimates of PCx."""
        return self.pxy_all[:, 0] / self.ncols

    @property
    def pcy_all(self) -> np.ndarray:
        """All estimates of PCy."""
        pcy_all = self.pxy_all[:, 1] / self.nrows
        if self.convention == "tsl":
            pcy_all = 1 - pcy_all
        return pcy_all

    @property
    def pcz_all(self) -> np.ndarray:
        """All estimates of PCz, scaled to fraction of detector height
        if `px_size` is not None.
        """
        line_lengths = self.line_lengths
        pcz = self.delta_z / ((line_lengths[1] / line_lengths[0]) - 1)
        if self.px_size is not None:
            pcz /= self.nrows * self.px_size * self.binning
        return pcz[self.pxy_within_detector]

    @property
    def pc_all(self) -> np.ndarray:
        """All estimates of PC."""
        return np.column_stack([self.pcx_all, self.pcy_all, self.pcz_all])

    @property
    def pc(self) -> np.ndarray:
        """The average PC calculated from all estimates."""
        return np.nanmean(self.pc_all, axis=0)

    def make_lines(self):
        """Draw lines between all points within a pattern and populate
        `self.lines`. Is first run upon initialization.
        """
        lines_in = _construct_lines_between_points(self.points[0])
        lines_out = _construct_lines_between_points(self.points[1])
        self._lines = np.stack([lines_in, lines_out])

    def plot(
        self,
        pattern_kwargs: dict = dict(cmap="gray"),
        line_kwargs: dict = dict(linewidth=2, zorder=1),
        scatter_kwargs: dict = dict(zorder=2),
        pc_kwargs: dict = dict(marker="*", s=300, facecolor="gold", edgecolor="k"),
        return_fig_ax: bool = False,
        **kwargs: dict,
    ) -> Union[None, Tuple[plt.Figure, List[plt.Axes]]]:
        """A convenience method of three images, the first two with the
        patterns with points and lines annotated, and the third with the
        calibration results.

        Parameters
        ----------
        pattern_kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.imshow`.
        line_kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.axline`.
        scatter_kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.scatter`.
        pc_kwargs
            Keyword arguments, along with `scatter_kwargs`, passed to
            :meth:`matplotlib.axes.Axes.scatter` when plotting the PCs.
        return_fig_ax
            Whether to return the figure and axes, default is False.
        kwargs
            Keyword arguments passed to
            :func:`matplotlib.pyplot.subplots`.

        Returns
        -------
        fig
            Figure, returned if `return_fig_ax` is True.
        ax
            Axes, returned if `return_fig_ax` is True.
        """
        pat1, pat2 = self.patterns
        points1, points2 = self.points
        px = self.pxy[0]
        py = self.pxy[1]
        pxy_all = self.pxy_all
        n_lines = self.n_lines
        lines_start = self.lines_start
        lines_end = self.lines_end
        lines_out_in_start = self.lines_out_in_start
        lines_out_in_end = self.lines_out_in_end

        ncols = 3
        for k, v in zip(["sharex", "sharey", "figsize"], [True, True, (20, 10)]):
            kwargs.setdefault(k, v)
        fig, ax = plt.subplots(ncols=ncols, **kwargs)
        ax[0].set_title("In (operating) position")
        ax[0].imshow(pat1, **pattern_kwargs)
        ax[0].scatter(points1[:, 0], points1[:, 1], **scatter_kwargs)
        for i in range(n_lines):
            start, end = lines_start[0, i], lines_end[0, i]
            ax[0].axline(start, end, linestyle="-", color=f"C{i}", **line_kwargs)

        ax[1].set_title("Out position")
        ax[1].imshow(pat2, **pattern_kwargs)
        ax[1].scatter(points2[:, 0], points2[:, 1], color="C1", **scatter_kwargs)
        for i in range(n_lines):
            start, end = lines_start[1, i], lines_end[1, i]
            ax[1].axline(start, end, linestyle="--", color=f"C{i}", **line_kwargs)

        ax[2].set_title("Projection center")
        ax[2].imshow(np.ones(self.shape), cmap="gray", vmin=0, vmax=2)
        ax[2].scatter(points1[:, 0], points1[:, 1], **scatter_kwargs)
        ax[2].scatter(points2[:, 0], points2[:, 1], **scatter_kwargs)
        ax[2].scatter(
            pxy_all[:, 0], pxy_all[:, 1], color="k", marker="*", **scatter_kwargs
        )
        for i in range(self.n_points):
            start, end = lines_out_in_start[i], lines_out_in_end[i]
            ax[2].axline(start, end, color=f"C{i}", **line_kwargs)

        for i in range(ncols):
            ax[i].scatter(px, py, **pc_kwargs, **scatter_kwargs)

        if return_fig_ax:
            return fig, ax

    def __repr__(self):
        name = self.__class__.__name__
        points = np.array_str(self.points, precision=0)
        pcx, pcy, pcz = self.pc
        return (
            f"{name}: (PCx, PCy, PCz) = ({pcx:.4f}, {pcy:.4f}, {pcz:.4f})\n"
            f"{self.n_points} points:\n{points}"
        )


def _get_intersection_from_lines(
    line1: Union[List[int], np.ndarray],
    line2: Union[List[int], np.ndarray],
) -> Tuple[float, float]:
    """line: [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    a1 = np.array([[x1, y1], [x2, y2]])
    a2 = np.array([[x3, y3], [x4, y4]])
    a3 = np.array([[x1, 1], [x2, 1]])
    a4 = np.array([[x3, 1], [x4, 1]])
    a5 = np.array([[y1, 1], [y2, 1]])
    a6 = np.array([[y3, 1], [y4, 1]])
    det_a1 = np.linalg.det(a1)
    det_a2 = np.linalg.det(a2)
    det_a3 = np.linalg.det(a3)
    det_a4 = np.linalg.det(a4)
    det_a5 = np.linalg.det(a5)
    det_a6 = np.linalg.det(a6)
    denom = np.linalg.det([[det_a3, det_a5], [det_a4, det_a6]])
    px = np.linalg.det([[det_a1, det_a3], [det_a2, det_a4]]) / denom
    py = np.linalg.det([[det_a1, det_a5], [det_a2, det_a6]]) / denom
    return px, py


def _line_lengths(lines_start: np.ndarray, lines_end: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(lines_start - lines_end), axis=1))


def _construct_lines_between_points(points: np.ndarray) -> np.ndarray:
    combs = list(combinations(range(len(points)), 2))
    start = np.zeros((len(combs), 2))
    end = np.zeros_like(start)
    for i, (s, e) in enumerate(combs):
        start[i] = points[s]
        end[i] = points[e]
    return np.column_stack([start, end])
