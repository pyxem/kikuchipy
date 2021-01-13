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

from typing import Union

from hyperspy.utils.markers import line_segment, point, text
import numpy as np


def get_line_segment_list(lines: Union[list, np.ndarray], **kwargs) -> list:
    """Return a list of line segment markers.

    Parameters
    ----------
    lines
        On the form [[x00, y00, x01, y01], [x10, y10, x11, y11], ...].
    kwargs
        Keyword arguments allowed by :func:`matplotlib.pyplot.axvline`.

    Returns
    -------
    marker_list : list
        List of
        :class:`hyperspy.drawing._markers.line_segment.LineSegment`.
    """
    lines = np.asarray(lines)
    if lines.ndim == 1:  # No navigation shape, one line
        lines = lines[np.newaxis, ...]

    marker_list = []
    for i in range(lines.shape[-2]):  # Iterate over bands
        if not np.allclose(lines[..., i, :], np.nan, equal_nan=True):
            x1 = lines[..., i, 0]
            y1 = lines[..., i, 1]
            x2 = lines[..., i, 2]
            y2 = lines[..., i, 3]
            marker_list.append(
                line_segment(x1=x1, y1=y1, x2=x2, y2=y2, **kwargs)
            )
    return marker_list


def get_point_list(points: Union[list, np.ndarray], **kwargs) -> list:
    """Return a list of point markers.

    Parameters
    ----------
    points
        On the form [[x0, y0], [x1, y1], ...].
    kwargs
        Keyword arguments allowed by :func:`matplotlib.pyplot.scatter`.

    Returns
    -------
    marker_list : list
        List of :class:`hyperspy.drawing._markers.point.Point`.
    """
    points = np.asarray(points)
    if points.ndim == 1:
        points = points[np.newaxis, ...]

    marker_list = []
    for i in range(points.shape[-2]):  # Iterate over zone axes
        if not np.allclose(points[..., i, :], np.nan, equal_nan=True):
            marker_list.append(
                point(x=points[..., i, 0], y=points[..., i, 1], **kwargs)
            )
    return marker_list


def get_text_list(
    texts: Union[list, np.ndarray],
    coordinates: Union[np.ndarray, list],
    **kwargs,
) -> list:
    """Return a list of text markers.

    Parameters
    ----------
    texts
        A list of texts.
    coordinates
        On the form [[x0, y0], [x1, y1], ...].
    kwargs
        Keyword arguments allowed by :func:`matplotlib.pyplot.axvline.`

    Returns
    -------
    marker_list : list
        List of :class:`hyperspy.drawing._markers.text.Text`.
    """
    coordinates = np.asarray(coordinates)
    if coordinates.ndim == 1:
        coordinates = coordinates[np.newaxis, ...]

    marker_list = []
    is_finite = np.isfinite(coordinates)[..., 0]
    coordinates[~is_finite] = -1
    for i in range(coordinates.shape[-2]):  # Iterate over zone axes
        if not np.allclose(coordinates[..., i, :], -1):  # All NaNs
            x = coordinates[..., i, 0]
            y = coordinates[..., i, 1]
            x[~is_finite[..., i]] = np.nan
            y[~is_finite[..., i]] = np.nan
            text_marker = text(x=x, y=y, text=texts[i], **kwargs,)
            # TODO: Pass "visible" parameter to text() when HyperSpy allows
            #  it (merges this PR
            #  https://github.com/hyperspy/hyperspy/pull/2558 and publishes
            #  a minor release with that update)
            # text_marker = text(
            #     x=coordinates[..., i, 0],
            #     y=coordinates[..., i, 1],
            #     text=texts[i],
            #     visible=is_finite[..., i],
            #     **kwargs
            # )
            marker_list.append(text_marker)
    return marker_list
