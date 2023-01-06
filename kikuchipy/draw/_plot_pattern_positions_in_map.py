# Copyright 2019-2023 The kikuchipy developers
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

"""Functions for plotting positions of patterns in a 2D map.

These functions were created specifically to plot positions of NORDIF
calibration patterns. These are acquired manually within a region of
interest within a larger area.
"""

from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def plot_pattern_positions_in_map(
    rc: np.ndarray,
    roi_shape: tuple,
    roi_origin: tuple = (0, 0),
    area_shape: Optional[tuple] = None,
    roi_image: Optional[np.ndarray] = None,
    area_image: Optional[np.ndarray] = None,
    axis: Optional[plt.Axes] = None,
    return_figure: bool = False,
    color: Optional[str] = "k",
) -> Optional[plt.Figure]:
    """Plot pattern positions in a 2D map within a region of interest
    (ROI), the ROI potentially within a larger area.

    Parameters
    ----------
    rc
        Position coordinates (row, column) in an array of shape (n, 2).
        If ``area_shape`` is passed, coordinates are assumed to be given
        with respect to the area origin, and so if ``roi_origin`` is
        passed, the origin is subtracted from the coordinates.
    roi_shape
        Shape of the ROI as (n rows, n columns).
    roi_origin
        Origin (row, column) of the ROI with respect to the area. If
        this and ``area_shape`` is passed, the origin is subtracted
        from the ``rc`` coordinates.
    area_shape
        Shape of the area including the ROI as (n rows, n columns). If
        this and ``roi_origin`` is passed, the origin is subtracted from
        the ``rc`` coordinates.
    roi_image
        Image to plot within the ROI, of the same aspect ratio.
    area_image
        Image to plot within the area, of the same aspect ratio.
    axis
        Existing Matplotlib axis to add the positions to. If not passed,
        a new figure will be created. If passed, only the coordinate
        markers and labels are added to the axis. E.g. ``roi_image`` or
        ``area_image`` will not be used.
    return_figure
        Whether to return the created figure. Default is ``False``.
    color
        Color of position markers and labels. Default is ``k``. Must be
        a valid Matplotlib color identifier.

    Returns
    -------
    fig
        Created figure, returned if ``return_figure=True``.
    """
    # Set different styling of rectangle(s), markers and text. Default
    # values assume no area_shape or images are passed.
    roi_rect_kw = dict(fc="none", lw=2, clip_on=False, zorder=5)
    roi_ny, roi_nx = roi_shape

    if isinstance(axis, plt.Axes):
        new_axis = False
        ax = axis
        fig = ax.figure
    else:
        new_axis = True
        fig, ax = plt.subplots()

    if new_axis and area_shape:
        area_ny, area_nx = area_shape
        roi_rect_kw.update(dict(ec="r", fc="none"))
        if isinstance(area_image, np.ndarray):
            area_image_flipped = area_image[::-1, :]  # Inverse y-axis
            ax.imshow(area_image_flipped, extent=(0, area_nx, 0, area_ny))
        ax.text(0, 0, s="Area", va="bottom")
        xlim = (0, area_nx)
        ylim = (0, area_ny)
    else:
        if roi_origin != (0, 0):
            rc = rc.copy()
            rc -= np.array(roi_origin)
            roi_origin = (0, 0)
        xlim = (0, roi_nx)
        ylim = (0, roi_ny)

    if new_axis:
        if isinstance(roi_image, np.ndarray):
            extent = (
                roi_origin[1],
                roi_origin[1] + roi_nx,
                roi_origin[0] + roi_ny,
                roi_origin[0],
            )
            ax.imshow(roi_image, extent=extent, zorder=1)
            roi_rect_kw.update(dict(fc="none"))

        rect_roi = mpatches.Rectangle(roi_origin[::-1], roi_nx, roi_ny, **roi_rect_kw)
        if area_shape:
            ax.text(*rect_roi.xy, s="Region of interest", va="bottom")
            ax.add_artist(rect_roi)

    for i, (r, c) in enumerate(rc):
        ax.scatter(c, r, s=50, marker="+", c=color)
        ax.text(c, r, str(i), va="bottom", ha="left", c=color)

    if new_axis:
        ax.set(xlim=xlim, ylim=ylim, xlabel="Column", ylabel="Row", aspect="equal")
        ax.invert_yaxis()

    if return_figure:
        return fig
