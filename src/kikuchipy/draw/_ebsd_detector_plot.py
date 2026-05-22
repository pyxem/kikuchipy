#
# Copyright 2019-2026 the kikuchipy developers
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
#

"""Plotting an EBSD detector and sample/detector geometry."""

from functools import cache
from typing import Any, Literal

from matplotlib import ticker
import matplotlib.axes as maxes
import matplotlib.figure as mfigure
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from kikuchipy.detectors._ebsd_detector import EBSDDetector

# Repeated in detector module: keep in sync!
DETECTOR_PLOT_FORMATS = Literal["detector", "gnomonic"]

BEAM_COLOR = "C2"
SAMPLE_COLOR = "C0"
PC_COLOR = "C1"
DETECTOR_COLOR = "C7"


@cache
def get_default_gnomonic_angles() -> np.ndarray:
    return np.linspace(10, 80, 8)


def get_gnomonic_circles(
    pcx: float, pcy: float, angles: np.ndarray | list[float], **kwargs
) -> list[mpatches.Circle]:
    """Return circle patches for gnomonic circles given by *angles* to
    plot on top of the EBSD detector.
    """
    tan_angles = np.tan(np.deg2rad(angles))
    default_kwds = {"alpha": 0.4, "edgecolor": "k", "facecolor": "None", "linewidth": 3}
    for k, v in default_kwds.items():
        kwargs.setdefault(k, v)
    circles = []
    for tan_angle in tan_angles:
        circle = mpatches.Circle((pcx, pcy), tan_angle, **kwargs)
        circles.append(circle)
    return circles


@cache
def get_default_projection_center_scatter_kwargs(
    s: float, zorder: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    circle_kwargs = {
        "marker": "o",
        "facecolor": "w",
        "edgecolor": "k",
        "label": "_pc_circle",
        "s": s,
        "zorder": zorder,
    }
    cross_kwargs = {
        "marker": "x",
        "color": "k",
        "linewidth": 1,
        "label": "_pc_cross",
        "s": 0.5 * s,
        "zorder": zorder,
    }
    return circle_kwargs, cross_kwargs


def plot_ebsd_detector(
    detector: EBSDDetector,
    coords_fmt: DETECTOR_PLOT_FORMATS,
    zoom: float,
    show_pc: bool,
    draw_gnomonic_circles: bool,
    pattern: np.ndarray | None,
    pattern_kwargs: dict,
    pc_kwargs: dict,
    gnomonic_circles_kwargs: dict,
    gnomonic_angles: np.ndarray | list[float] | None,
    ax: maxes.Axes | None = None,
) -> mfigure.Figure | mfigure.SubFigure:
    sy, sx = detector.shape
    if isinstance(pattern, np.ndarray) and pattern.shape != (sy, sx):
        raise ValueError(
            f"Pattern shape {pattern.shape} must equal the detector shape {(sy, sx)}"
        )

    fig, axis = set_up_figure_axis(ax=ax)

    # Y goes down for detector coordinates and up for gnomonic
    # coordinates
    if coords_fmt == "detector":
        pcx, pcy = detector.pc_average[:2]
        # Avoid multiplying by 0 if a default detector is used (of shape
        # (1, 1))
        pcy *= max([sy - 1, 1])
        pcx *= max([sx - 1, 1])
        xmin, xmax, ymax, ymin = detector.bounds
        xmax = max([xmax, 1])
        ymin = max([ymin, 1])
        x_label = "Detector X"
        y_label = "Detector Y"
    else:  # gnomonic
        pcx, pcy = (0, 0)
        xmin, xmax, ymin, ymax = detector._average_gnomonic_bounds
        x_label = "Gnomonic X"
        y_label = "Gnomonic Y"
        formatter = ticker.StrMethodFormatter("{x:.2f}")
        axis.xaxis.set_major_formatter(formatter)
        axis.yaxis.set_major_formatter(formatter)

    axis.set(
        xlabel=x_label,
        ylabel=y_label,
        xlim=(xmin * zoom, xmax * zoom),
        ylim=(ymin * zoom, ymax * zoom),
        aspect="equal",
    )

    if isinstance(pattern, np.ndarray):
        pattern_kwargs.setdefault("cmap", "gray")
        axis.imshow(pattern, extent=(xmin, xmax, ymin, ymax), **pattern_kwargs)
    else:
        # Draw a rectangle so that the detector bounds are visible even
        # after zooming out
        rect = mpatches.Rectangle(
            xy=(xmin, ymin),
            width=xmax - xmin,
            height=ymax - ymin,
            fc=DETECTOR_COLOR,
            zorder=-1,
        )
        axis.add_artist(rect)

    if show_pc:
        s = pc_kwargs.get("s", 100)
        zorder = pc_kwargs.get("zorder", 10)
        circle_kwargs, cross_kwargs = get_default_projection_center_scatter_kwargs(
            s=s, zorder=zorder
        )
        axis.scatter(x=pcx, y=pcy, **circle_kwargs)
        axis.scatter(x=pcx, y=pcy, **cross_kwargs)

    if draw_gnomonic_circles:
        if gnomonic_angles is None:
            gnomonic_angles = get_default_gnomonic_angles()
        circles = get_gnomonic_circles(
            pcx=pcx, pcy=pcy, angles=gnomonic_angles, **gnomonic_circles_kwargs
        )
        for circle in circles:
            axis.add_patch(circle)

    return fig


def plot_detector_sample_geometry_side_view(
    detector: EBSDDetector,
    ax: maxes.Axes | None = None,
    legend: bool = False,
    dimensionless: bool = False,
    **kwargs,
) -> mfigure.Figure | mfigure.SubFigure:
    """See the docstring of the EBSD detector method using this
    function.

    Coordinates are calculated in microns relative to the beam-sample
    interaction volume (0, 0).

    The figure plane is given by the microscope reference frame: Y west
    and Z north. Coordinates for elements are transformed to this
    reference frame after they are defined.
    """
    fig, ax = set_up_figure_axis(ax=ax, **kwargs)

    # Dimensions in mm and angles in radians
    height = detector.height
    L = detector.specimen_scintillator_distance[0]
    beam_length = height * 0.5
    half_sample_length = height * 0.3
    sigma = np.deg2rad(detector.sample_tilt)
    theta = np.deg2rad(detector.tilt)

    # Beam
    beam_end = np.zeros(2, dtype=np.float64)
    beam_start = np.array([0, -beam_length], dtype=np.float64)

    # Sample
    dx_sample = half_sample_length * np.cos(sigma)
    dy_sample = half_sample_length * np.sin(sigma)
    sample_end = np.array([dx_sample, dy_sample], dtype=np.float64)
    sample_start = -sample_end

    # Detector screen
    detector_z = np.array([np.cos(theta), np.sin(theta)])
    P = L * detector_z
    detector_y = np.array([-np.sin(theta), np.cos(theta)])

    # Screen extent
    pcy = detector.pc_average[1]
    screen_top = P - pcy * height * detector_y
    screen_bottom = P + (1 - pcy) * detector.height * detector_y

    # Negate Z so the Z axis points upward (opposite beam direction)
    def trans(point: np.ndarray) -> np.ndarray:
        p = point * 1e-3
        p[1] = -p[1]
        return p

    beam_start = trans(beam_start)
    beam_end = trans(beam_end)
    sample_start = trans(sample_start)
    sample_end = trans(sample_end)
    detector_start = trans(screen_top)
    detector_end = trans(screen_bottom)
    pc_pos = trans(P)
    source_pos = trans(np.zeros(2, dtype=np.float64))

    ax.annotate(
        "",
        xy=(beam_end[0], beam_end[1]),
        xytext=(beam_start[0], beam_start[1]),
        arrowprops={"arrowstyle": "->", "lw": 2, "color": BEAM_COLOR},
        zorder=5,
    )
    ax.plot(
        [sample_start[0], sample_end[0]],
        [sample_start[1], sample_end[1]],
        c=SAMPLE_COLOR,
        lw=6,
        label="Sample",
        zorder=4,
    )
    pc_circle_kwargs, pc_cross_kwargs = get_default_projection_center_scatter_kwargs(
        s=75, zorder=10
    )
    ax.scatter(x=pc_pos[0], y=pc_pos[1], **pc_circle_kwargs)
    ax.scatter(x=pc_pos[0], y=pc_pos[1], **pc_cross_kwargs)
    ax.plot(
        [detector_start[0], detector_end[0]],
        [detector_start[1], detector_end[1]],
        c=DETECTOR_COLOR,
        lw=4,
        label="Detector",
        zorder=4,
    )
    ax.plot(
        [source_pos[0], pc_pos[0]],
        [source_pos[1], pc_pos[1]],
        linestyle=":",
        color=BEAM_COLOR,
        alpha=0.5,
        zorder=0,
    )

    # Handle axis orientation: Y axis as x-coordinate (inverted so
    # detector appears on the left), Z as y-coordinate pointing up
    ax.set_aspect("equal")

    # Fix axis limits
    pad = get_axis_limit_pad(height)
    ax.set_xlim(pad, -pad)
    ax.set_ylim(-pad, pad)

    if legend:
        beam_handle = mlines.Line2D([], [], color=BEAM_COLOR, label="Beam")
        ax.legend(
            handles=[beam_handle, *ax.get_legend_handles_labels()[0]],
            loc="best",
            borderpad=max([plt.rcParams["legend.borderpad"], 0.6]),
        )

    if dimensionless:
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        unit = ""
    else:
        unit = " [mm]"
    ax.set_xlabel(f"\u2190 Microscope Y{unit}")
    ax.set_ylabel(f"Microscope Z{unit} \u2192")

    return fig


def update_detector_sample_geometry_side_view(
    detector: EBSDDetector,
    ax: maxes.Axes,
    legend: bool = False,
    dimensionless: bool = True,
) -> None:
    """Clear *ax* and redraw the side view for *detector*."""
    ax.clear()
    plot_detector_sample_geometry_side_view(
        detector, ax=ax, legend=legend, dimensionless=dimensionless
    )


def plot_detector_sample_geometry_top_view(
    detector: EBSDDetector,
    ax: maxes.Axes | None = None,
    legend: bool = False,
    dimensionless: bool = False,
    **kwargs,
) -> mfigure.Figure | mfigure.SubFigure:
    """See the docstring of the EBSD detector method using this function
    for further details.

    Coordinates are calculated in microns relative to the beam-sample
    interaction volume (0, 0).

    The figure plane is given by the microscope reference frame: X west
    and Y south. Coordinates for elements are transformed to this
    reference frame after they are defined.
    """
    fig, ax = set_up_figure_axis(ax=ax, **kwargs)

    # Dimensions in microns and angles in radians
    width = detector.width
    L = detector.specimen_scintillator_distance[0]
    sample_length = width * 0.6
    azimuthal = np.deg2rad(detector.azimuthal)

    beam = np.zeros(2)

    # PC in microscope X-Y coordinates, with the screen normal pointing
    # towards the interaction volume
    det_normal = np.array([np.sin(azimuthal), np.cos(azimuthal)])
    pc_pos = L * det_normal

    # Unit vector along the detector width, perpendicular to the normal
    width_dir = np.array([-np.cos(azimuthal), np.sin(azimuthal)])

    # Detector endpoints
    pcx = detector.pc_average[0]
    left_extent = pcx * width
    right_extent = (1 - pcx) * width
    det_left = pc_pos - left_extent * width_dir
    det_right = pc_pos + right_extent * width_dir

    # Sample surface: a line along X at Y=0
    sample_start = np.array([-sample_length / 2, 0])
    sample_end = np.array([sample_length / 2, 0])

    to_mm = 1e-3

    ax.scatter(
        beam[0] * to_mm,
        beam[1] * to_mm,
        marker="o",
        s=70,
        fc=BEAM_COLOR,
        ec="k",
        zorder=5,
        label="Beam",
    )
    ax.plot(
        [sample_start[0] * to_mm, sample_end[0] * to_mm],
        [sample_start[1] * to_mm, sample_end[1] * to_mm],
        c=SAMPLE_COLOR,
        lw=6,
        label="Sample",
        zorder=4,
    )
    pc_circle_kwargs, pc_cross_kwargs = get_default_projection_center_scatter_kwargs(
        s=75, zorder=10
    )
    ax.scatter(x=pc_pos[0] * to_mm, y=pc_pos[1] * to_mm, **pc_circle_kwargs)
    ax.scatter(x=pc_pos[0] * to_mm, y=pc_pos[1] * to_mm, **pc_cross_kwargs)
    ax.plot(
        [det_left[0] * to_mm, det_right[0] * to_mm],
        [det_left[1] * to_mm, det_right[1] * to_mm],
        c=DETECTOR_COLOR,
        lw=4,
        label="Detector",
        zorder=4,
    )
    ax.plot(
        [beam[0] * to_mm, pc_pos[0] * to_mm],
        [beam[1] * to_mm, pc_pos[1] * to_mm],
        linestyle=":",
        color=BEAM_COLOR,
        alpha=0.5,
        zorder=0,
    )

    ax.set_aspect("equal")

    # Fix axis limits
    pad = get_axis_limit_pad(width)
    ax.set_xlim(pad, -pad)
    ax.set_ylim(pad, -pad)

    if legend:
        ax.legend(loc="best")

    if dimensionless:
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        unit = ""
    else:
        unit = " [mm]"
    ax.set_xlabel(f"\u2190 Microscope X{unit}")
    ax.set_ylabel(f"Microscope Y{unit} \u2190")

    return fig


def get_axis_limit_pad(width: float) -> float:
    """Return sufficient axis limits for the detector to move around
    without moving outside the axis.
    """
    width_mm = width * 1e-3
    pad = 1.1 * width_mm
    return pad


def update_detector_sample_geometry_top_view(
    detector: EBSDDetector,
    ax: maxes.Axes,
    legend: bool = False,
    dimensionless: bool = True,
) -> None:
    """Clear *ax* and redraw the top view for *detector*."""
    ax.clear()
    plot_detector_sample_geometry_top_view(
        detector, ax=ax, legend=legend, dimensionless=dimensionless
    )


def update_detector_plane(
    detector: EBSDDetector,
    ax: maxes.Axes,
    coords_fmt: DETECTOR_PLOT_FORMATS,
    zoom: float,
    draw_gnomonic_circles: bool | None = None,
    show_pc: bool = True,
) -> None:
    """Clear *ax* and redraw the detector plane for *detector*."""
    if draw_gnomonic_circles is None:
        draw_gnomonic_circles = coords_fmt == "gnomonic"
    ax.clear()
    plot_ebsd_detector(
        detector,
        coords_fmt=coords_fmt,
        zoom=zoom,
        show_pc=show_pc,
        draw_gnomonic_circles=draw_gnomonic_circles,
        pattern=None,
        pattern_kwargs={},
        pc_kwargs={},
        gnomonic_circles_kwargs={},
        gnomonic_angles=None,
        ax=ax,
    )


def set_up_figure_axis(
    ax: maxes.Axes | None = None, **kwargs
) -> tuple[mfigure.Figure | mfigure.SubFigure, maxes.Axes]:
    if ax is None:
        w, h = plt.rcParams["figure.figsize"]
        kwargs.setdefault("layout", "constrained")
        kwargs.setdefault("figsize", (w, h))
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    return fig, ax
